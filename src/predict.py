import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
import json
import logging
from tqdm import tqdm


try:

    from model import CRNNModel
    from dataset import MorseDataset, collate_fn

    def decode_ctc_greedy(preds, index_map, blank_idx=0):
        decoded_batch = []
        for seq in preds: # preds: [Batch, Time]
            collapsed_seq = torch.unique_consecutive(seq)
            decoded_seq = [index_map.get(idx.item(), '?') for idx in collapsed_seq if idx.item() != blank_idx]
            decoded_batch.append("".join(decoded_seq))
        return decoded_batch
except ImportError as e:
    print(f"Ошибка импорта: {e}. Убедитесь, что model.py и dataset.py доступны.")
    exit(1)

# --- Настройка Логгера ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Парсер Аргументов ---
def parse_args():
    parser = argparse.ArgumentParser(description="Предсказание Морзе с помощью CRNN модели")
    parser.add_argument('--model-path', type=str, required=True, help='Путь к чекпоинту модели (.pth)')
    parser.add_argument('--input-csv', type=str, required=True, help='Путь к входному CSV файлу (test.csv)')
    parser.add_argument('--output-csv', type=str, default='submission.csv', help='Путь для сохранения CSV с предсказаниями')
    parser.add_argument('--data-root', type=str, default='..', help='Корневая директория данных для путей в input_csv')
    parser.add_argument('--batch-size', type=int, default=64, help='Размер батча')
    parser.add_argument('--num-workers', type=int, default=0, help='Количество воркеров DataLoader (0 для надежного порядка)')
    parser.add_argument('--device', type=str, default='auto', help='Устройство ("cuda", "cpu", "auto")')
    return parser.parse_args()

# --- Основная Функция Предсказания ---
def predict():
    args = parse_args()

    # Определение устройства
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Используемое устройство: {device}")

    # Загрузка чекпоинта
    model_path = os.path.abspath(args.model_path)
    logger.info(f"Загрузка чекпоинта из: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Файл чекпоинта не найден: {model_path}"); return

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except Exception as e:
        logger.error(f"Ошибка загрузки чекпоинта: {e}", exc_info=True); return

    # Извлечение конфигурации и карт символов
    try:
        config = checkpoint['config']
        char_map = checkpoint['char_map']
        index_map = {v: k for k, v in char_map.items()}
        num_classes = config['num_classes']
        blank_idx = next((idx for char, idx in char_map.items() if char == '-'), 0)
        logger.info("Конфигурация модели и карты символов загружены.")
    except KeyError as e:
        logger.error(f"Ошибка: Отсутствуют ключи в чекпоинте ('config', 'char_map'): {e}"); return

    # Инициализация модели
    logger.info("Инициализация модели...")
    try:
        model = CRNNModel(
            n_mels=config['n_mels'],
            num_classes=num_classes,
            rnn_hidden_size=config['rnn_hidden'],
            rnn_layers=config['rnn_layers'],
            dropout=config.get('dropout', 0.1)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Перевод в режим оценки
        logger.info("Модель успешно инициализирована.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {e}", exc_info=True); return

    # Подготовка данных
    input_csv_path = os.path.abspath(args.input_csv)
    abs_data_root = os.path.abspath(args.data_root)
    logger.info(f"Чтение входного CSV: {input_csv_path}")
    logger.info(f"Используется data_root: {abs_data_root}")

    if not os.path.exists(input_csv_path):
        logger.error(f"Входной CSV файл не найден: {input_csv_path}"); return

    try:
        # Dataset в режиме инференса
        predict_dataset = MorseDataset(csv_path=input_csv_path, char_map=char_map,
                                       data_root=args.data_root, is_inference=True)
        if len(predict_dataset) == 0:
            logger.warning("Входной CSV файл пуст.")
            pd.DataFrame(columns=['wav_filename', 'morse_prediction']).to_csv(args.output_csv, index=False)
            logger.info(f"Создан пустой файл {args.output_csv}"); return

        # DataLoader
        predict_loader = DataLoader(predict_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, collate_fn=collate_fn,
                                    pin_memory=(device.type == 'cuda'))
        logger.info(f"Создан DataLoader: {len(predict_dataset)} сэмплов, {len(predict_loader)} батчей.")
    except Exception as e:
        logger.error(f"Ошибка при создании Dataset/DataLoader: {e}", exc_info=True); return

    # Цикл Предсказания
    logger.info("Начало предсказания...")
    all_predictions = []
    predict_pbar = tqdm(predict_loader, desc="Предсказание", unit="батч")

    with torch.no_grad():
        for batch_data in predict_pbar:
            spectrograms, _, spec_lengths, _ = batch_data
            if spectrograms.numel() == 0: continue # Пропуск пустого батча

            spectrograms = spectrograms.to(device)
            # spec_lengths остаются на CPU

            try:
                outputs = model(spectrograms) # [Time, Batch, Classes]
                log_probs = F.log_softmax(outputs, dim=2) # [Time, Batch, Classes]
                # Декодирование
                preds_argmax = log_probs.cpu().argmax(dim=2).permute(1, 0) # [Batch, Time]
                batch_predictions = decode_ctc_greedy(preds_argmax, index_map, blank_idx)
                all_predictions.extend(batch_predictions)
            except Exception as e:
                 logger.error(f"Ошибка во время обработки батча: {e}", exc_info=False)
                 all_predictions.extend(["ERROR"] * spectrograms.size(0)) # Добавляем заглушки

    logger.info(f"Предсказание завершено. Получено {len(all_predictions)} предсказаний.")

    # Формирование и сохранение submission.csv
    try:
        input_df = pd.read_csv(input_csv_path)
        if len(all_predictions) != len(input_df):
            logger.error(f"Количество предсказаний ({len(all_predictions)}) не совпадает с количеством строк в CSV ({len(input_df)}). Результат может быть некорректным.")
            # Попытка скорректировать длину списка для сохранения
            predictions_adjusted = (all_predictions + ["PAD_ERROR"] * len(input_df))[:len(input_df)]
            input_df['morse_prediction'] = predictions_adjusted
        else:
             input_df['morse_prediction'] = all_predictions

        # Сохраняем только нужные колонки
        output_columns = [col for col in ['wav_filename', 'spectrogram_path'] if col in input_df.columns] + ['morse_prediction']
        submission_df = input_df[output_columns]

        output_csv_path = os.path.abspath(args.output_csv)
        submission_df.to_csv(output_csv_path, index=False)
        logger.info(f"Файл с предсказаниями сохранен в: {output_csv_path}")

    except Exception as e:
        logger.error(f"Ошибка при формировании/сохранении submission файла: {e}", exc_info=True)

if __name__ == '__main__':
    predict()