# src/train.py
import sys
import os
import functools

# --- Импорты стандартных и внешних библиотек ---
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import time
import argparse
import Levenshtein
import logging
from typing import Union, Optional # Для совместимости с Python 3.9

# --- Импорты из проекта (ОТНОСИТЕЛЬНЫЕ, т.к. запускаем с -m) ---
try:
    from .dataset import MorseDataset, collate_fn
    from .model import CRNNModel
except ImportError as e:
     # Эта ошибка не должна возникать при запуске с python -m src.train из Decoder/
     # но оставим для диагностики
     print(f"Критическая ошибка импорта dataset или model: {e}")
     print(f"Убедитесь, что файлы существуют и __init__.py на месте в src/.")
     # Попробуем вывести sys.path для дополнительной диагностики
     print(f"Текущий sys.path: {sys.path}")
     # Попробуем определить корень проекта и проверить пути
     _current_file_dir = os.path.dirname(os.path.abspath(__file__))
     _project_root_guess = os.path.abspath(os.path.join(_current_file_dir, '..'))
     print(f"Предполагаемый корень проекта: {_project_root_guess}")
     print(f"Существует ли {_project_root_guess}/src/dataset.py? {os.path.isfile(os.path.join(_project_root_guess, 'src', 'dataset.py'))}")
     print(f"Существует ли {_project_root_guess}/src/model.py? {os.path.isfile(os.path.join(_project_root_guess, 'src', 'model.py'))}")
     exit(1)

# --- Настройка Логгера ---
log_level = logging.INFO
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Используем __name__, чтобы логгер назывался src.train

# --- Конфигурация и Аргументы ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN for Morse Code Recognition (Online Processing)")

    # --- Пути (относительно папки src, т.к. там лежит скрипт) ---
    parser.add_argument('--train-csv', type=str, default='data/raw/train.csv',
                        help='Путь к CSV файлу для обучения (относительно папки src)')
    parser.add_argument('--val-csv', type=str, default='data/processed/val_processed.csv',
                        help='Путь к CSV файлу для валидации (относительно папки src)')
    parser.add_argument('--char-map', type=str, default='data/processed/char_maps/char_map.json',
                        help='Путь к JSON файлу char_map (относительно папки src)')
    parser.add_argument('--index-map', type=str, default='data/processed/char_maps/index_map.json',
                        help='Путь к JSON файлу index_map (относительно папки src)')
    parser.add_argument('--audio-dir', type=str, default='data/morse_dataset',
                        help='Путь к директории с аудиофайлами (.opus) (относительно папки src)')
    parser.add_argument('--save-dir', type=str, default='models/crnn_morse_online',
                        help='Директория для сохранения чекпоинтов (относительно папки src)')

    # --- Аргументы для колонок и расширения ---
    parser.add_argument('--audio-id-col', type=str, default='id',
                        help='Имя колонки в CSV с ID аудиофайла (без расширения)')
    parser.add_argument('--message-col', type=str, default='message',
                        help='Имя колонки в CSV с текстовой меткой (сообщением)')
    parser.add_argument('--audio-ext', type=str, default='.opus',
                        help='Расширение аудиофайлов (включая точку)')

    # --- Параметры генерации спектрограмм ---
    parser.add_argument('--sr', type=int, default=8000, help='Целевая частота дискретизации')
    parser.add_argument('--n-fft', type=int, default=512, help='Размер окна FFT')
    parser.add_argument('--hop-length', type=int, default=64, help='Шаг окна FFT')
    parser.add_argument('--n-mels', type=int, default=40, help='Количество Мел-фильтров (ВАЖНО: должно совпадать с моделью!)')
    parser.add_argument('--fmin', type=int, default=0, help='Мин. частота для Мел-фильтров')
    parser.add_argument('--fmax', type=int, default=None, help='Макс. частота для Мел-фильтров (None = sr/2)')
    parser.add_argument('--normalize-audio', action='store_true', default=True, help='Применять нормализацию аудио')
    parser.add_argument('--no-normalize-audio', action='store_false', dest='normalize_audio', help='Отключить нормализацию аудио')
    parser.add_argument('--apply-filter', action='store_true', default=False, help='Применять ФНЧ к аудио')
    parser.add_argument('--cutoff-freq', type=int, default=2400, help='Частота среза ФНЧ (если --apply-filter)')
    parser.add_argument('--filter-order', type=int, default=5, help='Порядок ФНЧ (если --apply-filter)')

    # --- Параметры аугментаций ---
    parser.add_argument('--use-augmentations', action='store_true', default=True, help='Применять аугментации на трейне')
    parser.add_argument('--no-augmentations', action='store_false', dest='use_augmentations', help='Отключить аугментации')
    parser.add_argument('--freq-mask-prob', type=float, default=0.5, help='Вероятность частотного маскирования')
    parser.add_argument('--time-mask-prob', type=float, default=0.5, help='Вероятность временного маскирования')
    parser.add_argument('--freq-mask-param', type=int, default=7, help='Макс. высота частотной маски')
    parser.add_argument('--time-mask-param', type=int, default=12, help='Макс. ширина временной маски')
    parser.add_argument('--num-freq-masks', type=int, default=1, help='Кол-во частотных масок')
    parser.add_argument('--num-time-masks', type=int, default=1, help='Кол-во временных масок')
    parser.add_argument('--mask-value', type=str, default='mean', help="Значение для маскирования ('mean' или float)")

    # --- Гиперпараметры обучения ---
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер батча')
    parser.add_argument('--lr', type=float, default=1e-4, help='Скорость обучения (learning rate)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Регуляризация весов (weight decay)')
    parser.add_argument('--num-workers', type=int, default=4, help='Количество воркеров для DataLoader')
    parser.add_argument('--patience', type=int, default=10, help='Терпение для ранней остановки (early stopping)')

    # --- Параметры модели ---
    parser.add_argument('--rnn-hidden', type=int, default=128, help='Размер скрытого состояния RNN')
    parser.add_argument('--rnn-layers', type=int, default=2, help='Количество слоев RNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Вероятность dropout')

    # --- Другое ---
    parser.add_argument('--seed', type=int, default=42, help='Случайное зерно для воспроизводимости')
    parser.add_argument('--log-interval', type=int, default=100, help='Интервал логирования шагов внутри эпохи')

    return parser.parse_args()

# --- Вспомогательные Функции ---
def load_json(path):
    """Загружает JSON файл."""
    # Определяем абсолютный путь относительно CWD, где запущен скрипт
    # Если запускать с python -m src.train из Decoder/, то CWD будет Decoder/
    # и argparse вернет пути как "../data/...", os.path.abspath сделает их абсолютными
    absolute_path = os.path.abspath(path)
    logger.info(f"Загрузка JSON из: {absolute_path}")
    try:
        with open(absolute_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Ошибка: JSON файл не найден по пути: {absolute_path}")
        exit(1)
    except Exception as e:
        logger.error(f"Ошибка при загрузке JSON из {absolute_path}: {e}", exc_info=True)
        exit(1)

def decode_ctc_greedy(preds, index_map, blank_idx=0):
    """Декодирует выход модели с помощью жадного CTC декодера."""
    decoded_batch = []
    preds_cpu = preds.cpu() # Убедимся, что на CPU
    for seq in preds_cpu: # preds ожидается формы [Batch, Time]
        collapsed_seq = torch.unique_consecutive(seq)
        decoded_seq = [index_map.get(str(idx.item()), '?') for idx in collapsed_seq if idx.item() != blank_idx] # Преобразуем ключ в str для index_map
        decoded_batch.append("".join(decoded_seq))
    return decoded_batch

def calculate_cer(preds_str, targets_str):
    """Вычисляет Character Error Rate (CER)."""
    total_dist = 0
    total_len = 0
    if not preds_str or not targets_str or len(preds_str) != len(targets_str):
        logger.warning(f"Несовпадение длин или пустые списки при расчете CER: preds={len(preds_str)}, targets={len(targets_str)}")
        return 1.0, len(targets_str) if targets_str else 1, len(targets_str) if targets_str else 1

    for pred, target in zip(preds_str, targets_str):
        dist = Levenshtein.distance(pred, target)
        total_dist += dist
        total_len += len(target)

    cer = total_dist / total_len if total_len > 0 else 0.0
    if total_dist > total_len > 0:
        logger.debug(f"CER > 1.0: dist={total_dist}, len={total_len}")
        # cer = min(cer, 1.0) # Опционально ограничить CER

    return cer, total_dist, total_len

# --- Основная Функция Обучения ---
def main():
    args = parse_args()

    current_working_directory = os.getcwd()
    logger.info(f"Текущая рабочая директория (CWD): {current_working_directory}")
    logger.info(f"Аргументы командной строки: {args}")

    # Воспроизводимость
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        logger.info("CUDA недоступна, используется CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    # Загрузка словарей
    logger.info("Загрузка карт символов...")
    char_map = load_json(args.char_map)
    index_map_str_keys = load_json(args.index_map) # Ключи будут строками
    # Преобразуем строковые ключи из JSON обратно в int для использования в decode_ctc_greedy
    try:
        # index_map нужен для декодирования: {int_index: 'char'}
        index_map = {int(k): v for k, v in index_map_str_keys.items()}
        # char_map нужен для датасета: {'char': int_index}
    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка преобразования ключей index_map в int: {e}. Проверьте файл {args.index_map}")
        return
    num_classes = len(char_map)
    blank_token_char = '-'
    blank_idx = char_map.get(blank_token_char, 0)
    logger.info(f"Количество классов: {num_classes}, Blank индекс ('{blank_token_char}'): {blank_idx}")

    # Создание конфигураций для ОНЛАЙН обработки
    logger.info("Формирование конфигураций для обработки данных...")
    spec_cfg = {
        'target_sr': args.sr, 'n_fft': args.n_fft, 'hop_length': args.hop_length,
        'n_mels': args.n_mels, 'fmin': args.fmin, 'fmax': args.fmax if args.fmax is not None else args.sr // 2,
        'normalize_audio': args.normalize_audio, 'apply_filter': args.apply_filter,
        'cutoff_freq_filter': args.cutoff_freq, 'filter_order': args.filter_order
    }
    logger.info(f"Конфигурация спектрограмм: {spec_cfg}")
    aug_cfg = None
    if args.use_augmentations:
        try: mask_val = float(args.mask_value)
        except ValueError: mask_val = 'mean' if args.mask_value.lower() == 'mean' else 'mean'
        aug_cfg = {
            'freq_mask_prob': args.freq_mask_prob, 'time_mask_prob': args.time_mask_prob,
            'freq_mask_param': args.freq_mask_param, 'time_mask_param': args.time_mask_param,
            'num_freq_masks': args.num_freq_masks, 'num_time_masks': args.num_time_masks,
            'mask_value': mask_val
        }
        logger.info(f"Конфигурация аугментаций: {aug_cfg}")
    else:
        logger.info("Аугментации отключены.")

    # Создание датасетов и загрузчиков
    logger.info("Создание датасетов и загрузчиков...")
    # Определяем абсолютный путь к аудио директории относительно CWD
    abs_audio_dir = os.path.abspath(args.audio_dir)
    logger.info(f"Абсолютный путь к директории аудио: {abs_audio_dir}")
    try:
        train_dataset = MorseDataset(
            csv_path=args.train_csv,
            char_map=char_map,
            audio_dir=args.audio_dir, # Передаем путь как есть (относительно CWD)
            spectrogram_cfg=spec_cfg,
            augment_cfg=aug_cfg,
            is_train=True,
            audio_filename_col=args.audio_id_col,
            label_col=args.message_col,
            audio_ext=args.audio_ext
        )
        val_dataset = MorseDataset(
            csv_path=args.val_csv,
            char_map=char_map,
            audio_dir=args.audio_dir,
            spectrogram_cfg=spec_cfg,
            augment_cfg=None,
            is_train=False,
            audio_filename_col=args.audio_id_col,
            label_col=args.message_col,
            audio_ext=args.audio_ext
        )
    except FileNotFoundError as e:
        logger.error(f"Ошибка: Не удалось найти CSV файл или директорию аудио: {e}", exc_info=True)
        return
    except ValueError as e:
         logger.error(f"Ошибка: Отсутствуют необходимые колонки в CSV или неверные конфиги: {e}", exc_info=True)
         return
    except Exception as e:
        logger.error(f"Неожиданная ошибка при создании датасета: {e}", exc_info=True)
        return

    logger.info(f"Сэмплов для обучения: {len(train_dataset)}, для валидации: {len(val_dataset)}")
    if len(train_dataset) == 0: logger.error("Тренировочный датасет пуст!"); return
    if len(val_dataset) == 0: logger.warning("Валидационный датасет пуст.")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=functools.partial(collate_fn, padding_value=0.0, label_padding_value=blank_idx), # <-- Исправлено
        pin_memory=pin_memory, persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=functools.partial(collate_fn, padding_value=0.0, label_padding_value=blank_idx), # <-- Исправлено
        pin_memory=pin_memory
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Инициализация модели
    logger.info("Инициализация модели...")
    try:
        model = CRNNModel(
            n_mels=args.n_mels,
            num_classes=num_classes,
            rnn_hidden_size=args.rnn_hidden,
            rnn_layers=args.rnn_layers,
            dropout=args.dropout
        ).to(device)
        if not hasattr(model, 'get_output_lengths'):
             logger.error("Критическая ошибка: Модель CRNNModel не имеет метода 'get_output_lengths'!")
             return
    except Exception as e:
         logger.error(f"Ошибка при инициализации модели CRNNModel: {e}", exc_info=True)
         return

    logger.info(f"Архитектура модели:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Количество обучаемых параметров: {num_params:,}")

    # Функция потерь, оптимизатор, планировщик
    criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience // 2, verbose=True)

    # Папка для сохранения
    abs_save_dir = os.path.abspath(args.save_dir)
    logger.info(f"Создание/проверка директории для сохранения моделей: {abs_save_dir}")
    os.makedirs(abs_save_dir, exist_ok=True)

    # Переменные для лучшей модели и ранней остановки
    best_val_cer = float('inf')
    epochs_no_improve = 0
    start_epoch = 1

    # TODO: Загрузка чекпоинта для возобновления

    # --- Цикл Обучения ---
    logger.info(f"Начало обучения на {args.epochs} эпох...")
    start_time_total = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"--- Эпоха {epoch}/{args.epochs} ---")
        epoch_start_time = time.time()

        # --- Тренировка ---
        model.train()
        total_train_loss = 0.0
        processed_batches = 0
        train_pbar = tqdm(train_loader, desc=f"Эпоха {epoch} Обучение", leave=False, unit="батч", ncols=100)

        for i, batch_data in enumerate(train_pbar):
            try:
                if not batch_data or not isinstance(batch_data, tuple) or len(batch_data) != 4:
                     logger.warning(f"Эпоха {epoch}, Тр.батч {i}: Пропуск некорректного батча.")
                     continue
                spectrograms, labels, spec_lengths, label_lengths = batch_data
                if spectrograms.numel() == 0 or labels.numel() == 0:
                    logger.warning(f"Эпоха {epoch}, Тр.батч {i}: Пропуск пустого батча.")
                    continue

                spectrograms = spectrograms.to(device, non_blocking=True)
                labels = labels.cpu()
                spec_lengths = spec_lengths.cpu()
                label_lengths = label_lengths.cpu()

                optimizer.zero_grad()
                outputs = model(spectrograms)

                if outputs.dim() != 3 or outputs.shape[1] != spectrograms.shape[0] or outputs.shape[2] != num_classes:
                     logger.error(f"Эпоха {epoch}, Тр.батч {i}: Неожиданная форма выхода модели: {outputs.shape}. Пропуск.")
                     continue

                log_probs = F.log_softmax(outputs, dim=2)
                output_lengths = model.get_output_lengths(spec_lengths).cpu()

                if torch.any(output_lengths <= 0):
                    logger.warning(f"Эпоха {epoch}, Тр.батч {i}: Нулевые/отрицательные output_lengths: {output_lengths}. Пропуск.")
                    continue

                valid_indices = output_lengths >= label_lengths
                if not torch.all(valid_indices):
                     problem_outputs = output_lengths[~valid_indices].tolist()
                     problem_labels = label_lengths[~valid_indices].tolist()
                     logger.warning(f"Эпоха {epoch}, Тр.батч {i}: Нарушение CTC (T >= L) для {len(problem_outputs)} сэмплов. Пропуск.")
                     logger.debug(f"   Проблемные (out_len, lbl_len): {list(zip(problem_outputs, problem_labels))}")
                     continue

                loss = criterion(log_probs, labels, output_lengths, label_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Эпоха {epoch}, Тр.батч {i}: Обнаружен NaN/Inf лосс ({loss.item()}). Пропуск backward/step.")
                    optimizer.zero_grad()
                    continue

                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) # Опционально
                optimizer.step()

                total_train_loss += loss.item()
                processed_batches += 1
                avg_loss = total_train_loss / processed_batches
                train_pbar.set_postfix(loss=f"{avg_loss:.4f}")

                if args.log_interval > 0 and (i + 1) % args.log_interval == 0:
                     current_lr = optimizer.param_groups[0]['lr']
                     logger.info(f"  Эпоха {epoch}, Шаг [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}), LR: {current_lr:.2e}")

            except Exception as e:
                 # Ловим любую другую ошибку в цикле обучения батча
                 logger.error(f"Эпоха {epoch}, Тр.батч {i}: Непредвиденная ошибка: {e}", exc_info=True)
                 continue # Пропускаем батч

        train_pbar.close()
        avg_train_loss = total_train_loss / processed_batches if processed_batches > 0 else float('inf')
        if processed_batches < len(train_loader):
             logger.warning(f"Эпоха {epoch}: Успешно обработано {processed_batches}/{len(train_loader)} тренировочных батчей.")
        logger.info(f"Эпоха {epoch} Средний Тренировочный Loss: {avg_train_loss:.4f}")

        # --- Валидация ---
        model.eval()
        total_val_loss = 0.0
        total_val_dist = 0
        total_val_len = 0
        processed_val_batches = 0
        val_pbar = tqdm(val_loader, desc=f"Эпоха {epoch} Валидация", leave=False, unit="батч", ncols=100)

        with torch.no_grad():
            for batch_data in val_pbar:
                try:
                    if not batch_data or not isinstance(batch_data, tuple) or len(batch_data) != 4:
                         logger.warning(f"Эпоха {epoch}, Вал.батч: Пропуск некорректного батча.")
                         continue
                    spectrograms, labels, spec_lengths, label_lengths = batch_data
                    if spectrograms.numel() == 0 or labels.numel() == 0:
                         logger.warning(f"Эпоха {epoch}, Вал.батч: Пропуск пустого батча.")
                         continue

                    spectrograms = spectrograms.to(device, non_blocking=True)
                    labels = labels.cpu()
                    spec_lengths = spec_lengths.cpu()
                    label_lengths = label_lengths.cpu()

                    outputs = model(spectrograms)

                    if outputs.dim() != 3 or outputs.shape[1] != spectrograms.shape[0] or outputs.shape[2] != num_classes:
                         logger.error(f"Эпоха {epoch}, Вал.батч: Неожиданная форма выхода: {outputs.shape}. Пропуск.")
                         continue

                    log_probs = F.log_softmax(outputs, dim=2)
                    output_lengths = model.get_output_lengths(spec_lengths).cpu()

                    # Расчет лосса (опционально, но полезно)
                    current_loss = float('nan') # Значение по умолчанию
                    if torch.any(output_lengths <= 0):
                        logger.warning(f"Эпоха {epoch}, Вал.батч: Нулевые/отрицательные output_lengths: {output_lengths}. Пропуск лосса.")
                    elif not torch.all(output_lengths >= label_lengths):
                         logger.warning(f"Эпоха {epoch}, Вал.батч: Нарушение CTC (T >= L). Пропуск лосса.")
                    else:
                         try:
                            loss = criterion(log_probs, labels, output_lengths, label_lengths)
                            if not (torch.isnan(loss) or torch.isinf(loss)):
                                current_loss = loss.item()
                                total_val_loss += current_loss
                            else: logger.warning(f"Эпоха {epoch}, Вал.батч: NaN/Inf лосс.")
                         except RuntimeError as e_loss:
                              logger.error(f"Эпоха {epoch}, Вал.батч: RuntimeError CTCLoss: {e_loss}")
                         except Exception as e_loss_other:
                              logger.error(f"Эпоха {epoch}, Вал.батч: Ошибка CTCLoss: {e_loss_other}", exc_info=True)

                    # Декодирование для CER
                    preds_str, targets_str = [], []
                    try:
                        preds_argmax = log_probs.argmax(dim=2).permute(1, 0) # [Batch, Time]
                        preds_str = decode_ctc_greedy(preds_argmax, index_map, blank_idx)
                    except Exception as e_dec_pred:
                         logger.error(f"Эпоха {epoch}, Вал.батч: Ошибка декодирования preds: {e_dec_pred}", exc_info=True)

                    try:
                        for i in range(labels.size(0)):
                            length = label_lengths[i].item()
                            target_indices = labels[i, :length].tolist()
                            targets_str.append("".join([index_map.get(idx, '?') for idx in target_indices]))
                    except Exception as e_dec_tgt:
                         logger.error(f"Эпоха {epoch}, Вал.батч: Ошибка декодирования targets: {e_dec_tgt}", exc_info=True)

                    # Расчет CER
                    if preds_str and targets_str and len(preds_str) == len(targets_str):
                         _, batch_dist, batch_len = calculate_cer(preds_str, targets_str)
                         total_val_dist += batch_dist
                         total_val_len += batch_len
                    else:
                         logger.warning("Пропуск расчета CER для батча из-за ошибок декодирования.")

                    processed_val_batches += 1
                    current_avg_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else 0.0
                    current_cer = total_val_dist / total_val_len if total_val_len > 0 else float('inf')
                    val_pbar.set_postfix(loss=f"{current_avg_loss:.4f}", cer=f"{current_cer:.4f}")

                except Exception as e:
                    logger.error(f"Эпоха {epoch}, Вал.батч: Непредвиденная ошибка: {e}", exc_info=True)
                    continue # Пропускаем батч

        val_pbar.close()

        avg_val_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else float('inf')
        val_cer = total_val_dist / total_val_len if total_val_len > 0 else float('inf')

        if processed_val_batches < len(val_loader):
            logger.warning(f"Эпоха {epoch}: Успешно обработано {processed_val_batches}/{len(val_loader)} валидационных батчей.")

        logger.info(f"Эпоха {epoch} Средний Валидационный Loss: {avg_val_loss:.4f}")
        logger.info(f"Эпоха {epoch} Валидационный CER: {val_cer:.4f} (Dist: {total_val_dist}, Len: {total_val_len})")

        # Обновление планировщика, сохранение моделей, ранняя остановка
        scheduler.step(val_cer)

        if val_cer < best_val_cer:
            best_val_cer = val_cer
            epochs_no_improve = 0
            save_path = os.path.join(abs_save_dir, 'best_model.pth')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_cer': best_val_cer,
                    'config': { # Параметры модели
                        'n_mels': args.n_mels, 'num_classes': num_classes,
                        'rnn_hidden_size': args.rnn_hidden, 'rnn_layers': args.rnn_layers,
                        'dropout': args.dropout,
                    },
                    'spectrogram_cfg': spec_cfg, # Параметры обработки данных
                    'char_map': char_map,        # Карта символов
                    'audio_id_col': args.audio_id_col, # Имена колонок и расширение
                    'message_col': args.message_col,
                    'audio_ext': args.audio_ext
                }, save_path)
                logger.info(f"Сохранена новая лучшая модель в {save_path} (CER: {best_val_cer:.4f})")
            except Exception as e:
                 logger.error(f"Ошибка при сохранении лучшей модели: {e}", exc_info=True)
        else:
            epochs_no_improve += 1
            logger.info(f"CER на валидации не улучшился {epochs_no_improve} эпох(и). Лучший CER: {best_val_cer:.4f}")

        # Сохранение последней модели
        last_save_path = os.path.join(abs_save_dir, 'last_model.pth')
        try:
            # Сохраняем те же данные, что и для лучшей модели, плюс last_val_cer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_cer': best_val_cer,
                'last_val_cer': val_cer, # Добавляем последний CER
                 'config': {
                    'n_mels': args.n_mels, 'num_classes': num_classes,
                    'rnn_hidden_size': args.rnn_hidden, 'rnn_layers': args.rnn_layers,
                    'dropout': args.dropout,
                 },
                'spectrogram_cfg': spec_cfg,
                'char_map': char_map,
                'audio_id_col': args.audio_id_col,
                'message_col': args.message_col,
                'audio_ext': args.audio_ext
            }, last_save_path)
        except Exception as e:
            logger.error(f"Ошибка при сохранении последней модели: {e}", exc_info=True)

        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Длительность эпохи {epoch}: {epoch_duration:.2f} секунд")

        if epochs_no_improve >= args.patience:
            logger.info(f"Ранняя остановка сработала после {args.patience} эпох без улучшения CER.")
            break

    # --- Завершение Обучения ---
    total_duration = time.time() - start_time_total
    logger.info(f"\nОбучение завершено за {total_duration / 60:.2f} минут.")
    logger.info(f"Лучший достигнутый CER на валидации: {best_val_cer:.4f}")
    logger.info(f"Лучшая модель сохранена в: {os.path.join(abs_save_dir, 'best_model.pth')}")
    logger.info(f"Последняя модель сохранена в: {os.path.join(abs_save_dir, 'last_model.pth')}")


# Точка входа в скрипт
if __name__ == '__main__':
    main()