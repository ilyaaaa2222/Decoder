# split_data.py
# (Поместите в корневую папку Decoder/)

import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
import logging

# --- Настройка Логгера ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Конфигурация ---
# *** ИЗМЕНЕНО ЗНАЧЕНИЕ ПО УМОЛЧАНИЮ ***
# Ожидаемые колонки в исходном файле.
EXPECTED_COLUMNS = ['id', 'message']

DEFAULT_INPUT_CSV = 'data/raw/train.csv' # Исходный файл для разделения (обычно test.csv)
DEFAULT_OUTPUT_DIR = 'data/processed'
DEFAULT_VAL_FILENAME = 'val_processed.csv'
DEFAULT_TEST_FILENAME = 'test_processed.csv'
DEFAULT_VAL_SPLIT_RATIO = 0.15 # 20% данных пойдут в val_processed.csv
DEFAULT_RANDOM_SEED = 42

def parse_args():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Разделение CSV файла на валидационный и тестовый наборы.")
    parser.add_argument('--input-csv', type=str, default=DEFAULT_INPUT_CSV,
                        help=f"Путь к исходному CSV файлу (относительно папки проекта). По умолчанию: {DEFAULT_INPUT_CSV}")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Путь к директории для сохранения выходных файлов. По умолчанию: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument('--val-file', type=str, default=DEFAULT_VAL_FILENAME,
                        help=f"Имя файла для валидационной выборки. По умолчанию: {DEFAULT_VAL_FILENAME}")
    parser.add_argument('--test-file', type=str, default=DEFAULT_TEST_FILENAME,
                        help=f"Имя файла для тестовой выборки. По умолчанию: {DEFAULT_TEST_FILENAME}")
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_VAL_SPLIT_RATIO,
                        help=f"Доля данных для валидационной выборки (от 0 до 1). По умолчанию: {DEFAULT_VAL_SPLIT_RATIO}")
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Случайное зерно для воспроизводимости разделения. По умолчанию: {DEFAULT_RANDOM_SEED}")
    # Добавим возможность переопределить ожидаемые колонки через аргументы, если нужно
    parser.add_argument('--expected-columns', type=str, nargs='+', default=EXPECTED_COLUMNS,
                        help=f'Список ожидаемых колонок в исходном CSV. По умолчанию: {" ".join(EXPECTED_COLUMNS)}')
    return parser.parse_args()

def main():
    """Основная функция скрипта."""
    args = parse_args()

    # Используем ожидаемые колонки из аргументов
    expected_cols = args.expected_columns
    logger.info(f"Ожидаемые колонки в исходном файле: {expected_cols}")

    # --- Проверка входных параметров ---
    if not (0 < args.val_ratio < 1):
        logger.error("Ошибка: Доля для валидации (val-ratio) должна быть между 0 и 1.")
        exit(1)

    # --- Построение путей ---
    base_dir = os.getcwd()
    input_path = os.path.join(base_dir, args.input_csv)
    output_dir_path = os.path.join(base_dir, args.output_dir)
    output_val_path = os.path.join(output_dir_path, args.val_file)
    output_test_path = os.path.join(output_dir_path, args.test_file)

    logger.info(f"Исходный файл: {input_path}")
    logger.info(f"Выходная директория: {output_dir_path}")
    logger.info(f"Файл валидации: {output_val_path}")
    logger.info(f"Файл теста: {output_test_path}")
    logger.info(f"Доля для валидации: {args.val_ratio:.2f}")
    logger.info(f"Random Seed: {args.seed}")

    # --- Создание выходной директории ---
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        logger.info(f"Выходная директория '{output_dir_path}' проверена/создана.")
    except OSError as e:
        logger.error(f"Ошибка при создании директории {output_dir_path}: {e}")
        exit(1)

    # --- Чтение исходного CSV ---
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Успешно прочитано {len(df)} строк из {input_path}.")

        # Проверка наличия ожидаемых колонок
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Ошибка: В файле {input_path} отсутствуют ожидаемые колонки: {missing_cols}.")
            logger.error(f"Ожидались колонки: {expected_cols}. Найдены: {list(df.columns)}")
            exit(1)

    except FileNotFoundError:
        logger.error(f"Ошибка: Исходный файл не найден по пути: {input_path}")
        exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Ошибка: Исходный файл {input_path} пуст.")
        exit(1)
    except Exception as e:
        logger.error(f"Ошибка при чтении CSV файла {input_path}: {e}", exc_info=True)
        exit(1)

    # --- Разделение данных ---
    try:
        test_df, val_df = train_test_split(
            df,
            test_size=args.val_ratio,
            random_state=args.seed,
            shuffle=True
        )
        logger.info(f"Данные разделены: {len(test_df)} строк для теста, {len(val_df)} строк для валидации.")

        if len(test_df) == 0 or len(val_df) == 0:
             logger.warning("Предупреждение: Одна из выборок (тестовая или валидационная) получилась пустой.")

    except Exception as e:
        logger.error(f"Ошибка при разделении данных: {e}", exc_info=True)
        exit(1)

    # --- Сохранение результатов ---
    try:
        # Убедимся, что сохраняем только нужные колонки (на всякий случай)
        test_df[expected_cols].to_csv(output_test_path, index=False, encoding='utf-8')
        logger.info(f"Тестовая выборка ({len(test_df)} строк) сохранена в: {output_test_path}")

        val_df[expected_cols].to_csv(output_val_path, index=False, encoding='utf-8')
        logger.info(f"Валидационная выборка ({len(val_df)} строк) сохранена в: {output_val_path}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении CSV файлов: {e}", exc_info=True)
        exit(1)

    logger.info("Разделение данных успешно завершено.")

if __name__ == "__main__":
    main()