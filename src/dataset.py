
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Optional, Dict, List, Any # Добавлен Any для словарей конфигурации

# --- Импорт функций обработки ---
# Убедись, что папка Decoder/ является корневой для импорта или настроен PYTHONPATH
try:
    from data_processing.generate_spectrograms import generate_mel_spectrogram
    from data_processing.augmentations import apply_spectrogram_augmentations
except ImportError as e:
     print("Ошибка импорта функций из data_processing. Убедитесь, что структура папок верна и Decoder/ доступен для импорта.")
     print(e)
     # Можно либо поднять исключение, либо продолжить, но Dataset не будет работать
     raise

# Настройка logging (оставляем как есть, хорошо)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Класс Dataset для онлайн-обработки ---
class MorseDataset(Dataset):
    """
    Кастомный Dataset для онлайн-загрузки и обработки аудиофайлов Морзе.
    Генерирует мел-спектрограммы и применяет аугментации на лету.

    Args:
        csv_path (str): Путь к CSV файлу (train.csv или test.csv).
                        Ожидаемые колонки: 'filename' (или заданная `audio_filename_col`),
                        и 'morse_code' (или заданная `label_col`) для режима обучения.
        char_map (Dict[str, int]): Словарь для преобразования символов в индексы.
                                   Ожидается, что индекс 0 зарезервирован для CTC blank.
        audio_dir (str): Путь к директории, содержащей аудиофайлы (.opus).
        spectrogram_cfg (Dict[str, Any]): Словарь с параметрами для функции
                                          `generate_mel_spectrogram`.
        augment_cfg (Optional[Dict[str, Any]], optional): Словарь с параметрами для функции
                                                           `apply_spectrogram_augmentations`.
                                                           Применяется только если `is_train=True`.
                                                           Defaults to None.
        is_train (bool, optional): Флаг режима обучения. Если True, загружаются метки
                                   и применяются аугментации (если `augment_cfg` задан).
                                   Defaults to True.
        audio_filename_col (str, optional): Имя колонки в CSV с именем аудиофайла.
                                            Defaults to 'filename'.
        label_col (str, optional): Имя колонки в CSV с текстовой меткой (морзе).
                                   Defaults to 'morse_code'.
        blank_token (str, optional): Символ, представляющий blank токен в char_map.
                                     Defaults to '-'. Если не найден, используется индекс 0.
    """
    def __init__(self,
                 csv_path: str,
                 char_map: Dict[str, int],
                 audio_dir: str,
                 spectrogram_cfg: Dict[str, Any],
                 augment_cfg: Optional[Dict[str, Any]] = None,
                 is_train: bool = True,
                 audio_filename_col: str = 'id',      # <-- ПРАВИЛЬНО
                 label_col: str = 'message',        # <-- ПРАВИЛЬНО
                 audio_ext: str = '.opus',          # <-- ДОБАВЛЕНО
                 blank_token: str = '-'):

        self.csv_path = csv_path
        self.char_map = char_map
        self.audio_dir = os.path.abspath(audio_dir)
        self.spectrogram_cfg = spectrogram_cfg
        self.augment_cfg = augment_cfg if is_train else None
        self.is_train = is_train
        self.audio_filename_col = audio_filename_col
        self.label_col = label_col
        self.audio_ext = audio_ext                # <-- ДОБАВЛЕНО СОХРАНЕНИЕ
        self.blank_token_index = char_map.get(blank_token, 0)

        logger.info(f"Инициализация MorseDataset (Онлайн-обработка)...")
        logger.info(f"  CSV: {self.csv_path}")
        logger.info(f"  Директория аудио: {self.audio_dir}")
        logger.info(f"  Режим обучения: {self.is_train}")
        logger.info(f"  Конфиг спектрограмм: {self.spectrogram_cfg}")
        if self.augment_cfg:
            logger.info(f"  Конфиг аугментаций: {self.augment_cfg}")
        else:
            logger.info("  Аугментации отключены.")


        # Проверки существования путей
        if not os.path.exists(self.csv_path):
             logger.error(f"Файл CSV не найден: {self.csv_path}")
             raise FileNotFoundError(f"Файл CSV не найден: {self.csv_path}")
        if not os.path.isdir(self.audio_dir):
             logger.error(f"Директория аудио не найдена: {self.audio_dir}")
             raise FileNotFoundError(f"Директория аудио не найдена: {self.audio_dir}")

        # Определяем индекс blank токена (логика осталась)
        self.blank_token_index = self.char_map.get(blank_token, 0)
        # ... (можно добавить предупреждения как в оригинале)

        # Загрузка DataFrame
        try:
            self.data_frame = pd.read_csv(self.csv_path)
            # Проверка наличия необходимых колонок
            required_cols = [self.audio_filename_col]
            if self.is_train:
                # Колонка с метками обязательна только для обучения
                required_cols.append(self.label_col)

            missing_cols = [col for col in required_cols if col not in self.data_frame.columns]
            if missing_cols:
                 message = f"В CSV файле {self.csv_path} отсутствуют необходимые колонки: {missing_cols}"
                 logger.error(message)
                 raise ValueError(message)

            # Проверка на пустые значения в критичных колонках
            if self.data_frame[self.audio_filename_col].isnull().any():
                 logger.warning(f"В колонке '{self.audio_filename_col}' файла {self.csv_path} есть пустые значения (NaN). Эти строки будут пропущены.")
            if self.is_train and self.label_col in self.data_frame.columns and self.data_frame[self.label_col].isnull().any():
                 logger.warning(f"В колонке '{self.label_col}' файла {self.csv_path} есть пустые значения (NaN). Для них будут возвращаться пустые метки.")

        except pd.errors.EmptyDataError:
            logger.warning(f"CSV файл {self.csv_path} пуст или содержит только заголовки.")
            self.data_frame = pd.DataFrame(columns=required_cols)
        except Exception as e:
            logger.error(f"Ошибка чтения CSV файла {self.csv_path}: {e}", exc_info=True)
            raise IOError(f"Ошибка чтения CSV файла {self.csv_path}: {e}")

        logger.info(f"MorseDataset инициализирован. Сэмплов: {len(self.data_frame)}")

    def __len__(self) -> int:
        """Возвращает количество сэмплов в датасете."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Загружает аудиофайл, генерирует спектрограмму, применяет аугментации (если нужно),
        кодирует метку и возвращает тензоры.

        Args:
            idx (int): Индекс сэмпла.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Кортеж (спектрограмма, метка) или None в случае ошибки.
        """
        if torch.is_tensor(idx):
            idx = idx.item()

        # Проверка индекса
        if not (0 <= idx < len(self.data_frame)):
            logger.warning(f"Запрошен неверный индекс {idx}, размер датасета {len(self.data_frame)}.")
            return None

        row = self.data_frame.iloc[idx]
        audio_filename = row.get(self.audio_filename_col)
        full_audio_path = None # Инициализируем для логов/ошибок

        try:
            # Проверка имени файла из CSV
            if pd.isna(audio_filename) or not isinstance(audio_filename, str) or not audio_filename:
                 logger.debug(f"[idx={idx}] Неверное или отсутствующее имя аудиофайла в CSV: '{audio_filename}'. Пропуск.")
                 return None

            # --- Построение пути к АУДИО файлу ---
            full_audio_path = os.path.join(self.audio_dir, audio_filename)
            logger.debug(f"[idx={idx}] Попытка загрузки аудио: {full_audio_path}")

            # --- ШАГ 1: Генерация спектрограммы ---
            # Используем параметры из self.spectrogram_cfg
            spectrogram = generate_mel_spectrogram(
                audio_path=full_audio_path,
                **self.spectrogram_cfg
            )

            # Обработка ошибки генерации (generate_mel_spectrogram вернет None)
            if spectrogram is None:
                logger.warning(f"[idx={idx}] Не удалось сгенерировать спектрограмму для файла {full_audio_path}. Пропуск.")
                # Сама функция generate_mel_spectrogram уже залогировала детали ошибки
                return None

            # --- ШАГ 2: Применение аугментаций (только для обучения) ---
            if self.is_train and self.augment_cfg:
                 logger.debug(f"[idx={idx}] Применение аугментаций...")
                 spectrogram = apply_spectrogram_augmentations(
                     spectrogram=spectrogram,
                     **self.augment_cfg # Используем параметры из self.augment_cfg
                 )

            # --- ШАГ 3: Конвертация спектрограммы в тензор ---
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
            # Ожидаемая форма: [n_mels, time_steps]

            # --- ШАГ 4: Обработка меток ---
            label_tensor = torch.empty(0, dtype=torch.long) # Пустой тензор по умолчанию (для инференса)
            if self.is_train:
                morse_code = row.get(self.label_col, '') # Используем .get для надежности
                if pd.isna(morse_code) or not isinstance(morse_code, str):
                    morse_code = "" # Обрабатываем NaN или не-строки как пустую строку

                label_indices = []
                for char in morse_code:
                    index = self.char_map.get(char)
                    if index is None:
                        # Символ не найден в словаре, можно заменить на blank или пропустить
                        # logger.warning(f"[idx={idx}] Символ '{char}' не найден в char_map. Используется blank ({self.blank_token_index}).")
                        label_indices.append(self.blank_token_index)
                    else:
                        label_indices.append(index)

                label_tensor = torch.tensor(label_indices, dtype=torch.long)

            logger.debug(f"[idx={idx}] Обработка завершена. Форма спектрограммы: {spectrogram_tensor.shape}, Длина метки: {len(label_tensor)}")
            return spectrogram_tensor, label_tensor

        except FileNotFoundError:
             # Эта ошибка может возникнуть, если generate_mel_spectrogram не поймает ее внутри
             logger.warning(f"[idx={idx}] Файл аудио НЕ НАЙДЕН: {full_audio_path}. Пропуск.")
             return None
        except Exception as e:
            # Ловим любые другие неожиданные ошибки при обработке этого сэмпла
            path_repr = full_audio_path or audio_filename or 'N/A'
            # Используем exc_info=True для вывода полного стектрейса при отладке
            logger.error(f"Неожиданная ошибка при обработке сэмпла [idx={idx}], аудио: {path_repr}: {e}. Пропуск.", exc_info=False)
            return None


# --- Функция для сборки батча (Collate Function) ---
# Эта функция не требует изменений, т.к. она работает с результатом __getitem__
# и уже умеет обрабатывать None и паддить последовательности разной длины.
# Убедись, что label_padding_value (обычно 0) соответствует твоему blank_token_index.
def collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
               padding_value: float = 0.0, # Значение для паддинга спектрограмм
               label_padding_value: int = 0  # Значение для паддинга меток (обычно индекс blank/pad)
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Собирает список сэмплов от MorseDataset в батч.
    Паддит спектрограммы и метки. Обрабатывает None значения в батче.

    Args:
        batch: Список кортежей (спектрограмма, метка) или None.
        padding_value: Значение для паддинга спектрограмм.
        label_padding_value: Значение для паддинга меток.

    Returns:
        Кортеж из:
        - spectrograms_padded: Тензор спектрограмм [batch, n_mels, max_time_steps].
        - labels_padded: Тензор меток [batch, max_label_len].
        - spectrogram_lengths: Тензор реальных длин спектрограмм [batch].
        - label_lengths: Тензор реальных длин меток [batch].
    """
    # Фильтруем None значения (если __getitem__ вернул None из-за ошибки)
    original_batch_size = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_count = original_batch_size - len(batch)
    if filtered_count > 0:
        logger.debug(f"collate_fn: Отфильтровано {filtered_count} None-элементов из батча размером {original_batch_size}.")

    # Если после фильтрации батч пуст
    if not batch:
        logger.warning("collate_fn: Пустой батч после фильтрации None.")
        return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # Разделяем данные
    try:
        spectrograms = [item[0] for item in batch]
        labels = [item[1] for item in batch]
    except IndexError:
         logger.error("collate_fn: Ошибка при распаковке элементов батча. Возможно, __getitem__ вернул не кортеж.", exc_info=True)
         return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)


    # Вычисляем реальные длины ДО паддинга
    # .shape[1] т.к. спектрограммы [n_mels, time_steps]
    spectrogram_lengths = torch.tensor([spec.shape[1] for spec in spectrograms], dtype=torch.long)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # Паддинг спектрограмм
    # pad_sequence ожидает [seq_len, *] -> транспонируем [time_steps, n_mels]
    try:
        spectrograms_padded = pad_sequence([s.transpose(0, 1) for s in spectrograms],
                                           batch_first=True, # -> [batch, max_time_steps, n_mels]
                                           padding_value=padding_value)
        # Транспонируем обратно -> [batch, n_mels, max_time_steps]
        spectrograms_padded = spectrograms_padded.transpose(1, 2)
    except Exception as e:
        logger.error(f"Ошибка при паддинге спектрограмм в collate_fn: {e}", exc_info=True)
        for i, s in enumerate(spectrograms): logger.error(f"  Форма спектрограммы {i}: {s.shape}")
        return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    # Паддинг меток
    try:
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=label_padding_value)
    except Exception as e:
         logger.error(f"Ошибка при паддинге меток в collate_fn: {e}", exc_info=True)
         return torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return spectrograms_padded, labels_padded, spectrogram_lengths, label_lengths