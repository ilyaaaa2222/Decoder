# data_processing/augmentations.py

import numpy as np
import random
import warnings
from typing import Union, Optional, Dict, Any

# Попытка импорта scipy для фильтрации (все еще может быть полезно для фильтрации аудио ДО спектрограммы)
try:
    from scipy.signal import butter, filtfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    # Предупреждение можно вывести в функциях, которые его используют


def apply_db_threshold_on_spectrogram(
    mel_spec_db: np.ndarray,
    db_threshold: float = -40.0,
    db_fill_value: float = -80.0
) -> np.ndarray:
    
    if not isinstance(mel_spec_db, np.ndarray):
        warnings.warn(f"Ожидался numpy массив для apply_db_threshold_on_spectrogram, получен {type(mel_spec_db)}", stacklevel=2)
        return mel_spec_db # Возвращаем без изменений

    cleaned_spec = mel_spec_db.copy() # Работаем с копией
    cleaned_spec[cleaned_spec < db_threshold] = db_fill_value
    return cleaned_spec

def apply_frequency_band_mask(
    spectrogram: np.ndarray,
    lowcut_mel_idx: Optional[int] = None,
    highcut_mel_idx: Optional[int] = None,
    fill_value: Optional[float] = None
) -> np.ndarray:
    
    if not isinstance(spectrogram, np.ndarray):
        warnings.warn(f"Ожидался numpy массив для apply_frequency_band_mask, получен {type(spectrogram)}", stacklevel=2)
        return spectrogram

    n_mels = spectrogram.shape[0]
    masked_spec = spectrogram.copy()

    # Определяем значение для заполнения
    current_fill_value = fill_value
    if current_fill_value is None:
        try:
            current_fill_value = np.min(spectrogram)
            if not np.isfinite(current_fill_value): current_fill_value = -80.0 # Запасное значение
        except ValueError: # Если спектрограмма пустая
             current_fill_value = -80.0


    # Обрезка снизу
    if lowcut_mel_idx is not None and lowcut_mel_idx > 0:
        actual_lowcut = min(lowcut_mel_idx, n_mels) # Ограничиваем индексом
        masked_spec[:actual_lowcut, :] = current_fill_value

    # Обрезка сверху
    if highcut_mel_idx is not None and highcut_mel_idx < n_mels - 1:
        actual_highcut = max(0, highcut_mel_idx) # Ограничиваем индексом
        masked_spec[actual_highcut + 1:, :] = current_fill_value

    return masked_spec


def apply_spectrogram_cleaning( # Переименованная функция
    spectrogram: np.ndarray, # Ожидаем спектрограмму УЖЕ В ДБ!
    # --- Параметры для Частотной Маски (замена фильтра спектрограммы) ---
    apply_freq_masking: bool = False, # Применять ли обрезку частот (по умолчанию ВЫКЛ)
    lowcut_mel_idx: Optional[int] = 5,    # Индекс Мел-бина для нижней частоты
    highcut_mel_idx: Optional[int] = 35,   # Индекс Мел-бина для верхней частоты
    mask_fill_value: Optional[float] = None, # Значение для заполнения обрезанных частот
    # --- Параметры для Порога ---
    apply_threshold: bool = True, # По умолчанию порог ВКЛЮЧЕН
    db_threshold: float = -40.0,  # Значение порога
    db_fill_value: float = -80.0  # Значение для заполнения тишины
) -> np.ndarray:
   
    if not isinstance(spectrogram, np.ndarray):
        warnings.warn(f"apply_spectrogram_cleaning ожидал numpy массив, получен {type(spectrogram)}. Возврат без изменений.", stacklevel=2)
        return spectrogram

    working_spec = spectrogram.copy() # Работаем с копией

    # Шаг 1: Маскирование частотных бинов (если включено)
    if apply_freq_masking:
        working_spec = apply_frequency_band_mask(
            spectrogram=working_spec,
            lowcut_mel_idx=lowcut_mel_idx,
            highcut_mel_idx=highcut_mel_idx,
            fill_value=mask_fill_value
        )

    # Шаг 2: Пороговая обработка (если включено)
    if apply_threshold:
        working_spec = apply_db_threshold_on_spectrogram(
            mel_spec_db=working_spec,
            db_threshold=db_threshold,
            db_fill_value=db_fill_value
        )

    return working_spec


def apply_frequency_mask(
    spectrogram: np.ndarray,
    max_mask_height: int,
    num_masks: int,
    mask_value: Optional[float] = None 
) -> np.ndarray:
    """
    Применяет маскирование по частоте к спектрограмме (SpecAugment).
    (код функции без изменений)
    """
    augmented_spec = spectrogram.copy() # Работаем с копией
    num_mels = augmented_spec.shape[0]
    if mask_value is None:
        mask_value = augmented_spec.mean()
        if not np.isfinite(mask_value): mask_value = 0.0 # Доп. проверка

    if num_mels == 0 or max_mask_height <= 0 or num_masks <= 0:
        return augmented_spec

    max_mask_height = min(max_mask_height, num_mels)

    for _ in range(num_masks):
        f = random.randint(0, max_mask_height)
        if f == 0: continue
        f0 = random.randint(0, num_mels - f)
        augmented_spec[f0:f0 + f, :] = mask_value

    return augmented_spec


# ... (код apply_time_mask) ...
def apply_time_mask(
    spectrogram: np.ndarray,
    max_mask_width: int,
    num_masks: int,
    mask_value: Optional[float] = None # <--- Исправлено
) -> np.ndarray:
    """
    Применяет маскирование по времени к спектрограмме (SpecAugment).
    (код функции без изменений)
    """
    augmented_spec = spectrogram.copy() # Работаем с копией
    num_steps = augmented_spec.shape[1]
    if mask_value is None:
        mask_value = augmented_spec.mean()
        if not np.isfinite(mask_value): mask_value = 0.0 # Доп. проверка

    if num_steps == 0 or max_mask_width <= 0 or num_masks <= 0:
        return augmented_spec

    max_mask_width = min(max_mask_width, num_steps)

    for _ in range(num_masks):
        t = random.randint(0, max_mask_width)
        if t == 0: continue
        t0 = random.randint(0, num_steps - t)
        augmented_spec[:, t0:t0 + t] = mask_value

    return augmented_spec


# --- Обновленная основная функция с более мягкими параметрами ---
def apply_spectrogram_augmentations(
    spectrogram: np.ndarray,
    freq_mask_prob: float = 0.5,
    time_mask_prob: float = 0.5,
    freq_mask_param: int = 7,
    time_mask_param: int = 12,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
    mask_value: Union[float, str] = 'mean' # <--- Исправлено
) -> np.ndarray:
    """
    Применяет набор аугментаций (SpecAugment) к спектрограмме с
    более консервативными параметрами по умолчанию.

    Args:
        spectrogram (np.ndarray): Входная спектрограмма [n_mels, time_steps].
        freq_mask_prob (float): Вероятность применения частотного маскирования.
        time_mask_prob (float): Вероятность применения временного маскирования.
        freq_mask_param (int): Параметр F для частотной маски (макс. высота).
        time_mask_param (int): Параметр T для временной маски (макс. ширина).
        num_freq_masks (int): Количество частотных масок (m_F).
        num_time_masks (int): Количество временных масок (m_T).
        mask_value (float | str): Значение для маскирования. 'mean' для среднего по спектрограмме,
                                  иначе использовать переданное float значение.

    Returns:
        np.ndarray: Аугментированная спектрограмма.
    """
    working_spec = spectrogram.copy()

    fill_value = None
    if isinstance(mask_value, (int, float)):
        fill_value = float(mask_value)
    elif mask_value == 'mean':
        fill_value = working_spec.mean()
        if not np.isfinite(fill_value):
             fill_value = 0.0
    # else: fill_value останется None, и вложенные функции сами вычислят среднее

    if random.random() < freq_mask_prob:
        working_spec = apply_frequency_mask(
            spectrogram=working_spec,
            max_mask_height=freq_mask_param,
            num_masks=num_freq_masks,
            mask_value=fill_value
        )

    if random.random() < time_mask_prob:
        working_spec = apply_time_mask(
            spectrogram=working_spec,
            max_mask_width=time_mask_param,
            num_masks=num_time_masks,
            mask_value=fill_value
        )

    return working_spec
