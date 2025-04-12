# data_processing/augmentations.py

import numpy as np
import random
from typing import Union, Optional

# Функции apply_frequency_mask и apply_time_mask остаются без изменений
# ... (код apply_frequency_mask) ...
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
