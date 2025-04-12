# data_processing/generate_spectrogram.py

import librosa
import librosa.util # Импортируем для normalize
import numpy as np
import os
import warnings
import sys
from typing import Union

# Попытка импорта scipy для фильтрации
try:
    from scipy.signal import butter, filtfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

def _apply_lowpass_filter(y_data: np.ndarray, cutoff: int, fs: int, order: int = 5) -> np.ndarray:
    # ... (код функции не изменился) ...
    if not _SCIPY_AVAILABLE:
        warnings.warn("Библиотека scipy не найдена. Фильтрация НЧ будет пропущена.", ImportWarning, stacklevel=2) # Добавил stacklevel
        return y_data
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0:
             warnings.warn(f"Частота среза ({cutoff} Hz) >= частоты Найквиста ({nyquist} Hz). Фильтрация пропущена.", stacklevel=2)
             return y_data
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y_filtered_data = filtfilt(b, a, y_data)
        return y_filtered_data
    except Exception as e:
        warnings.warn(f"Ошибка при применении фильтра: {e}. Возвращаются исходные данные.", stacklevel=2)
        return y_data

def generate_mel_spectrogram(
    audio_path: str,
    target_sr: int = 8000,
    n_fft: int = 512,
    hop_length: int = 64,
    n_mels: int = 40,
    fmin: int = 0,
    fmax: int = None,
    apply_filter: bool = False,
    cutoff_freq_filter: int = 2400,
    filter_order: int = 5,
    normalize_audio: bool = True # <<< Новый параметр
) -> Union[np.ndarray, None]:
    """
    Загружает аудиофайл, опционально нормализует громкость, опционально
    применяет ФНЧ, вычисляет Мел-спектрограмму и конвертирует ее в дБ.

    Args:
        # ... (остальные аргументы как раньше) ...
        normalize_audio (bool): Применять ли пиковую нормализацию к аудиосигналу
                                перед генерацией спектрограммы.

    Returns:
        np.ndarray | None: Логарифмическая Мел-спектрограмма (формат [n_mels, time_steps])
                           в виде NumPy массива float32, или None в случае ошибки.
    """
    if fmax is None:
        fmax = target_sr / 2

    try:
        # 1. Загрузка и ресемплинг аудио
        y, sr_loaded = librosa.load(audio_path, sr=target_sr, mono=True)

        if y.size == 0:
            warnings.warn(f"Предупреждение: Загружен пустой аудиофайл: {os.path.basename(audio_path)}", stacklevel=2)
            return None

        # 2. Нормализация аудио (waveform) <<< НОВЫЙ ШАГ
        if normalize_audio:
            # librosa.util.normalize выполняет пиковую нормализацию (макс. абс. значение = 1.0)
            y = librosa.util.normalize(y)
            # Проверка на случай, если аудио было полностью нулевым
            if np.all(y == 0):
                 warnings.warn(f"Предупреждение: Аудио стало нулевым после нормализации (возможно, исходный файл был тишиной): {os.path.basename(audio_path)}", stacklevel=2)
                 # Можно вернуть None или пустую спектрограмму, зависит от желаемого поведения.
                 # Вернем None для консистентности с другими ошибками.
                 return None


        # 3. Опциональная фильтрация
        if apply_filter:
             y = _apply_lowpass_filter(y, cutoff_freq_filter, target_sr, order=filter_order)
             # Дополнительная проверка на нулевой сигнал после фильтрации (маловероятно, но возможно)
             if np.all(y == 0):
                  warnings.warn(f"Предупреждение: Аудио стало нулевым после фильтрации: {os.path.basename(audio_path)}", stacklevel=2)
                  return None

        # 4. Генерация Мел-спектрограммы
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )

        # 5. Конвертация в дБ
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        if mel_spec_db.shape[1] == 0:
            warnings.warn(f"Предупреждение: Получена спектрограмма нулевой длины для: {os.path.basename(audio_path)}", stacklevel=2)
            return None

        return mel_spec_db.astype(np.float32)

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден - {audio_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nОшибка при обработке файла {os.path.basename(audio_path)}: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc(file=sys.stderr)
        return None