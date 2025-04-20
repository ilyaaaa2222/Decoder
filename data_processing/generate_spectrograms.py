# data_processing/generate_spectrograms.py

import librosa
import librosa.util
import numpy as np
import os
import warnings
import sys
from typing import Union, Optional

# --- ИМПОРТ ТОЛЬКО НУЖНОЙ ФУНКЦИИ ОЧИСТКИ СПЕКТРОГРАММЫ ---
# Убедись, что файл spectrogram_processing.py существует в той же папке data_processing/
try:
    # Импортируем ТОЛЬКО функцию для обработки спектрограммы
    from .spectrogram_processing import apply_db_threshold_on_spectrogram
except ImportError:
    print("Ошибка импорта функции apply_db_threshold_on_spectrogram из data_processing.spectrogram_processing")
    # Заглушка, если импорт не удался
    def apply_db_threshold_on_spectrogram(spec, **kwargs):
        warnings.warn("Не удалось импортировать apply_db_threshold_on_spectrogram! Пороговая обработка не будет работать.", ImportWarning)
        return spec
# -----------------------------

# --- ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ФИЛЬТРАЦИИ (ОСТАЕТСЯ ЗДЕСЬ) ---
# Попытка импорта scipy для фильтрации
try:
    from scipy.signal import butter, filtfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    # Предупреждение будет внутри _apply_bandpass_filter

def _apply_bandpass_filter(y_data: np.ndarray, lowcut: int, highcut: int, fs: int, order: int = 5) -> np.ndarray:
    """Применяет полосовой фильтр Баттерворта."""
    if not _SCIPY_AVAILABLE:
        warnings.warn("Библиотека scipy не найдена. Полосовая фильтрация пропущена.", ImportWarning, stacklevel=2)
        return y_data
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        if low <= 0 or high >= 1.0 or low >= high:
            warnings.warn(f"Некорректные частоты среза для полосового фильтра (low={lowcut}, high={highcut}, nyquist={nyquist}). Фильтрация пропущена.", stacklevel=2)
            return y_data
        try:
            b, a = butter(order, [low, high], btype='bandpass', analog=False)
        except ValueError: # Для старых версий scipy
             b, a = butter(order, [low, high], btype='band', analog=False)
        y_filtered_data = filtfilt(b, a, y_data)
        return y_filtered_data
    except Exception as e:
        warnings.warn(f"Ошибка при применении полосового фильтра: {e}. Возвращаются исходные данные.", stacklevel=2)
        return y_data
# --- КОНЕЦ ВСПОМОГАТЕЛЬНОЙ ФУНКЦИИ ---

def generate_mel_spectrogram(
    audio_path: str,
    target_sr: int = 8000,
    n_fft: int = 512,
    hop_length: int = 64,
    n_mels: int = 40,
    fmin: int = 0,
    fmax: int = None,
    normalize_audio: bool = True,
    # --- Параметры фильтрации АУДИО ---
    apply_bandpass_filter: bool = False, # По умолчанию ВЫКЛЮЧЕНА
    lowcut_freq: int = 500,
    highcut_freq: int = 1500,
    filter_order: int = 5,
    # --- Параметры пороговой обработки СПЕКТРОГРАММЫ ---
    apply_db_threshold: bool = False,  # По умолчанию ВЫКЛЮЧЕНА
    db_threshold: float = -40.0,
    db_fill_value: float = -80.0
) -> Union[np.ndarray, None]:
    """
    Загружает аудио, нормализует, опционально фильтрует аудио,
    вычисляет Мел-спектрограмму, конвертирует в дБ и опционально
    применяет пороговую обработку к спектрограмме.
    """
    if fmax is None:
        fmax = target_sr / 2

    try:
        # 1. Загрузка и ресемплинг
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if y.size == 0: return None

        # 2. Нормализация
        if normalize_audio:
            y = librosa.util.normalize(y)
            if np.all(y == 0): return None

        # 3. Фильтрация АУДИО (если включено)
        if apply_bandpass_filter:
            y_processed = _apply_bandpass_filter(
                y,
                lowcut=lowcut_freq,
                highcut=highcut_freq,
                fs=target_sr,
                order=filter_order
            )
            if np.all(y_processed == 0):
                 warnings.warn(f"Аудио стало нулевым после фильтрации для {os.path.basename(audio_path)}", stacklevel=2)
                 return None
        else:
            y_processed = y # Используем исходный (или нормализованный) сигнал

        # 4. Генерация Мел-спектрограммы из обработанного y_processed
        mel_spec = librosa.feature.melspectrogram(
            y=y_processed,
            sr=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            fmin=fmin, fmax=fmax
        )

        # 5. Конвертация в дБ
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 6. Пороговая обработка СПЕКТРОГРАММЫ (вызов импортированной функции)
        if apply_db_threshold:
            mel_spec_db = apply_db_threshold_on_spectrogram(
                mel_spec_db=mel_spec_db,
                db_threshold=db_threshold,
                db_fill_value=db_fill_value
            )

        if mel_spec_db.shape[1] == 0: return None

        return mel_spec_db.astype(np.float32)

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден - {audio_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\nОшибка при обработке файла {os.path.basename(audio_path)}: {e}", file=sys.stderr)
        return None
