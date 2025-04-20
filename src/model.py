# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Настроим логгер для этого модуля
model_logger = logging.getLogger(__name__)
if not model_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    model_logger.addHandler(handler)
    # Уровень INFO по умолчанию, можно изменить на DEBUG для детальной отладки самой модели
    model_logger.setLevel(logging.INFO)

class CRNNModelModified(nn.Module):
    """
    Модифицированная CRNN модель с более глубокой/широкой CNN и дополнительным FC слоем.
    Предназначена для извлечения лучших признаков и потенциального преодоления плато лосса.
    """
    def __init__(self, n_mels, num_classes, rnn_hidden_size=256, rnn_layers=2, dropout=0.1, bidirectional=True):
        """
        Инициализация модифицированной CRNN модели.

        Args:
            n_mels (int): Количество Мел-фильтров (высота входной спектрограммы).
            num_classes (int): Количество классов (уникальные символы + CTC blank).
            rnn_hidden_size (int): Размер скрытого состояния в RNN.
            rnn_layers (int): Количество слоев в RNN.
            dropout (float): Вероятность dropout в RNN и перед FC слоями.
            bidirectional (bool): Использовать ли двунаправленный RNN.
        """
        super().__init__()
        # Проверка аргументов
        if not isinstance(n_mels, int) or n_mels <= 0:
            raise ValueError("n_mels должен быть положительным целым числом.")
        if not isinstance(num_classes, int) or num_classes <= 1:
             raise ValueError("num_classes должен быть целым числом > 1.")
        if not isinstance(rnn_hidden_size, int) or rnn_hidden_size <= 0:
             raise ValueError("rnn_hidden_size должен быть положительным целым числом.")
        if not isinstance(rnn_layers, int) or rnn_layers <= 0:
             raise ValueError("rnn_layers должен быть положительным целым числом.")

        model_logger.info(f"Инициализация CRNNModelModified с параметрами: n_mels={n_mels}, num_classes={num_classes}, rnn_hidden_size={rnn_hidden_size}, rnn_layers={rnn_layers}, dropout={dropout}, bidirectional={bidirectional}")
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional

        # --- Модифицированная CNN часть ---
        self.cnn = nn.Sequential(
            # Слой 1: -> (N, 32, n_mels/2, Time/2)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Слой 2: -> (N, 64, n_mels/4, Time/4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # Слой 3: -> (N, 128, n_mels/8, Time/4)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)), # Сжимаем только высоту

            # Слой 4: -> (N, 256, n_mels/16, Time/4)
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # Сжимаем только высоту
        )
        model_logger.info("CNN часть создана.")

        # --- Расчет размера выхода CNN для RNN ---
        try:
            with torch.no_grad():
                # Используем небольшой пример времени, чтобы не тратить много памяти
                dummy_input = torch.randn(1, 1, self.n_mels, 32)
                cnn_out = self.cnn(dummy_input)
                self.cnn_output_features = cnn_out.shape[1] # Каналы
                self.cnn_output_height = cnn_out.shape[2]   # Высота
                self.rnn_input_size = self.cnn_output_features * self.cnn_output_height
                model_logger.info(f"Расчетный выход CNN: Каналы={self.cnn_output_features}, Высота={self.cnn_output_height}")
                model_logger.info(f"Расчетный размер входа для RNN (Features): {self.rnn_input_size}")
                if self.cnn_output_height <= 0:
                     raise ValueError(f"Высота выхода CNN стала <= 0 ({self.cnn_output_height}). Упростите CNN или проверьте n_mels.")
                if self.rnn_input_size <= 0:
                     raise ValueError(f"Размер входа для RNN стал <= 0 ({self.rnn_input_size}). Проверьте архитектуру CNN.")
        except Exception as e:
            model_logger.error(f"КРИТИЧЕСКАЯ ОШИБКА при расчете выхода CNN: {e}", exc_info=True)
            raise RuntimeError("Не удалось определить размер выхода CNN для RNN.") from e

        # --- RNN часть (GRU) ---
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,       # Важно для reshape ниже
            dropout=dropout if rnn_layers > 1 else 0,
            bidirectional=bidirectional
        )
        model_logger.info("RNN (GRU) часть создана.")

        # --- Классификатор (FC слои) ---
        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size
        # Промежуточный FC слой
        self.fc1 = nn.Linear(rnn_output_size, rnn_output_size // 2)
        self.fc_dropout = nn.Dropout(dropout)
        # Выходной FC слой
        self.fc2 = nn.Linear(rnn_output_size // 2, num_classes)
        model_logger.info("FC часть создана.")

    def forward(self, x):
        """
        Прямой проход данных через модифицированную модель.
        Args:
            x (torch.Tensor): Вход [N, H, W] или [N, 1, H, W].
        Returns:
            torch.Tensor: Выход [W_new, N, NumClasses] для CTC.
        """
        # 0. Проверка и добавление размерности канала
        if x.dim() == 3:
            x = x.unsqueeze(1) # [N, H, W] -> [N, 1, H, W]
        elif x.dim() != 4 or x.shape[1] != 1:
             raise ValueError(f"Неожиданная форма входа в forward: {x.shape}. Ожидалась [N, H, W] или [N, 1, H, W].")

        # 1. CNN
        # Вход: [N, 1, H, W]
        x = self.cnn(x)
        # Выход: [N, C_out, H_new, W_new]

        # 2. Reshape для RNN (batch_first=True)
        N, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2) # -> [N, W_new, C_out, H_new]
        x = x.reshape(N, W, C * H) # -> [N, W_new(Time), Features=C*H]
        # Проверка совпадения размера признаков (на всякий случай)
        if x.shape[2] != self.rnn_input_size:
             raise RuntimeError(f"Несовпадение размера признаков для RNN! Ожидалось: {self.rnn_input_size}, Получено: {x.shape[2]}. Возможно, ошибка в __init__ или forward.")

        # 3. RNN
        # Вход: [N, Time, Features]
        x, _ = self.rnn(x)
        # Выход: [N, Time, rnn_output_size]

        # 4. Классификатор
        x = self.fc1(x)
        x = F.relu(x) # Используем ReLU для промежуточной активации
        x = self.fc_dropout(x)
        x = self.fc2(x)
        # Выход: [N, Time, NumClasses]

        # 5. Permute для CTC Loss
        # [N, Time, NumClasses] -> [Time, N, NumClasses]
        x = x.permute(1, 0, 2)

        return x

    def get_output_lengths(self, input_lengths):
        """
        Рассчитывает длину последовательности (временной оси) после прохождения через CNN.
        Args:
            input_lengths (torch.Tensor): Тензор с исходными длинами временной оси.
        Returns:
            torch.Tensor: Тензор с длинами временной оси после CNN.
        """
        if not isinstance(input_lengths, torch.Tensor):
             input_lengths = torch.tensor(input_lengths) # Преобразуем, если нужно

        lengths = input_lengths.clone().float()

        # Применяем формулу ТОЛЬКО для слоев, меняющих временную размерность (W)
        # Формула: L_out = floor((L_in + 2*p - d*(k-1) - 1)/s + 1)

        # 1. После первого MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # params W: k=2, s=2, p=0, d=1
        lengths = torch.floor((lengths + 2*0 - 1*(2-1) - 1) / 2 + 1)

        # 2. После второго MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # params W: k=2, s=2, p=0, d=1
        lengths = torch.floor((lengths + 2*0 - 1*(2-1) - 1) / 2 + 1)

        # Остальные слои MaxPool2d(2,1) имеют stride[1]=1, поэтому не меняют W

        # Преобразуем и гарантируем минимальную длину 1
        output_lengths = lengths.long()
        output_lengths = torch.clamp(output_lengths, min=1)

        # Проверки
        if torch.any(output_lengths <= 0):
             model_logger.warning(f"Обнаружена нулевая выходная длина! Input: {input_lengths.tolist()}, Output: {output_lengths.tolist()}")
        if output_lengths.shape != input_lengths.shape:
            model_logger.error(f"Размерность тензора длин изменилась! In: {input_lengths.shape}, Out: {output_lengths.shape}. Возвращаем исходные.")
            return input_lengths.long()

        # Отладка: можно раскомментировать для проверки
        # model_logger.debug(f"get_output_lengths: Input={input_lengths.tolist()}, Output={output_lengths.tolist()}")

        return output_lengths
