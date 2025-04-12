# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CRNNModel(nn.Module):
    def __init__(self, n_mels, num_classes, rnn_hidden_size=128, rnn_layers=2, dropout=0.1):
        """
        Инициализация CRNN модели.

        Args:
            n_mels (int): Количество Мел-фильтров (высота входной спектрограммы).
            num_classes (int): Количество классов (уникальные символы + CTC blank).
            rnn_hidden_size (int): Размер скрытого состояния в RNN.
            rnn_layers (int): Количество слоев в RNN.
            dropout (float): Вероятность dropout в RNN и перед линейным слоем.
        """
        super().__init__()
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers

        # --- CNN Часть ---
        # Вход: [batch, 1, n_mels, time_steps]
        self.cnn = nn.Sequential(
            # Слой 1
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # Уменьшаем высоту и время -> [b, 32, n_mels/2, time/2]

            # Слой 2
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # -> [b, 64, n_mels/4, time/4]

            # Слой 3
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Можно добавить еще MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) если нужно еще уменьшить время

            # Слой 4 (опционально, может уменьшить высоту до 1, если n_mels позволяет)
            # Убедимся, что высота после пулингов не стала < 1
            # Пример: если n_mels=40, после 2х пулингов (2,2) высота будет 10.
            # Добавим свертку, которая сожмет высоту до 1
            nn.Conv2d(128, 128, kernel_size=(n_mels // 4, 3), stride=(1, 1), padding=(0, 1)), # Ядро по высоте = оставшаяся высота
            nn.BatchNorm2d(128),
            nn.ReLU() # -> [b, 128, 1, time/4]
        )

        # Рассчитываем размер входа для RNN
        # После CNN: [batch, channels, height=1, width=time/4]
        # Нам нужно: [batch, channels*height, width]
        # RNN ожидает: [seq_len(width), batch, features(channels*height)]
        # В нашем случае height = 1 после последнего Conv
        self.rnn_input_size = 128 * 1 # channels * height

        # --- RNN Часть ---
        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False, # Ожидаем [seq_len, batch, features]
            dropout=dropout if rnn_layers > 1 else 0 # Dropout между слоями RNN
        )

        # --- Линейный Слой (Классификатор) ---
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes) # *2 т.к. bidirectional
        self.dropout_fc = nn.Dropout(dropout)

    def forward(self, x):
        """
        Прямой проход данных через модель.

        Args:
            x (torch.Tensor): Входной тензор спектрограмм [batch, n_mels, time_steps].

        Returns:
            torch.Tensor: Выход модели [time_steps_out, batch, num_classes] (до log_softmax).
                          time_steps_out - длина последовательности после CNN.
        """
        # Добавляем канал: [batch, n_mels, time_steps] -> [batch, 1, n_mels, time_steps]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Прогоняем через CNN
        x = self.cnn(x) # -> [batch, channels, height=1, width=time/4]

        # Подготовка к RNN: [batch, channels, height, width] -> [width, batch, channels*height]
        batch_size, channels, height, width = x.size()
        x = x.permute(3, 0, 1, 2) # [width, batch, channels, height]
        x = x.view(width, batch_size, channels * height) # [width(seq_len), batch, features]

        # Прогоняем через RNN
        x, _ = self.rnn(x) # -> [seq_len, batch, rnn_hidden_size * 2]

        # Прогоняем через линейный слой
        x = self.dropout_fc(x)
        x = self.fc(x) # -> [seq_len, batch, num_classes]

        return x

    def get_output_lengths(self, input_lengths):
        """
        Вычисляет длины выходных последовательностей после CNN части.
        Это **приблизительный** расчет, основанный на stride пулинг слоев.
        Точный расчет зависит от padding и kernel_size.
        Здесь мы считаем, что каждый MaxPool2d(stride=(2,2)) уменьшает время в 2 раза.

        Args:
            input_lengths (torch.Tensor): Тензор с исходными длинами спектрограмм [batch].

        Returns:
            torch.Tensor: Тензор с длинами последовательностей после CNN [batch].
        """
        # Пройдемся по слоям CNN и найдем все пулинги с уменьшением времени
        lengths = input_lengths.clone().float()
        for layer in self.cnn:
            if isinstance(layer, nn.MaxPool2d):
                # Учитываем stride по временной оси (индекс 3 для kernel_size/stride кортежа (H, W))
                stride_time = layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride
                kernel_time = layer.kernel_size[1] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                padding_time = layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding

                # Формула расчета выходной длины после пулинга/свертки
                lengths = torch.floor((lengths + 2 * padding_time - (kernel_time - 1) - 1) / stride_time + 1)

            # Если есть Conv2d со stride по времени > 1, тоже нужно учесть
            # (В нашей текущей архитектуре stride везде (1,1) у Conv2d)

        # Важно: CTCLoss ожидает целочисленные длины
        return lengths.long()