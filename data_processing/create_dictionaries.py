# Пример в prepare_environment.py или аналогичном
import pandas as pd
import json
from dataset import create_char_maps # Импортируем функцию

TRAIN_CSV = 'processed/csv/train_processed.csv'
CHAR_MAP_FILE = 'processed/char_maps/char_map.json'
INDEX_MAP_FILE = 'processed/char_maps/index_map.json'
NUM_CLASSES_FILE = 'processed/char_maps/num_classes.txt' # Можно сохранить и кол-во классов

print("Создание/проверка словарей символов...")

try:
    df_train = pd.read_csv(TRAIN_CSV)
    if df_train.empty:
         raise ValueError(f"Файл {TRAIN_CSV} пуст.")

    char_map, index_map, num_classes = create_char_maps(df_train)

    # Создаем папку, если ее нет
    import os
    os.makedirs(os.path.dirname(CHAR_MAP_FILE), exist_ok=True)

    # Сохраняем словари в JSON
    with open(CHAR_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False, indent=4)
    print(f"Словарь char_map сохранен в {CHAR_MAP_FILE}")

    # index_map ключи - это int, JSON их преобразует в str при сохранении.
    # При загрузке нужно будет преобразовать обратно в int.
    with open(INDEX_MAP_FILE, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, ensure_ascii=False, indent=4)
    print(f"Словарь index_map сохранен в {INDEX_MAP_FILE}")

    # Сохраняем количество классов
    with open(NUM_CLASSES_FILE, 'w') as f:
        f.write(str(num_classes))
    print(f"Количество классов ({num_classes}) сохранено в {NUM_CLASSES_FILE}")

except FileNotFoundError:
    print(f"Ошибка: Не найден файл {TRAIN_CSV}")
except Exception as e:
    print(f"Ошибка при создании или сохранении словарей: {e}")