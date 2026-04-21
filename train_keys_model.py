import os
import time
import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.model import GRUSeparator
from app.utils import split_and_save, AudioEffectDataset, spectral_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути для датасета клавиш
BASE_DIR = "data"
KEYS_RAW_DIR = os.path.join(BASE_DIR, "keys", "raw")
KEYS_PROCESSED_DIR = os.path.join(BASE_DIR, "keys", "processed")

# Директории для сегментов
SEGMENTS_RAW_DIR = os.path.join(BASE_DIR, "keys_segments", "raw")
SEGMENTS_PROCESSED_DIR = os.path.join(BASE_DIR, "keys_segments", "processed")

os.makedirs(SEGMENTS_RAW_DIR, exist_ok=True)
os.makedirs(SEGMENTS_PROCESSED_DIR, exist_ok=True)

print("="*70)
print("Обучение модели обработки клавиш (Keys Model Training)")
print("="*70)

# Получаем список файлов и сопоставляем пары
def match_file_pairs(raw_dir, processed_dir):
    """
    Сопоставляет raw и processed файлы по базовому имени.
    Ожидается что файлы называются:
      - raw: name_raw.wav или name.wav
      - processed: name_processed.wav
    """
    raw_files = glob.glob(os.path.join(raw_dir, "*.wav"))
    processed_files = glob.glob(os.path.join(processed_dir, "*.wav"))
    
    pairs = []
    unmatched_raw = []
    
    for raw_path in raw_files:
        raw_name = Path(raw_path).stem  # например '1939_raw'
        # Убираем '_raw' суффикс если есть
        base_name = raw_name.replace('_raw', '')
        
        matched = False
        for proc_path in processed_files:
            proc_name = Path(proc_path).stem  # например '1939_processed'
            
            # Проверяем если processed файл содержит базовое имя и '_processed'
            if base_name in proc_name and '_processed' in proc_name:
                pairs.append((raw_path, proc_path))
                matched = True
                print(f"  ✓ Пара: {Path(raw_path).name} → {Path(proc_path).name}")
                break
        
        if not matched:
            unmatched_raw.append(raw_path)
            print(f"  ⚠️  Не найдена пара для: {Path(raw_path).name}")
    
    return pairs, unmatched_raw

print("\nСопоставление пар raw → processed:")
file_pairs, unmatched = match_file_pairs(KEYS_RAW_DIR, KEYS_PROCESSED_DIR)

if not file_pairs:
    print("ERROR: Не найдено пар файлов!")
    exit(1)

print(f"\nНайдено пар: {len(file_pairs)}")
if unmatched:
    print(f"⚠️  Без пары: {len(unmatched)} файлов")

print("\n" + "="*70)
print("ЭТАП 1: Разбиение аудио на 15-секундные сегменты")
print("="*70)

# Разбиваем каждый файл на 15-секундные сегменты
SEGMENT_DURATION = 15.0  # 15 секунд
SAMPLE_RATE = 48000  # Единый sample rate для всех файлов

# Проверяем, есть ли уже сегменты
raw_segments = glob.glob(os.path.join(SEGMENTS_RAW_DIR, "*.wav"))
processed_segments = glob.glob(os.path.join(SEGMENTS_PROCESSED_DIR, "*.wav"))

if len(raw_segments) > 0 and len(processed_segments) > 0:
    print("\n✓ Сегменты уже созданы ранее!")
    print(f"  Найдено сегментов сырых записей: {len(raw_segments)}")
    print(f"  Найдено сегментов обработанных записей: {len(processed_segments)}")
    print("  Пропускаем повторное разбиение.")
else:
    print("\nСегменты не найдены, создаю новые...")
    print("\nОбработка сырых записей...")
    for raw_file, _ in file_pairs:
        split_and_save(raw_file, SEGMENTS_RAW_DIR, segment_duration=SEGMENT_DURATION, sample_rate=SAMPLE_RATE)

    print("\nОбработка обработанных записей...")
    for _, proc_file in file_pairs:
        split_and_save(proc_file, SEGMENTS_PROCESSED_DIR, segment_duration=SEGMENT_DURATION, sample_rate=SAMPLE_RATE)

print("\n" + "="*70)
print("ЭТАП 2: Подготовка датасета")
print("="*70)

# Создаем датасет
dataset = AudioEffectDataset(SEGMENTS_RAW_DIR, SEGMENTS_PROCESSED_DIR, sample_rate=SAMPLE_RATE)
print(f"\nОбщее количество сегментов: {len(dataset)}")

# Загружаем данные батчами
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
print("Размер батча: 4")
print(f"Количество батчей: {len(loader)}")

# Инициализируем модель
print("\n" + "="*70)
print("ЭТАП 3: Инициализация модели")
print("="*70)

model = GRUSeparator().to(device)
print("Модель: GRUSeparator")
print(f"Используется устройство: {device}")

# Функция потерь и оптимизатор
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("Функция потерь: MSELoss + Spectral Loss")
print("Оптимизатор: Adam (lr=1e-3)")

# Функция для обучения эпохи
def train_epoch(model, loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 70)

    for batch_i, (clean_mel, processed_mel) in enumerate(loader, 1):
        clean_mel = clean_mel.to(device)
        processed_mel = processed_mel.to(device)

        optimizer.zero_grad()

        t0 = time.time()
        output = model(clean_mel)
        
        # Комбинированная функция потерь
        loss_mse = criterion(output, processed_mel)
        loss_spec = spectral_loss(
            output.detach(),
            processed_mel.detach()
        )

        loss = loss_mse + 0.5 * loss_spec

        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        batch_time = time.time() - t0
        total_loss += loss.item()

        if batch_i % 5 == 0 or batch_i == len(loader):
            print(
                f"Batch {batch_i:3d}/{len(loader):3d}  "
                f"| MSE={loss_mse.item():.4f}  "
                f"| Spec={loss_spec.item():.4f}  "
                f"| Total={loss.item():.4f}  "
                f"| Time={batch_time:.1f}s"
            )

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} завершен: средняя потеря = {avg_loss:.4f}")
    return avg_loss


# Обучение модели
print("\n" + "="*70)
print("ЭТАП 4: Обучение модели на 30 эпох")
print("="*70)

num_epochs = 30
epoch_losses = []

start_time = time.time()

try:
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, loader, criterion, optimizer, epoch, num_epochs)
        epoch_losses.append(avg_loss)
        
        # Каждые 10 эпох выводим статистику
        if epoch % 10 == 0:
            print(f"\n>>> Прогресс: {epoch}/{num_epochs} эпох завершено")

except KeyboardInterrupt:
    print("\n\nОбучение прервано пользователем.")
except Exception as e:
    print(f"\n\nОшибка при обучении: {e}")
    raise

total_time = time.time() - start_time

# Сохранение весов модели
print("\n" + "="*70)
print("ЭТАП 5: Сохранение модели")
print("="*70)

model_path = "model_weights_keys_2204.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✓ Модель сохранена: {model_path}")

# Выводим статистику
print("\n" + "="*70)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*70)
print(f"Количество эпох: {num_epochs}")
print(f"Начальная потеря: {epoch_losses[0]:.4f}")
print(f"Финальная потеря: {epoch_losses[-1]:.4f}")
print(f"Улучшение: {((epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100):.1f}%")
print(f"Общее время обучения: {total_time/60:.1f} минут ({total_time:.0f} секунд)")
print(f"Среднее время per epoch: {total_time/num_epochs:.1f} сек")
print("\n✓ Обучение завершено успешно!")
print("="*70)
