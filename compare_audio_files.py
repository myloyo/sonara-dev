#!/usr/bin/env python3
"""
Скрипт для сравнения оригинального и обработанного аудиофайлов.
Анализирует различные метрики качества и различия между файлами.

Использование:
    poetry run python compare_audio_files.py <original_file> <processed_file>

Пример:
    poetry run python compare_audio_files.py data/bass/ref1_long.wav res7.wav
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Параметры анализа
SAMPLE_RATE = 48000
N_FFT = 2048
HOP_LENGTH = 512


def load_audio(path):
    """Загружает аудиофайл."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    
    audio, sr = librosa.load(path, sr=SAMPLE_RATE)
    return audio, sr


def compute_snr(original, processed):
    """
    Вычисляет SNR (Signal-to-Noise Ratio) между файлами.
    Положительное SNR означает, что сигнал сохранился.
    Отрицательное SNR означает, что добавлены шумы/артефакты.
    """
    # Нормализуем оба файла к одинаковой громкости
    orig_norm = original / (np.max(np.abs(original)) + 1e-8)
    proc_norm = processed / (np.max(np.abs(processed)) + 1e-8)
    
    # Обрезаем до одинаковой длины
    min_len = min(len(orig_norm), len(proc_norm))
    orig_norm = orig_norm[:min_len]
    proc_norm = proc_norm[:min_len]
    
    # Разница как "шум"
    noise = orig_norm - proc_norm
    signal_power = np.sum(orig_norm ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power < 1e-10:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def compute_spectral_difference(original, processed, sr=SAMPLE_RATE):
    """
    Вычисляет разницу в спектральном содержании.
    """
    # STFT обоих файлов
    orig_stft = np.abs(librosa.stft(original, n_fft=N_FFT, hop_length=HOP_LENGTH))
    proc_stft = np.abs(librosa.stft(processed, n_fft=N_FFT, hop_length=HOP_LENGTH))
    
    # Нормализуем
    orig_stft = orig_stft / (np.max(orig_stft) + 1e-8)
    proc_stft = proc_stft / (np.max(proc_stft) + 1e-8)
    
    # Обрезаем до одинакового размера
    min_freq = min(orig_stft.shape[0], proc_stft.shape[0])
    min_time = min(orig_stft.shape[1], proc_stft.shape[1])
    
    orig_stft = orig_stft[:min_freq, :min_time]
    proc_stft = proc_stft[:min_freq, :min_time]
    
    # Разница
    diff = np.abs(orig_stft - proc_stft)
    
    return {
        'mean_diff': np.mean(diff),
        'max_diff': np.max(diff),
        'std_diff': np.std(diff),
        'correlation': np.corrcoef(orig_stft.flatten(), proc_stft.flatten())[0, 1]
    }


def compute_frequency_analysis(original, processed, sr=SAMPLE_RATE):
    """
    Анализирует частотное содержание файлов.
    """
    # Spectral centroids
    orig_centroid = librosa.feature.spectral_centroid(y=original, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    proc_centroid = librosa.feature.spectral_centroid(y=processed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # Spectral rolloff
    orig_rolloff = librosa.feature.spectral_rolloff(y=original, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    proc_rolloff = librosa.feature.spectral_rolloff(y=processed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # Spectral bandwidth
    orig_bandwidth = librosa.feature.spectral_bandwidth(y=original, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    proc_bandwidth = librosa.feature.spectral_bandwidth(y=processed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    
    # RMS energy
    orig_rms = librosa.feature.rms(y=original, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    proc_rms = librosa.feature.rms(y=processed, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    
    return {
        'centroid': {
            'original': np.mean(orig_centroid),
            'processed': np.mean(proc_centroid),
            'diff': np.mean(proc_centroid) - np.mean(orig_centroid)
        },
        'rolloff': {
            'original': np.mean(orig_rolloff),
            'processed': np.mean(proc_rolloff),
            'diff': np.mean(proc_rolloff) - np.mean(orig_rolloff)
        },
        'bandwidth': {
            'original': np.mean(orig_bandwidth),
            'processed': np.mean(proc_bandwidth),
            'diff': np.mean(proc_bandwidth) - np.mean(orig_bandwidth)
        },
        'rms': {
            'original': np.mean(orig_rms),
            'processed': np.mean(proc_rms),
            'diff': np.mean(proc_rms) - np.mean(orig_rms)
        }
    }


def compute_high_frequency_energy(original, processed, sr=SAMPLE_RATE, threshold=5000):
    """
    Вычисляет энергию в высокочастотном диапазоне (> threshold Hz).
    """
    # STFT
    orig_stft = librosa.stft(original, n_fft=N_FFT, hop_length=HOP_LENGTH)
    proc_stft = librosa.stft(processed, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Частотные бины
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    
    # Индексы высоких частот
    high_freq_mask = freqs > threshold
    high_freq_indices = np.where(high_freq_mask)[0]
    
    # Энергия в высоких частотах
    orig_high_energy = np.mean(np.abs(orig_stft[high_freq_indices, :]) ** 2)
    proc_high_energy = np.mean(np.abs(proc_stft[high_freq_indices, :]) ** 2)
    
    # Отношение
    if orig_high_energy > 1e-10:
        ratio = proc_high_energy / orig_high_energy
    else:
        ratio = float('inf')
    
    return {
        'original': orig_high_energy,
        'processed': proc_high_energy,
        'ratio': ratio,
        'reduction_db': 10 * np.log10(orig_high_energy / (proc_high_energy + 1e-10)) if proc_high_energy > 0 else float('inf')
    }


def compute_zero_crossing_rate(original, processed, sr=SAMPLE_RATE):
    """
    Вычисляет rate пересечения нуля (индикатор шума/высоких частот).
    """
    orig_zcr = librosa.feature.zero_crossing_rate(original, frame_length=2048, hop_length=512)[0]
    proc_zcr = librosa.feature.zero_crossing_rate(processed, frame_length=2048, hop_length=512)[0]
    
    return {
        'original': np.mean(orig_zcr),
        'processed': np.mean(proc_zcr),
        'diff': np.mean(proc_zcr) - np.mean(orig_zcr),
        'diff_percent': (np.mean(proc_zcr) - np.mean(orig_zcr)) / (np.mean(orig_zcr) + 1e-8) * 100
    }


def analyze_audio_files(original_path, processed_path):
    """
    Полный анализ двух аудиофайлов.
    """
    print("="*70)
    print("АНАЛИЗ АУДИОФАЙЛОВ")
    print("="*70)
    
    # Загрузка
    print(f"\nЗагрузка файлов...")
    print(f"  Оригинал: {original_path}")
    print(f"  Обработанный: {processed_path}")
    
    original, sr = load_audio(original_path)
    processed, _ = load_audio(processed_path)
    
    print(f"\n  Оригинал: {len(original)} сэмплов ({len(original)/sr:.2f} сек)")
    print(f"  Обработанный: {len(processed)} сэмплов ({len(processed)/sr:.2f} сек)")
    
    # SNR
    print(f"\n{'='*70}")
    print("1. SIGNAL-TO-NOISE RATIO (SNR)")
    print("="*70)
    snr = compute_snr(original, processed)
    print(f"  SNR: {snr:.2f} dB")
    if snr > 20:
        print("  ✓ Отлично: файлы очень похожи")
    elif snr > 10:
        print("  ○ Хорошо: заметные различия")
    elif snr > 0:
        print("  ⚠ Различия значительные")
    else:
        print("  ✗ Плохо: файлы сильно отличаются (возможны артефакты)")
    
    # Спектральная разница
    print(f"\n{'='*70}")
    print("2. СПЕКТРАЛЬНАЯ РАЗНИЦА")
    print("="*70)
    spectral = compute_spectral_difference(original, processed, sr)
    print(f"  Средняя разница: {spectral['mean_diff']:.4f}")
    print(f"  Максимальная разница: {spectral['max_diff']:.4f}")
    print(f"  Стандартное отклонение: {spectral['std_diff']:.4f}")
    print(f"  Корреляция: {spectral['correlation']:.4f}")
    if spectral['correlation'] > 0.95:
        print("  ✓ Отличная корреляция")
    elif spectral['correlation'] > 0.8:
        print("  ○ Хорошая корреляция")
    else:
        print("  ⚠ Низкая корреляция - спектры сильно отличаются")
    
    # Частотный анализ
    print(f"\n{'='*70}")
    print("3. ЧАСТОТНЫЙ АНАЛИЗ")
    print("="*70)
    freq = compute_frequency_analysis(original, processed, sr)
    
    print("\n  Spectral Centroid (центр масс спектра):")
    print(f"    Оригинал: {freq['centroid']['original']:.1f} Hz")
    print(f"    Обработанный: {freq['centroid']['processed']:.1f} Hz")
    print(f"    Изменение: {freq['centroid']['diff']:+.1f} Hz")
    
    print("\n  Spectral Rolloff (граница 85% энергии):")
    print(f"    Оригинал: {freq['rolloff']['original']:.1f} Hz")
    print(f"    Обработанный: {freq['rolloff']['processed']:.1f} Hz")
    print(f"    Изменение: {freq['rolloff']['diff']:+.1f} Hz")
    
    print("\n  Spectral Bandwidth (ширина спектра):")
    print(f"    Оригинал: {freq['bandwidth']['original']:.1f} Hz")
    print(f"    Обработанный: {freq['bandwidth']['processed']:.1f} Hz")
    print(f"    Изменение: {freq['bandwidth']['diff']:+.1f} Hz")
    
    print("\n  RMS Energy (громкость):")
    print(f"    Оригинал: {freq['rms']['original']:.4f}")
    print(f"    Обработанный: {freq['rms']['processed']:.4f}")
    print(f"    Изменение: {freq['rms']['diff']:+.4f}")
    
    # Анализ высоких частот
    print(f"\n{'='*70}")
    print("4. АНАЛИЗ ВЫСОКИХ ЧАСТОТ (> 5 kHz)")
    print("="*70)
    high_freq = compute_high_frequency_energy(original, processed, sr, threshold=5000)
    print(f"  Энергия ВЧ (оригинал): {high_freq['original']:.6f}")
    print(f"  Энергия ВЧ (обработанный): {high_freq['processed']:.6f}")
    print(f"  Отношение (proc/orig): {high_freq['ratio']:.2f}")
    print(f"  Подавление ВЧ: {high_freq['reduction_db']:.2f} dB")
    
    if high_freq['ratio'] < 0.5:
        print("  ✓ ВЧ подавлены (хорошо для баса)")
    elif high_freq['ratio'] < 1.0:
        print("  ○ ВЧ немного подавлены")
    elif high_freq['ratio'] < 2.0:
        print("  ○ ВЧ почти не изменились")
    else:
        print("  ⚠ ВЧ усилены (возможный источник шума)")
    
    # Zero Crossing Rate
    print(f"\n{'='*70}")
    print("5. ZERO CROSSING RATE (индикатор шума)")
    print("="*70)
    zcr = compute_zero_crossing_rate(original, processed, sr)
    print(f"  ZCR (оригинал): {zcr['original']:.4f}")
    print(f"  ZCR (обработанный): {zcr['processed']:.4f}")
    print(f"  Изменение: {zcr['diff']:+.4f} ({zcr['diff_percent']:+.1f}%)")
    
    if zcr['diff_percent'] > 50:
        print("  ⚠ ZCR значительно вырос - возможны высокочастотные артефакты")
    elif zcr['diff_percent'] > 20:
        print("  ○ ZCR немного вырос")
    elif zcr['diff_percent'] > -20:
        print("  ✓ ZCR почти не изменился")
    else:
        print("  ✓ ZCR уменьшился - ВЧ подавлены")
    
    # Итоговый вывод
    print(f"\n{'='*70}")
    print("ИТОГОВЫЙ ВЫВОД")
    print("="*70)
    
    # Оценка качества
    score = 0
    max_score = 100
    
    # SNR оценка
    if snr > 20:
        score += 25
    elif snr > 10:
        score += 15
    elif snr > 0:
        score += 5
    
    # Корреляция
    if spectral['correlation'] > 0.95:
        score += 25
    elif spectral['correlation'] > 0.8:
        score += 15
    elif spectral['correlation'] > 0.6:
        score += 5
    
    # ВЧ подавление
    if high_freq['ratio'] < 0.5:
        score += 25
    elif high_freq['ratio'] < 1.0:
        score += 15
    elif high_freq['ratio'] < 2.0:
        score += 5
    
    # ZCR
    if zcr['diff_percent'] < 20:
        score += 25
    elif zcr['diff_percent'] < 50:
        score += 15
    elif zcr['diff_percent'] < 100:
        score += 5
    
    print(f"\n  Оценка качества обработки: {score}/{max_score}")
    
    if score >= 80:
        print("  ✓ ОТЛИЧНО: Обработка работает корректно")
    elif score >= 60:
        print("  ○ ХОРОШО: Обработка работает, но есть небольшие артефакты")
    elif score >= 40:
        print("  ⚠ УДОВЛЕТВОРИТЕЛЬНО: Заметные артефакты обработки")
    else:
        print("  ✗ ПЛОХО: Обработка добавляет значительные артефакты")
    
    print("\n" + "="*70)
    
    return {
        'snr': snr,
        'spectral': spectral,
        'frequency': freq,
        'high_freq': high_freq,
        'zcr': zcr,
        'score': score
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nОшибка: укажите два файла для сравнения")
        print("\nПример:")
        print("  poetry run python compare_audio_files.py original.wav processed.wav")
        sys.exit(1)
    
    original_file = sys.argv[1]
    processed_file = sys.argv[2]
    
    if not os.path.exists(original_file):
        print(f"Ошибка: файл не найден: {original_file}")
        sys.exit(1)
    
    if not os.path.exists(processed_file):
        print(f"Ошибка: файл не найден: {processed_file}")
        sys.exit(1)
    
    analyze_audio_files(original_file, processed_file)


if __name__ == "__main__":
    main()
