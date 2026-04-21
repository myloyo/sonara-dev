"""
Анализ аудиоданных для датасета keyboard effects
Проверка консистентности данных: частоты, амплитуда, громкость
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Пути к данным
DATA_DIR = Path("./data/keys")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def analyze_audio_file(file_path):
    """Анализ одного аудиофайла"""
    try:
        # Загрузка аудио
        y, sr = librosa.load(file_path, sr=None)
        
        # Основные характеристики
        duration = len(y) / sr
        
        # Амплитуда
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_max = float(np.max(rms))
        rms_min = float(np.min(rms))
        
        # Пиковая амплитуда
        peak_amplitude = float(np.max(np.abs(y)))
        
        # Частотные характеристики (спектральный центр)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = float(np.mean(spectral_centroids))
        spectral_centroid_std = float(np.std(spectral_centroids))
        
        # Спектральный контраст
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = [float(x) for x in np.mean(spectral_contrast, axis=1)]
        
        # Частоты (спектральный rolloff)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
        
        # MFCC (для характеристики тембра)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = [float(np.mean(mfcc)) for mfcc in mfccs]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))
        
        # Громкость в dB
        db = librosa.amplitude_to_db(np.abs(y), ref=np.max)
        db_mean = float(np.mean(db))
        db_max = float(np.max(db))
        
        return {
            "file": str(file_path),
            "sample_rate": sr,
            "duration_sec": round(duration, 3),
            "samples": len(y),
            "amplitude": {
                "rms_mean": round(rms_mean, 6),
                "rms_std": round(rms_std, 6),
                "rms_max": round(rms_max, 6),
                "rms_min": round(rms_min, 6),
                "peak": round(peak_amplitude, 6)
            },
            "loudness_db": {
                "mean": round(db_mean, 2),
                "max": round(db_max, 2)
            },
            "frequency": {
                "spectral_centroid_mean": round(spectral_centroid_mean, 2),
                "spectral_centroid_std": round(spectral_centroid_std, 2),
                "spectral_rolloff_mean": round(spectral_rolloff_mean, 2)
            },
            "timbre": {
                "zcr_mean": round(zcr_mean, 6),
                "mfcc_mean": [round(m, 2) for m in mfcc_mean]
            },
            "spectral_contrast_mean": [round(s, 2) for s in spectral_contrast_mean]
        }
    except Exception as e:
        import traceback
        return {"file": str(file_path), "error": str(e), "traceback": traceback.format_exc()}


def plot_comparison(analyses, output_path="./data/audio_analysis.png"):
    """Визуализация сравнения файлов"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Audio Data Consistency Analysis', fontsize=16)
    
    files = [Path(a['file']).name for a in analyses if 'error' not in a]
    
    if not files:
        print("Нет данных для визуализации")
        plt.close()
        return
    
    # 1. RMS (энергия сигнала)
    rms_means = [a['amplitude']['rms_mean'] for a in analyses if 'error' not in a]
    axes[0, 0].bar(files, rms_means, color='steelblue')
    axes[0, 0].set_title('RMS Mean (Energy)')
    axes[0, 0].set_ylabel('RMS')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Peak Amplitude
    peaks = [a['amplitude']['peak'] for a in analyses if 'error' not in a]
    axes[0, 1].bar(files, peaks, color='coral')
    axes[0, 1].set_title('Peak Amplitude')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Loudness (dB)
    db_means = [a['loudness_db']['mean'] for a in analyses if 'error' not in a]
    axes[0, 2].bar(files, db_means, color='seagreen')
    axes[0, 2].set_title('Mean Loudness (dB)')
    axes[0, 2].set_ylabel('dB')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Spectral Centroid
    centroids = [a['frequency']['spectral_centroid_mean'] for a in analyses if 'error' not in a]
    axes[1, 0].bar(files, centroids, color='goldenrod')
    axes[1, 0].set_title('Spectral Centroid (Brightness)')
    axes[1, 0].set_ylabel('Hz')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Zero Crossing Rate
    zcrs = [a['timbre']['zcr_mean'] for a in analyses if 'error' not in a]
    axes[1, 1].bar(files, zcrs, color='mediumpurple')
    axes[1, 1].set_title('Zero Crossing Rate')
    axes[1, 1].set_ylabel('ZCR')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Duration
    durations = [a['duration_sec'] for a in analyses if 'error' not in a]
    axes[1, 2].bar(files, durations, color='crimson')
    axes[1, 2].set_title('Duration')
    axes[1, 2].set_ylabel('Seconds')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График сохранен: {output_path}")


def check_consistency(analyses):
    """Проверка консистентности данных"""
    valid = [a for a in analyses if 'error' not in a]
    
    if len(valid) < 2:
        return {"status": "insufficient_data", "message": "Недостаточно файлов для анализа"}
    
    # Статистика вариации
    rms_values = [a['amplitude']['rms_mean'] for a in valid]
    peak_values = [a['amplitude']['peak'] for a in valid]
    db_values = [a['loudness_db']['mean'] for a in valid]
    centroid_values = [a['frequency']['spectral_centroid_mean'] for a in valid]
    
    # Коэффициент вариации (CV = std/mean)
    def cv(values):
        mean = np.mean(values)
        if mean == 0:
            return 0
        return np.std(values) / mean
    
    results = {
        "amplitude_consistency": {
            "rms_cv": round(cv(rms_values), 4),
            "peak_cv": round(cv(peak_values), 4),
            "assessment": "good" if cv(rms_values) < 0.3 else "needs_normalization"
        },
        "loudness_consistency": {
            "db_range": round(max(db_values) - min(db_values), 2),
            "assessment": "good" if max(db_values) - min(db_values) < 10 else "needs_normalization"
        },
        "frequency_consistency": {
            "centroid_cv": round(cv(centroid_values), 4),
            "assessment": "good" if cv(centroid_values) < 0.3 else "varied_content"
        },
        "sample_rate_consistency": len(set(a['sample_rate'] for a in valid)) == 1,
        "sample_rates": list(set(a['sample_rate'] for a in valid))
    }
    
    return results


def main():
    print("=" * 60)
    print("АНАЛИЗ АУДИОДАТАСЕТА ДЛЯ KEYBOARD EFFECTS")
    print("=" * 60)
    
    # Сбор всех аудиофайлов
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
        audio_files.extend(DATA_DIR.glob(f"**/{ext}"))
    
    print(f"\nНайдено аудиофайлов: {len(audio_files)}")
    
    # Анализ каждого файла
    analyses = []
    for file_path in sorted(audio_files):
        print(f"\nАнализ: {file_path}")
        analysis = analyze_audio_file(file_path)
        analyses.append(analysis)
        
        if 'error' not in analysis:
            print(f"  Sample Rate: {analysis['sample_rate']} Hz")
            print(f"  Duration: {analysis['duration_sec']} sec")
            print(f"  RMS Mean: {analysis['amplitude']['rms_mean']:.6f}")
            print(f"  Peak Amplitude: {analysis['amplitude']['peak']:.6f}")
            print(f"  Loudness (dB): {analysis['loudness_db']['mean']:.2f} dB")
            print(f"  Spectral Centroid: {analysis['frequency']['spectral_centroid_mean']:.2f} Hz")
        else:
            print(f"  ОШИБКА: {analysis['error']}")
            print(f"  Traceback: {analysis.get('traceback', 'N/A')}")
    
    # Сохранение результатов в JSON
    results_path = DATA_DIR / "analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(analyses, f, indent=2)
    print(f"\nРезультаты сохранены: {results_path}")
    
    # Проверка консистентности
    print("\n" + "=" * 60)
    print("ПРОВЕРКА КОНСИСТЕНТНОСТИ")
    print("=" * 60)
    
    consistency = check_consistency(analyses)
    print(json.dumps(consistency, indent=2))
    
    # Визуализация
    plot_comparison(analyses)
    
    # Рекомендации
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ")
    print("=" * 60)
    
    valid = [a for a in analyses if 'error' not in a]
    
    if len(valid) < 2:
        print("⚠️  Недостаточно данных для анализа консистентности")
        print("   → Добавьте больше аудиофайлов в датасет")
    else:
        if consistency['amplitude_consistency']['assessment'] == 'needs_normalization':
            print("⚠️  Амплитуда: Требуется нормализация громкости между сэмплами")
            print("   → Используйте RMS нормализацию или peak normalization")
        
        if consistency['loudness_consistency']['assessment'] == 'needs_normalization':
            print("⚠️  Громкость: Большой разброс уровней громкости (dB)")
            print("   → Примените loudness normalization к целевому уровню (например, -18 dB)")
        
        if not consistency['sample_rate_consistency']:
            print(f"⚠️  Sample Rate: Разные частоты дискретизации {consistency['sample_rates']}")
            print("   → Конвертируйте все файлы к единому sample rate (рекомендуется 48000 Hz)")
        
        if consistency['frequency_consistency']['assessment'] == 'varied_content':
            print("ℹ️  Частоты: Различный спектральный состав (это нормально для разных клавиш)")
            print("   → Модель должна обучаться на разнообразных частотных паттернах")
    
    # Общие рекомендации
    print("\n📋 ОБЩИЕ РЕКОМЕНДАЦИИ:")
    print("1. Для обрезки частот: используйте highpass/lowpass фильтры (80Hz - 15kHz для клавиш)")
    print("2. Для усиления громкости: примените компрессию + лимитер")
    print("3. Для обучения модели: создайте пары (raw, processed) с консистентными параметрами")
    print(f"4. Текущий размер датасета: {len(valid)} файлов")
    
    return analyses, consistency


if __name__ == "__main__":
    main()
