"""
Скрипт для обработки аудио с помощью улучшенной U-Net модели клавиш с шумоподавлением
Использование: python process_audio_keys_improved_denoise.py input_audio.wav output_audio.wav

Добавлено пост-процессирование для удаления лишних шумов:
- Spectral gating (пороговая обработка спектрограммы)
- Smooth filtering (сглаживание по времени)
- High-frequency noise reduction
"""

import os
import sys
import torch
import librosa
import soundfile as sf
import numpy as np
from scipy.ndimage import median_filter

from app.model_unet_improved import ImprovedUNetSeparator
from app.utils_unet import stft_spectrogram, stft_to_audio

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "model_weights_keys_improved_unet.pth"
SAMPLE_RATE = 48000

# Параметры шумоподавления - МЯГКИЙ режим по умолчанию
NOISE_REDUCTION_CONFIG = {
    # Порог для спектрального гейтирования (очень мягкий)
    "spectral_gate_threshold": 0.005,
    
    # Мягкий порог (плавное затухание около порога)
    "soft_gate_threshold": 0.02,
    
    # Минимальный уровень сигнала (защита от полного обнуления)
    "min_signal_level": 0.0001,
    
    # High-frequency noise reduction (мягкое ослабление)
    "high_freq_cutoff": 0.9,  # Только самые высокие частоты
    "high_freq_attenuation": 0.85,  # Очень мягкое ослабление
    
    # Median filtering для сглаживания (маленькое окно)
    "median_filter_size": 1,  # Выключено по умолчанию
    
    # Temporal smoothing (сглаживание по времени)
    "temporal_smooth": False,  # Выключено по умолчанию
    "temporal_window": 3,
}


def apply_noise_reduction(magnitude_norm, config=None):
    """
    Применяет шумоподавление к спектрограмме.
    
    Args:
        magnitude_norm: Нормализованная спектрограмма (freq, time)
        config: Словарь с параметрами шумоподавления
    
    Returns:
        Обработанная спектрограмма
    """
    if config is None:
        config = NOISE_REDUCTION_CONFIG
    
    output = magnitude_norm.copy()
    
    # 1. Spectral Gating - удаляем очень тихие участки
    threshold = config["spectral_gate_threshold"]
    soft_threshold = config["soft_gate_threshold"]
    min_level = config["min_signal_level"]
    
    # Жёсткий порог - обнуляем всё что ниже
    output[output < threshold] = 0
    
    # Мягкий порог - плавное затухание
    mask = (output >= threshold) & (output < soft_threshold)
    if np.any(mask):
        # Плавная функция затухания (sigmoid)
        values = output[mask]
        smooth_mask = 1.0 / (1.0 + np.exp(-(values - (threshold + soft_threshold) / 2) * 50))
        output[mask] = values * smooth_mask
    
    # 2. High-frequency noise reduction - ослабляем высокие частоты
    high_freq_cutoff = config["high_freq_cutoff"]
    high_freq_attenuation = config["high_freq_attenuation"]
    
    freq_bins = output.shape[0]
    cutoff_bin = int(freq_bins * high_freq_cutoff)
    
    if cutoff_bin < freq_bins:
        # Плавное затухание к высоким частотам
        freq_axis = np.linspace(0, 1, freq_bins)
        attenuation_curve = np.ones(freq_bins)
        attenuation_curve[cutoff_bin:] = np.linspace(1, high_freq_attenuation, freq_bins - cutoff_bin)
        output = output * attenuation_curve[:, np.newaxis]
    
    # 3. Median filtering - удаляем одиночные выбросы
    median_size = config["median_filter_size"]
    if median_size > 1:
        output = median_filter(output, size=(median_size, median_size))
    
    # 4. Temporal smoothing - сглаживание по времени
    if config.get("temporal_smooth", False):
        window = config.get("temporal_window", 5)
        from scipy.ndimage import uniform_filter1d
        output = uniform_filter1d(output, size=window, axis=1)
    
    # Защита от полного обнуления
    output = np.clip(output, min_level, 1.0)
    
    return output


def load_model():
    """Загружает обученную модель"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    
    model = ImprovedUNetSeparator(input_size=1025, base_channels=32, dropout_rate=0.1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print(f"✓ Модель загружена из {MODEL_PATH}")
    print(f"  Устройство: {device}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Количество параметров: {num_params:,}")
    
    return model


def process_audio(model, input_path, output_path, chunk_size=15, debug=False, 
                  reference_path=None, use_denoise=True, noise_config=None):
    """
    Обрабатывает аудиофайл моделью с шумоподавлением и overlap-add для плавных переходов.
    """
    print(f"\nЗагружаю аудио: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")
    
    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
    print(f"✓ Аудио загружено: {len(audio)} семплов ({len(audio)/sr:.1f} сек)")
    
    if use_denoise:
        print(f"🔇 Шумоподавление: ВКЛЮЧЕНО")
        if debug and noise_config:
            print(f"  Порог спектрального гейта: {noise_config['spectral_gate_threshold']}")
            print(f"  Ослабление высоких частот: {noise_config['high_freq_attenuation']}")
    else:
        print(f"🔇 Шумоподавление: ВЫКЛЮЧЕНО")
    
    reference_mag = None
    if debug and reference_path and os.path.exists(reference_path):
        print(f"\nЗагружаю reference: {reference_path}")
        ref_audio, _ = librosa.load(reference_path, sr=SAMPLE_RATE)
        reference_mag, _ = stft_spectrogram(ref_audio, sr)
    
    if len(audio) < chunk_size * SAMPLE_RATE:
        print("Обрабатываю аудио целиком...")
        magnitude_norm, phase = stft_spectrogram(audio, sr)
        
        if debug:
            print(f"\n[DEBUG] Вход: shape={magnitude_norm.shape}, range=[{magnitude_norm.min():.4f}, {magnitude_norm.max():.4f}]")
        
        mag_tensor = torch.tensor(magnitude_norm).unsqueeze(0).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            output_mag = model(mag_tensor)
        
        output_mag = output_mag.squeeze(0).squeeze(0).cpu().numpy()
        output_mag = np.clip(output_mag, 0, 1)
        
        if use_denoise:
            output_mag = apply_noise_reduction(output_mag, config=noise_config)
        
        if debug:
            print(f"[DEBUG] Выход: shape={output_mag.shape}, range=[{output_mag.min():.4f}, {output_mag.max():.4f}]")
            print(f"[DEBUG] Ненулевых элементов: {np.sum(output_mag > 0.01)} / {output_mag.size} ({100*np.sum(output_mag > 0.01)/output_mag.size:.1f}%)")
        
        orig_time = output_mag.shape[1]
        if output_mag.shape[0] < 1025:
            output_mag_padded = np.zeros((1025, orig_time), dtype=output_mag.dtype)
            output_mag_padded[:output_mag.shape[0], :] = output_mag
            output_mag = output_mag_padded
        
        if phase.shape[0] != output_mag.shape[0]:
            phase_padded = np.zeros((output_mag.shape[0], phase.shape[1]), dtype=phase.dtype)
            phase_padded[:phase.shape[0], :] = phase
            phase = phase_padded
        if phase.shape[1] != output_mag.shape[1]:
            phase = phase[:, :output_mag.shape[1]]
        
        output_audio = stft_to_audio(output_mag, phase, sr)
    
    else:
        # OVERLAP-ADD обработка с плавными переходами
        chunk_samples = int(chunk_size * SAMPLE_RATE)
        overlap_samples = int(chunk_samples * 0.25)  # 25% перекрытие
        hop_samples = chunk_samples - overlap_samples
        
        output_audio = np.zeros(len(audio))
        window = np.hanning(2 * overlap_samples)  # Hann window для crossfade
        
        num_chunks = (len(audio) - chunk_samples) // hop_samples + 1
        print(f"Обрабатываю аудио с overlap-add ({num_chunks} чанков x {chunk_size}s, overlap=25%)...")
        
        for i in range(num_chunks):
            start = i * hop_samples
            end = start + chunk_samples
            
            if end > len(audio):
                end = len(audio)
                start = max(0, end - chunk_samples)
            
            chunk = audio[start:end]
            
            # Паддим чанк если нужно до полного размера
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            magnitude_norm, phase = stft_spectrogram(chunk, sr)
            
            mag_tensor = torch.tensor(magnitude_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                output_mag = model(mag_tensor)
            
            output_mag = output_mag.squeeze(0).squeeze(0).cpu().numpy()
            output_mag = np.clip(output_mag, 0, 1)
            
            if use_denoise:
                output_mag = apply_noise_reduction(output_mag, config=noise_config)
            
            orig_time = output_mag.shape[1]
            if output_mag.shape[0] < 1025:
                output_mag_padded = np.zeros((1025, orig_time), dtype=output_mag.dtype)
                output_mag_padded[:output_mag.shape[0], :] = output_mag
                output_mag = output_mag_padded
            
            if phase.shape[0] != output_mag.shape[0]:
                phase_padded = np.zeros((output_mag.shape[0], phase.shape[1]), dtype=phase.dtype)
                phase_padded[:phase.shape[0], :] = phase
                phase = phase_padded
            if phase.shape[1] != output_mag.shape[1]:
                phase = phase[:, :output_mag.shape[1]]
            
            output_chunk = stft_to_audio(output_mag, phase, sr)
            
            # Паддим output_chunk до размера чанка если нужно
            if len(output_chunk) < chunk_samples:
                output_chunk = np.pad(output_chunk, (0, chunk_samples - len(output_chunk)), mode='constant')
            elif len(output_chunk) > chunk_samples:
                output_chunk = output_chunk[:chunk_samples]
            
            # Применяем окно для overlap-add
            actual_chunk_len = end - start
            
            if i == 0:
                # Первый чанк - только fade out на конце
                if overlap_samples < actual_chunk_len:
                    fade_out = np.linspace(1, 0, overlap_samples)
                    output_chunk[:overlap_samples] *= np.linspace(0, 1, overlap_samples)  # fade in от предыдущего
                    output_chunk[-overlap_samples:] *= fade_out
                output_audio[start:end] += output_chunk[:actual_chunk_len]
            elif i == num_chunks - 1:
                # Последний чанк - только fade in на начале
                if overlap_samples < actual_chunk_len:
                    fade_in = np.linspace(0, 1, overlap_samples)
                    output_chunk[:overlap_samples] *= fade_in
                output_audio[start:end] += output_chunk[:actual_chunk_len]
            else:
                # Средние чанки - fade in и fade out
                if overlap_samples * 2 < actual_chunk_len:
                    fade_in = np.linspace(0, 1, overlap_samples)
                    fade_out = np.linspace(1, 0, overlap_samples)
                    output_chunk[:overlap_samples] *= fade_in
                    output_chunk[-overlap_samples:] *= fade_out
                output_audio[start:end] += output_chunk[:actual_chunk_len]
            
            progress = (i + 1) / num_chunks * 100
            print(f"  [{progress:5.1f}%] Обработано: {end/SAMPLE_RATE:.1f}s")
        
        # Нормализуем на overlap regions (где было сложение)
        # Создаем маску overlap regions
        overlap_mask = np.zeros(len(audio))
        for i in range(num_chunks):
            start = i * hop_samples
            end = min(start + chunk_samples, len(audio))
            overlap_mask[start:end] += 1
        
        # Делим на количество перекрытий (избегаем деления на 0)
        overlap_mask = np.maximum(overlap_mask, 1)
        output_audio = output_audio / overlap_mask
    
    max_val = np.max(np.abs(output_audio))
    if max_val > 1.0:
        output_audio = output_audio / max_val
    
    sf.write(output_path, output_audio, sr)
    print(f"\n✓ Результат сохранён: {output_path}")
    print(f"  Длина: {len(output_audio)} семплов ({len(output_audio)/sr:.1f} сек)")


def main():
    if len(sys.argv) < 3:
        print("="*70)
        print("Обработка аудио с улучшенной U-Net моделью + ШУМОПОДАВЛЕНИЕ")
        print("="*70)
        print("\nИспользование:")
        print("  python process_audio_keys_improved_denoise.py <input> <output> [--debug] [--no-denoise]")
        print("\nПримеры:")
        print("  python process_audio_keys_improved_denoise.py raw.wav output.wav")
        print("  python process_audio_keys_improved_denoise.py raw.wav output.wav --no-denoise")
        print("  python process_audio_keys_improved_denoise.py raw.wav output.wav --debug")
        print("\nМодель: Improved U-Net с Attention Gates")
        print("Шумоподавление: Spectral Gating + High-freq reduction + Median filter")
        print("="*70)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    debug = "--debug" in sys.argv
    use_denoise = "--no-denoise" not in sys.argv
    
    # Можно переопределить параметры через командную строку
    noise_config = NOISE_REDUCTION_CONFIG.copy()
    
    # Парсим кастомные параметры
    for i, arg in enumerate(sys.argv):
        if arg == "--gate-threshold" and i + 1 < len(sys.argv):
            noise_config["spectral_gate_threshold"] = float(sys.argv[i + 1])
        elif arg == "--high-freq-atten" and i + 1 < len(sys.argv):
            noise_config["high_freq_attenuation"] = float(sys.argv[i + 1])
    
    print("="*70)
    print("Обработка аудио клавиш улучшенной U-Net моделью")
    if use_denoise:
        print("[РЕЖИМ: С ШУМОПОДАВЛЕНИЕМ]")
    else:
        print("[РЕЖИМ: БЕЗ ШУМОПОДАВЛЕНИЯ]")
    if debug:
        print("[РЕЖИМ ОТЛАДКИ ВКЛЮЧЕН]")
    print("="*70)
    
    try:
        model = load_model()
        process_audio(model, input_file, output_file, debug=debug, 
                     use_denoise=use_denoise, noise_config=noise_config)
        
        print("\n" + "="*70)
        print("✓ Обработка успешно завершена!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
