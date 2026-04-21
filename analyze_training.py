import librosa
import numpy as np

sr = 44100

# Загружаем файлы
ref, _ = librosa.load('data/keys/reference.wav', sr=sr)
output, _ = librosa.load('keys_processing_output.wav', sr=sr)
expected, _ = librosa.load('reference render 001.wav', sr=sr)

print("="*60)
print("АМПЛИТУДА И ГРОМКОСТЬ")
print("="*60)

def analyze_audio(y, name):
    rms = np.sqrt(np.mean(y**2))
    peak = np.max(np.abs(y))
    db = 20 * np.log10(rms + 1e-8)
    print(f'{name:25} | RMS: {rms:7.4f} | Peak: {peak:7.4f} | dB: {db:6.1f}')

analyze_audio(ref, "Input (reference.wav)")
analyze_audio(output, "Model output")
analyze_audio(expected, "Expected (reference render 001.wav)")
analyze_audio(processed_train, "Training target")

print("\n" + "="*60)
print("СПЕКТРАЛЬНЫЙ АНАЛИЗ (STFT)")
print("="*60)

def spectral_analysis(y, name):
    D = librosa.stft(y)
    mag = np.abs(D)
    
    # Энергия по полосам
    low = mag[:50].sum()      # 0-500 Hz
    mid = mag[50:500].sum()   # 500-5k Hz  
    high = mag[500:].sum()    # 5k+ Hz
    
    total = low + mid + high
    print(f'\n{name}:')
    print(f'  Низкие (0-500Hz):    {low:10.1f} ({100*low/total:5.1f}%)')
    print(f'  Средние (500-5kHz):  {mid:10.1f} ({100*mid/total:5.1f}%)')
    print(f'  Высокие (5kHz+):     {high:10.1f} ({100*high/total:5.1f}%)')
    return low, mid, high, total

print('\nИнпут:')
spectral_analysis(ref, "reference.wav")
print('\nОутпут модели:')
spectral_analysis(output, "Model output")
print('\nОжидаемый результат:')
spectral_analysis(expected, "Expected")
print('\nТренировочный пример:')
spectral_analysis(processed_train, "Training target")

print("\n" + "="*60)
print("РАЗНИЦА МЕЖДУ ОЖИДАНИЕМ И РЕАЛЬНОСТЬЮ")
print("="*60)

ref_rms = np.sqrt(np.mean(ref**2))
out_rms = np.sqrt(np.mean(output**2))
exp_rms = np.sqrt(np.mean(expected**2))

print(f'Модель усилила на: {20*np.log10(out_rms/ref_rms):.1f} dB')
print(f'Ожидается усилить на: {20*np.log10(exp_rms/ref_rms):.1f} dB')
print(f'РАЗНИЦА: {20*np.log10(exp_rms/out_rms):.1f} dB (не хватает)')
