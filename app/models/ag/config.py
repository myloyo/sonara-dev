import os

# Путь к весам модели акустической гитары
MODEL_PATH = os.getenv("AG_MODEL_PATH", "model_weights_ag_improved_unet.pth")

# Параметры обработки звука
SAMPLE_RATE = 48000

# Параметры модели (совпадают с основной моделью)
MODEL_CONFIG = {
    "input_size": 1025,
    "base_channels": 32,
    "dropout_rate": 0.1
}

# Параметры STFT
N_FFT = 2048
HOP = 512
WIN_LENGTH = 2048
WINDOW = "hann"

# Параметры обработки overlap-add
CHUNK_SIZE = 15  # секунды
OVERLAP_RATIO = 0.3  # 30% перекрытия

# Параметры пост-процессинга для акустической гитары (мягкая обработка)
POST_PROCESSING = {
    "smoothing_factor": 0.15,  # меньше сглаживания для сохранения деталей
    "gating_threshold": 0.04,  # выше порог для щадящего режима
    "gating_floor": 0.2,  # больше сохраняем тихих деталей
    "low_freq_boost": 1.15,  # умеренное усиление низких частот
    "high_freq_attenuation": 0.92,  # мягкое ослабление высоких частот
    "lowpass_cutoff": 10000  # 10 kHz для сохранения воздушности
}

# Устройство
DEVICE = os.getenv("DEVICE", "cuda")
