import torch.nn as nn


class GRUSeparator(nn.Module):
    """
    BiDirectional GRU-based model для обработки аудио эффектов (STFT-based sound effects processing).
    
    Архитектура:
    - Двунаправленный GRU для захватывания контекста в обе стороны
    - Layer Normalization для стабилизации обучения
    - Многослойные FC слои с ReLU активацией
    - Residual connection (skip connection) для улучшения градиентного потока
    - Регуляризация через Dropout и Batch Normalization
    - Входные данные: STFT спектрограммы (1025 частотных бинов, 44.1 kHz, N_FFT=2048, HOP=512)
    
    Args:
        input_size (int): Размер входных признаков (STFT бины), default=1025 (N_FFT//2 + 1)
        hidden_size (int): Размер скрытого состояния GRU, default=256
        num_layers (int): Количество слоев GRU, default=2
        fc_hidden (int): Размер скрытых слоев FC, default=512
        dropout_rate (float): Вероятность Dropout, default=0.2
    """
    
    def __init__(self, input_size=1025, hidden_size=256, num_layers=2, 
                 fc_hidden=512, dropout_rate=0.2):
        super(GRUSeparator, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Layer Normalization перед GRU для стабилизации входов
        self.layer_norm_input = nn.LayerNorm(input_size)
        
        # Двунаправленный GRU (выход будет hidden_size * 2)
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Layer Normalization после GRU
        self.layer_norm_gru = nn.LayerNorm(hidden_size * 2)
        
        # FC слои с улучшенной структурой
        self.fc = nn.Sequential(
            # Первый слой: расширение
            nn.Linear(hidden_size * 2, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Второй слой: промежуточный
            nn.Linear(fc_hidden, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # Третий слой: сжатие к исходному размеру
            nn.Linear(fc_hidden, input_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for name, param in self.named_parameters():
            if 'weight_hh' in name or 'weight_ih' in name:
                # Ортогональная инициализация для GRU
                nn.init.orthogonal_(param.data)
            elif 'weight' in name and 'gru' not in name:
                # Xavier инициализация для FC слоев (только для 2D+ тензоров)
                if param.data.dim() >= 2:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 1, freq, time) - мел-спектрограмма
            
        Returns:
            output: Tensor того же размера (batch, 1, freq, time)
        """
        # Проверка входных размеров
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, 1, freq, time), got {x.dim()}D")
        
        # Распаковка: (batch, 1, freq, time)
        batch_size, _, freq_bins, time_steps = x.shape
        
        # Проверка размерности частотных бинов
        if freq_bins != self.input_size:
            raise ValueError(
                f"Expected freq_bins={self.input_size}, "
                f"got {freq_bins}"
            )
        
        # Трансформация:  (batch, 1, freq, time) → (batch, freq, time) → (batch, time, freq)
        x = x.squeeze(1)  # (batch, freq, time)
        x = x.permute(0, 2, 1)  # (batch, time, freq)
        
        # Layer Normalization на входе
        x = self.layer_norm_input(x)  # (batch, time, freq)
        
        # GRU обработка
        gru_out, _ = self.gru(x)  # (batch, time, hidden*2)
        
        # Layer Normalization после GRU
        gru_out = self.layer_norm_gru(gru_out)  # (batch, time, hidden*2)
        
        # Сохраняем исходный выход для skip connection
        skip = x  # (batch, time, freq)
        
        # FC слои обработка
        # BatchNorm1d требует (batch, features, ...), поэтому трансформируем
        fc_in = gru_out.reshape(batch_size * time_steps, -1)  # (batch*time, hidden*2)
        fc_out = self.fc(fc_in)  # (batch*time, freq)
        fc_out = fc_out.reshape(batch_size, time_steps, self.input_size)  # (batch, time, freq)
        
        # Skip connection (residual): добавляем исходный сигнал
        output = fc_out + skip  # (batch, time, freq)
        
        # Восстанавливаем оригинальную форму: (batch, 1, freq, time)
        output = output.permute(0, 2, 1)  # (batch, freq, time)
        output = output.unsqueeze(1)  # (batch, 1, freq, time)
        
        return output
    
    def __repr__(self):
        """Красивый вывод архитектуры модели"""
        return (
            f"GRUSeparator(\n"
            f"  input_size={self.input_size},\n"
            f"  hidden_size={self.hidden_size},\n"
            f"  gru_layers=2 (bidirectional),\n"
            f"  fc_hidden=256,\n"
            f"  features: LayerNorm + BiGRU + LayerNorm + FC(3 слоя) + Skip Connection\n"
            f")"
        )
