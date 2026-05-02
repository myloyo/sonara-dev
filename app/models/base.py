"""
Базовый интерфейс для всех обработчиков аудио.
"""
from abc import ABC, abstractmethod
import io


class BaseProcessor(ABC):
    """
    Абстрактный класс для процессоров аудио.
    Все инструменты должны реализовывать этот интерфейс.
    """
    
    @abstractmethod
    def process(self, input_bytes: bytes, output_format: str = "WAV") -> io.BytesIO:
        """
        Обработать аудиофайл.
        
        Args:
            input_bytes: Содержимое аудиофайла в байтах
            output_format: Формат выходного файла (WAV, MP3, FLAC и т.д.)
            
        Returns:
            io.BytesIO: Буфер с обработанным аудио
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """Загрузить модель (вызывается один раз при инициализации)."""
        pass
