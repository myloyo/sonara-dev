from contextlib import asynccontextmanager
import threading
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from .kafka_service import kafka_consumer_loop
from .model_service_unet import process_audio_file_improved


consumer_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global consumer_thread
    print("[App] Starting ML Audio Processor")
    consumer_thread = threading.Thread(target=kafka_consumer_loop, daemon=True)
    consumer_thread.start()
    print("[App] Kafka consumer started")
    yield
    print("[App] Shutting down")


app = FastAPI(title="ML Audio Processor", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "ML Audio Processor"}


@app.get("/")
def root():
    return {"service": "ML Audio Processor", "version": "1.0", "mode": "worker"}


@app.post("/process-keys")
async def process_keys(file: UploadFile = File(...)):
    """
    Обработка аудиофайла моделью Improved UNet для клавиш.
    Возвращает обработанный файл в том же формате, что и входной.
    """
    contents = await file.read()
    filename = file.filename or "audio.wav"
    ext = filename.split('.')[-1].upper()
    
    # Маппинг расширений на форматы soundfile
    format_map = {
        "WAV": "WAV",
        "MP3": "MP3",
        "FLAC": "FLAC",
        "OGG": "OGG",
        "M4A": "M4A",
        "AIFF": "AIFF",
    }
    output_format = format_map.get(ext, "WAV")
    
    mime_map = {
        "WAV": "audio/wav",
        "MP3": "audio/mpeg",
        "FLAC": "audio/flac",
        "OGG": "audio/ogg",
        "M4A": "audio/mp4",
        "AIFF": "audio/aiff",
    }
    media_type = mime_map.get(output_format, "audio/wav")
    
    result_buf = process_audio_file_improved(contents, output_format=output_format)
    
    output_filename = filename.rsplit('.', 1)[0] + "_processed." + ext.lower()
    
    return StreamingResponse(
        result_buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={output_filename}"}
    )
