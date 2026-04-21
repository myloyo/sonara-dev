# ML Audio Processor

Асинхронный Kafka worker для обработки аудиофайлов с моделью GRU.

## Быстрый старт

```bash
docker-compose up -d
curl http://localhost:8000/health
```

## Архитектура

```
Kafka (job.prepared)
  ↓
ML Service (скачать → обработать → загрузить)
  ↓
Kafka (job.completed/failed)
```

1. Backend отправляет: `{"jobId": "xxx", "inputKey": "input/xxx.wav"}`
2. ML Service скачивает из MinIO, обрабатывает, загружает результат
3. Обновляет Backend через PUT и публикует результат в Kafka

## Сервисы

- ML Service: http://localhost:8000 (/health, /)
- MinIO Console: http://localhost:9001 (minio/minio123)
- Kafka: localhost:9092

## Структура

```
app/
  main.py           - FastAPI (endpoints, lifespan)
  config.py         - Переменные окружения
  kafka_service.py  - Consumer/producer логика
  minio_service.py  - Download/upload файлов
  model_service.py  - Загрузка модели, обработка
  model.py          - GRU архитектура
  utils.py          - Утилиты аудио
```

## Конфигурация

Переменные в `app/config.py`:

- `KAFKA_BOOTSTRAP` (default: kafka:9092)
- `MINIO_ENDPOINT` (default: minio:9000)
- `BACKEND_URL` (default: http://backend:8080/api/jobs)
- `BUCKET` (default: audio-files)
- `DEVICE` (default: cuda)

## Backend интеграция

See: [BACKEND_INTEGRATION.md](BACKEND_INTEGRATION.md)
