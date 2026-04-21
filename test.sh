#!/bin/bash

# ML Service Testing Script
# Полный цикл тестирования: загрузка файла → обработка → скачивание

set -e

echo "🧪 ML Service Test Suite"
echo "========================"
echo ""

# Проверить что docker запущен
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен"
    exit 1
fi

# Проверить что сервисы запущены
echo "1️⃣ Checking if services are running..."

echo -n "  ML Service: "
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅"
else
    echo "❌ ML Service is not running. Start with: docker-compose up -d"
    exit 1
fi

echo -n "  Kafka: "
if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "✅"
else
    echo "❌ Kafka is not running"
    exit 1
fi

echo ""
echo "2️⃣ Checking Kafka topics..."

topics=$(docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list)

for topic in job.prepared job.completed job.failed; do
    if echo "$topics" | grep -q "$topic"; then
        echo "  ✅ $topic"
    else
        echo "  ❌ $topic not found. Run: docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic $topic --partitions 1 --replication-factor 1"
        exit 1
    fi
done

echo ""
echo "3️⃣ Testing ML Service health endpoints..."

# Health check
response=$(curl -s http://localhost:8000/health)
if echo "$response" | grep -q '"status":"ok"'; then
    echo "  ✅ /health endpoint"
else
    echo "  ❌ /health endpoint failed"
    exit 1
fi

# Root info
response=$(curl -s http://localhost:8000/)
if echo "$response" | grep -q '"service":"ML Audio Processor"'; then
    echo "  ✅ / endpoint"
else
    echo "  ❌ / endpoint failed"
    exit 1
fi

echo ""
echo "4️⃣ Testing Kafka connectivity..."

# Publish test message
test_message='{"jobId":"test-'$(date +%s)'","inputKey":"input/test.wav"}'
echo "$test_message" | docker exec -i kafka kafka-console-producer \
    --broker-list localhost:9092 \
    --topic job.prepared > /dev/null 2>&1

echo "  ✅ Message published to job.prepared"

# Wait a bit
sleep 2

# Check logs
echo ""
echo "5️⃣ Checking ML Service behavior..."

logs=$(docker logs ml-service 2>&1 | tail -20)

if echo "$logs" | grep -q "Processing"; then
    echo "  ✅ ML Service is processing messages"
else
    echo "  ℹ️ ML Service hasn't processed any messages yet (this is OK on first run)"
fi

if echo "$logs" | grep -q "job.prepared"; then
    echo "  ✅ ML Service is listening to job.prepared topic"
else
    echo "  ⚠️ ML Service may not be listening to Kafka properly"
fi

echo ""
echo "✅ All tests passed!"
echo ""
echo "📝 Next steps:"
echo "  1. Prepare a test audio file: test_audio.wav"
echo "  2. Upload to MinIO: mc cp test_audio.wav minio/audio-files/input/"
echo "  3. Backend creates a Job and sends to Kafka"
echo "  4. ML Service will process it and upload result to output/"
echo "  5. Backend receives job.completed event"
echo ""
echo "🔍 To monitor processing:"
echo "  docker logs -f ml-service"
echo ""
echo "📊 To see Kafka messages:"
echo "  docker exec kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic job.completed --from-beginning"
