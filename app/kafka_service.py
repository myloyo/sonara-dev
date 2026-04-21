import json
import time
import requests
from confluent_kafka import Consumer, Producer
from .config import (
    KAFKA_BOOTSTRAP,
    INPUT_TOPIC,
    OUTPUT_TOPIC_OK,
    OUTPUT_TOPIC_FAIL,
    BACKEND_URL,
    BACKEND_TIMEOUT,
)
from .minio_service import download_file, upload_file
from .model_service import process_audio_file


def get_kafka_consumer():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "group.id": "ml-service",
        "auto.offset.reset": "earliest",
    })
    consumer.subscribe([INPUT_TOPIC])
    return consumer


def get_kafka_producer():
    return Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})


def publish_result(job_id: str, output_key: str, success: bool, error_msg: str = None):
    producer = get_kafka_producer()
    topic = OUTPUT_TOPIC_OK if success else OUTPUT_TOPIC_FAIL
    
    message = {
        "jobId": job_id,
    }
    
    if success:
        message["outputKey"] = output_key
    else:
        message["error"] = error_msg
    
    producer.produce(
        topic,
        json.dumps(message).encode()
    )
    producer.flush()


def update_backend_job(job_id: str, status: str, output_key: str = None, error_msg: str = None):
    try:
        payload = {"status": status}
        if output_key:
            payload["outputKey"] = output_key
        if error_msg:
            payload["errorMessage"] = error_msg
        
        requests.put(
            f"{BACKEND_URL}/{job_id}",
            json=payload,
            timeout=BACKEND_TIMEOUT
        )
        print(f"[Backend] Updated job {job_id} with status {status}")
    except Exception as e:
        print(f"[Backend] Error updating job {job_id}: {e}")


def process_job(message):
    data = json.loads(message.value().decode())
    job_id = data["jobId"]
    input_key = data["inputKey"]
    output_key = f"output/{job_id}.wav"
    
    try:
        print(f"[Job] Processing job {job_id}")
        
        print(f"[Download] Downloading {input_key}")
        input_bytes = download_file(input_key)
        print(f"[Download] Downloaded {len(input_bytes)} bytes")
        
        print(f"[ML] Processing with model")
        result_buf = process_audio_file(input_bytes)
        result_buf.seek(0)
        result_bytes = result_buf.getvalue()
        print(f"[ML] Processing completed, output size: {len(result_bytes)} bytes")
        
        print(f"[Upload] Uploading result to {output_key}")
        result_buf.seek(0)
        upload_file(output_key, result_buf, result_bytes)
        print(f"[Upload] Uploaded successfully")
        
        update_backend_job(job_id, "Completed", output_key=output_key)
        publish_result(job_id, output_key, success=True)
        print(f"[Success] Job {job_id} completed")
        
    except Exception as e:
        print(f"[Error] Job {job_id} failed: {e}")
        update_backend_job(job_id, "Failed", error_msg=str(e))
        publish_result(job_id, None, success=False, error_msg=str(e))


def kafka_consumer_loop():
    print("[Consumer] Starting...")
    
    max_retries = 5
    retry_count = 0
    consumer = None
    
    while retry_count < max_retries:
        try:
            consumer = get_kafka_consumer()
            print("[Consumer] Connected to Kafka")
            break
        except Exception as e:
            retry_count += 1
            print(f"[Consumer] Connection attempt {retry_count}/{max_retries} failed: {e}")
            time.sleep(5)
    
    if consumer is None:
        print("[Consumer] Failed to connect after retries")
        return
    
    while True:
        try:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"[Consumer] Error: {msg.error()}")
                continue
            
            process_job(msg)
        except Exception as e:
            print(f"[Consumer] Loop error: {e}")
            time.sleep(1)
