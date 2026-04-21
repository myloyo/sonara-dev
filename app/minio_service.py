from minio import Minio
from .config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, BUCKET


def get_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def download_file(key: str) -> bytes:
    client = get_minio_client()
    response = client.get_object(BUCKET, key)
    data = response.read()
    response.close()
    return data


def upload_file(key: str, file_stream, file_size: int):
    client = get_minio_client()
    client.put_object(
        BUCKET,
        key,
        file_stream,
        length=file_size,
        content_type="audio/wav",
    )
