FROM python:3.13-slim
WORKDIR /app/backend
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV DLL_DATA_DIR=/data
EXPOSE 8000
# downloads + verifies the connectome on first start (cached in the /data volume)
CMD ["sh", "-c", "python -m scripts.fetch_data && uvicorn dll.api.app:app --host 0.0.0.0 --port 8000"]
