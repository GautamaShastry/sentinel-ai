FROM python:3.11-slim

WORKDIR /app

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-shm0 \
    libxcb-render0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
ENV PYTHONPATH=/app
EXPOSE 9101
CMD ["python", "-m", "app.worker"]
