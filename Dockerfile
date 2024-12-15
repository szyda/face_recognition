FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libboost-python-dev \
    libboost-all-dev \
    libx11-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip --upgrade && pip install --upgrade pip

COPY requirements.txt /app/requirements.txt
COPY backend/ /app/backend/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "backend/app.py"]

