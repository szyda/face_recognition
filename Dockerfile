FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y \
    cmake \
    libboost-all-dev \
    g++ \
    libx11-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    && apt-get clean

RUN python -m ensurepip --upgrade && \
    pip install --upgrade pip

COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "backend/app.py"]

