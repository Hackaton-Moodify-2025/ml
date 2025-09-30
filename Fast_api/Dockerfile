# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Системные зависимости (по минимуму)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements для лучшего кеширования
COPY requirements_api.txt /tmp/requirements_api.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements_api.txt \
    && pip install fastapi uvicorn[standard]

# Копируем весь проект (нужен корень для импорта fast_analysis_api.py)
COPY . /app

# Открываем порт
EXPOSE 8000

# Запуск Uvicorn сервера
CMD ["uvicorn", "Fast_api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]



