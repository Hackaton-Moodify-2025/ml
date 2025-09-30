# Gazprombank Reviews API

FastAPI-сервис для анализа отзывов о банке с использованием различных LLM API.

## Возможности

- Анализ отзывов с определением тем и тональности
- Поддержка OpenAI, YandexGPT и GigaChat API
- Два набора тем: базовый (9 тем) и расширенный (22 темы)
- REST API с автоматической документацией

## Быстрый запуск

### 1. Настройка переменных окружения

Создайте файл `.env` в корне проекта:

```bash
# Для OpenAI
OPENAI_API_KEY=your_openai_api_key

# Для YandexGPT
YANDEX_API_KEY=your_yandex_api_key
YANDEX_FOLDER_ID=your_yandex_folder_id

# Для GigaChat
GIGACHAT_TOKEN=your_gigachat_token
```

### 2. Запуск через Docker Compose

```bash
# Сборка и запуск
docker-compose up --build

# Запуск в фоне
docker-compose up -d --build
```

### 3. Проверка работы

- API доступно по адресу: http://localhost:8000
- Документация: http://localhost:8000/docs
- Проверка здоровья: http://localhost:8000/health

## Использование API

### Анализ отзывов

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "api_provider": "openai",
    "model_name": "gpt-3.5-turbo",
    "topics_version": "old",
    "reviews": [
      {
        "id": "1",
        "title": "Отличное обслуживание",
        "text": "Очень доволен работой отделения банка"
      }
    ]
  }'
```

### Параметры запроса

- `api_provider`: "openai", "yandex", "gigachat"
- `model_name`: название модели (для OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4o")
- `topics_version`: "old" (9 тем) или "new" (22 темы)
- `reviews`: массив отзывов с полями `id`, `title`, `text`

## Структура проекта

```
Fast_api/
├── app/
│   └── main.py          # FastAPI приложение
├── Dockerfile           # Docker образ
├── .dockerignore        # Исключения для Docker
└── README.md           # Документация
```

## Разработка

### Локальный запуск без Docker

```bash
# Установка зависимостей
pip install -r requirements_api.txt
pip install fastapi uvicorn[standard]

# Запуск сервера
uvicorn Fast_api.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Сборка Docker образа

```bash
# Сборка образа
docker build -f Fast_api/Dockerfile -t gpb-reviews-api .

# Запуск контейнера
docker run -p 8000:8000 --env-file .env gpb-reviews-api
```

## Логи и мониторинг

```bash
# Просмотр логов
docker-compose logs -f api

# Остановка сервиса
docker-compose down
```
