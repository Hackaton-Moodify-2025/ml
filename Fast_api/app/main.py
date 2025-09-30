from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union


# ===== Схемы запроса/ответа =====
class InputReview(BaseModel):
    id: Union[int, str]
    text: str


class AnalyzeFileRequest(BaseModel):
    data: List[InputReview]


class PredictionItem(BaseModel):
    id: Union[int, str]
    topics: List[str]
    sentiments: List[str]


class AnalyzeFileResponse(BaseModel):
    predictions: List[PredictionItem]
    total: int


app = FastAPI(title="Gazprombank Reviews API", version="0.2.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ===== Заглушка модели =====
TOPICS = [
    "Отделения",
    "Карты",
    "Депозиты и вклады",
    "Переводы и платежи",
    "Мобильное приложение",
    "Служба поддержки",
    "Премиум-обслуживание",
    "Кэшбек и бонусы",
    "Ипотека",
]

SENTIMENTS = ["положительно", "отрицательно", "нейтрально"]


def model_stub_predict_topics_and_sentiments(text: str) -> List[dict]:
    """Простая эвристическая заглушка вместо ML-модели.
    Возвращает список {topic, sentiment} для демонстрации контракта.
    """
    normalized = text.lower()

    inferred: List[dict] = []

    if any(w in normalized for w in ["карта", "карту", "карты", "картой"]):
        inferred.append({"topic": "Карты", "sentiment": "нейтрально"})

    if any(w in normalized for w in ["мобильное приложение", "приложение", "зависает", "лаг"]):
        inferred.append({"topic": "Мобильное приложение", "sentiment": "отрицательно"})

    if any(w in normalized for w in ["отделени", "офис", "обслуживан"]):
        inferred.append({"topic": "Отделения", "sentiment": "положительно"})

    if any(w in normalized for w in ["кредит", "лимит"]):
        inferred.append({"topic": "Карты", "sentiment": "нейтрально"})

    # Если ничего не нашли — возвращаем пустой список (модель потом заменит)
    return inferred


@app.post("/analyze", response_model=AnalyzeFileResponse)
def analyze_file(req: AnalyzeFileRequest) -> AnalyzeFileResponse:
    predictions: List[PredictionItem] = []

    for item in req.data:
        topic_sentiments = model_stub_predict_topics_and_sentiments(item.text)
        prediction = PredictionItem(
            id=item.id,
            topics=[ts["topic"] for ts in topic_sentiments],
            sentiments=[ts["sentiment"] for ts in topic_sentiments],
        )
        predictions.append(prediction)

    return AnalyzeFileResponse(predictions=predictions, total=len(predictions))



