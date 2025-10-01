from typing import List, Optional
from transformers import pipeline
import torch

DEFAULT_MODEL = "blanchefort/rubert-base-cased-sentiment"   # 3-class ru sentiment

def load_sentiment_pipeline(model_name: str = DEFAULT_MODEL, device: Optional[int] = None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device,
        truncation=True,       # <--- режем тексты при токенизации
        max_length=512,        # <--- BERT поддерживает только 512 токенов
        padding=True,          # <--- батчируется ровно
        return_all_scores=False
    )

def predict_sentiment(texts: List[str],
                      model_name: str = DEFAULT_MODEL,
                      batch_size: int = 32) -> List[str]:
    """Возвращает метки из набора: negative / neutral / positive."""
    clf = load_sentiment_pipeline(model_name=model_name)
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        preds = clf(chunk, batch_size=batch_size)   # truncation/max_length теперь внутри пайплайна
        for p in preds:
            label = (p.get("label") or "").lower()
            if label in {"negative", "neg"}:
                out.append("Негатив")
            elif label in {"neutral", "neu"}:
                out.append("Нейтрально")
            elif label in {"positive", "pos"}:
                out.append("Положительно")
            else:
                out.append("Нейтрально")
    return out
