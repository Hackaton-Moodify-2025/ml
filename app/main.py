from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Импортируем свои утилиты
from src.encode import encode_long
from src.sentiment import predict_sentiment  # внутри должен быть truncation=True, max_length=512

# ==== Конфиг (жёсткие дефолты) ====
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/latest"))
POOLING = os.getenv("POOLING", "max")
MAX_CHARS = int(os.getenv("MAX_CHARS", "1000"))
TOPK = int(os.getenv("TOPK", "5"))
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "blanchefort/rubert-base-cased-sentiment")
SENTIMENT_BATCH = int(os.getenv("SENTIMENT_BATCH", "32"))

# Порог уверенности и «умный» фильтр для категорий
MIN_CONF  = 0.15   # строгий порог показа
MIN_FLOOR = 0.10   # нижний порог, если есть хороший зазор
MARGIN    = 0.05   # зазор с соседней категорией
TAU       = 0.03   # температура softmax

# Настройки аспектного сентимента (назначение предложений категориям)
SENT_MIN_SIM   = 0.12  # минимальная схожесть предложения с категорией, чтобы «привязать» его к ней
SENT_MAX_CHARS = 400
# лёгкий бонус к похожести, если в клаузе встречается якорное слово для категории
KEYWORD_BONUS  = 0.10

# ==== Pydantic модели ====

class ReviewIn(BaseModel):
    id: str | int
    text: str = Field(min_length=1)

class ReviewsIn(BaseModel):
    data: List[ReviewIn]

class ReviewOut(BaseModel):
    id: str | int
    topics: List[Optional[str]]        # длиной TOPK, порядок cat1..catK
    sentiments: List[Optional[str]]        # длиной TOPK, порядок sent1..sentK, выровнен с categories
   

class ReviewsOut(BaseModel):
    predict: List[ReviewOut]

# ==== Приложение ====
app = FastAPI(title="Banking Topics Inference API", version="1.2")

# Глобальные объекты (грузим при старте)
EMBEDDER: Optional[SentenceTransformer] = None
CLASS_VECS: Optional[np.ndarray] = None  # [num_classes, dim]
CLASS_NAMES: Optional[list[str]] = None

# ==== Хелперы ====

def _load_artifacts(model_dir: Path):
    assert (model_dir / "class_vecs.npy").exists(), f"Missing {model_dir/'class_vecs.npy'}"
    assert (model_dir / "class_names.json").exists(), f"Missing {model_dir/'class_names.json'}"
    assert (model_dir / "embedder").exists(), f"Missing {model_dir/'embedder'} directory"

    class_vecs = np.load(model_dir / "class_vecs.npy")
    with open(model_dir / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    embedder = SentenceTransformer(str(model_dir / "embedder"))
    return embedder, class_vecs, class_names

def _build_texts(rows: List[ReviewIn]) -> list[str]:
    return [r.text.strip() for r in rows]

def _softmax_cos_row(x: np.ndarray, tau: float) -> np.ndarray:
    """Softmax поверх косинусов: меньше tau → распределение острее."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - x.max()) / max(tau, 1e-9)
    ez = np.exp(z)
    return ez / ez.sum()

def split_into_sents(text: str) -> list[str]:
    """Бьём на клаузы: по .?! и дополнительно по контрастивным союзам ('но', 'однако', 'а', 'зато')."""
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"[.!?]+[\s\n]+", t)
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        sub = re.split(r",?\s+(?:но|однако|зато|а)\s+", p, flags=re.IGNORECASE)
        for s in sub:
            s = s.strip(" ,;")
            if s:
                chunks.append(s)
    return chunks

def majority_label(labels: list[str], default: str = "Нейтрально") -> str:
    if not labels:
        return default
    norm = []
    for l in labels:
        l = (l or "").lower()
        if l in {"negative", "neg", "негатив", "отрицательно"}:
            norm.append("Отрицательно")
        elif l in {"neutral", "neu", "нейтрально"}:
            norm.append("Нейтрально")
        elif l in {"positive", "pos", "позитив", "положительно"}:
            norm.append("Положительно")
        else:
            norm.append("Нейтрально")
    from collections import Counter
    return Counter(norm).most_common(1)[0][0]

def keyword_bonus(clause: str, cat_name: str) -> float:
    """Небольшой бонус к похожести, если в тексте клауы и имени категории есть якорные слова."""
    if not clause or not cat_name:
        return 0.0
    c = clause.lower()
    n = cat_name.lower()
    anchors = {
        "прилож": ["прилож", "мобиль"],
        "обслуж": ["обслуж", "сервис"],
        "поддерж": ["поддерж", "саппорт", "колл-центр", "колл центр"],
        "кэшбэк": ["кэшбэк", "cashback", "бонус"],
        "карта": ["карта", "дебетов", "кредитн"],
        "перевод": ["перевод", "платёж", "оплата"],
        "вклад": ["вклад", "депозит", "накопит"],
    }
    for key, keys in anchors.items():
        if key in n and any(k in c for k in keys):
            return KEYWORD_BONUS
    return 0.0

# ==== События жизненного цикла ====

@app.on_event("startup")
def on_startup():
    global EMBEDDER, CLASS_VECS, CLASS_NAMES
    try:
        EMBEDDER, CLASS_VECS, CLASS_NAMES = _load_artifacts(MODEL_DIR)
    except Exception as e:
        raise RuntimeError(f"Failed to load artifacts from {MODEL_DIR}: {e}")

@app.get("/health")
def health():
    ok = all([EMBEDDER is not None, CLASS_VECS is not None, CLASS_NAMES is not None])
    return {
        "status": "ok" if ok else "broken",
        "classes": len(CLASS_NAMES) if CLASS_NAMES else 0,
        "thresholds": {
            "MIN_CONF": MIN_CONF, "MIN_FLOOR": MIN_FLOOR, "MARGIN": MARGIN, "TAU": TAU,
            "SENT_MIN_SIM": SENT_MIN_SIM, "SENT_MAX_CHARS": SENT_MAX_CHARS, "KEYWORD_BONUS": KEYWORD_BONUS
        },
    }

# ==== Основной эндпоинт ====

@app.post("/predict", response_model=ReviewsOut)
def predict(body: ReviewsIn):
    if EMBEDDER is None or CLASS_VECS is None or CLASS_NAMES is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not body.data:
        return ReviewsOut(data=[])

    # 1) Тексты
    texts = _build_texts(body.data)

    # 2) Эмбеддинги (целиком)
    try:
        embs = encode_long(texts, EMBEDDER, pooling=POOLING, max_chars=MAX_CHARS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3) Косинусы + вероятности
    sims = cosine_similarity(embs, CLASS_VECS)      # [N, C]
    order = np.argsort(sims, axis=1)[:, ::-1]       # индексы по убыванию

    # 4) Общий сентимент
    try:
        sentiments = predict_sentiment(texts, model_name=SENTIMENT_MODEL, batch_size=SENTIMENT_BATCH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment failed: {e}")

    # 5) Ответ
    out_rows: List[ReviewOut] = []
    for i, rev in enumerate(body.data):
        probs_i = _softmax_cos_row(sims[i], tau=TAU)
        top_idx = order[i, :TOPK]

        names: List[Optional[str]] = []
        probs: List[Optional[float]] = []

        # применяем умный фильтр (последняя позиция — только MIN_CONF)
        k_last = len(top_idx) - 1
        for j, idx in enumerate(top_idx):
            p = float(probs_i[idx])

            if j == k_last:
                cond = p >= MIN_CONF
            else:
                next_p = float(probs_i[top_idx[j+1]])
                cond_conf   = p >= MIN_CONF
                cond_margin = (p >= MIN_FLOOR) and ((p - next_p) >= MARGIN)
                cond = cond_conf or cond_margin

            names.append(CLASS_NAMES[idx] if cond else None)
            probs.append(p)

        # Дополняем до 5 слотов
        while len(names) < 5:
            names.append(None)
            probs.append(None)

        # ===== АСПЕКТНЫЙ СЕНТИМЕНТ =====
        sents = split_into_sents(texts[i])
        cat_sent_labels = [None, None, None, None, None]

        if sents:
            # эмбеддинги предложений (короткие куски)
            sent_embs = encode_long(sents, EMBEDDER, pooling=POOLING, max_chars=SENT_MAX_CHARS)

            # локальные индексы выбранных категорий
            chosen_loc_idx  = [k for k, nm in enumerate(names) if nm]
            chosen_real_idx = [top_idx[k] for k in chosen_loc_idx]

            if chosen_real_idx:
                sub_class_vecs = CLASS_VECS[chosen_real_idx]                 # [K_sel, dim]
                sent_sims = cosine_similarity(sent_embs, sub_class_vecs)     # [S, K_sel]

                # keyword bonus
                for si, clause in enumerate(sents):
                    for cj, loc_k in enumerate(chosen_loc_idx):
                        cat_name = names[loc_k]
                        if cat_name:
                            sent_sims[si, cj] += keyword_bonus(clause, cat_name)

                best_idx = np.argmax(sent_sims, axis=1)                      # для каждого предложения — лучшая категория
                best_val = np.take_along_axis(sent_sims, best_idx[:, None], axis=1).ravel()

                # bucket: локальный индекс категории -> список предложений
                from collections import defaultdict
                bucket: dict[int, List[int]] = defaultdict(list)
                for si, (bj, val) in enumerate(zip(best_idx, best_val)):
                    if float(val) >= SENT_MIN_SIM:
                        loc_k = chosen_loc_idx[bj]  # маппим обратно к 0..4
                        bucket[loc_k].append(si)

                # сентимент по каждому «ведру»
                for loc_k, sent_ids in bucket.items():
                    seg_texts = [sents[x] for x in sent_ids]
                    seg_labels = predict_sentiment(seg_texts, model_name=SENTIMENT_MODEL, batch_size=SENTIMENT_BATCH)
                    cat_sent_labels[loc_k] = majority_label(seg_labels, default="Нейтрально")

        out_rows.append(ReviewOut(
            id=rev.id,
            text=rev.text,
            topics=names[:TOPK],
            sentiments=cat_sent_labels[:TOPK],
            sentiment=sentiments[i],
        ))

    return ReviewsOut(predict=out_rows)
