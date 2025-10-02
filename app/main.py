from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from src.encode import encode_long

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import snapshot_download
from transformers import pipeline
import torch

# ==== Конфиг ====
MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/latest"))
POOLING = os.getenv("POOLING", "max")
MAX_CHARS = int(os.getenv("MAX_CHARS", "1000"))
TOPK = int(os.getenv("TOPK", "5"))

# Батчи
EMB_BATCH = int(os.getenv("EMB_BATCH", "32"))
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "blanchefort/rubert-base-cased-sentiment")
SENTIMENT_BATCH = int(os.getenv("SENTIMENT_BATCH", "64"))
SENT_MAX_LEN = int(os.getenv("SENT_MAX_LEN", "256"))

# HF repo с эмбеддером
HF_REPO = os.getenv("HF_REPO", "moomive/my-banking-embedder")
HF_TOKEN = os.getenv("HF_TOKEN")  # нужен, если репо приватный

# Порог уверенности и «умный» фильтр для категорий
MIN_CONF  = float(os.getenv("MIN_CONF",  "0.10"))  # строгий порог показа
MIN_FLOOR = float(os.getenv("MIN_FLOOR", "0.08"))  # нижний порог, если есть хороший зазор
MARGIN    = float(os.getenv("MARGIN",    "0.03"))  # зазор с соседней категорией
TAU       = float(os.getenv("TAU",       "0.05"))  # температура softmax

# Настройки аспектного сентимента
SENT_MIN_SIM        = float(os.getenv("SENT_MIN_SIM",       "0.10"))
SENT_MULTI_DELTA    = float(os.getenv("SENT_MULTI_DELTA",   "0.03"))
SENT_BACKFILL_TOPN  = int(os.getenv("SENT_BACKFILL_TOPN",   "1"))
SENT_MAX_CHARS      = int(os.getenv("SENT_MAX_CHARS",       "300"))
KEYWORD_BONUS       = float(os.getenv("KEYWORD_BONUS",      "0.10"))

# ==== Pydantic модели ====
class ReviewIn(BaseModel):
    id: str | int
    text: Optional[str] = ""  # допускаем null -> ""

class ReviewsIn(BaseModel):
    data: List[ReviewIn]

class ReviewOut(BaseModel):
    id: str | int
    topics: List[str]      # без null, только выбранные категории
    sentiments: List[str]  # без null, та же длина, что и topics

class ReviewsOut(BaseModel):
    predict: List[ReviewOut]

# ==== Приложение ====
app = FastAPI(title="Banking Topics Inference API", version="1.3-fast")

# Глобальные объекты
EMBEDDER: Optional[SentenceTransformer] = None
CLASS_VECS: Optional[np.ndarray] = None
CLASS_NAMES: Optional[List[str]] = None
SENT_CLF = None  # глобальный sentiment pipeline

# ==== Хелперы ====
def _load_artifacts(model_dir: Path) -> Tuple[SentenceTransformer, np.ndarray, List[str]]:
    assert (model_dir / "class_vecs.npy").exists(), f"Missing {model_dir/'class_vecs.npy'}"
    assert (model_dir / "class_names.json").exists(), f"Missing {model_dir/'class_names.json'}"
    assert (model_dir / "embedder").exists(), f"Missing {model_dir/'embedder'} directory"

    class_vecs = np.load(model_dir / "class_vecs.npy")
    with open(model_dir / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    embedder = SentenceTransformer(str(model_dir / "embedder"))
    return embedder, class_vecs, class_names

def _build_texts(rows: List[ReviewIn]) -> List[str]:
    return [(r.text or "").strip() for r in rows]

def _softmax_cos_row(x: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = (x - x.max()) / max(tau, 1e-9)
    ez = np.exp(z)
    return ez / ez.sum()

def split_into_sents(text: str) -> List[str]:
    """Бьём на клаузы: по .?! и дополнительно по контрастивным союзам ('но', 'однако', 'а', 'зато')."""
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"[.!?]+[\s\n]+", t)
    chunks: List[str] = []
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

def majority_label(labels: List[str], default: str = "Нейтрально") -> str:
    if not labels:
        return default
    norm = []
    for l in labels:
        l = (l or "").lower()
        if l.startswith("pos") or l in {"положительно", "позитив"}:
            norm.append("Положительно")
        elif l.startswith("neg") or l in {"негатив", "отрицательно"}:
            norm.append("Отрицательно")
        else:
            norm.append("Нейтрально")
    from collections import Counter
    return Counter(norm).most_common(1)[0][0]

def keyword_bonus(clause: str, cat_name: str) -> float:
    if not clause or not cat_name:
        return 0.0
    c = clause.lower()
    n = cat_name.lower()
    patterns = {
        r"прилож": [r"\bприлож\w*", r"\bмобиль\w*"],
        r"обслуж|сервис": [r"\bобслуж\w*", r"\bсервис\w*"],
        r"поддерж": [r"\bподдерж\w*", r"\bсаппорт\w*", r"колл[\-\s]?центр"],
        r"кэшб|cashback|бонус": [r"\bкэшб\w*", r"\bcashback\b", r"\bбонус\w*"],
        r"карт": [r"\bкарт\w*", r"\bкредит\w*", r"\bдебет\w*"],
        r"перевод|платеж|оплат": [r"\bперевод\w*", r"\bплат[её]ж\w*", r"\bоплат\w*"],
        r"вклад|депозит|накоп": [r"\bвклад\w*", r"\bдепозит\w*", r"\bнакоп\w*"],
        r"union ?pay": [r"\bunion ?pay\b"],
    }
    for cat_pat, clause_pats in patterns.items():
        if re.search(cat_pat, n):
            if any(re.search(p, c) for p in clause_pats):
                return KEYWORD_BONUS
    return 0.0

def ensure_embedder(model_dir: Path):
    emb_dir = model_dir / "embedder"
    if emb_dir.exists() and any(emb_dir.iterdir()):
        return
    if not HF_REPO:
        raise RuntimeError("HF_REPO is not set. Cannot download embedder from Hugging Face.")
    print(f"[INFO] Downloading embedder from HF: {HF_REPO} -> {emb_dir}")
    emb_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="model",
        local_dir=str(emb_dir),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

def run_sentiment(batch_texts: List[str]) -> List[str]:
    """Глобальный быстрый сентимент (batched). Возвращает русские метки."""
    if not batch_texts:
        return []
    preds = SENT_CLF(batch_texts, truncation=True, max_length=SENT_MAX_LEN, batch_size=SENTIMENT_BATCH)
    out = []
    for p in preds:
        lab = (p.get("label") or "").lower()
        if lab.startswith("pos"):
            out.append("Положительно")
        elif lab.startswith("neg"):
            out.append("Отрицательно")
        else:
            out.append("Нейтрально")
    return out

# ==== События жизненного цикла ====
@app.on_event("startup")
def on_startup():
    global EMBEDDER, CLASS_VECS, CLASS_NAMES, SENT_CLF

    # 1) гарантируем, что embedder есть (если нет — подтянем с HF)
    ensure_embedder(MODEL_DIR)

    # 2) загрузка артефактов
    EMBEDDER, CLASS_VECS, CLASS_NAMES = _load_artifacts(MODEL_DIR)

    # 3) инициализация sentiment-пайплайна 1 раз
    # предпочитаем MPS на Apple, затем CUDA, иначе CPU
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = 0 if torch.cuda.is_available() else -1
    global_pipe = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=device
    )
    SENT_CLF = global_pipe

@app.get("/health")
def health():
    ok = all([EMBEDDER is not None, CLASS_VECS is not None, CLASS_NAMES is not None, SENT_CLF is not None])
    return {
        "status": "ok" if ok else "broken",
        "classes": len(CLASS_NAMES) if CLASS_NAMES else 0,
        "hf_repo": HF_REPO or None,
        "batch": {"EMB_BATCH": EMB_BATCH, "SENTIMENT_BATCH": SENTIMENT_BATCH, "SENT_MAX_LEN": SENT_MAX_LEN},
        "thresholds": {
            "MIN_CONF": MIN_CONF, "MIN_FLOOR": MIN_FLOOR, "MARGIN": MARGIN, "TAU": TAU,
            "SENT_MIN_SIM": SENT_MIN_SIM, "SENT_MULTI_DELTA": SENT_MULTI_DELTA,
            "SENT_BACKFILL_TOPN": SENT_BACKFILL_TOPN, "SENT_MAX_CHARS": SENT_MAX_CHARS,
            "KEYWORD_BONUS": KEYWORD_BONUS
        },
    }

# ==== Основной эндпоинт ====
@app.post("/predict", response_model=ReviewsOut)
def predict(body: ReviewsIn):
    if EMBEDDER is None or CLASS_VECS is None or CLASS_NAMES is None or SENT_CLF is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not body.data:
        return ReviewsOut(predict=[])

    # 1) Тексты
    texts = _build_texts(body.data)

    # 2) Эмбеддинги (целиком)
    try:
        embs = encode_long(texts, EMBEDDER, pooling=POOLING, max_chars=MAX_CHARS, batch_size=EMB_BATCH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # 3) Косинусы + вероятности
    sims = cosine_similarity(embs, CLASS_VECS)   # [N, C]
    order = np.argsort(sims, axis=1)[:, ::-1]

    # 4) Ответ
    out_rows: List[ReviewOut] = []
    for i, rev in enumerate(body.data):
        probs_i = _softmax_cos_row(sims[i], tau=TAU)
        top_idx = order[i, :TOPK]

        # === Выбираем категории (умный фильтр) ===
        chosen_real_idx: List[int] = []
        chosen_names: List[str] = []
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
            if cond:
                chosen_real_idx.append(idx)
                chosen_names.append(CLASS_NAMES[idx])

        # Если ни одна не прошла — пустые массивы
        if not chosen_real_idx:
            
            best_idx = int(order[i, 0])
            topics = [CLASS_NAMES[best_idx]]

            # аспектный сентимент: весь текст как одна клауза
            sents = split_into_sents(texts[i]) or [texts[i]]
            # можно сократить до 1-2 клауз, если длинно:
            sents = sents[:1]
            seg_lab = run_sentiment(sents)
            aspects_sent = [majority_label(seg_lab, default="Нейтрально")]

            out_rows.append(ReviewOut(id=rev.id, topics=topics, sentiments=aspects_sent))
            continue
            

        topics = chosen_names[:]  # без null

        # === Аспектный сентимент по выбранным категориям ===
        sents = split_into_sents(texts[i])
        if not sents:
            aspects_sent = ["Нейтрально"] * len(topics)
        else:
            # эмбеддинги клауз
            sent_embs = encode_long(sents, EMBEDDER, pooling=POOLING, max_chars=SENT_MAX_CHARS, batch_size=EMB_BATCH)

            sub_class_vecs = CLASS_VECS[chosen_real_idx]               # [K_sel, dim]
            sent_sims = cosine_similarity(sent_embs, sub_class_vecs)   # [S, K_sel]

            # keyword bonus
            for si, clause in enumerate(sents):
                for cj, cat_name in enumerate(topics):
                    if cat_name:
                        sent_sims[si, cj] += keyword_bonus(clause, cat_name)

            # мультиназначение с порогами
            from collections import defaultdict
            bucket: Dict[int, List[int]] = defaultdict(list)
            row_max = sent_sims.max(axis=1, keepdims=True)
            mask = (sent_sims >= (row_max - SENT_MULTI_DELTA)) & (sent_sims >= SENT_MIN_SIM)

            for si in range(sent_sims.shape[0]):
                cands = np.where(mask[si])[0].tolist()
                for cj in cands:
                    bucket[cj].append(si)

            # бэкфилл: гарантируем хотя бы одну клаузу на категорию
            if SENT_BACKFILL_TOPN > 0:
                for cj in range(len(topics)):
                    if not bucket[cj]:
                        best_si = int(np.argmax(sent_sims[:, cj]))
                        bucket[cj].append(best_si)

            # единый батч сентимента по всем сегментам
            all_seg_items: List[Tuple[int, str]] = []
            for cj in range(len(topics)):
                for si in bucket.get(cj, []):
                    all_seg_items.append((cj, sents[si]))

            if all_seg_items:
                seg_labels = run_sentiment([t for _, t in all_seg_items])
                per_cat: Dict[int, List[str]] = {}
                for (cj, _), lab in zip(all_seg_items, seg_labels):
                    per_cat.setdefault(cj, []).append(lab)
                aspects_sent = [
                    majority_label(per_cat.get(cj, []), default="Нейтрально")
                    for cj in range(len(topics))
                ]
            else:
                aspects_sent = ["Нейтрально"] * len(topics)

        out_rows.append(ReviewOut(id=rev.id, topics=topics, sentiments=aspects_sent))

    return ReviewsOut(predict=out_rows)
