import os, json, argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .encode import encode_long
from .sentiment import predict_sentiment  # важно: внутри truncation=True/max_length в пайплайне

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------- helpers ----------

def _read_json_flex(path: str, root_key: Optional[str] = None) -> pd.DataFrame:
    p = Path(path)
    assert p.exists(), f"Файл не найден: {p.resolve()}"
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
    except Exception:
        # JSONL
        return pd.read_json(p, lines=True)

    if root_key and isinstance(data, dict) and root_key in data and isinstance(data[root_key], list):
        return pd.DataFrame(data[root_key])

    for key in ["reviews", "data", "items", "records"]:
        if isinstance(data, dict) and key in data and isinstance(data[key], list):
            return pd.DataFrame(data[key])

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict) and len(data) == 1:
        only_val = next(iter(data.values()))
        if isinstance(only_val, list):
            return pd.DataFrame(only_val)

    return pd.json_normalize(data)


def _resolve_candidates(df: pd.DataFrame, primary: Optional[str], fallbacks: List[str], desc: str) -> List[str]:
    if primary:
        if primary in df.columns:
            return [primary] + [c for c in fallbacks if c != primary]
        else:
            print(f"[WARN] Колонка для {desc!r} не найдена: {primary!r}. "
                  f"Доступные: {list(df.columns)}. Использую синонимы: {fallbacks}")
    return fallbacks


def _pick_column(df: pd.DataFrame, candidates: List[str], *, required: bool) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].astype("string").fillna("")
    if required:
        raise KeyError("Не найден столбец. Ожидались любые из: "
                       + ", ".join(candidates) + f". Доступные: {list(df.columns)}")
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _softmax_cos(x: np.ndarray, tau: float = 0.07) -> np.ndarray:
    """Softmax поверх косинусных сходств: чем меньше tau, тем острее распределение."""
    x = np.asarray(x, dtype=np.float64)
    z = (x - x.max()) / max(tau, 1e-9)
    ez = np.exp(z)
    return ez / ez.sum()


# ---------- main ----------

def main(args):
    md = Path(args.model_dir)

    # 1) Артефакты zero-shot классификатора
    class_vecs  = np.load(md / "class_vecs.npy")
    with open(md / "class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    embedder    = SentenceTransformer(str(md / "embedder"))

    # 2) Входные данные
    df = _read_json_flex(args.input_path, root_key=args.root_key)

    title_fallbacks = ["title", "review_title", "headline", "subject", "name"]
    text_fallbacks  = ["text", "review_text", "content", "body", "message", "comment", "review"]

    title_candidates = _resolve_candidates(df, args.title_col, title_fallbacks, "заголовка")
    text_candidates  = _resolve_candidates(df, args.text_col,  text_fallbacks,  "текста")

    title_s = _pick_column(df, title_candidates, required=False)
    text_s  = _pick_column(df, text_candidates,  required=True)

    texts = (
        (title_s.where(title_s.str.len() > 0, other="") + ". " + text_s)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )
    if len(texts) == 0:
        raise ValueError("Входной файл не содержит ни одной строки с title/text.")

    # 3) Эмбеддинги документов (с чанкингом)
    emb_docs = encode_long(texts, embedder, pooling=args.pooling, max_chars=args.max_chars)

    # 4) Косинусы → top-k с «умным» порогом уверенности
    sims  = cosine_similarity(emb_docs, class_vecs)   # [N, C]
    order = np.argsort(sims, axis=1)[:, ::-1]

    rows = []
    for i, _ in enumerate(texts):
        rid = df.loc[i, "id"] if "id" in df.columns else i
        row = {"id": rid}

        probs_i = _softmax_cos(sims[i], tau=args.tau)  # (C,)

        # пробегаем top-k
        for k in range(min(args.topk, sims.shape[1])):
            idx = order[i, k]
            p   = float(probs_i[idx])

            if k == K - 1:
                # последняя позиция в топе: только строгий порог
                cond = p >= args.min_conf
            else:
                next_idx = order[i, k+1]
                next_p   = float(probs_i[next_idx])
                cond_conf   = p >= args.min_conf
                cond_margin = (p >= args.min_floor) and ((p - next_p) >= args.margin)
                cond = cond_conf or cond_margin

            row[f"top{k+1}_prob"] = p
            if cond:
                row[f"top{k+1}_class"] = class_names[idx]
                row[f"top{k+1}_sim"]   = float(sims[i, idx])
            else:
                row[f"top{k+1}_class"] = None
                row[f"top{k+1}_sim"]   = None

        rows.append(row)

    out = pd.DataFrame(rows)

    # 5) Сентимент
    sentiments = predict_sentiment(
        texts,
        model_name=args.sentiment_model,
        batch_size=args.sentiment_batch_size,
        # max_length=args.sentiment_max_length,  # раскомментируй, если добавила этот параметр в sentiment.py
    )
    out["sentiment"] = sentiments

    # 6) (опц.) вернуть исходный текст
    if args.with_text:
        if "id" in df.columns:
            review_col = (title_s.fillna("").astype(str) + ". " + text_s.fillna("").astype(str)) \
                .str.replace(r"\s+", " ", regex=True).str.strip()
            out = out.merge(pd.DataFrame({"id": df["id"], "review": review_col}), on="id", how="left")
        else:
            out = out.merge(pd.DataFrame({"id": range(len(texts)), "review": texts}), on="id", how="left")

    # 7) Сохранение
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.lower().endswith(".xlsx"):
        out.to_excel(args.output_path, index=False)
    else:
        out.to_csv(args.output_path, index=False)
    print(f"[OK] saved: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="artifacts/latest")
    ap.add_argument("--input_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, default="predictions.xlsx")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pooling", type=str, default="max", choices=["mean","max","attn_len"])
    ap.add_argument("--max_chars", type=int, default=1000)

    # Пороги и температура softmax
    ap.add_argument("--min_conf", type=float, default=0.15)
    ap.add_argument("--min_floor", type=float, default=0.10)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.03)

    # Сентимент
    ap.add_argument("--sentiment_model", type=str, default="blanchefort/rubert-base-cased-sentiment")
    ap.add_argument("--sentiment_batch_size", type=int, default=32)
    # ap.add_argument("--sentiment_max_length", type=int, default=512)  # если добавишь в sentiment.py

    # Включить текст в вывод
    ap.add_argument("--with_text", action="store_true", help="добавить колонку review в вывод")

    # Имена колонок и корневой ключ
    ap.add_argument("--title-col", type=str, default=None, help="имя колонки с заголовком (если отличается)")
    ap.add_argument("--text-col", type=str, default=None, help="имя колонки с текстом (если отличается)")
    ap.add_argument("--root-key", type=str, default=None, help="корневой ключ JSON (например, data/reviews)")

    args = ap.parse_args()
    main(args)
