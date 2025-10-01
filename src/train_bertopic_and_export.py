import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

from .sentiment import predict_sentiment
from .final_classes import FINAL_CLASSES_EX
from .utils_text import prepare_text, softmax_cos
from .encode import encode_long

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------- utils: чтение и выбор колонок ----------

def _read_json_flex(path: str) -> pd.DataFrame:
    """
    Чтение JSON или JSONL в DataFrame.
    Поддерживает формат {"reviews": [...]}.
    """
    p = Path(path)
    assert p.exists(), f"Файл не найден: {p.resolve()}"

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except Exception:
            # fallback: JSONL
            return pd.read_json(p, lines=True)

    # если словарь с ключом reviews
    if isinstance(data, dict) and "reviews" in data:
        return pd.DataFrame(data["reviews"])
    # если список словарей
    elif isinstance(data, list):
        return pd.DataFrame(data)
    # если сразу табличка
    else:
        return pd.json_normalize(data)




def _pick_column(df: pd.DataFrame, candidates: list[str], *, required: bool) -> pd.Series:
    """
    Берёт первый существующий столбец из candidates как Series(str).
    Если не найден и required=True — бросаем KeyError с понятным описанием.
    Если required=False — возвращаем пустую строковую серию нужной длины.
    """
    for c in candidates:
        if c in df.columns:
            return df[c].astype("string").fillna("")
    if required:
        raise KeyError(
            "Не найден столбец с основным текстом. Ожидались любые из: "
            + ", ".join(candidates)
            + f". Доступные колонки: {list(df.columns)}"
        )
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def normalize_topic_words(words_scores, keep=12):
    # words_scores: List[Tuple[str, score]] (берём top_n_words из BERTopic)
    if not words_scores:
        return ""
    seen, out = set(), []
    for w, _ in words_scores:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) == keep:
            break
    return ", ".join(out)


def build_class_matrix(embedder):
    class_names = list(FINAL_CLASSES_EX.keys())
    class_vecs = []
    for variants in FINAL_CLASSES_EX.values():
        vv = [s for s in variants if isinstance(s, str) and s.strip()]
        vecs = embedder.encode(vv, normalize_embeddings=True)
        class_vecs.append(vecs.mean(axis=0))
    return class_names, np.stack(class_vecs)


def main(args):
    mlflow.set_experiment("banking-topics-bertopic")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # 1) Загрузка данных (гибко)
        df = _read_json_flex(args.data_path)

        # 1.1) Определяем столбцы заголовка/текста
        # можно явно задать через флаги, иначе ищем по синонимам
        title_candidates = [args.title_col] if args.title_col else [
            "title", "review_title", "headline", "subject", "name"
        ]
        text_candidates = [args.text_col] if args.text_col else [
            "text", "review_text", "content", "body", "message", "comment", "review"
        ]

        title_s = _pick_column(df, title_candidates, required=False)
        text_s = _pick_column(df, text_candidates, required=True)

        # сырые и очищенные тексты
        # docs_ctx — контекст для эмбеддингов (заголовок + текст)
        docs_ctx = (
            (title_s.where(title_s.str.len() > 0, other="") + ". " + text_s)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .tolist()
        )
        # doc — очищенный текст для BERTopic (используем ваш prepare_text)
        df["doc"] = pd.Series(
            (prepare_text(ttl, txt) for ttl, txt in zip(title_s.tolist(), text_s.tolist())),
            index=df.index,
            dtype="string",
        )
        docs_clean = df["doc"].astype(str).tolist()

        # 2) Эмбеддер + эмбеддинги (с чанкингом)
        embedder = SentenceTransformer(args.embedder_model)
        embeddings = encode_long(docs_ctx, embedder, pooling=args.pooling, max_chars=args.max_chars)

        # 3) Vectorizer для названий тем (чистим «воду»)
        custom_stopwords = {
            "банк","банка","банке","банку","банков",
            "газпромбанк","газпром",
            "карта","карты","картой","карте",
            "клиент","клиенты","приложение","приложения",
            "деньги","средства","номер","сегодня","вчера","теперь","просто"
        }
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.4,
            token_pattern=r"(?u)\b[\w\-]+\b",
            stop_words=list(custom_stopwords)
        )

        # 4) Кластеризация: KMeans (через hdbscan_model=) или HDBSCAN
        if args.use_kmeans:
            cluster_model = KMeans(n_clusters=args.n_clusters, random_state=42)
            topic_model = BERTopic(
                hdbscan_model=cluster_model,   # для BERTopic>=0.17 возможно прокидывать sklearn-кластеризатор
                vectorizer_model=vectorizer_model,
                language="multilingual",
                min_topic_size=args.min_topic_size,
                calculate_probabilities=True
            )
        else:
            from hdbscan import HDBSCAN
            hdbscan_model = HDBSCAN(min_cluster_size=args.min_topic_size, min_samples=3, prediction_data=True)
            topic_model = BERTopic(
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language="multilingual",
                calculate_probabilities=True
            )

        topics, probs = topic_model.fit_transform(docs_clean, embeddings=embeddings)
        if not args.use_kmeans:
            # Попробуем «дораспределить» -1 (только для HDBSCAN)
            topics = topic_model.reduce_outliers(docs_clean, topics, strategy="c-tf-idf")

        # 5) Улучшаем представление тем (MMR), IN-PLACE
        rep_model = MaximalMarginalRelevance(diversity=0.6)
        _ = topic_model.update_topics(
            docs=docs_clean,
            vectorizer_model=vectorizer_model,
            representation_model=rep_model,
            n_gram_range=(1, 3),
            top_n_words=12
        )

        # 6) Таблицы
        topic_info = topic_model.get_topic_info()
        doc_info   = topic_model.get_document_info(docs_clean)

        # 7) Маппинг тем -> бизнес-классы (через эмбеддинги топ-слов темы)
        class_names, class_vecs = build_class_matrix(embedder)

        rows = []
        ti = topic_info.set_index("Topic")
        topic_ids = [t for t in ti.index.tolist() if t != -1]
        for t in topic_ids:
            words_scores = topic_model.get_topic(t) or []
            label = normalize_topic_words(words_scores, keep=12)
            t_vec = embedder.encode(label or "тема", normalize_embeddings=True)
            sims = cosine_similarity([t_vec], class_vecs)[0]
            order = np.argsort(sims)[::-1]
            probs_soft = softmax_cos(sims, tau=0.07)
            rows.append({
                "topic_id": t,
                "topic_name": ti.loc[t, "Name"],
                "topic_keywords": label,
                "top1_class": class_names[order[0]], "top1_sim": float(sims[order[0]]), "top1_prob": float(probs_soft[order[0]]),
                "top2_class": class_names[order[1]], "top2_sim": float(sims[order[1]]),
                "top3_class": class_names[order[2]], "top3_sim": float(sims[order[2]]),
            })
        topic_map_df = pd.DataFrame(rows).sort_values(["top1_class","top1_sim"], ascending=[True, False])

        # 8) Документный датасет: id, отзыв, top3 класса (через doc-level zero-shot)
        sims_docs = cosine_similarity(embeddings, class_vecs)
        order = np.argsort(sims_docs, axis=1)[:, ::-1]
        topk = args.topk_classes

        doc_rows = []
        for i in range(len(df)):
            r = {
                "id": df.loc[i, "id"] if "id" in df.columns else i,
                "review": docs_ctx[i]
            }
            for k in range(topk):
                idx = order[i, k]
                r[f"doc_top{k+1}_class"] = class_names[idx]
                r[f"doc_top{k+1}_sim"]   = float(sims_docs[i, idx])
            # добавим тему BERTopic для справки
            r["topic_id"] = int(topics[i])
            r["topic_name"] = ti.loc[topics[i], "Name"] if topics[i] in ti.index else "прочее"
            doc_rows.append(r)
        doc_topk_df = pd.DataFrame(doc_rows)

        # сентимент
        sent_labels = predict_sentiment(docs_ctx, model_name=args.sentiment_model, batch_size=32)
        doc_topk_df["sentiment"] = sent_labels

        # 9) Suspect / распределения
        SUS = args.suspect_topic_sim
        suspect_topics = topic_map_df.query("top1_sim < @SUS")
        dist_docs   = doc_topk_df["doc_top1_class"].value_counts().rename_axis("class").reset_index(name="docs")
        dist_topics = topic_map_df["top1_class"].value_counts().rename_axis("class").reset_index(name="topics")

        # 10) Сохранение артефактов
        out_root = Path(args.out_dir)
        out_run  = out_root / "latest"
        out_run.mkdir(parents=True, exist_ok=True)

        # Excel с несколькими листами
        excel_path = out_run / "bertopic_outputs.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            topic_info.to_excel(writer, sheet_name="topics_summary", index=False)
            topic_map_df.to_excel(writer, sheet_name="topics_top3_classes", index=False)
            doc_topk_df.to_excel(writer, sheet_name="documents_topk", index=False)
            suspect_topics.to_excel(writer, sheet_name="suspect_topics", index=False)
            dist_docs.to_excel(writer, sheet_name="dist_docs", index=False)
            dist_topics.to_excel(writer, sheet_name="dist_topics", index=False)

        # сохраняем модель/векторизатор/классы/эмбеддер
        topic_model.save(str(out_run / "bertopic_model"))
        with open(out_run / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False)
        np.save(out_run / "class_vecs.npy", class_vecs)
        embedder.save(str(out_run / "embedder"))

        # логируем в MLflow
        mlflow.log_artifacts(str(out_run))
        print(f"[OK] saved artifacts to: {out_run.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--embedder_model", type=str, default="BAAI/bge-m3")
    ap.add_argument("--pooling", type=str, default="max", choices=["mean","max","attn_len"])
    ap.add_argument("--max_chars", type=int, default=1000)
    ap.add_argument("--use_kmeans", type=lambda x: str(x).lower() == "true", default=True)
    ap.add_argument("--n_clusters", type=int, default=30)
    ap.add_argument("--min_topic_size", type=int, default=15)
    ap.add_argument("--topk_classes", type=int, default=3)
    ap.add_argument("--sentiment_model", type=str, default="blanchefort/rubert-base-cased-sentiment")
    ap.add_argument("--suspect_topic_sim", type=float, default=0.45)
    # новые флаги для явного указания колонок
    ap.add_argument("--title-col", type=str, default=None, help="Имя колонки с заголовком (опц.)")
    ap.add_argument("--text-col", type=str, default=None, help="Имя колонки с текстом (если не задано, ищем по синонимам)")
    args = ap.parse_args()
    main(args)
