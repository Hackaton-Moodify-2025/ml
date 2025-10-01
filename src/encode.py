import re
from typing import List, Sequence
import numpy as np

def chunk_text(s: str, max_chars: int = 1000) -> List[str]:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s: return []
    return [s[i:i+max_chars] for i in range(0, len(s), max_chars)]

def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    denom = np.sqrt((mat**2).sum(axis=axis, keepdims=True)) + eps
    return mat / denom

def pool_mean(chunk_vecs: np.ndarray) -> np.ndarray:
    return chunk_vecs.mean(axis=0)

def pool_max(chunk_vecs: np.ndarray) -> np.ndarray:
    return chunk_vecs.max(axis=0)

def pool_attn_len(chunk_vecs: np.ndarray, chunks: Sequence[str], tau: float = 0.5) -> np.ndarray:
    lengths = np.array([max(len(c), 1) for c in chunks], dtype=float)
    scores = lengths / lengths.max()
    scores = scores / max(tau, 1e-6)
    scores = np.exp(scores - scores.max())
    w = scores / scores.sum()
    return (chunk_vecs * w[:, None]).sum(axis=0)

def encode_long(texts, model, pooling="max", max_chars=1000, tau=0.5, normalize_chunks=True):
    vecs = []
    for t in texts:
        chunks = chunk_text(t, max_chars=max_chars) or [""]
        chunk_vecs = model.encode(chunks, normalize_embeddings=normalize_chunks)
        if normalize_chunks and isinstance(chunk_vecs, np.ndarray):
            chunk_vecs = l2_normalize(chunk_vecs, axis=1)
        if pooling == "mean":
            pooled = pool_mean(chunk_vecs)
        elif pooling == "attn_len":
            pooled = pool_attn_len(chunk_vecs, chunks, tau=tau)
        else:
            pooled = pool_max(chunk_vecs)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
        vecs.append(pooled)
    return np.stack(vecs)
