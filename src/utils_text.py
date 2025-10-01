import re
from typing import List
import numpy as np

NORM_MAP = {
    r"\b(cash[\s-]?back|кэш[- ]?б?е?к)\b": "кэшбэк",
    r"\b(premium|озон\s*premium|ozon\s*premium)\b": "ozon_premium",
    r"\b(push|пуш[- ]?уведомления?)\b": "push_уведомления",
    r"\b(app|апп)\b": "мобильное_приложение",
    r"\b(тинькофф|tinkoff|tcs)\b": "tinkoff",
    r"\b(gazprom\s*bonus|газпром\s*бонус)\b": "gazprom_bonus",
    r"\b(union\s*pay|unionpay|юнион\s*пей)\b": "unionpay",
}

def normalize_anglicisms(s: str) -> str:
    s2 = s or ""
    for pat, repl in NORM_MAP.items():
        s2 = re.sub(pat, repl, s2, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s2).strip()

try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    HAVE_MORPH = True
except Exception:
    morph = None
    HAVE_MORPH = False

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    RU_STOPS = set(stopwords.words("russian"))
except Exception:
    RU_STOPS = set()

KEEP_RAW = {"tinkoff", "ozon_premium", "gazprom_bonus", "push_уведомления", "мобильное_приложение", "unionpay"}

def normalize_ru(text: str) -> str:
    out = []
    for w in (text or "").split():
        lw = w.lower()
        if lw in KEEP_RAW or re.search(r"[A-Za-z]", lw):
            if lw not in RU_STOPS:
                out.append(lw); continue
        if HAVE_MORPH:
            p = morph.parse(lw)[0]
            if p.tag.POS in {"PREP", "CONJ", "PRCL"}:  # предлоги/союзы/частицы
                continue
            lemma = p.normal_form
            if lemma and lemma not in RU_STOPS and not re.search(r"\d", lemma):
                out.append(lemma)
        else:
            if lw not in RU_STOPS and not re.search(r"\d", lw):
                out.append(lw)
    return " ".join(out)

def basic_clean(s: str) -> str:
    s = (s or "").replace("\u200b", " ")
    return re.sub(r"\s+", " ", s).strip()

def prepare_text(title: str, text: str) -> str:
    full = f"{(title or '').strip()}. {(text or '').strip()}".strip(". ").strip()
    full = basic_clean(full)
    full = normalize_anglicisms(full)
    full = normalize_ru(full)
    full = basic_clean(full)
    return full

def softmax_cos(sims: np.ndarray, tau: float = 0.07) -> np.ndarray:
    x = np.array(sims, dtype=float) / float(tau)
    x -= x.max()
    p = np.exp(x); p /= p.sum()
    return p
