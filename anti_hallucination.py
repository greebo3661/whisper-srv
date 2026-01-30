import re
import logging
from collections import deque
from typing import List, Optional, Dict, Any

from config import (
    MIN_WORD_CONFIDENCE,
    MAX_REPEAT_WINDOW_SEC,
    REPEAT_SIMILARITY_THRESHOLD,
    INTRA_REPEAT_NGRAM,
    INTRA_REPEAT_MAX_RATIO,
)

logger = logging.getLogger(__name__)


def _normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[\s]+", " ", t)
    t = re.sub(r"[^\w\s]+", "", t)
    return t.strip()


def _jaccard_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    inter = len(sa & sb)
    union = len(sa | sb) or 1
    return inter / union


def filter_low_confidence_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Отбрасываем сегменты с низкой средней уверенностью по словам."""
    filtered = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            filtered.append(seg)
            continue
        probs = [w.get("probability", 1.0) for w in words]
        avg_prob = sum(probs) / max(len(probs), 1)
        if avg_prob >= MIN_WORD_CONFIDENCE:
            filtered.append(seg)
        else:
            logger.info(
                f"Фильтр prob: сегмент '{seg.get('text','')[:30]}' выкинут ({avg_prob:.2f} < {MIN_WORD_CONFIDENCE})"
            )
    return filtered


def filter_repetitions(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Удаляем повторы сегментов в скользящем окне по времени."""
    window = deque()
    result = []

    for seg in segments:
        seg_text = seg.get("text", "")
        norm = _normalize_text(seg_text)
        if not norm:
            result.append(seg)
            continue

        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        seg_mid = (seg_start + seg_end) / 2.0

        while window and seg_mid - window[0][0] > MAX_REPEAT_WINDOW_SEC:
            window.popleft()

        is_repeat = False
        for _, prev_norm in window:
            sim = _jaccard_similarity(norm, prev_norm)
            if sim >= REPEAT_SIMILARITY_THRESHOLD:
                is_repeat = True
                logger.info(
                    f"Фильтр repeat: сегмент '{seg_text[:30]}' выкинут как повтор (sim={sim:.2f})"
                )
                break

        if not is_repeat:
            result.append(seg)
            window.append((seg_mid, norm))

    return result


def filter_intra_segment_repeats(seg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Отбрасываем/обрезаем сегменты, где большая часть текста — повторяющиеся n-граммы.
    """
    text = seg.get("text", "")
    norm = _normalize_text(text)
    if not norm:
        return seg

    tokens = norm.split()
    if len(tokens) < INTRA_REPEAT_NGRAM * 2:
        return seg

    # строим n-граммы
    n = INTRA_REPEAT_NGRAM
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    freq = {}
    for ng in ngrams:
        freq[ng] = freq.get(ng, 0) + 1

    total_ngrams = len(ngrams)
    max_count = max(freq.values()) if freq else 0
    max_ratio = max_count / max(total_ngrams, 1)

    if max_ratio >= INTRA_REPEAT_MAX_RATIO:
        # сегмент почти полностью состоит из одной повторяющейся n-граммы
        logger.info(
            f"Фильтр intra-repeat: сегмент '{text[:30]}' выкинут (max_ratio={max_ratio:.2f})"
        )
        return None

    return seg


def apply_anti_hallucination(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Композитный фильтр: prob -> внутрисегментные повторы -> межсегментные повторы."""
    segs = segments or []

    segs = filter_low_confidence_words(segs)

    trimmed = []
    for seg in segs:
        new_seg = filter_intra_segment_repeats(seg)
        if new_seg is not None:
            trimmed.append(new_seg)
    segs = trimmed

    segs = filter_repetitions(segs)

    return segs
