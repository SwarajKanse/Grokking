"""
Pairwise Feature Engineering Module.
Amazon ML Challenge 2026: Business Entity Resolution.

Computes fine-grained string similarity, numeric, missingness, and metadata features
for candidate pairs surviving Phase 2 blocking.
Optimized for high throughput (>5,000 pairs/sec) and minimal peak memory (<2 GB).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import NormalizedAddress, NormalizedName, NormalizedRecord

DIGIT_TOKEN_RE = re.compile(r"\d+")

# Country frequency mapping from EDA base rates (with fallback for unseen countries like France)
COUNTRY_FREQ_MAP: dict[str, float] = {
    "us": 0.60,
    "in": 0.40,
    "india": 0.40,
    "united states": 0.60,
}
DEFAULT_COUNTRY_FREQ: float = 0.15


def get_char_qgrams(text: str, q: int = 3) -> set[str]:
    """Generates character q-grams for fine-grained substring overlap."""
    if len(text) < q:
        return {text} if text else set()
    return {text[i : i + q] for i in range(len(text) - q + 1)}


def extract_digit_tokens(text: str) -> set[str]:
    """
    Extracts all standalone digit sequences from text for position-agnostic numeric matching,
    normalizing leading zeros so e.g. '006201' matches '6201'.
    """
    if not text:
        return set()
    return {t.lstrip("0") or "0" for t in DIGIT_TOKEN_RE.findall(text)}


def compute_pair_features(
    s1_rec: NormalizedRecord,
    cand_rec: NormalizedRecord,
    blocking_score: float = 0.0,
    candidate_rank: int = 1,
    candidate_count: int = 1,
    top_score: float = 0.0,
    cand_source: int = 2,
) -> dict[str, Any]:
    """
    Computes all pairwise features for a single (Source 1, Candidate) pair.
    Uses sentinel value (-1.0) when fields are missing to distinguish from genuine low similarity (0.0).
    """
    n1 = s1_rec.name
    n2 = cand_rec.name
    a1 = s1_rec.address
    a2 = cand_rec.address

    has_n1 = n1.has_name and bool(n1.norm)
    has_n2 = n2.has_name and bool(n2.norm)
    both_name = has_n1 and has_n2

    has_a1 = a1.has_address and bool(a1.norm)
    has_a2 = a2.has_address and bool(a2.norm)
    both_addr = has_a1 and has_a2

    # 1. Name Features
    if both_name:
        name_lev_sim = float(Levenshtein.normalized_similarity(n1.norm, n2.norm))
        name_jw_sim = float(JaroWinkler.similarity(n1.norm, n2.norm))
        name_token_sort = float(fuzz.token_sort_ratio(n1.norm, n2.norm) / 100.0)
        name_token_set = float(fuzz.token_set_ratio(n1.norm, n2.norm) / 100.0)

        toks1 = set(n1.tokens)
        toks2 = set(n2.tokens)
        name_union = len(toks1 | toks2)
        name_inter = len(toks1 & toks2)
        name_word_jaccard = float(name_inter / name_union) if name_union > 0 else 0.0
        name_common_tokens = name_inter

        qg1 = get_char_qgrams(n1.clean_norm or n1.norm, q=3)
        qg2 = get_char_qgrams(n2.clean_norm or n2.norm, q=3)
        qg_union = len(qg1 | qg2)
        name_char3_jaccard = float(len(qg1 & qg2) / qg_union) if qg_union > 0 else 0.0

        l1, l2 = len(n1.norm), len(n2.norm)
        name_len_diff = abs(l1 - l2)
        name_len_ratio = float(min(l1, l2) / max(l1, l2)) if max(l1, l2) > 0 else 1.0

        clean1 = n1.clean_norm or n1.norm
        clean2 = n2.clean_norm or n2.norm
        name_exact_clean = 1.0 if (clean1 and clean1 == clean2) else 0.0
        name_exact_raw = 1.0 if (n1.raw and n1.raw.strip().lower() == n2.raw.strip().lower()) else 0.0

        suf1 = n1.legal_suffix
        suf2 = n2.legal_suffix
        name_suffix_match = 1.0 if (suf1 and suf1 == suf2) else (0.5 if (not suf1 and not suf2) else 0.0)
    else:
        name_lev_sim = -1.0
        name_jw_sim = -1.0
        name_token_sort = -1.0
        name_token_set = -1.0
        name_word_jaccard = -1.0
        name_char3_jaccard = -1.0
        name_common_tokens = -1
        name_len_diff = -1
        name_len_ratio = -1.0
        name_exact_clean = -1.0
        name_exact_raw = -1.0
        name_suffix_match = -1.0

    # 2. Address Features
    if both_addr:
        addr_lev_sim = float(Levenshtein.normalized_similarity(a1.norm, a2.norm))
        addr_jw_sim = float(JaroWinkler.similarity(a1.norm, a2.norm))
        addr_token_sort = float(fuzz.token_sort_ratio(a1.norm, a2.norm) / 100.0)
        addr_token_set = float(fuzz.token_set_ratio(a1.norm, a2.norm) / 100.0)

        atoks1 = set(a1.tokens)
        atoks2 = set(a2.tokens)
        a_union = len(atoks1 | atoks2)
        a_inter = len(atoks1 & atoks2)
        addr_word_jaccard = float(a_inter / a_union) if a_union > 0 else 0.0
        addr_common_tokens = a_inter

        al1, al2 = len(a1.norm), len(a2.norm)
        addr_len_diff = abs(al1 - al2)
        addr_exact_norm = 1.0 if (a1.norm and a1.norm == a2.norm) else 0.0

        # Position-agnostic numeric overlap across entire address
        d1 = extract_digit_tokens(a1.raw or a1.norm)
        d2 = extract_digit_tokens(a2.raw or a2.norm)
        d_union = len(d1 | d2)
        d_inter = len(d1 & d2)
        digits_overlap = 1.0 if d_inter > 0 else 0.0
        digits_jaccard = float(d_inter / d_union) if d_union > 0 else 0.0
    else:
        addr_lev_sim = -1.0
        addr_jw_sim = -1.0
        addr_token_sort = -1.0
        addr_token_set = -1.0
        addr_word_jaccard = -1.0
        addr_common_tokens = -1
        addr_len_diff = -1
        addr_exact_norm = -1.0
        digits_overlap = -1.0
        digits_jaccard = -1.0

    # 3. Numeric & Locality Features
    p1 = a1.postal_code or ""
    p2 = a2.postal_code or ""
    both_postal = bool(p1 and p2)
    postal_exact = 1.0 if (both_postal and p1 == p2) else (0.0 if both_postal else -1.0)
    postal_sim = float(Levenshtein.normalized_similarity(p1, p2)) if both_postal else -1.0

    st1 = (a1.street_number or "").lstrip("0") or ("0" if a1.street_number else "")
    st2 = (a2.street_number or "").lstrip("0") or ("0" if a2.street_number else "")
    both_street = bool(st1 and st2)
    street_num_exact = 1.0 if (both_street and st1 == st2) else (0.0 if both_street else -1.0)

    # 4. Country Features (Safe categorical encoding with unseen fallback)
    c1 = s1_rec.country.strip().lower()
    c2 = cand_rec.country.strip().lower()
    same_country = 1.0 if (c1 and c1 == c2) else 0.0
    country_freq = COUNTRY_FREQ_MAP.get(c1, DEFAULT_COUNTRY_FREQ)

    # 5. Metadata & Blocking Signals
    score_gap = float(top_score - blocking_score) if top_score >= blocking_score else 0.0

    return {
        # Name similarities
        "name_lev_sim": np.float32(name_lev_sim),
        "name_jw_sim": np.float32(name_jw_sim),
        "name_token_sort": np.float32(name_token_sort),
        "name_token_set": np.float32(name_token_set),
        "name_word_jaccard": np.float32(name_word_jaccard),
        "name_char3_jaccard": np.float32(name_char3_jaccard),
        "name_common_tokens": np.int16(name_common_tokens),
        "name_len_diff": np.int16(name_len_diff),
        "name_len_ratio": np.float32(name_len_ratio),
        "name_exact_clean": np.int8(name_exact_clean),
        "name_exact_raw": np.int8(name_exact_raw),
        "name_suffix_match": np.float32(name_suffix_match),
        # Address similarities
        "addr_lev_sim": np.float32(addr_lev_sim),
        "addr_jw_sim": np.float32(addr_jw_sim),
        "addr_token_sort": np.float32(addr_token_sort),
        "addr_token_set": np.float32(addr_token_set),
        "addr_word_jaccard": np.float32(addr_word_jaccard),
        "addr_common_tokens": np.int16(addr_common_tokens),
        "addr_len_diff": np.int16(addr_len_diff),
        "addr_exact_norm": np.int8(addr_exact_norm),
        # Numeric & Locality
        "postal_exact": np.int8(postal_exact),
        "postal_sim": np.float32(postal_sim),
        "street_num_exact": np.int8(street_num_exact),
        "digits_overlap": np.int8(digits_overlap),
        "digits_jaccard": np.float32(digits_jaccard),
        # Missingness flags
        "has_name_s1": np.int8(1 if has_n1 else 0),
        "has_name_cand": np.int8(1 if has_n2 else 0),
        "both_have_name": np.int8(1 if both_name else 0),
        "has_addr_s1": np.int8(1 if has_a1 else 0),
        "has_addr_cand": np.int8(1 if has_a2 else 0),
        "both_have_address": np.int8(1 if both_addr else 0),
        "has_postal_s1": np.int8(1 if p1 else 0),
        "has_postal_cand": np.int8(1 if p2 else 0),
        "both_have_postal": np.int8(1 if both_postal else 0),
        # Country
        "same_country": np.int8(same_country),
        "country_freq": np.float32(country_freq),
        # Meta & Blocking
        "blocking_score": np.float32(blocking_score),
        "candidate_rank": np.int16(candidate_rank),
        "candidate_count": np.int16(candidate_count),
        "score_gap_to_top": np.float32(score_gap),
        "is_source2": np.int8(1 if cand_source == 2 else 0),
        "is_source3": np.int8(1 if cand_source == 3 else 0),
    }


FEATURE_COLUMN_NAMES: list[str] = [
    "name_lev_sim", "name_jw_sim", "name_token_sort", "name_token_set",
    "name_word_jaccard", "name_char3_jaccard", "name_common_tokens",
    "name_len_diff", "name_len_ratio", "name_exact_clean", "name_exact_raw",
    "name_suffix_match", "addr_lev_sim", "addr_jw_sim", "addr_token_sort",
    "addr_token_set", "addr_word_jaccard", "addr_common_tokens", "addr_len_diff",
    "addr_exact_norm", "postal_exact", "postal_sim", "street_num_exact",
    "digits_overlap", "digits_jaccard", "has_name_s1", "has_name_cand",
    "both_have_name", "has_addr_s1", "has_addr_cand", "both_have_address",
    "has_postal_s1", "has_postal_cand", "both_have_postal", "same_country",
    "country_freq", "blocking_score", "candidate_rank", "candidate_count",
    "score_gap_to_top", "is_source2", "is_source3"
]


def build_pair_features_df(
    pairs_df: pd.DataFrame,
    s1_lookup: dict[str, NormalizedRecord],
    cand_lookup: dict[str, NormalizedRecord],
    chunk_size: int = 50000,
) -> pd.DataFrame:
    """
    Extracts features for all candidate pairs in chunked fashion to maintain low peak RSS.
    Returns DataFrame containing identifier columns plus all 42 numerical/flag features.
    """
    chunks: list[pd.DataFrame] = []
    total_pairs = len(pairs_df)

    # Compute top score per source1_entity_id for relative gap calculation
    top_scores = pairs_df.groupby("source1_entity_id")["blocking_score"].max().to_dict()
    cand_counts = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].count().to_dict()

    for start_idx in range(0, total_pairs, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pairs)
        chunk_slice = pairs_df.iloc[start_idx:end_idx]

        s1_ids = chunk_slice["source1_entity_id"].tolist()
        cand_ids = chunk_slice["candidate_entity_id"].tolist()
        scores = chunk_slice["blocking_score"].tolist() if "blocking_score" in chunk_slice else [0.0] * len(chunk_slice)
        ranks = chunk_slice["candidate_rank"].tolist() if "candidate_rank" in chunk_slice else list(range(1, len(chunk_slice) + 1))
        has_label = "label" in chunk_slice
        labels = chunk_slice["label"].tolist() if has_label else None

        records_list: list[dict[str, Any]] = []
        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            cand_id = cand_ids[i]
            score = float(scores[i])
            rank = int(ranks[i])

            s1_rec = s1_lookup.get(s1_id)
            cand_rec = cand_lookup.get(cand_id)
            if not s1_rec or not cand_rec:
                continue

            top_sc = top_scores.get(s1_id, score)
            count = cand_counts.get(s1_id, 1)
            source_tag = 3 if cand_id.startswith("S3-") else 2

            feat = compute_pair_features(
                s1_rec=s1_rec,
                cand_rec=cand_rec,
                blocking_score=score,
                candidate_rank=rank,
                candidate_count=count,
                top_score=top_sc,
                cand_source=source_tag,
            )
            feat["source1_entity_id"] = s1_id
            feat["candidate_entity_id"] = cand_id
            if has_label and labels is not None:
                feat["label"] = np.int8(labels[i])

            records_list.append(feat)

        if records_list:
            chunk_df = pd.DataFrame(records_list)
            chunks.append(chunk_df)

    if not chunks:
        cols = ["source1_entity_id", "candidate_entity_id"] + FEATURE_COLUMN_NAMES
        return pd.DataFrame(columns=cols)

    return pd.concat(chunks, ignore_index=True)
