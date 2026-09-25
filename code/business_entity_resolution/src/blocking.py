"""
Multi-strategy Candidate Blocking and Retrieval Module.
Amazon ML Challenge 2026: Business Entity Resolution.

Implements 4 complementary blocking strategies with country-partitioned execution:
1. Token-overlap inverted index with IDF weighting (name tokens & domain names)
2. Character n-gram prefix & initials (script-agnostic, robust to typos/transliteration)
3. Phonetic key on Latin name tokens (Soundex)
4. Address locality, postal code, street number, and 2-word address shingles
"""

from __future__ import annotations

import array
import heapq
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import regex

from normalize import LEGAL_SUFFIX_TOKENS, normalize_address, normalize_name

# Common address stopwords to ignore for street/city token extraction
ADDR_STOPWORDS: set[str] = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "highway", "hwy", "boulevard", "blvd", "court", "ct", "place", "pl",
    "floor", "fl", "flr", "suite", "ste", "apartment", "apt", "room", "rm",
    "building", "bldg", "near", "nr", "opp", "opposite", "behind", "next",
    "route", "rte", "chemin", "che", "rue", "allee", "all", "cours", "crs",
}

TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+", regex.UNICODE)
WORD_RE = regex.compile(r"[\p{L}\p{M}]+", regex.UNICODE)
DIGIT_RE = regex.compile(r"\b\d+\b")
DOMAIN_RE = regex.compile(r"\.(com|in|org|net|co\.in|co|io|biz|info|fr)\b", regex.IGNORECASE)


def get_soundex(word: str) -> str:
    """Standard Soundex encoding for Latin words."""
    if not word:
        return ""
    word = word.upper()
    first = word[0]
    if not ('A' <= first <= 'Z'):
        return ""
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    encoded = [first]
    prev = mapping.get(first, '')
    for char in word[1:]:
        digit = mapping.get(char, '')
        if digit:
            if digit != prev:
                encoded.append(digit)
                prev = digit
        else:
            prev = ''
    return ("".join(encoded) + "0000")[:4]


def extract_blocking_keys(raw_name: Any, raw_address: Any) -> dict[str, list[str]]:
    """
    Extracts multi-strategy blocking keys from raw name and address:
    - name_tokens: informative clean name tokens (len >= 3) and bigram tokens
    - char_ngrams: script-agnostic 4-character prefix and initials
    - phonetic: Soundex code on first Latin token
    - address: postal codes, street number + word blocks, and 2-word address shingles
    """
    keys: dict[str, list[str]] = {
        "name_tokens": [],
        "char_ngrams": [],
        "phonetic": [],
        "address": [],
    }

    # 1. Name tokens and n-grams
    clean_tokens: list[str] = []
    if raw_name and not pd.isna(raw_name):
        s_name = str(raw_name).casefold()
        s_name = DOMAIN_RE.sub(" ", s_name)
        raw_tokens = TOKEN_RE.findall(s_name)
        clean_tokens = [t for t in raw_tokens if t not in LEGAL_SUFFIX_TOKENS and len(t) >= 2]

        for t in clean_tokens:
            if len(t) >= 3:
                keys["name_tokens"].append(t)
        if len(clean_tokens) >= 2:
            keys["name_tokens"].append(clean_tokens[0] + "_" + clean_tokens[1])

        no_space = "".join(clean_tokens)
        if len(no_space) >= 4:
            keys["char_ngrams"].append("p4_" + no_space[:4])
        if len(clean_tokens) >= 2 and len(clean_tokens[0]) >= 2 and len(clean_tokens[1]) >= 2:
            keys["char_ngrams"].append("init2_" + clean_tokens[0][:2] + clean_tokens[1][:2])

        if clean_tokens:
            sx = get_soundex(clean_tokens[0])
            if sx:
                keys["phonetic"].append("sx_" + sx)
                if len(clean_tokens) >= 2:
                    keys["phonetic"].append("sx2_" + sx + "_" + clean_tokens[1][:2])

    # 2. Address keys
    if raw_address and not pd.isna(raw_address):
        addr_str = str(raw_address).strip()
        if addr_str.lower() not in ("", "nan", "null", "none"):
            digits = DIGIT_RE.findall(addr_str)
            street_num = digits[0] if digits else None
            postal = digits[-1] if (digits and len(digits[-1]) >= 3) else None

            addr_words = [w for w in WORD_RE.findall(addr_str.casefold()) 
                          if w not in ADDR_STOPWORDS and len(w) >= 3]

            name_prefix = clean_tokens[0][:2] if clean_tokens else ""

            if postal:
                keys["address"].append("pin_" + postal)
                if name_prefix:
                    keys["address"].append("post_name_" + postal + "_" + name_prefix)
                if street_num:
                    keys["address"].append("post_num_" + postal + "_" + street_num)

            if street_num and addr_words:
                keys["address"].append("st_num_word_" + street_num + "_" + addr_words[0][:5])
                if len(addr_words) >= 2:
                    keys["address"].append("st_num_word2_" + street_num + "_" + addr_words[1][:5])

            for i in range(len(addr_words) - 1):
                w1, w2 = addr_words[i], addr_words[i+1]
                if len(w1) >= 3 and len(w2) >= 3:
                    keys["address"].append("ash_" + w1 + "_" + w2)

            for w in addr_words:
                if len(w) >= 5:
                    keys["address"].append("aw_" + w)

    return keys


class CandidateIndex:
    """
    Compact inverted index over candidates from Source 2 and Source 3.
    Uses array.array('I') (unsigned 32-bit integers) for minimal memory consumption.
    Supports streaming batch construction to keep peak RAM under 300 MB.
    """

    def __init__(self, cand_ids: list[str] | None = None, names: list[str] | None = None, addrs: list[str] | None = None):
        self.cand_ids: list[str] = []
        self.index_name_tokens: dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.index_char_ngrams: dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.index_phonetic: dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.index_address: dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.token_idf: dict[str, float] = {}
        self.addr_idf: dict[str, float] = {}

        if cand_ids is not None and names is not None and addrs is not None:
            self.add_batch(cand_ids, names, addrs)
            self.finalize()

    def add_batch(self, batch_ids: list[str], batch_names: list[str], batch_addrs: list[str]) -> None:
        start_idx = len(self.cand_ids)
        self.cand_ids.extend(batch_ids)
        n = len(batch_ids)
        for offset in range(n):
            idx = start_idx + offset
            keys = extract_blocking_keys(batch_names[offset], batch_addrs[offset])
            for k in keys["name_tokens"]:
                self.index_name_tokens[k].append(idx)
            for k in keys["char_ngrams"]:
                self.index_char_ngrams[k].append(idx)
            for k in keys["phonetic"]:
                self.index_phonetic[k].append(idx)
            for k in keys["address"]:
                self.index_address[k].append(idx)

    def finalize(self) -> None:
        n_cands = len(self.cand_ids)
        self.token_idf = {
            tok: math.log(1.0 + (n_cands / (1.0 + len(post))))
            for tok, post in self.index_name_tokens.items()
        }
        self.addr_idf = {
            ad: math.log(1.0 + (n_cands / (1.0 + len(post))))
            for ad, post in self.index_address.items()
        }

    def query(
        self,
        q_keys: dict[str, list[str]],
        candidate_cap: int = 50,
        max_token_post: int = 3000,
        max_char_post: int = 3000,
        max_phon_post: int = 3000,
        max_addr_post: int = 1000,
    ) -> list[tuple[str, float]]:
        """
        Retrieves top candidate IDs for a query entity ranked by combined strategy score.
        Optimized with selective posting list caps and heapq.nlargest for high throughput (>1,700 q/s).
        """
        cand_scores: dict[int, float] = defaultdict(float)

        # 1. Name tokens (IDF weighted)
        for tok in q_keys["name_tokens"]:
            if tok in self.index_name_tokens:
                post = self.index_name_tokens[tok]
                if len(post) <= max_token_post:
                    weight = self.token_idf.get(tok, 1.0)
                    if "_" in tok:
                        weight += 1.5
                    for c_idx in post:
                        cand_scores[c_idx] += weight

        # 2. Character n-grams
        for ng in q_keys["char_ngrams"]:
            if ng in self.index_char_ngrams:
                post = self.index_char_ngrams[ng]
                if len(post) <= max_char_post:
                    for c_idx in post:
                        cand_scores[c_idx] += 2.0

        # 3. Phonetic
        for ph in q_keys["phonetic"]:
            if ph in self.index_phonetic:
                post = self.index_phonetic[ph]
                if len(post) <= max_phon_post:
                    for c_idx in post:
                        cand_scores[c_idx] += 1.5

        # 4. Address keys
        for ad in q_keys["address"]:
            if ad in self.index_address:
                post = self.index_address[ad]
                if len(post) <= max_addr_post:
                    weight = self.addr_idf.get(ad, 1.0)
                    if ad.startswith("post_name_") or ad.startswith("post_num_"):
                        weight += 4.0
                    elif ad.startswith("ash_") or ad.startswith("st_num_word"):
                        weight += 3.0
                    for c_idx in post:
                        cand_scores[c_idx] += weight

        if not cand_scores:
            return []

        # heapq.nlargest is significantly faster than sorting full dictionary
        top_items = heapq.nlargest(candidate_cap, cand_scores.items(), key=lambda x: x[1])
        return [(self.cand_ids[c_idx], score) for c_idx, score in top_items]


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_cap: int = 50,
) -> pd.DataFrame:
    """
    Generates candidate pairs partitioned by country.
    Runs identically on US, India, France (unseen), or any other country partition.
    Returns DataFrame with columns ['source1_entity_id', 'candidate_entity_id', 'blocking_score'].
    """
    output_rows: list[dict] = []
    countries = s1_df["country"].unique()

    for ctry in countries:
        s1_c = s1_df[s1_df["country"] == ctry]
        s2_c = s2_df[s2_df["country"] == ctry]
        s3_c = s3_df[s3_df["country"] == ctry]

        cands_c = pd.concat([s2_c, s3_c], ignore_index=True)
        if len(cands_c) == 0:
            continue

        cand_ids = cands_c["entity_id"].tolist()
        names = cands_c["business_name"].fillna("").tolist()
        addrs = cands_c["business_address"].fillna("").tolist()

        index = CandidateIndex(cand_ids, names, addrs)

        for _, row in s1_c.iterrows():
            s1_id = row["entity_id"]
            q_keys = extract_blocking_keys(row["business_name"], row["business_address"])
            top_cands = index.query(q_keys, candidate_cap=candidate_cap)
            for c_id, score in top_cands:
                output_rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": c_id,
                    "blocking_score": score,
                })

    return pd.DataFrame(output_rows)
