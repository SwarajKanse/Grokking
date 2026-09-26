"""
Memory Stage Breakdown Diagnostic Script.
Runs the exact small slice (2,500 entities, 124,633 candidate pairs)
and logs Current RSS and Peak RSS at each discrete pipeline stage:
1. Baseline
2. Validation split & Source 1 load
3. Source 1 normalization
4. Source 2 raw load & normalization
5. Source 3 raw load & normalization
6. Country grouping / partitioning
7. Inverted index construction (US)
8. Inverted index construction (India)
9. Querying candidates for 2,500 entities (124,633 pairs)
10. Pairwise feature extraction
"""

from __future__ import annotations

import gc
import os
import resource
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "business_entity_resolution", "src")))

from blocking import CandidateIndex, extract_blocking_keys
from features import (
    build_pair_features_df,
    compute_pair_features,
    extract_digit_tokens,
)
from normalize import NormalizedRecord, normalize_address, normalize_name
from extract_features import load_normalized_lookup


def get_mem_mb() -> tuple[float, float]:
    """Returns (current_rss_mb, peak_rss_mb) on Linux."""
    curr_mb = 0.0
    try:
        with open("/proc/self/statm", "r") as f:
            parts = f.read().split()
            pages = int(parts[1])
            curr_mb = (pages * os.sysconf("SC_PAGE_SIZE")) / (1024.0 * 1024.0)
    except Exception:
        pass
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return curr_mb, peak_mb


def print_stage(stage_num: int, stage_name: str, prev_curr: float) -> tuple[float, float]:
    gc.collect()
    curr_mb, peak_mb = get_mem_mb()
    delta_mb = curr_mb - prev_curr
    print(f"| Stage {stage_num:02d} | {stage_name:<45} | {curr_mb:9.2f} MB | {delta_mb:+9.2f} MB | {peak_mb:9.2f} MB |")
    return curr_mb, peak_mb


def main():
    print("=" * 105)
    print("PHASE 3 MEMORY BREAKDOWN AUDIT — DISCRETE STAGE MEASUREMENT ON AZURE VM")
    print("=" * 105)
    print(f"| Stage    | {'Pipeline Stage Description':<45} | Current RSS | Stage Delta | Peak RSS    |")
    print("|" + "-" * 10 + "|" + "-" * 47 + "|" + "-" * 13 + "|" + "-" * 13 + "|" + "-" * 13 + "|")

    curr_mb, peak_mb = print_stage(0, "Process Baseline (Python runtime + imports)", 0.0)

    # 1. Load Validation Split
    val_parquet_path = "experiments/val_split.parquet"
    val_df = pd.read_parquet(val_parquet_path)
    curr_mb, peak_mb = print_stage(1, "Load val_split.parquet (25k entities)", curr_mb)

    # 2. Load train_source1.tsv
    s1_tsv = "dataset/train/train_source1.tsv"
    s1_df = pd.read_csv(s1_tsv, sep="\t", dtype=str).fillna("")
    curr_mb, peak_mb = print_stage(2, "Load train_source1.tsv (2.2M rows pandas DF)", curr_mb)

    # 3. Merge validation entities
    val_merged = val_df.merge(
        s1_df[["entity_id", "business_name", "business_address"]],
        left_on="source1_entity_id",
        right_on="entity_id",
        how="inner",
    )
    # Free s1_df to see if verify_phase3 held onto it
    del s1_df
    curr_mb, peak_mb = print_stage(3, "Merge val_merged & drop s1_df", curr_mb)

    # 4. Normalize Source 1 validation records
    s1_lookup: dict[str, NormalizedRecord] = {}
    for e_id, ctry, raw_name, raw_addr in zip(
        val_merged["source1_entity_id"],
        val_merged["country"],
        val_merged["business_name"],
        val_merged["business_address"],
    ):
        s1_lookup[e_id] = NormalizedRecord(
            entity_id=e_id,
            country=ctry,
            name=normalize_name(raw_name),
            address=normalize_address(raw_addr),
        )
    curr_mb, peak_mb = print_stage(4, "Normalize 25k S1 records (s1_lookup)", curr_mb)

    # 5. Load and normalize Source 2
    s2_path = "dataset/train/train_source2.tsv"
    t0 = time.time()
    s2_lookup = load_normalized_lookup(s2_path, source_tag="source2")
    curr_mb, peak_mb = print_stage(5, f"Load & normalize S2 ({len(s2_lookup):,} records)", curr_mb)

    # 6. Load and normalize Source 3
    s3_path = "dataset/train/train_source3.tsv"
    t0 = time.time()
    s3_lookup = load_normalized_lookup(s3_path, source_tag="source3")
    curr_mb, peak_mb = print_stage(6, f"Load & normalize S3 ({len(s3_lookup):,} records)", curr_mb)

    # Combine into cand_lookup
    cand_lookup: dict[str, NormalizedRecord] = {}
    cand_lookup.update(s2_lookup)
    cand_lookup.update(s3_lookup)
    del s2_lookup, s3_lookup
    curr_mb, peak_mb = print_stage(7, f"Combine into cand_lookup ({len(cand_lookup):,} records)", curr_mb)

    # 8. Group candidates by country
    country_candidates: dict[str, list[NormalizedRecord]] = {}
    for c_rec in cand_lookup.values():
        c_code = c_rec.country.strip().lower()
        if c_code not in country_candidates:
            country_candidates[c_code] = []
        country_candidates[c_code].append(c_rec)
    curr_mb, peak_mb = print_stage(8, "Group cand_lookup into country_candidates lists", curr_mb)

    # 9. Build index for US
    us_cands = country_candidates.get("us", [])
    c_ids_us = [r.entity_id for r in us_cands]
    names_us = [r.name.raw for r in us_cands]
    addrs_us = [r.address.raw for r in us_cands]
    curr_mb, peak_mb = print_stage(9, f"Derived lists for US index ({len(us_cands):,} records)", curr_mb)

    idx_us = CandidateIndex(c_ids_us, names_us, addrs_us)
    del c_ids_us, names_us, addrs_us
    curr_mb, peak_mb = print_stage(10, "Construct CandidateIndex(US)", curr_mb)

    # 10. Build index for India
    in_cands = country_candidates.get("india", [])
    c_ids_in = [r.entity_id for r in in_cands]
    names_in = [r.name.raw for r in in_cands]
    addrs_in = [r.address.raw for r in in_cands]
    curr_mb, peak_mb = print_stage(11, f"Derived lists for India index ({len(in_cands):,} records)", curr_mb)

    idx_in = CandidateIndex(c_ids_in, names_in, addrs_in)
    del c_ids_in, names_in, addrs_in
    curr_mb, peak_mb = print_stage(12, "Construct CandidateIndex(India)", curr_mb)

    # Free country_candidates list container (since cand_lookup already has them)
    del country_candidates
    curr_mb, peak_mb = print_stage(13, "Free country_candidates container", curr_mb)

    # 11. Small Slice Generation (2,500 entities)
    slice_sample_size = 2500
    slice_df = val_merged.sample(n=slice_sample_size, random_state=42).reset_index(drop=True)

    gt_lookup: dict[str, set[str]] = {}
    for s1, matched_str in zip(val_merged["source1_entity_id"], val_merged["matched_entity_ids"]):
        if pd.isna(matched_str) or not matched_str:
            gt_lookup[s1] = set()
        else:
            gt_lookup[s1] = set(x.strip() for x in str(matched_str).split(",") if x.strip())

    indexes = {"us": idx_us, "india": idx_in}
    slice_pairs_list: list[dict[str, Any]] = []
    for s1_id, ctry, b_name, b_addr in zip(
        slice_df["source1_entity_id"],
        slice_df["country"],
        slice_df["business_name"],
        slice_df["business_address"],
    ):
        ctry_clean = ctry.strip().lower()
        idx = indexes.get(ctry_clean)
        if not idx:
            continue
        q_keys = extract_blocking_keys(b_name, b_addr)
        top_cands = idx.query(q_keys, candidate_cap=50)
        s1_true_matches = gt_lookup.get(s1_id, set())

        for rank, (cand_id, score) in enumerate(top_cands, 1):
            slice_pairs_list.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "blocking_score": score,
                "candidate_rank": rank,
                "label": 1 if cand_id in s1_true_matches else 0,
            })
    curr_mb, peak_mb = print_stage(14, f"Generate slice_pairs_list ({len(slice_pairs_list):,} dicts)", curr_mb)

    slice_pairs_df = pd.DataFrame(slice_pairs_list)
    del slice_pairs_list
    curr_mb, peak_mb = print_stage(15, f"Convert to slice_pairs_df ({len(slice_pairs_df):,} rows)", curr_mb)

    # 12. Feature Extraction on Slice
    t_feat0 = time.time()
    slice_features_df = build_pair_features_df(slice_pairs_df, s1_lookup, cand_lookup, chunk_size=25000)
    feat_time = time.time() - t_feat0
    curr_mb, peak_mb = print_stage(16, f"Feature extraction on slice ({len(slice_features_df):,} rows, {feat_time:.2f}s)", curr_mb)

    print("=" * 105)
    print("SUMMARY OF OBSERVED PEAK RSS & BOTTLENECK IDENTIFICATION")
    print(f"Final Current RSS: {curr_mb:9.2f} MB ({curr_mb/1024:.2f} GB)")
    print(f"Final Peak RSS:    {peak_mb:9.2f} MB ({peak_mb/1024:.2f} GB)")
    print("=" * 105)


if __name__ == "__main__":
    main()
