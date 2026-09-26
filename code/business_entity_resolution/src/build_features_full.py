"""
Full-scale Candidate Blocking and Pairwise Feature Extraction Pipeline.
Amazon ML Challenge 2026: Business Entity Resolution.

Architecture:
- Country-partitioned execution for minimal peak memory (<8 GB RSS).
- Stage 1: Blocking index construction + top-K retrieval.
- Stage 2: Streaming candidate normalization + pairwise feature extraction in 100k chunks.
- Stage 3: Streaming append to Parquet via pyarrow.parquet.ParquetWriter.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

def safe_print(*args, **kwargs):
    """Prints with flush=True and ignores BrokenPipeError if stdout pipe is closed."""
    try:
        sys.stdout.write(" ".join(str(a) for a in args) + kwargs.get("end", "\n"))
        sys.stdout.flush()
    except (BrokenPipeError, IOError):
        pass

print = safe_print

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocking import CandidateIndex, extract_blocking_keys
from features import FEATURE_COLUMN_NAMES, compute_pair_features
from normalize import NormalizedRecord, normalize_record


def get_peak_rss_mb() -> float:
    """Returns peak resident set size in megabytes."""
    try:
        import resource
        # On Linux ru_maxrss is in kilobytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            return 0.0


def build_arrow_schema(is_labeled: bool) -> tuple[list[str], pa.Schema]:
    """Builds the exact PyArrow schema using dummy feature extraction."""
    expected_cols = ["source1_entity_id", "candidate_entity_id"] + list(FEATURE_COLUMN_NAMES)
    if is_labeled:
        expected_cols.append("label")

    dummy_s1 = normalize_record("dummy_s1", "Acme Corp", "123 Main St", "US")
    dummy_c1 = normalize_record("dummy_c1", "Acme Corp", "123 Main St", "US")
    dummy_feat = compute_pair_features(dummy_s1, dummy_c1)
    dummy_feat["source1_entity_id"] = "dummy_s1"
    dummy_feat["candidate_entity_id"] = "dummy_c1"
    if is_labeled:
        dummy_feat["label"] = np.int8(0)

    dummy_df = pd.DataFrame([dummy_feat])[expected_cols]
    schema = pa.Schema.from_pandas(dummy_df, preserve_index=False)
    return expected_cols, schema


def run_pipeline(
    split: str,
    candidate_cap: int = 100,
    num_entities: Optional[int] = None,
    seed: int = 42,
    chunk_size: int = 500000,
    write_batch_size: int = 100000,
    selected_countries: Optional[list[str]] = None,
    custom_output_path: Optional[str] = None,
) -> str:
    """Executes the full-scale candidate blocking and feature extraction pipeline."""
    t_start = time.time()
    os.makedirs("experiments", exist_ok=True)
    output_path = custom_output_path or f"experiments/{split}_features_k{candidate_cap}.parquet"

    print("=" * 70)
    print(f"BER FULL-SCALE FEATURE PIPELINE: split={split}, K={candidate_cap}, seed={seed}")
    print(f"Output path: {output_path}")
    print("=" * 70)

    is_labeled = split in ("train", "val")
    expected_columns, arrow_schema = build_arrow_schema(is_labeled)

    # 1. Determine paths
    if split == "test":
        s1_path = "dataset/test/test_source1.tsv"
        s2_path = "dataset/test/test_source2.tsv"
        s3_path = "dataset/test/test_source3.tsv"
        gt_path = None
    else:
        s1_path = "dataset/train/train_source1.tsv"
        s2_path = "dataset/train/train_source2.tsv"
        s3_path = "dataset/train/train_source3.tsv"
        gt_path = "dataset/train/train_ground_truth.tsv"

    # 2. Load Source 1 entities for this split
    print(f"\n[Step 1] Loading Source 1 entities from {s1_path}...")
    s1_full = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    print(f"Loaded {len(s1_full):,} Source 1 entities.")

    if split == "train":
        val_split_path = "experiments/val_split.parquet"
        if os.path.exists(val_split_path):
            val_df = pd.read_parquet(val_split_path)
            val_ids = set(val_df["source1_entity_id"])
            print(f"Excluding {len(val_ids):,} validation entities...")
            s1_full = s1_full[~s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
            print(f"Remaining training entities: {len(s1_full):,}")
        else:
            print("Warning: experiments/val_split.parquet not found; no validation entities excluded.")

        if num_entities is not None and num_entities < len(s1_full):
            print(f"Sampling {num_entities:,} entities with seed={seed}...")
            s1_full = s1_full.sample(n=num_entities, random_state=seed).reset_index(drop=True)
        print(f"Final training S1 entities to process: {len(s1_full):,}")

    elif split == "val":
        val_split_path = "experiments/val_split.parquet"
        if not os.path.exists(val_split_path):
            raise FileNotFoundError(f"Required validation split {val_split_path} not found.")
        val_df = pd.read_parquet(val_split_path)
        val_ids = set(val_df["source1_entity_id"])
        print(f"Filtering to {len(val_ids):,} validation entities...")
        s1_full = s1_full[s1_full["entity_id"].isin(val_ids)].reset_index(drop=True)
        print(f"Final validation S1 entities to process: {len(s1_full):,}")

    # 3. Load Ground Truth if labeled
    gt_map: dict[str, set[str]] = {}
    if is_labeled and gt_path and os.path.exists(gt_path):
        print(f"\n[Step 2] Loading ground truth from {gt_path}...")
        t_gt = time.time()
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        s1_id_set = set(s1_full["entity_id"])
        for row in gt_df.itertuples(index=False):
            s1_id = row.source1_entity_id
            if s1_id in s1_id_set:
                matched = row.matched_entity_ids
                gt_map[s1_id] = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
        del gt_df
        gc.collect()
        print(f"Loaded ground truth for {len(gt_map):,} entities in {time.time()-t_gt:.2f}s.")

    # 4. Process Country by Country
    if selected_countries:
        countries = [c for c in selected_countries if c in s1_full["country"].unique()]
    else:
        countries = [c for c in s1_full["country"].unique() if c]
    print(f"\n[Step 3] Processing countries: {countries}")

    total_pairs_all = 0
    total_positives_all = 0
    writer: Optional[pq.ParquetWriter] = None

    # Track metrics per stage for final summary
    country_stats = []

    for ctry in countries:
        t_c_start = time.time()
        print(f"\n{'='*40}\nProcessing Country: {ctry}\n{'='*40}")

        s1_c = s1_full[s1_full["country"] == ctry].reset_index(drop=True)
        num_s1 = len(s1_c)
        if num_s1 == 0:
            print(f"No S1 entities for {ctry}, skipping.")
            continue

        # --- STAGE 1: Blocking ---
        print(f"Stage 1: Building CandidateIndex for {ctry}...")
        t_idx_start = time.time()
        index = CandidateIndex()

        # Stream candidate source data (S2 + S3)
        cand_count_loaded = 0
        for src_path in [s2_path, s3_path]:
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, chunksize=chunk_size):
                mask = chunk["country"] == ctry
                c_chunk = chunk[mask]
                if not c_chunk.empty:
                    batch_ids = c_chunk["entity_id"].tolist()
                    batch_names = c_chunk["business_name"].fillna("").tolist()
                    batch_addrs = c_chunk["business_address"].fillna("").tolist()
                    index.add_batch(batch_ids, batch_names, batch_addrs)
                    cand_count_loaded += len(batch_ids)

        index.finalize()
        t_idx = time.time() - t_idx_start
        print(f"CandidateIndex built with {cand_count_loaded:,} candidates in {t_idx:.1f}s. Peak RSS: {get_peak_rss_mb():.1f} MB")

        # Query candidates for each S1 entity
        print(f"Stage 1: Querying top-{candidate_cap} candidates for {num_s1:,} S1 entities...")
        t_query_start = time.time()
        cand_to_pairs: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
        top_scores: dict[str, float] = {}
        cand_counts: dict[str, int] = {}
        num_pairs_c = 0

        for row in s1_c.itertuples(index=False):
            s1_id = row.entity_id
            q_keys = extract_blocking_keys(row.business_name, row.business_address)
            top_cands = index.query(q_keys, candidate_cap=candidate_cap)
            if not top_cands:
                continue

            top_scores[s1_id] = top_cands[0][1]
            cand_counts[s1_id] = len(top_cands)
            num_pairs_c += len(top_cands)

            for rank, (cand_id, score) in enumerate(top_cands, 1):
                cand_to_pairs[cand_id].append((s1_id, score, rank))

        t_query = time.time() - t_query_start
        q_rate = num_s1 / max(t_query, 0.001)
        print(f"Querying completed in {t_query:.1f}s ({q_rate:.0f} entities/s). Total pairs: {num_pairs_c:,}.")

        # Free CandidateIndex
        del index
        gc.collect()
        print(f"Country {ctry}: {num_s1} entities, {num_pairs_c} candidate pairs, index freed. Peak RSS: {get_peak_rss_mb():.1f} MB")

        # --- STAGE 2: Feature Extraction ---
        print(f"Stage 2: Normalizing S1 records for {ctry}...")
        s1_lookup: dict[str, NormalizedRecord] = {}
        for row in s1_c.itertuples(index=False):
            s1_lookup[row.entity_id] = normalize_record(
                entity_id=row.entity_id,
                business_name=row.business_name,
                business_address=row.business_address,
                country=row.country,
            )

        needed_cand_ids = set(cand_to_pairs.keys())
        print(f"Unique candidates needed for {ctry}: {len(needed_cand_ids):,}.")

        print(f"Stage 2: Streaming through candidates to compute pairwise features...")
        t_feat_start = time.time()
        feature_buffer: list[dict[str, Any]] = []
        pairs_written_c = 0
        positives_c = 0
        total_computed = 0

        for src_path in [s2_path, s3_path]:
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, chunksize=chunk_size):
                mask = (chunk["country"] == ctry) & (chunk["entity_id"].isin(needed_cand_ids))
                matching_rows = chunk[mask]
                if matching_rows.empty:
                    continue

                for row in matching_rows.itertuples(index=False):
                    cand_id = row.entity_id
                    pairs_for_cand = cand_to_pairs.get(cand_id)
                    if not pairs_for_cand:
                        continue

                    cand_rec = normalize_record(
                        entity_id=cand_id,
                        business_name=row.business_name,
                        business_address=row.business_address,
                        country=row.country,
                    )
                    cand_source = 3 if cand_id.startswith("S3-") else 2

                    for s1_id, score, rank in pairs_for_cand:
                        s1_rec = s1_lookup.get(s1_id)
                        if s1_rec is None:
                            continue

                        feat = compute_pair_features(
                            s1_rec=s1_rec,
                            cand_rec=cand_rec,
                            blocking_score=score,
                            candidate_rank=rank,
                            candidate_count=cand_counts.get(s1_id, 1),
                            top_score=top_scores.get(s1_id, score),
                            cand_source=cand_source,
                        )
                        feat["source1_entity_id"] = s1_id
                        feat["candidate_entity_id"] = cand_id

                        if is_labeled:
                            lbl = 1 if (s1_id in gt_map and cand_id in gt_map[s1_id]) else 0
                            feat["label"] = np.int8(lbl)
                            if lbl == 1:
                                positives_c += 1

                        feature_buffer.append(feat)
                        total_computed += 1

                        if total_computed % 50000 == 0:
                            elapsed_feat = time.time() - t_feat_start
                            rate = total_computed / max(elapsed_feat, 0.001)
                            print(f"  Computed {total_computed:,} feature rows ({rate:.0f} pairs/s). Peak RSS: {get_peak_rss_mb():.1f} MB")

                        if len(feature_buffer) >= write_batch_size:
                            chunk_df = pd.DataFrame(feature_buffer)[expected_columns]
                            table = pa.Table.from_pandas(chunk_df, schema=arrow_schema, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(output_path, arrow_schema, compression="snappy")
                            writer.write_table(table)
                            pairs_written_c += len(chunk_df)
                            feature_buffer.clear()

                    # Free candidate pairs from memory
                    del cand_to_pairs[cand_id]
                    needed_cand_ids.discard(cand_id)

        # Flush remaining buffer
        if feature_buffer:
            chunk_df = pd.DataFrame(feature_buffer)[expected_columns]
            table = pa.Table.from_pandas(chunk_df, schema=arrow_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, arrow_schema, compression="snappy")
            writer.write_table(table)
            pairs_written_c += len(chunk_df)
            feature_buffer.clear()

        t_feat = time.time() - t_feat_start
        print(f"Country {ctry} features complete: {pairs_written_c:,} pairs written in {t_feat:.1f}s.")
        if is_labeled:
            pos_rate = (positives_c / pairs_written_c * 100.0) if pairs_written_c > 0 else 0.0
            print(f"  Positive pairs: {positives_c:,} ({pos_rate:.2f}%)")

        # Cleanup country structures
        del s1_lookup
        del cand_to_pairs
        del needed_cand_ids
        del top_scores
        del cand_counts
        del s1_c
        gc.collect()

        total_pairs_all += pairs_written_c
        total_positives_all += positives_c
        c_time = time.time() - t_c_start
        country_stats.append({
            "country": ctry,
            "entities": num_s1,
            "pairs": pairs_written_c,
            "positives": positives_c,
            "time_sec": c_time,
            "peak_rss_mb": get_peak_rss_mb(),
        })

    # Close ParquetWriter
    if writer is not None:
        writer.close()
    else:
        empty_df = pd.DataFrame(columns=expected_columns)
        table = pa.Table.from_pandas(empty_df, schema=arrow_schema, preserve_index=False)
        pq.write_table(table, output_path, compression="snappy")

    total_time = time.time() - t_start
    file_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)

    # --- STAGE 3: Final Summary ---
    print("\n" + "=" * 70)
    print(f"PIPELINE SUMMARY: split={split}, K={candidate_cap}")
    print("=" * 70)
    for stat in country_stats:
        pos_str = f", Positives: {stat['positives']:,} ({stat['positives']/max(stat['pairs'], 1)*100:.2f}%)" if is_labeled else ""
        print(f"Country {stat['country']}: {stat['entities']:,} entities, {stat['pairs']:,} pairs{pos_str}, Time: {stat['time_sec']:.1f}s, Peak RSS: {stat['peak_rss_mb']:.1f} MB")
    print("-" * 70)
    pos_all_str = f", Total positives: {total_positives_all:,} ({total_positives_all/max(total_pairs_all, 1)*100:.2f}%)" if is_labeled else ""
    print(f"Total pairs written: {total_pairs_all:,}{pos_all_str}")
    print(f"Output file: {output_path} ({file_size_mb:.2f} MB)")
    print(f"Total wall-clock time: {total_time:.1f}s ({total_time/60.0:.2f} minutes)")
    print(f"Peak RSS: {get_peak_rss_mb():.1f} MB")
    print("=" * 70)

    return output_path


def merge_parquets(file_list: list[str], output_path: str) -> None:
    """Concatenates multiple Parquet files preserving schema and row groups."""
    print(f"Merging {len(file_list)} Parquet files into {output_path}...")
    writer = None
    total_rows = 0
    for fpath in file_list:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File to merge not found: {fpath}")
        pf = pq.ParquetFile(fpath)
        print(f"  Reading {fpath} ({pf.metadata.num_rows:,} rows across {pf.metadata.num_row_groups} row groups)...")
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += len(table)
    if writer is not None:
        writer.close()
    meta = pq.read_metadata(output_path)
    file_size_mb = os.path.getsize(output_path) / (1024.0 * 1024.0)
    print(f"Successfully merged {meta.num_rows:,} total rows into {output_path} ({file_size_mb:.2f} MB).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-Scale Candidate Blocking and Feature Extraction")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train", help="Data split to process")
    parser.add_argument("--k", type=int, default=100, help="Candidate cap per entity (default: 100)")
    parser.add_argument("--num-entities", type=int, default=None, help="Number of S1 entities to sample (train split only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--countries", nargs="+", default=None, help="Specific countries to process (e.g. US)")
    parser.add_argument("--output", type=str, default=None, help="Custom output parquet path")
    parser.add_argument("--merge", nargs="+", default=None, help="Merge list of Parquet files into --output")

    args = parser.parse_args()

    if args.merge:
        if not args.output:
            parser.error("--output is required when using --merge")
        merge_parquets(args.merge, args.output)
        return

    run_pipeline(
        split=args.split,
        candidate_cap=args.k,
        num_entities=args.num_entities,
        seed=args.seed,
        selected_countries=args.countries,
        custom_output_path=args.output,
    )


if __name__ == "__main__":
    main()
