"""
Production End-to-End Test Inference Pipeline (run_inference.py).
Amazon ML Challenge 2026: Business Entity Resolution.

Executes streaming candidate generation (candidate_cap=10000, 99.92% recall),
dynamic pairwise feature computation with rank/count clipping, and ensemble
inference across all 1,732,544 test entities in US, India, and France.

Produces:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv

Validates outputs using utils/validate_submission.py before reporting completion.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import functools
import gc
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

print = functools.partial(print, flush=True)

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from blocking import CandidateIndex, extract_blocking_keys
from features import FEATURE_COLUMN_NAMES, compute_pair_features
from normalize import NormalizedRecord, normalize_record


def get_peak_rss_mb() -> float:
    """Returns peak memory usage of process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_inference(
    model_dir: str = "experiments/",
    test_dir: str = "dataset/test/",
    output_dir: str = "output/",
    candidate_cap: int = 10000,
    chunk_size: int = 500000,
    predict_batch_size: int = 50000,
) -> None:
    t_total_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PRODUCTION TEST INFERENCE PIPELINE (NO-CAP K=10000, FULL TEST SET)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Load Ensemble Configuration & Models
    # -------------------------------------------------------------------------
    config_path = os.path.join(model_dir, "ensemble_config.txt")
    threshold_path = os.path.join(model_dir, "threshold_v2.txt")

    if not os.path.exists(config_path):
        # Fallback to threshold_v1 if v2 does not exist
        if os.path.exists(os.path.join(model_dir, "threshold_v1.txt")):
            threshold_path = os.path.join(model_dir, "threshold_v1.txt")
        config_line = "lgb_only||0.97"
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config_line = f.read().strip()

    parts = config_line.split("|")
    method = parts[0] if len(parts) > 0 else "lgb_only"
    weights_str = parts[1] if len(parts) > 1 else ""
    default_thresh = float(parts[2]) if len(parts) > 2 and parts[2] else 0.97

    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = float(f.read().strip())
    else:
        threshold = default_thresh

    print(f"\n[Step 1] Loading models with configuration: method='{method}', threshold={threshold:.3f}")

    lgb_booster = None
    xgb_booster = None
    cat_model = None
    weights = [1.0, 0.0, 0.0]

    if method == "weighted":
        weights = [float(x) for x in weights_str.split(",")]
        print(f"Ensemble weights: LGB={weights[0]:.2f}, XGB={weights[1]:.2f}, CatBoost={weights[2]:.2f}")

        lgb_file = os.path.join(model_dir, "lgbm_model_v2.txt")
        if not os.path.exists(lgb_file):
            lgb_file = os.path.join(model_dir, "lgbm_model_v1.txt")
        lgb_booster = lgb.Booster(model_file=lgb_file)
        print(f"Loaded LightGBM model from {lgb_file}")

        xgb_file = os.path.join(model_dir, "xgb_model_v2.json")
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_file)
        print(f"Loaded XGBoost model from {xgb_file}")

        cat_file = os.path.join(model_dir, "catboost_model_v2.cbm")
        cat_model = CatBoostClassifier()
        cat_model.load_model(cat_file)
        print(f"Loaded CatBoost model from {cat_file}")

    elif method == "xgb_only":
        xgb_file = os.path.join(model_dir, "xgb_model_v2.json")
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_file)
        weights = [0.0, 1.0, 0.0]
        print(f"Loaded XGBoost model from {xgb_file}")

    elif method == "cat_only":
        cat_file = os.path.join(model_dir, "catboost_model_v2.cbm")
        cat_model = CatBoostClassifier()
        cat_model.load_model(cat_file)
        weights = [0.0, 0.0, 1.0]
        print(f"Loaded CatBoost model from {cat_file}")

    else:  # lgb_only
        lgb_file = os.path.join(model_dir, "lgbm_model_v2.txt")
        if not os.path.exists(lgb_file):
            lgb_file = os.path.join(model_dir, "lgbm_model_v1.txt")
        lgb_booster = lgb.Booster(model_file=lgb_file)
        weights = [1.0, 0.0, 0.0]
        print(f"Loaded LightGBM model from {lgb_file}")

    # -------------------------------------------------------------------------
    # Step 2: Read All Test Source 1 Entities
    # -------------------------------------------------------------------------
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    print(f"\n[Step 2] Reading test S1 entities from {s1_path}...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    all_test_s1_ids = sorted(s1_df["entity_id"].dropna().unique().tolist())
    total_test_entities = len(all_test_s1_ids)
    print(f"Found {total_test_entities:,} unique Source 1 entities in test set.")
    assert total_test_entities == 1732544, f"Expected 1,732,544 test entities, found {total_test_entities}"

    country_to_s1 = defaultdict(list)
    for row in s1_df.itertuples(index=False):
        c = row.country if pd.notna(row.country) and row.country else "UNKNOWN"
        country_to_s1[c].append(row)

    print(f"Country breakdown in test set: {dict((k, len(v)) for k, v in country_to_s1.items())}")
    del s1_df
    gc.collect()

    # Data structures for results
    matching_results: dict[str, set[str]] = defaultdict(set)
    # Temporary directory for streaming candidate pairs to disk per country
    tmp_cands_dir = os.path.join(output_dir, "tmp_candidates")
    os.makedirs(tmp_cands_dir, exist_ok=True)

    country_summary_stats = []

    # -------------------------------------------------------------------------
    # Step 3: Process Each Country Partition
    # -------------------------------------------------------------------------
    for ctry, s1_rows in country_to_s1.items():
        t_ctry_start = time.time()
        num_s1_c = len(s1_rows)
        print("\n" + "=" * 70)
        print(f"Processing Country: {ctry} ({num_s1_c:,} S1 entities)")
        print("=" * 70)

        # --- Stage 3a: Blocking CandidateIndex ---
        t_idx_start = time.time()
        cand_ids: list[str] = []
        cand_names: list[str] = []
        cand_addrs: list[str] = []

        print(f"Building CandidateIndex for {ctry} from S2 and S3...")
        for src_path in [s2_path, s3_path]:
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, chunksize=chunk_size):
                mask = chunk["country"] == ctry
                c_chunk = chunk[mask]
                if c_chunk.empty:
                    continue
                cand_ids.extend(c_chunk["entity_id"].tolist())
                cand_names.extend(c_chunk["business_name"].fillna("").tolist())
                cand_addrs.extend(c_chunk["business_address"].fillna("").tolist())

        t_idx_load = time.time() - t_idx_start
        print(f"Loaded {len(cand_ids):,} candidate records for {ctry} in {t_idx_load:.1f}s.")

        index = CandidateIndex(cand_ids, cand_names, cand_addrs)
        del cand_names, cand_addrs
        gc.collect()

        t_idx_build = time.time() - t_idx_start
        print(f"CandidateIndex built for {ctry} in {t_idx_build:.1f}s. Peak RSS: {get_peak_rss_mb():.1f} MB")

        # --- Stage 3b: Querying Top Candidates ---
        print(f"Querying top-{candidate_cap} candidates for {num_s1_c:,} entities...")
        t_query_start = time.time()
        cand_to_pairs = defaultdict(list)
        top_scores: dict[str, float] = {}
        cand_counts: dict[str, int] = {}
        max_cands_single = 0
        total_pairs_c = 0

        # Stream candidate_pairs for this country to temporary file
        cands_tmp_file = os.path.join(tmp_cands_dir, f"cands_{ctry}.tsv")
        with open(cands_tmp_file, "w", encoding="utf-8") as f_cands:
            for s1_row in s1_rows:
                s1_id = s1_row.entity_id
                raw_name = s1_row.business_name if pd.notna(s1_row.business_name) else ""
                raw_addr = s1_row.business_address if pd.notna(s1_row.business_address) else ""

                q_keys = extract_blocking_keys(raw_name, raw_addr)
                top_cands = index.query(q_keys, candidate_cap=candidate_cap)
                if not top_cands:
                    f_cands.write(f"{s1_id}\t\n")
                    continue

                c_list = [c_id for c_id, _ in top_cands]
                f_cands.write(f"{s1_id}\t{','.join(c_list)}\n")

                num_c = len(top_cands)
                if num_c > max_cands_single:
                    max_cands_single = num_c
                total_pairs_c += num_c

                top_scores[s1_id] = top_cands[0][1]
                cand_counts[s1_id] = num_c

                for rank, (cand_id, score) in enumerate(top_cands, 1):
                    cand_to_pairs[cand_id].append((s1_id, score, rank))

        t_query = time.time() - t_query_start
        print(f"Querying complete in {t_query:.1f}s ({num_s1_c/max(t_query, 0.001):.0f} q/s).")
        print(f"Country {ctry}: {total_pairs_c:,} pairs (avg {total_pairs_c/num_s1_c:.1f}, max {max_cands_single}).")

        # Free CandidateIndex immediately
        del index
        gc.collect()
        t_blocking_total = time.time() - t_ctry_start
        print(f"Index freed for {ctry}. Peak RSS: {get_peak_rss_mb():.1f} MB")

        # --- Stage 3c: S1 Normalization ---
        print(f"Normalizing S1 records for {ctry}...")
        s1_lookup: dict[str, NormalizedRecord] = {}
        for s1_row in s1_rows:
            s1_lookup[s1_row.entity_id] = normalize_record(
                entity_id=s1_row.entity_id,
                business_name=s1_row.business_name,
                business_address=s1_row.business_address,
                country=s1_row.country,
            )

        needed_cand_ids = set(cand_to_pairs.keys())
        print(f"Unique candidates needed for {ctry}: {len(needed_cand_ids):,}.")

        # --- Stage 3d: Feature Extraction & Prediction Buffer ---
        print(f"Streaming through candidate sources for feature computation & prediction...")
        t_infer_start = time.time()
        feat_rows: list[dict[str, Any]] = []
        pair_meta: list[tuple[str, str]] = []  # (s1_id, cand_id)
        pairs_predicted_c = 0

        feature_cols = [c for c in FEATURE_COLUMN_NAMES]

        def flush_and_predict():
            nonlocal feat_rows, pair_meta, pairs_predicted_c
            if not feat_rows:
                return

            df_feat = pd.DataFrame(feat_rows)
            X_mat = df_feat[feature_cols].values.astype(np.float32)

            if method == "weighted":
                p_lgb = lgb_booster.predict(X_mat)
                p_xgb = xgb_booster.predict(xgb.DMatrix(X_mat, feature_names=feature_cols))
                p_cat = cat_model.predict_proba(X_mat)[:, 1]
                prob = weights[0] * p_lgb + weights[1] * p_xgb + weights[2] * p_cat
            elif method == "xgb_only":
                prob = xgb_booster.predict(xgb.DMatrix(X_mat, feature_names=feature_cols))
            elif method == "cat_only":
                prob = cat_model.predict_proba(X_mat)[:, 1]
            else:
                prob = lgb_booster.predict(X_mat)

            for i in range(len(prob)):
                if prob[i] >= threshold:
                    s1_i, cand_i = pair_meta[i]
                    matching_results[s1_i].add(cand_i)

            pairs_predicted_c += len(prob)
            if pairs_predicted_c % 200000 < predict_batch_size:
                print(f"  Predicted {pairs_predicted_c:,}/{total_pairs_c:,} pairs ({pairs_predicted_c/(time.time()-t_infer_start):.0f} p/s). RSS: {get_peak_rss_mb():.1f} MB")

            feat_rows.clear()
            pair_meta.clear()

        for src_path in [s2_path, s3_path]:
            for chunk in pd.read_csv(src_path, sep="\t", dtype=str, chunksize=chunk_size):
                mask = (chunk["country"] == ctry) & (chunk["entity_id"].isin(needed_cand_ids))
                c_chunk = chunk[mask]
                if c_chunk.empty:
                    continue

                for row in c_chunk.itertuples(index=False):
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
                            candidate_rank=min(rank, 100),
                            candidate_count=min(cand_counts.get(s1_id, 1), 200),
                            top_score=top_scores.get(s1_id, score),
                            cand_source=cand_source,
                        )
                        feat_rows.append(feat)
                        pair_meta.append((s1_id, cand_id))

                        if len(feat_rows) >= predict_batch_size:
                            flush_and_predict()

        flush_and_predict()

        # Free country resources
        del s1_lookup, cand_to_pairs, needed_cand_ids, top_scores, cand_counts
        gc.collect()

        t_infer_total = time.time() - t_infer_start
        matched_entities_c = sum(1 for r in s1_rows if len(matching_results[r.entity_id]) > 0)
        singletons_c = num_s1_c - matched_entities_c
        singleton_pct = (singletons_c / num_s1_c) * 100.0

        country_summary_stats.append({
            "country": ctry,
            "entities": num_s1_c,
            "pairs": total_pairs_c,
            "avg_cands": total_pairs_c / max(num_s1_c, 1),
            "max_cands": max_cands_single,
            "matched_entities": matched_entities_c,
            "singletons": singletons_c,
            "singleton_pct": singleton_pct,
            "blocking_time": t_blocking_total,
            "infer_time": t_infer_total,
        })

        print(f"Country {ctry} Done: {matched_entities_c:,} matched entities, {singletons_c:,} singletons ({singleton_pct:.2f}%), Blocking: {t_blocking_total:.1f}s, Inference: {t_infer_total:.1f}s.")

    # -------------------------------------------------------------------------
    # Step 4: Write Final Output Files
    # -------------------------------------------------------------------------
    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print("\n" + "=" * 70)
    print("[Step 4] Writing Submission Output Files...")
    print("=" * 70)

    # Pre-write verification
    print("Running pre-write sanity assertions...")
    assert len(all_test_s1_ids) == 1732544, "Wrong entity count in all_test_s1_ids!"
    for s1_id in all_test_s1_ids[:1000]:
        matched = matching_results.get(s1_id, set())
        for m in matched:
            assert not m.startswith("S1-"), f"Found forbidden S1- prefix in matches: {m}"
    print("Pre-write checks passed.")

    # Write matching_results.tsv
    print(f"Writing {matching_tsv_path}...")
    with open(matching_tsv_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_test_s1_ids:
            matched = matching_results.get(s1_id, set())
            if matched:
                sorted_matches = sorted(list(matched))
                f_out.write(f"{s1_id}\t{','.join(sorted_matches)}\n")
            else:
                f_out.write(f"{s1_id}\t\n")

    # Assemble candidate_pairs.tsv from country temporary files
    print(f"Assembling {candidate_tsv_path}...")
    # Load all country candidate files into memory or stream in entity order
    s1_to_cand_line: dict[str, str] = {}
    for ctry in country_to_s1.keys():
        cands_tmp_file = os.path.join(tmp_cands_dir, f"cands_{ctry}.tsv")
        if os.path.exists(cands_tmp_file):
            with open(cands_tmp_file, "r", encoding="utf-8") as f_c:
                for line in f_c:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        s1_to_cand_line[parts[0]] = parts[1]
            os.remove(cands_tmp_file)

    with open(candidate_tsv_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_test_s1_ids:
            cand_str = s1_to_cand_line.get(s1_id, "")
            f_out.write(f"{s1_id}\t{cand_str}\n")

    del s1_to_cand_line
    if os.path.exists(tmp_cands_dir):
        try:
            os.rmdir(tmp_cands_dir)
        except OSError:
            pass
    gc.collect()

    # -------------------------------------------------------------------------
    # Step 5: Run Submission Validator
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[Step 5] Running Official Submission Validator...")
    print("=" * 70)
    validator_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", matching_tsv_path,
        "--candidate", candidate_tsv_path,
        "--test-dir", test_dir,
    ]
    val_proc = subprocess.run(validator_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print("Validator stderr:")
        print(val_proc.stderr)

    if val_proc.returncode != 0:
        raise RuntimeError(f"VALIDATOR FAILED with return code {val_proc.returncode}!")
    print("VALIDATOR PASSED: Files are 100% compliant with competition scorer.")

    # -------------------------------------------------------------------------
    # Step 6: Print Comprehensive Summary
    # -------------------------------------------------------------------------
    total_time = time.time() - t_total_start
    matching_size_mb = os.path.getsize(matching_tsv_path) / (1024.0 * 1024.0)
    candidate_size_mb = os.path.getsize(candidate_tsv_path) / (1024.0 * 1024.0)
    total_matched_entities = sum(s["matched_entities"] for s in country_summary_stats)
    total_singletons = sum(s["singletons"] for s in country_summary_stats)
    total_pairs_all = sum(s["pairs"] for s in country_summary_stats)
    overall_singleton_pct = (total_singletons / total_test_entities) * 100.0

    print("\n" + "=" * 80)
    print("TEST INFERENCE SUMMARY")
    print("=" * 80)
    print(f"{'Country':<10} | {'S1 Entities':<12} | {'Pairs':<12} | {'Avg/S1':<8} | {'Max':<6} | {'Matched':<10} | {'Singletons':<10} | {'Sing %':<8} | {'Blocking(s)':<11} | {'Infer(s)':<9}")
    print("-" * 105)
    for s in country_summary_stats:
        print(f"{s['country']:<10} | {s['entities']:<12,} | {s['pairs']:<12,} | {s['avg_cands']:<8.1f} | {s['max_cands']:<6} | {s['matched_entities']:<10,} | {s['singletons']:<10,} | {s['singleton_pct']:<7.2f}% | {s['blocking_time']:<11.1f} | {s['infer_time']:<9.1f}")
    print("-" * 105)
    print(f"Total Test Entities:             {total_test_entities:,}")
    print(f"Total Candidate Pairs Generated: {total_pairs_all:,} (average {total_pairs_all/total_test_entities:.1f}/entity)")
    print(f"Entities with Predicted Matches: {total_matched_entities:,}")
    print(f"Total Predicted Singletons:      {total_singletons:,} ({overall_singleton_pct:.2f}% vs 5.58% train baseline)")
    print(f"Validator Result:                PASS")
    print(f"Output File matching_results:    {matching_tsv_path} ({matching_size_mb:.2f} MB)")
    print(f"Output File candidate_pairs:     {candidate_tsv_path} ({candidate_size_mb:.2f} MB)")
    print(f"Total End-to-End Runtime:        {total_time:.1f}s ({total_time/3600.0:.2f} hours)")
    print(f"Peak RSS Footprint:              {get_peak_rss_mb():.1f} MB")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Production Test Inference Pipeline")
    parser.add_argument("--model-dir", default="experiments/", help="Model directory")
    parser.add_argument("--test-dir", default="dataset/test/", help="Test data directory")
    parser.add_argument("--output-dir", default="output/", help="Output directory")
    parser.add_argument("--candidate-cap", type=int, default=10000, help="Candidate cap per entity (default 10000)")
    args = parser.parse_args()

    run_inference(
        model_dir=args.model_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        candidate_cap=args.candidate_cap,
    )
