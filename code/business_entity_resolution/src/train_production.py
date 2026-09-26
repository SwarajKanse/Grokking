"""
Production Matching Model Training & Validation Pipeline.
Amazon ML Challenge 2026: Business Entity Resolution.

Trains a full-scale LightGBM pairwise classifier on 200K entities (K=100 candidates),
evaluates on held-out validation set (25K entities), sweeps decision threshold
from 0.10 to 0.95 in 0.01 increments, saves the model checkpoint and threshold,
and logs measured metrics to experiments/log.md.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure unbuffered stdout flushing for real-time progress logging
print = functools.partial(print, flush=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import FEATURE_COLUMN_NAMES
from train_matching import compute_entity_f05, evaluate_predictions


def train_production_model(
    train_parquet_path: str = "experiments/train_features_k100.parquet",
    val_parquet_path: str = "experiments/val_features_k100.parquet",
    val_split_path: str = "experiments/val_split.parquet",
    gt_tsv_path: str = "dataset/train/train_ground_truth.tsv",
    model_output_path: str = "experiments/lgbm_model_v1.txt",
    threshold_output_path: str = "experiments/threshold_v1.txt",
    log_path: str = "experiments/log.md",
) -> dict[str, Any]:
    """Trains production LightGBM model and performs fine-grained threshold optimization."""
    t_start = time.time()
    print("=" * 70)
    print("PRODUCTION MATCHING MODEL TRAINING (200K ENTITIES, K=100)")
    print("=" * 70)

    # 1. Load Training Data
    if not os.path.exists(train_parquet_path):
        raise FileNotFoundError(f"Training features {train_parquet_path} not found.")
    print(f"\n[1/6] Loading training features from {train_parquet_path}...")
    t_load_train = time.time()
    train_df = pd.read_parquet(train_parquet_path)
    print(f"Loaded {len(train_df):,} training candidate pairs in {time.time()-t_load_train:.1f}s.")

    # 2. Load Validation Data
    if not os.path.exists(val_parquet_path):
        raise FileNotFoundError(f"Validation features {val_parquet_path} not found.")
    print(f"\n[2/6] Loading validation features from {val_parquet_path}...")
    t_load_val = time.time()
    val_df = pd.read_parquet(val_parquet_path)
    print(f"Loaded {len(val_df):,} validation candidate pairs in {time.time()-t_load_val:.1f}s.")

    # 3. Prepare Feature Matrices
    feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in train_df.columns]
    print(f"Using {len(feature_cols)} features for model training.")

    X_train = train_df[feature_cols]
    y_train = train_df["label"].values.astype(np.int8)

    X_val = val_df[feature_cols]
    y_val = val_df["label"].values.astype(np.int8)

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    scale_pos_weight = float(num_neg / max(num_pos, 1))

    print(f"Training labels: {num_pos:,} positive pairs ({num_pos/len(y_train)*100:.2f}%), {num_neg:,} negative pairs.")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    print(f"Validation labels: {int(y_val.sum()):,} positive pairs ({int(y_val.sum())/len(y_val)*100:.2f}%), {len(y_val)-int(y_val.sum()):,} negative pairs.")

    # 4. Train Single Production LightGBM Model
    print("\n[3/6] Training LightGBM Production Model...")
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    print("Model hyperparameters:")
    for k, v in lgb_params.items():
        print(f"  {k}: {v}")

    model = lgb.LGBMClassifier(**lgb_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    t_train_start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=callbacks,
    )
    t_train = time.time() - t_train_start
    best_iter = getattr(model, "best_iteration_", model.n_estimators)
    print(f"\nModel training completed in {t_train:.1f}s ({t_train/60.0:.2f} min). Best iteration: {best_iter}.")

    # 5. Predict on Validation Set & Evaluate Discrimination
    print("\n[4/6] Evaluating on Held-Out Validation Set...")
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_probs))
    val_ap = float(average_precision_score(y_val, val_probs))

    print(f"Validation ROC-AUC: {val_auc:.5f}")
    print(f"Validation PR-AUC (Average Precision): {val_ap:.5f}")

    # Top 20 Feature Importances
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    print("\n=== TOP 20 MOST INFORMATIVE FEATURES (LightGBM split gain) ===")
    for rank_idx, r in fi_df.head(20).iterrows():
        print(f"  {rank_idx+1:2d}. {r['feature']:<25}: {r['importance']:.1f}")

    # 6. Fine-Grained Threshold Sweep (0.10 to 0.95, step 0.01)
    print("\n[5/6] Performing fine-grained threshold sweep for Macro F0.5 optimization...")
    print(f"Loading ground truth from {gt_tsv_path}...")
    gt_df = pd.read_csv(gt_tsv_path, sep="\t", dtype=str).fillna("")
    gt_map: dict[str, set[str]] = {}
    for row in gt_df.itertuples(index=False):
        s1 = row.source1_entity_id
        matched = row.matched_entity_ids
        gt_map[s1] = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
    del gt_df
    gc.collect()

    # Determine full set of validation entity IDs (to ensure singletons with 0 cands are included)
    if os.path.exists(val_split_path):
        val_split_df = pd.read_parquet(val_split_path)
        all_s1_ids = list(val_split_df["source1_entity_id"].unique())
        print(f"Evaluating across all {len(all_s1_ids):,} entities from {val_split_path}.")
    else:
        all_s1_ids = list(val_df["source1_entity_id"].unique())
        print(f"Evaluating across {len(all_s1_ids):,} unique entities found in validation features.")

    # Group validation candidates by source1_entity_id once
    val_df["pred_score"] = val_probs
    s1_candidates = val_df.groupby("source1_entity_id")[["candidate_entity_id", "pred_score"]].apply(
        lambda g: list(zip(g["candidate_entity_id"], g["pred_score"]))
    ).to_dict()

    thresholds = [round(x, 2) for x in np.arange(0.10, 0.96, 0.01)]
    best_thresh = 0.50
    best_metrics: dict[str, float] = {}
    best_f05 = -1.0
    sweep_records = []

    print("\n---------------------------------------------------------------------------------------------------------")
    print(f"{'Thresh':<8} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<14} | {'Non-Single F0.5':<15}")
    print("---------------------------------------------------------------------------------------------------------")

    for th in thresholds:
        pred_map: dict[str, set[str]] = {}
        for s1_id in all_s1_ids:
            cands = s1_candidates.get(s1_id, [])
            passing = {c_id for c_id, sc in cands if sc >= th}
            pred_map[s1_id] = passing

        metrics = evaluate_predictions(gt_map, pred_map, all_s1_ids)
        f05 = metrics["macro_f05"]
        sweep_records.append((th, metrics))

        print(f"  {th:.2f}   |   {f05:.4f}   |   {metrics['macro_precision']:.4f}   |   {metrics['macro_recall']:.4f}   |    {metrics['singleton_accuracy']:.4f}     |     {metrics['non_singleton_f05']:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            best_metrics = metrics

    print("---------------------------------------------------------------------------------------------------------")
    print(f"\n=======================================================")
    print(f"OPTIMAL THRESHOLD:       {best_thresh:.2f}")
    print(f"Best Macro F0.5:         {best_metrics['macro_f05']:.4f}")
    print(f"Macro Precision:         {best_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:            {best_metrics['macro_recall']:.4f}")
    print(f"Singleton Accuracy:      {best_metrics['singleton_accuracy']:.4f}")
    print(f"Non-Singleton F0.5:      {best_metrics['non_singleton_f05']:.4f}")
    print(f"=======================================================")

    # 7. Save Model and Threshold
    print("\n[6/6] Saving Artifacts and Logging Results...")
    os.makedirs(os.path.dirname(model_output_path) or ".", exist_ok=True)
    model.booster_.save_model(model_output_path)
    print(f"Saved LightGBM model booster to {model_output_path}")

    with open(threshold_output_path, "w", encoding="utf-8") as f:
        f.write(f"{best_thresh:.2f}\n")
    print(f"Saved optimal decision threshold ({best_thresh:.2f}) to {threshold_output_path}")

    # 8. Update experiments/log.md
    if os.path.exists(log_path):
        update_log_file(
            log_path=log_path,
            best_thresh=best_thresh,
            best_metrics=best_metrics,
            fi_df=fi_df,
            val_auc=val_auc,
            val_ap=val_ap,
            num_train_pairs=len(train_df),
            num_val_pairs=len(val_df),
            best_iteration=best_iter,
            scale_pos_weight=scale_pos_weight,
            sweep_records=sweep_records,
        )

    total_time = time.time() - t_start
    print(f"\nTotal Pipeline Execution Time: {total_time:.1f}s ({total_time/60.0:.2f} min)")

    return {
        "best_threshold": best_thresh,
        "best_metrics": best_metrics,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "best_iteration": best_iter,
        "feature_importances": fi_df,
        "sweep_records": sweep_records,
    }


def update_log_file(
    log_path: str,
    best_thresh: float,
    best_metrics: dict[str, float],
    fi_df: pd.DataFrame,
    val_auc: float,
    val_ap: float,
    num_train_pairs: int,
    num_val_pairs: int,
    best_iteration: int,
    scale_pos_weight: float,
    sweep_records: list[tuple[float, dict[str, float]]],
) -> None:
    """Updates experiments/log.md with the production model run results."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    top_feats_str = ", ".join([f"{r['feature']} ({r['importance']:.0f})" for _, r in fi_df.head(4).iterrows()])

    # Table row for the main summary table
    table_row = (
        f"| {today} | Production (K=100) | LightGBM production (200K entities, {num_train_pairs:,} train pairs, K=100) | "
        f"~0.9124 (@K=100) | {best_metrics['macro_precision']:.4f} | {best_metrics['macro_recall']:.4f} | "
        f"{best_metrics['macro_f05']:.4f} | {best_metrics['singleton_accuracy']:.4f} | "
        f"ROC-AUC={val_auc:.4f}, PR-AUC={val_ap:.4f}, optimal threshold={best_thresh:.2f}, scale_pos_weight={scale_pos_weight:.2f}, top features: {top_feats_str} |\n"
    )

    # Detailed section
    detailed_lines = [
        f"\n---\n\n## Production Model v1 (200K Entities, K=100 Candidates)\n\n",
        f"### 1. Training Setup & Architecture\n",
        f"- **Model:** LightGBM Binary Classifier (`LGBMClassifier`, `boosting_type='gbdt'`, `n_estimators=2000`, `learning_rate=0.03`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.7`, `colsample_bytree=0.7`, `scale_pos_weight={scale_pos_weight:.4f}`).\n",
        f"- **Training Dataset:** 200,000 Source 1 training entities, {num_train_pairs:,} candidate pairs (`experiments/train_features_k100.parquet`).\n",
        f"- **Validation Dataset:** 25,000 held-out stratified validation entities, {num_val_pairs:,} candidate pairs (`experiments/val_features_k100.parquet`).\n",
        f"- **Best Iteration:** {best_iteration} (with early stopping patience=50).\n",
        f"- **Model Checkpoint:** `experiments/lgbm_model_v1.txt`.\n",
        f"- **Optimal Decision Threshold File:** `experiments/threshold_v1.txt`.\n\n",
        f"### 2. Discrimination Metrics on Held-Out Validation Set\n\n",
        f"| Metric | Value |\n",
        f"|---|---|\n",
        f"| ROC-AUC | {val_auc:.5f} |\n",
        f"| PR-AUC (Average Precision) | {val_ap:.5f} |\n\n",
        f"### 3. Top 20 Most Informative Features (LightGBM Split Gain)\n\n",
        f"| Rank | Feature | Importance (Split Gain) |\n",
        f"|---|---|---|\n",
    ]

    for rank_idx, r in fi_df.head(20).iterrows():
        detailed_lines.append(f"| {rank_idx+1} | `{r['feature']}` | {r['importance']:.1f} |\n")

    detailed_lines.append(f"\n### 4. Fine-Grained Decision Threshold Sweep (0.10 to 0.95)\n\n")
    detailed_lines.append(f"| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy | Non-Singleton F0.5 |\n")
    detailed_lines.append(f"|---|---|---|---|---|---|\n")

    for th, m in sweep_records:
        bold = "**" if th == best_thresh else ""
        detailed_lines.append(
            f"| {bold}{th:.2f}{bold} | {bold}{m['macro_f05']:.4f}{bold} | {m['macro_precision']:.4f} | "
            f"{m['macro_recall']:.4f} | {m['singleton_accuracy']:.4f} | {m['non_singleton_f05']:.4f} |\n"
        )

    detailed_lines.append(f"\n### 5. Optimal Threshold Summary\n")
    detailed_lines.append(f"- **Optimal Decision Threshold:** `{best_thresh:.2f}`\n")
    detailed_lines.append(f"- **Best Macro F0.5:** `{best_metrics['macro_f05']:.4f}`\n")
    detailed_lines.append(f"- **Macro Precision:** `{best_metrics['macro_precision']:.4f}`\n")
    detailed_lines.append(f"- **Macro Recall:** `{best_metrics['macro_recall']:.4f}`\n")
    detailed_lines.append(f"- **Singleton Accuracy:** `{best_metrics['singleton_accuracy']:.4f}`\n")
    detailed_lines.append(f"- **Non-Singleton F0.5:** `{best_metrics['non_singleton_f05']:.4f}`\n")

    detailed_str = "".join(detailed_lines)

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Insert table row after Phase 4 row
        if "| Phase 4 |" in content:
            parts = content.split("| Phase 4 |")
            first_part = parts[0]
            rest = "| Phase 4 |" + parts[1]
            lines = rest.split("\n", 1)
            new_content = first_part + lines[0] + "\n" + table_row + lines[1] + detailed_str
        else:
            new_content = content + "\n" + table_row + detailed_str

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {log_path} with production model experiment results.")
    except Exception as e:
        print(f"Warning: Failed to update {log_path}: {e}")


def main() -> None:
    train_production_model()


if __name__ == "__main__":
    main()
