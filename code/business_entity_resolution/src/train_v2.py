"""
Production 3-Model Ensemble Training & Optimization Pipeline (train_v2.py).
Amazon ML Challenge 2026: Business Entity Resolution.

Trains three complementary gradient boosted models on 20M candidate pairs:
  1. LightGBM (aggressive: 8000 trees, 255 leaves, lr=0.015)
  2. XGBoost (hist: 5000 trees, depth=9, max_bin=512, lr=0.02)
  3. CatBoost (5000 trees, depth=9, auto_class_weights='Balanced', lr=0.02)

Evaluates on held-out no-cap validation features (val_features_nocap.parquet),
sweeps decision thresholds for individual models and ensemble strategies:
  A) Equal average (1/3 each)
  B) Weighted (0.40, 0.35, 0.25)
  C) Weighted (0.50, 0.25, 0.25)
  D) Optimal grid search (weights summing to 1.0)

Exports best checkpoints, optimal configuration, and logs all measured metrics.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import gc
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

print = functools.partial(print, flush=True)

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from features import FEATURE_COLUMN_NAMES
from train_matching import evaluate_predictions


def get_peak_rss_mb() -> float:
    """Returns current process peak RSS in Megabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def sweep_thresholds_for_probs(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    gt_map: dict[str, set[str]],
    all_s1_ids: list[str],
    threshold_list: list[float],
    model_name: str = "Model",
) -> tuple[float, dict[str, float]]:
    """Evaluates metrics across threshold_list and returns (best_threshold, best_metrics)."""
    t0 = time.time()
    val_df_local = pd.DataFrame({
        "source1_entity_id": val_df["source1_entity_id"],
        "candidate_entity_id": val_df["candidate_entity_id"],
        "pred_score": probs,
    })

    s1_candidates = val_df_local.groupby("source1_entity_id")[["candidate_entity_id", "pred_score"]].apply(
        lambda g: list(zip(g["candidate_entity_id"], g["pred_score"]))
    ).to_dict()

    best_thresh = 0.50
    best_f05 = -1.0
    best_metrics: dict[str, float] = {}

    for th in threshold_list:
        pred_map: dict[str, set[str]] = {}
        for s1_id in all_s1_ids:
            cands = s1_candidates.get(s1_id, [])
            pred_map[s1_id] = {c_id for c_id, sc in cands if sc >= th}

        metrics = evaluate_predictions(gt_map, pred_map, all_s1_ids)
        f05 = metrics["macro_f05"]
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = th
            best_metrics = metrics

    dt = time.time() - t0
    print(f"  [{model_name}] Optimal Threshold: {best_thresh:.3f} | Macro F0.5: {best_f05:.4f} | Prec: {best_metrics['macro_precision']:.4f} | Recall: {best_metrics['macro_recall']:.4f} | Singleton Acc: {best_metrics['singleton_accuracy']:.4f} ({dt:.1f}s)")
    return best_thresh, best_metrics


def train_ensemble_v2(
    train_parquet_path: str = "experiments/train_features_k100.parquet",
    val_parquet_path: str = "experiments/val_features_nocap.parquet",
    val_split_path: str = "experiments/val_split.parquet",
    gt_tsv_path: str = "dataset/train/train_ground_truth.tsv",
    output_dir: str = "experiments/",
    log_path: str = "experiments/log.md",
) -> dict[str, Any]:
    """Trains 3 models, evaluates ensemble combinations, and saves all artifacts."""
    t_pipeline_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("PHASE C: 3-MODEL ENSEMBLE PRODUCTION TRAINING (LGBM + XGB + CATBOOST)")
    print("=" * 70)

    # 1. Load Training Data
    if not os.path.exists(train_parquet_path):
        raise FileNotFoundError(f"Training features {train_parquet_path} not found.")
    print(f"\n[1/7] Loading training features from {train_parquet_path}...")
    t_load = time.time()
    train_df = pd.read_parquet(train_parquet_path)
    print(f"Loaded {len(train_df):,} training candidate pairs in {time.time()-t_load:.1f}s.")

    # 2. Load Validation Data (No-Cap)
    if not os.path.exists(val_parquet_path):
        raise FileNotFoundError(f"Validation features {val_parquet_path} not found.")
    print(f"\n[2/7] Loading validation features from {val_parquet_path}...")
    t_load = time.time()
    val_df = pd.read_parquet(val_parquet_path)
    print(f"Loaded {len(val_df):,} validation candidate pairs in {time.time()-t_load:.1f}s.")

    # 3. Clip candidate_rank and candidate_count on val_df
    print("\n[3/7] Aligning feature distributions (clipping rank <= 100, count <= 200)...")
    val_df["candidate_rank"] = val_df["candidate_rank"].clip(upper=100)
    val_df["candidate_count"] = val_df["candidate_count"].clip(upper=200)

    feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in train_df.columns]
    print(f"Feature count: {len(feature_cols)} features.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df["label"].values.astype(np.int8)

    X_val = val_df[feature_cols].copy()
    y_val = val_df["label"].values.astype(np.int8)

    del train_df
    gc.collect()

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    scale_pos_weight = float(num_neg / max(num_pos, 1))

    print(f"Train set: {len(X_train):,} pairs ({num_pos:,} pos, {num_neg:,} neg, scale_pos_weight={scale_pos_weight:.4f})")
    print(f"Val set:   {len(X_val):,} pairs ({int(y_val.sum()):,} pos, {len(y_val)-int(y_val.sum()):,} neg)")
    print(f"Peak RSS after loading: {get_peak_rss_mb():.1f} MB")

    # Load Ground Truth and Validation S1 IDs
    print("\nLoading ground truth for threshold optimization...")
    gt_df = pd.read_csv(gt_tsv_path, sep="\t")
    gt_map: dict[str, set[str]] = {}
    for row in gt_df.itertuples(index=False):
        matched = str(row.matched_entity_ids) if pd.notna(row.matched_entity_ids) else ""
        gt_map[row.source1_entity_id] = set(matched.split(",")) if matched and matched != "nan" else set()

    if os.path.exists(val_split_path):
        val_split_df = pd.read_parquet(val_split_path)
        all_s1_ids = list(val_split_df["source1_entity_id"].unique())
    else:
        all_s1_ids = list(val_df["source1_entity_id"].unique())
    print(f"Validation entity pool: {len(all_s1_ids):,} entities.")

    # Threshold List: 0.10 to 0.90 (0.02), 0.90 to 0.96 (0.01), 0.96 to 0.995 (0.005)
    thresholds_coarse = [round(x, 2) for x in np.arange(0.10, 0.90, 0.02)]
    thresholds_mid = [round(x, 2) for x in np.arange(0.90, 0.96, 0.01)]
    thresholds_fine = [round(x, 3) for x in np.arange(0.96, 0.996, 0.005)]
    threshold_list = sorted(list(set(thresholds_coarse + thresholds_mid + thresholds_fine)))

    # Dictionary to collect results for comparison table
    results_table: list[dict[str, Any]] = []

    # =========================================================================
    # MODEL A: LightGBM (Aggressive)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4/7] Training Model A: LightGBM (Aggressive: 8000 trees, num_leaves=255)...")
    print("=" * 70)
    t_lgb = time.time()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 8000,
        "learning_rate": 0.015,
        "num_leaves": 255,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_jobs": -1,
        "max_bin": 511,
        "verbose": -1,
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False), lgb.log_evaluation(period=200)],
    )
    best_iter_lgb = lgb_model.best_iteration_ or lgb_params["n_estimators"]
    print(f"LightGBM trained in {time.time()-t_lgb:.1f}s. Best iteration: {best_iter_lgb}")

    lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_prob)
    lgb_ap = average_precision_score(y_val, lgb_prob)
    print(f"LightGBM Validation ROC-AUC: {lgb_auc:.5f} | PR-AUC: {lgb_ap:.5f}")

    lgb_model_path = os.path.join(output_dir, "lgbm_model_v2.txt")
    lgb_model.booster_.save_model(lgb_model_path)
    print(f"Saved LightGBM model to {lgb_model_path}")

    del lgb_model
    gc.collect()
    print(f"Peak RSS after LightGBM: {get_peak_rss_mb():.1f} MB")

    th_lgb, m_lgb = sweep_thresholds_for_probs(val_df, lgb_prob, gt_map, all_s1_ids, threshold_list, "LightGBM v2")
    results_table.append({
        "Model": "LightGBM v2",
        "ROC-AUC": lgb_auc,
        "PR-AUC": lgb_ap,
        "Threshold": th_lgb,
        "F0.5": m_lgb["macro_f05"],
        "Precision": m_lgb["macro_precision"],
        "Recall": m_lgb["macro_recall"],
        "Singleton Acc": m_lgb["singleton_accuracy"],
    })

    # =========================================================================
    # MODEL B: XGBoost (tree_method='hist')
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5/7] Training Model B: XGBoost (Hist: 5000 trees, depth=9)...")
    print("=" * 70)
    t_xgb = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=5000,
        learning_rate=0.02,
        max_depth=9,
        min_child_weight=50,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=200,
        tree_method="hist",
        max_bin=512,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )
    best_iter_xgb = xgb_model.best_iteration
    print(f"XGBoost trained in {time.time()-t_xgb:.1f}s. Best iteration: {best_iter_xgb}")

    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_prob)
    xgb_ap = average_precision_score(y_val, xgb_prob)
    print(f"XGBoost Validation ROC-AUC: {xgb_auc:.5f} | PR-AUC: {xgb_ap:.5f}")

    xgb_model_path = os.path.join(output_dir, "xgb_model_v2.json")
    xgb_model.save_model(xgb_model_path)
    print(f"Saved XGBoost model to {xgb_model_path}")

    del xgb_model
    gc.collect()
    print(f"Peak RSS after XGBoost: {get_peak_rss_mb():.1f} MB")

    th_xgb, m_xgb = sweep_thresholds_for_probs(val_df, xgb_prob, gt_map, all_s1_ids, threshold_list, "XGBoost v2")
    results_table.append({
        "Model": "XGBoost v2",
        "ROC-AUC": xgb_auc,
        "PR-AUC": xgb_ap,
        "Threshold": th_xgb,
        "F0.5": m_xgb["macro_f05"],
        "Precision": m_xgb["macro_precision"],
        "Recall": m_xgb["macro_recall"],
        "Singleton Acc": m_xgb["singleton_accuracy"],
    })

    # =========================================================================
    # MODEL C: CatBoost
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6/7] Training Model C: CatBoost (5000 iterations, depth=9)...")
    print("=" * 70)
    t_cat = time.time()
    cat_model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=9,
        l2_leaf_reg=3.0,
        subsample=0.7,
        auto_class_weights="Balanced",
        early_stopping_rounds=200,
        random_seed=42,
        thread_count=-1,
        verbose=200,
    )
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=200,
    )
    best_iter_cat = cat_model.get_best_iteration()
    print(f"CatBoost trained in {time.time()-t_cat:.1f}s. Best iteration: {best_iter_cat}")

    cat_prob = cat_model.predict_proba(X_val)[:, 1]
    cat_auc = roc_auc_score(y_val, cat_prob)
    cat_ap = average_precision_score(y_val, cat_prob)
    print(f"CatBoost Validation ROC-AUC: {cat_auc:.5f} | PR-AUC: {cat_ap:.5f}")

    cat_model_path = os.path.join(output_dir, "catboost_model_v2.cbm")
    cat_model.save_model(cat_model_path)
    print(f"Saved CatBoost model to {cat_model_path}")

    del cat_model
    del X_train, y_train
    gc.collect()
    print(f"Peak RSS after CatBoost & Freeing Train Set: {get_peak_rss_mb():.1f} MB")

    th_cat, m_cat = sweep_thresholds_for_probs(val_df, cat_prob, gt_map, all_s1_ids, threshold_list, "CatBoost v2")
    results_table.append({
        "Model": "CatBoost v2",
        "ROC-AUC": cat_auc,
        "PR-AUC": cat_ap,
        "Threshold": th_cat,
        "F0.5": m_cat["macro_f05"],
        "Precision": m_cat["macro_precision"],
        "Recall": m_cat["macro_recall"],
        "Singleton Acc": m_cat["singleton_accuracy"],
    })

    # =========================================================================
    # ENSEMBLE EVALUATION & GRID SEARCH
    # =========================================================================
    print("\n" + "=" * 70)
    print("[7/7] Evaluating Ensemble Combinations & Optimizing Weights...")
    print("=" * 70)

    # Strategy A: Equal average
    prob_equal = (lgb_prob + xgb_prob + cat_prob) / 3.0
    th_eq, m_eq = sweep_thresholds_for_probs(val_df, prob_equal, gt_map, all_s1_ids, threshold_list, "Equal Ensemble (1/3, 1/3, 1/3)")
    results_table.append({
        "Model": "Equal Ensemble (0.33, 0.33, 0.33)",
        "ROC-AUC": roc_auc_score(y_val, prob_equal),
        "PR-AUC": average_precision_score(y_val, prob_equal),
        "Threshold": th_eq,
        "F0.5": m_eq["macro_f05"],
        "Precision": m_eq["macro_precision"],
        "Recall": m_eq["macro_recall"],
        "Singleton Acc": m_eq["singleton_accuracy"],
    })

    # Strategy B: Weighted (0.40, 0.35, 0.25)
    prob_w_b = 0.40 * lgb_prob + 0.35 * xgb_prob + 0.25 * cat_prob
    th_wb, m_wb = sweep_thresholds_for_probs(val_df, prob_w_b, gt_map, all_s1_ids, threshold_list, "Weighted B (0.40, 0.35, 0.25)")
    results_table.append({
        "Model": "Weighted B (0.40, 0.35, 0.25)",
        "ROC-AUC": roc_auc_score(y_val, prob_w_b),
        "PR-AUC": average_precision_score(y_val, prob_w_b),
        "Threshold": th_wb,
        "F0.5": m_wb["macro_f05"],
        "Precision": m_wb["macro_precision"],
        "Recall": m_wb["macro_recall"],
        "Singleton Acc": m_wb["singleton_accuracy"],
    })

    # Strategy C: Weighted (0.50, 0.25, 0.25)
    prob_w_c = 0.50 * lgb_prob + 0.25 * xgb_prob + 0.25 * cat_prob
    th_wc, m_wc = sweep_thresholds_for_probs(val_df, prob_w_c, gt_map, all_s1_ids, threshold_list, "Weighted C (0.50, 0.25, 0.25)")
    results_table.append({
        "Model": "Weighted C (0.50, 0.25, 0.25)",
        "ROC-AUC": roc_auc_score(y_val, prob_w_c),
        "PR-AUC": average_precision_score(y_val, prob_w_c),
        "Threshold": th_wc,
        "F0.5": m_wc["macro_f05"],
        "Precision": m_wc["macro_precision"],
        "Recall": m_wc["macro_recall"],
        "Singleton Acc": m_wc["singleton_accuracy"],
    })

    # Strategy D: Fine Grid Search (w1, w2, w3 summing to 1.0 from 0.20 to 0.60)
    print("\nRunning Weight Grid Search across (w_lgb, w_xgb, w_cat)...")
    best_grid_f05 = -1.0
    best_grid_weights = (0.34, 0.33, 0.33)
    best_grid_thresh = 0.50
    best_grid_metrics: dict[str, float] = {}

    weight_candidates = [round(w, 2) for w in np.arange(0.20, 0.65, 0.05)]
    valid_combos = []
    for w1 in weight_candidates:
        for w2 in weight_candidates:
            w3 = round(1.0 - w1 - w2, 2)
            if 0.15 <= w3 <= 0.60:
                valid_combos.append((w1, w2, w3))

    print(f"Testing {len(valid_combos)} weight combinations...")
    for w1, w2, w3 in valid_combos:
        blend_prob = w1 * lgb_prob + w2 * xgb_prob + w3 * cat_prob
        th_g, m_g = sweep_thresholds_for_probs(
            val_df, blend_prob, gt_map, all_s1_ids, threshold_list, f"Grid ({w1:.2f}, {w2:.2f}, {w3:.2f})"
        )
        if m_g["macro_f05"] > best_grid_f05:
            best_grid_f05 = m_g["macro_f05"]
            best_grid_weights = (w1, w2, w3)
            best_grid_thresh = th_g
            best_grid_metrics = m_g

    prob_best_grid = best_grid_weights[0] * lgb_prob + best_grid_weights[1] * xgb_prob + best_grid_weights[2] * cat_prob
    results_table.append({
        "Model": f"Best Weighted Ensemble ({best_grid_weights[0]:.2f}, {best_grid_weights[1]:.2f}, {best_grid_weights[2]:.2f})",
        "ROC-AUC": roc_auc_score(y_val, prob_best_grid),
        "PR-AUC": average_precision_score(y_val, prob_best_grid),
        "Threshold": best_grid_thresh,
        "F0.5": best_grid_metrics["macro_f05"],
        "Precision": best_grid_metrics["macro_precision"],
        "Recall": best_grid_metrics["macro_recall"],
        "Singleton Acc": best_grid_metrics["singleton_accuracy"],
    })

    # Pick the absolute winning configuration across all tested rows
    winning_row = max(results_table, key=lambda r: r["F0.5"])
    print("\n" + "=" * 70)
    print("WINNING CONFIGURATION")
    print("=" * 70)
    print(f"Model/Ensemble:      {winning_row['Model']}")
    print(f"Optimal Threshold:   {winning_row['Threshold']:.3f}")
    print(f"Best Macro F0.5:     {winning_row['F0.5']:.4f}")
    print(f"Macro Precision:     {winning_row['Precision']:.4f}")
    print(f"Macro Recall:        {winning_row['Recall']:.4f}")
    print(f"Singleton Accuracy:  {winning_row['Singleton Acc']:.4f}")
    print(f"ROC-AUC:             {winning_row['ROC-AUC']:.5f}")
    print(f"PR-AUC:              {winning_row['PR-AUC']:.5f}")
    print("=" * 70)

    # Save threshold_v2.txt
    threshold_path = os.path.join(output_dir, "threshold_v2.txt")
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(f"{winning_row['Threshold']:.3f}\n")
    print(f"Saved optimal threshold to {threshold_path}")

    # Save ensemble_config.txt
    config_path = os.path.join(output_dir, "ensemble_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        if "Best Weighted" in winning_row["Model"]:
            f.write(f"weighted|{best_grid_weights[0]},{best_grid_weights[1]},{best_grid_weights[2]}|{winning_row['Threshold']:.3f}\n")
        elif "Equal" in winning_row["Model"]:
            f.write(f"weighted|0.3333,0.3333,0.3334|{winning_row['Threshold']:.3f}\n")
        elif "Weighted B" in winning_row["Model"]:
            f.write(f"weighted|0.40,0.35,0.25|{winning_row['Threshold']:.3f}\n")
        elif "Weighted C" in winning_row["Model"]:
            f.write(f"weighted|0.50,0.25,0.25|{winning_row['Threshold']:.3f}\n")
        elif "LightGBM" in winning_row["Model"]:
            f.write(f"lgb_only||{winning_row['Threshold']:.3f}\n")
        elif "XGBoost" in winning_row["Model"]:
            f.write(f"xgb_only||{winning_row['Threshold']:.3f}\n")
        elif "CatBoost" in winning_row["Model"]:
            f.write(f"cat_only||{winning_row['Threshold']:.3f}\n")
        else:
            f.write(f"weighted|{best_grid_weights[0]},{best_grid_weights[1]},{best_grid_weights[2]}|{winning_row['Threshold']:.3f}\n")
    print(f"Saved ensemble configuration to {config_path}")

    # Append to experiments/log.md
    if os.path.exists(log_path):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write("## Phase C: 3-Model Ensemble (LightGBM v2 + XGBoost v2 + CatBoost v2)\n\n")
            f.write(f"**Execution Date:** {datetime.date.today().isoformat()}\n\n")
            f.write("### Model Ensemble Comparison Table (Val Features No-Cap)\n\n")
            f.write("| Model | ROC-AUC | PR-AUC | Best Threshold | Macro F0.5 | Precision | Recall | Singleton Acc |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
            f.write("| Baseline (LGB v1, K=50) | 0.9992 | 0.9894 | 0.650 | 0.9074 | 0.9486 | 0.8282 | 0.9033 |\n")
            f.write("| Baseline (LGB v1, K=100) | 0.9997 | 0.9917 | 0.970 | 0.9250 | 0.9574 | 0.8627 | 0.9169 |\n")
            for r in results_table:
                f.write(f"| {r['Model']} | {r['ROC-AUC']:.5f} | {r['PR-AUC']:.5f} | {r['Threshold']:.3f} | **{r['F0.5']:.4f}** | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['Singleton Acc']:.4f} |\n")
            f.write(f"\n**Winning Configuration:** `{winning_row['Model']}` at threshold `{winning_row['Threshold']:.3f}` with **Macro F0.5 = {winning_row['F0.5']:.4f}**.\n")
        print(f"Updated {log_path} with Phase C results.")

    total_time = time.time() - t_pipeline_start
    print(f"\nPhase C Complete in {total_time:.1f}s ({total_time/60.0:.2f} minutes).")

    return {
        "results_table": results_table,
        "winning_config": winning_row,
    }


if __name__ == "__main__":
    train_ensemble_v2()
