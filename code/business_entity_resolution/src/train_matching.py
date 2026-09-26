"""
Phase 4 & 5 Matching Model Training & Validation Harness.
Amazon ML Challenge 2026: Business Entity Resolution.

Trains a LightGBM pairwise matching classifier with GroupKFold cross-validation
(grouped by source1_entity_id), sweeps the decision threshold to maximize macro F0.5,
and reports precision, recall, singleton accuracy, and non-singleton F0.5.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from features import FEATURE_COLUMN_NAMES


def compute_entity_f05(
    true_set: set[str],
    pred_set: set[str],
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """
    Computes Precision, Recall, and F0.5 for a single Source 1 entity.
    Handles singletons (empty true_set) and empty predictions explicitly per spec:
    - If true is empty and pred is empty: P=1, R=1, F0.5=1 (correct singleton).
    - If true is non-empty and pred is empty: P=0, R=0, F0.5=0 (missed match).
    - If true is empty and pred is non-empty: P=0, R=0, F0.5=0 (false merge).
    """
    if len(true_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0
    if len(true_set) == 0 or len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    inter = len(true_set & pred_set)
    p = inter / len(pred_set)
    r = inter / len(true_set)

    beta_sq = beta * beta
    denom = (beta_sq * p) + r
    f_beta = ((1.0 + beta_sq) * p * r / denom) if denom > 0 else 0.0
    return p, r, f_beta


def evaluate_predictions(
    ground_truth_map: dict[str, set[str]],
    predictions_map: dict[str, set[str]],
    all_s1_ids: list[str],
) -> dict[str, float]:
    """
    Computes macro-averaged metrics across all Source 1 entities in the split.
    Reports overall Macro F0.5, Precision, Recall, Singleton Accuracy, and Non-Singleton F0.5.
    """
    p_list: list[float] = []
    r_list: list[float] = []
    f_list: list[float] = []

    singleton_correct: int = 0
    singleton_total: int = 0
    non_singleton_f: list[float] = []

    for s1_id in all_s1_ids:
        true_set = ground_truth_map.get(s1_id, set())
        pred_set = predictions_map.get(s1_id, set())

        p, r, f05 = compute_entity_f05(true_set, pred_set, beta=0.5)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f05)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_f.append(f05)

    macro_p = float(np.mean(p_list)) if p_list else 0.0
    macro_r = float(np.mean(r_list)) if r_list else 0.0
    macro_f = float(np.mean(f_list)) if f_list else 0.0
    singleton_acc = float(singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    non_singleton_f_mean = float(np.mean(non_singleton_f)) if non_singleton_f else 0.0

    return {
        "macro_f05": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "non_singleton_f05": non_singleton_f_mean,
        "num_entities": len(all_s1_ids),
        "num_singletons": singleton_total,
        "num_non_singletons": len(non_singleton_f),
    }


def train_and_evaluate_matching(
    features_parquet_path: str,
    ground_truth_tsv_path: str,
    model_output_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Trains LightGBM classifier with GroupKFold cross-validation, evaluates AUC/PR-AUC,
    sweeps threshold to maximize Macro F0.5, and reports detailed validation breakdown.
    """
    print(f"Loading features from {features_parquet_path}...")
    df = pd.read_parquet(features_parquet_path)
    print(f"Loaded {len(df):,} candidate pair instances.")

    feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].values
    groups = df["source1_entity_id"].values
    all_s1_ids = list(df["source1_entity_id"].unique())

    print(f"Loading ground truth from {ground_truth_tsv_path}...")
    gt_df = pd.read_csv(ground_truth_tsv_path, sep="\t", dtype=str).fillna("")
    gt_map: dict[str, set[str]] = {}
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        cands = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1] = set(cands)

    positives = int(y.sum())
    print(f"Pair labels: {positives:,} positive pairs, {len(y)-positives:,} negative pairs.")

    # 3-Fold GroupKFold Cross Validation
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(len(df), dtype=np.float32)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 350,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    print("\nExecuting GroupKFold Cross-Validation...")
    feature_importances = np.zeros(len(feature_cols), dtype=np.float64)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        feature_importances += model.feature_importances_ / 3.0

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_ap = average_precision_score(y_val, val_probs)
        print(f"  Fold {fold}: ROC-AUC = {fold_auc:.4f}, PR-AUC = {fold_ap:.4f}")

    total_auc = roc_auc_score(y, oof_preds)
    total_ap = average_precision_score(y, oof_preds)
    print(f"\nOverall Out-of-Fold: ROC-AUC = {total_auc:.4f}, PR-AUC (Average Precision) = {total_ap:.4f}")

    # Feature Importance Ranking
    fi_df = pd.DataFrame({"feature": feature_cols, "importance": feature_importances})
    fi_df = fi_df.sort_values(by="importance", ascending=False).reset_index(drop=True)
    print("\n=== TOP 15 MOST INFORMATIVE FEATURES (LightGBM split gain) ===")
    for _, r in fi_df.head(15).iterrows():
        print(f"  {r['feature']:<25}: {r['importance']:.1f}")

    # Threshold Sweep to Maximize Macro F0.5
    print("\nSweeping decision thresholds for Macro F0.5 optimization...")
    df["pred_score"] = oof_preds
    thresholds = np.arange(0.15, 0.90, 0.05)

    best_thresh = 0.50
    best_metrics: dict[str, float] = {}
    best_f05 = -1.0

    # Group dataframe by source1_entity_id once for fast thresholding
    s1_candidates = df.groupby("source1_entity_id")[["candidate_entity_id", "pred_score"]].apply(
        lambda g: list(zip(g["candidate_entity_id"], g["pred_score"]))
    ).to_dict()

    for th in thresholds:
        pred_map: dict[str, set[str]] = {}
        for s1_id in all_s1_ids:
            cands = s1_candidates.get(s1_id, [])
            passing = {c_id for c_id, sc in cands if sc >= th}
            pred_map[s1_id] = passing

        metrics = evaluate_predictions(gt_map, pred_map, all_s1_ids)
        f05 = metrics["macro_f05"]
        print(f"  Threshold {th:.2f}: Macro F0.5 = {f05:.4f} | Prec = {metrics['macro_precision']:.4f} | Rec = {metrics['macro_recall']:.4f} | Singleton Acc = {metrics['singleton_accuracy']:.4f} | Non-Singleton F0.5 = {metrics['non_singleton_f05']:.4f}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(th)
            best_metrics = metrics

    print(f"\n=======================================================")
    print(f"OPTIMAL THRESHOLD: {best_thresh:.2f}")
    print(f"Best Macro F0.5:         {best_metrics['macro_f05']:.4f}")
    print(f"Macro Precision:         {best_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:            {best_metrics['macro_recall']:.4f}")
    print(f"Singleton Accuracy:      {best_metrics['singleton_accuracy']:.4f}")
    print(f"Non-Singleton F0.5:      {best_metrics['non_singleton_f05']:.4f}")
    print(f"=======================================================")

    return {
        "best_threshold": best_thresh,
        "metrics": best_metrics,
        "feature_importances": fi_df,
        "auc": total_auc,
        "pr_auc": total_ap,
    }


if __name__ == "__main__":
    feat_path = "experiments/val_features.parquet"
    gt_path = "dataset/train/train_ground_truth.tsv"
    if not os.path.exists(feat_path):
        print(f"Features file {feat_path} not found. Run extract_features.py first.")
        sys.exit(1)

    train_and_evaluate_matching(feat_path, gt_path)
