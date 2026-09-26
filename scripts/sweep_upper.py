import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import time

sys.path.insert(0, "code/business_entity_resolution/src")
from features import FEATURE_COLUMN_NAMES
from train_matching import evaluate_predictions

print("Loading validation data and model...")
val_df = pd.read_parquet("experiments/val_features_k100.parquet")
bst = lgb.Booster(model_file="experiments/lgbm_model_v1.txt")

feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in val_df.columns]
print(f"Using {len(feature_cols)} features. Predicting validation probabilities...")
val_probs = bst.predict(val_df[feature_cols])
val_df["pred_score"] = val_probs

print("Loading ground truth...")
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")
gt_map = {}
for _, row in gt_df.iterrows():
    s1_id = row["source1_entity_id"]
    matched = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
    gt_map[s1_id] = set(matched.split(",")) if matched and matched != "nan" else set()

val_split_df = pd.read_parquet("experiments/val_split.parquet")
all_s1_ids = list(val_split_df["source1_entity_id"].unique())

print("Grouping candidate pairs...")
s1_candidates = val_df.groupby("source1_entity_id")[["candidate_entity_id", "pred_score"]].apply(
    lambda g: list(zip(g["candidate_entity_id"], g["pred_score"]))
).to_dict()

thresholds = [round(x, 3) for x in np.arange(0.920, 0.996, 0.005)]
print("-" * 105)
print(f"{'Thresh':<8} | {'Macro F0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<14} | {'Non-Single F0.5':<15}")
print("-" * 105)

best_thresh = 0.95
best_f05 = -1.0
best_metrics = {}

for th in thresholds:
    pred_map = {}
    for s1_id in all_s1_ids:
        cands = s1_candidates.get(s1_id, [])
        pred_map[s1_id] = {c_id for c_id, sc in cands if sc >= th}
    metrics = evaluate_predictions(gt_map, pred_map, all_s1_ids)
    f05 = metrics["macro_f05"]
    print(f"  {th:.3f}  |   {f05:.4f}   |   {metrics['macro_precision']:.4f}   |   {metrics['macro_recall']:.4f}   |    {metrics['singleton_accuracy']:.4f}     |     {metrics['non_singleton_f05']:.4f}")
    if f05 > best_f05:
        best_f05 = f05
        best_thresh = th
        best_metrics = metrics

print("-" * 105)
print(f"OPTIMAL UPPER THRESHOLD: {best_thresh}")
print(f"Best Macro F0.5: {best_metrics['macro_f05']:.4f}")
print(f"Macro Precision: {best_metrics['macro_precision']:.4f}")
print(f"Macro Recall:    {best_metrics['macro_recall']:.4f}")
print(f"Singleton Acc:   {best_metrics['singleton_accuracy']:.4f}")
print(f"Non-Single F0.5: {best_metrics['non_singleton_f05']:.4f}")
