#!/usr/bin/env python3
"""Quick retrain LGB v2 on existing K=100 features."""
import sys, os, time, gc, functools
sys.path.insert(0, "code/business_entity_resolution/src")
print = functools.partial(print, flush=True)

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from features import FEATURE_COLUMN_NAMES
from train_matching import evaluate_predictions

print("=" * 70)
print("QUICK TRAIN v2: LightGBM 5000 trees, lr=0.02, leaves=127")
print("=" * 70)

print("\n[1] Loading training features...")
train_df = pd.read_parquet("experiments/train_features_k100.parquet")
print(f"    {len(train_df):,} training pairs loaded.")

print("[2] Loading validation features (K=100)...")
val_df = pd.read_parquet("experiments/val_features_k100.parquet")
print(f"    {len(val_df):,} validation pairs loaded.")

feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in train_df.columns]
print(f"[3] Using {len(feature_cols)} features.")

X_train = train_df[feature_cols].values
y_train = train_df["label"].values
X_val = val_df[feature_cols].values
y_val = val_df["label"].values

n_pos = int(y_train.sum())
n_neg = len(y_train) - n_pos
spw = n_neg / n_pos
print(f"    Train: {len(y_train):,} pairs ({n_pos:,} pos, {n_neg:,} neg)")
print(f"    Val:   {len(y_val):,} pairs ({int(y_val.sum()):,} pos)")
print(f"    scale_pos_weight = {spw:.4f}")

del train_df
gc.collect()

print(f"\n[4] Training LightGBM v2...")
t0 = time.time()
model = lgb.LGBMClassifier(
    n_estimators=5000, learning_rate=0.02, num_leaves=127,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
    max_bin=511, random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(X_train, y_train,
    eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(200)])
train_time = time.time() - t0
print(f"\n    Training complete: {train_time:.0f}s, best_iteration={model.best_iteration_}")

del X_train, y_train
gc.collect()

print("\n[5] Evaluating on validation set...")
val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)
prauc = average_precision_score(y_val, val_probs)
print(f"    ROC-AUC:  {auc:.6f}")
print(f"    PR-AUC:   {prauc:.6f}")

gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
gt_map = {}
for _, row in gt_df.iterrows():
    s1 = row["source1_entity_id"]
    matched = row["matched_entity_ids"].strip()
    gt_map[s1] = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
del gt_df

val_df["pred_score"] = val_probs
all_s1 = sorted(val_df["source1_entity_id"].unique())
s1_cands = {}
for s1_id, grp in val_df.groupby("source1_entity_id"):
    s1_cands[s1_id] = list(zip(grp["candidate_entity_id"], grp["pred_score"]))

print("\n[6] Threshold sweep:")
print(f"    {'Threshold':>10} {'F0.5':>8} {'Prec':>8} {'Recall':>8} {'SingAcc':>8}")
best_f05, best_th, best_m = -1, 0.5, {}
thresholds = list(np.arange(0.50, 0.96, 0.01)) + list(np.arange(0.96, 1.001, 0.005))
for th in thresholds:
    pred_map = {s1_id: {c for c, s in s1_cands.get(s1_id, []) if s >= th} for s1_id in all_s1}
    m = evaluate_predictions(gt_map, pred_map, all_s1)
    f = m["macro_f05"]
    if f > best_f05:
        best_f05, best_th, best_m = f, th, m
    if th >= 0.90:
        print(f"    {th:10.3f} {f:8.4f} {m['macro_precision']:8.4f} "
              f"{m['macro_recall']:8.4f} {m['singleton_accuracy']:8.4f}")

print(f"\n*** BEST: threshold={best_th:.3f}, Macro F0.5={best_f05:.4f} ***")
print(f"    Precision:      {best_m['macro_precision']:.4f}")
print(f"    Recall:         {best_m['macro_recall']:.4f}")
print(f"    Singleton Acc:  {best_m['singleton_accuracy']:.4f}")
print(f"    v1 baseline:    F0.5=0.9250 (threshold=0.97)")
print(f"    v2 improvement: +{best_f05 - 0.9250:.4f}")

model.booster_.save_model("experiments/lgbm_model_v2.txt")
with open("experiments/threshold_v2.txt", "w") as f:
    f.write(f"{best_th:.3f}\n")
with open("experiments/ensemble_config.txt", "w") as f:
    f.write(f"lgb_only||{best_th:.3f}\n")
with open("experiments/feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))

print(f"\n[7] Saved all model files.")

with open("experiments/log.md", "a") as f:
    f.write(f"\n### Quick Train v2 — {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"- LightGBM, 5000 trees (stopped at {model.best_iteration_}), lr=0.02, num_leaves=127\n")
    f.write(f"- Train: {19902872} pairs (K=100), Val: {2488078} pairs (K=100)\n")
    f.write(f"- ROC-AUC: {auc:.6f}, PR-AUC: {prauc:.6f}\n")
    f.write(f"- **Macro F0.5: {best_f05:.4f}** (threshold={best_th:.3f})\n")
    f.write(f"- Precision: {best_m['macro_precision']:.4f}, Recall: {best_m['macro_recall']:.4f}\n")
    f.write(f"- Singleton Acc: {best_m['singleton_accuracy']:.4f}\n")
    f.write(f"- Training time: {train_time:.0f}s\n")
print("Done.")
