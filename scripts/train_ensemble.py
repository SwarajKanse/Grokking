#!/usr/bin/env python3
"""Train 3-model ensemble (LGB + XGB + CatBoost) on combined training data."""
import sys, os, time, gc, functools
sys.path.insert(0, "code/business_entity_resolution/src")
print = functools.partial(print, flush=True)

import lightgbm as lgb
import xgboost as xgb_lib
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from features import FEATURE_COLUMN_NAMES
from train_matching import evaluate_predictions

print("=" * 70)
print("ENSEMBLE TRAINING: LGB + XGB + CatBoost")
print("=" * 70)

# Load training features
print("\n[1] Loading training features...")
df1 = pd.read_parquet("experiments/train_features_k100.parquet")
print(f"    Original: {len(df1):,} pairs")

extra_path = "experiments/train_features_extra300k.parquet"
if os.path.exists(extra_path):
    df2 = pd.read_parquet(extra_path)
    print(f"    Extra 300K: {len(df2):,} pairs")
    train_df = pd.concat([df1, df2], ignore_index=True)
    del df1, df2
else:
    print("    WARNING: Extra features not found, using original only.")
    train_df = df1
    del df1
gc.collect()
print(f"    Combined: {len(train_df):,} pairs")

print("[2] Loading validation features...")
val_df = pd.read_parquet("experiments/val_features_k100.parquet")
print(f"    Val: {len(val_df):,} pairs")

feature_cols = [c for c in FEATURE_COLUMN_NAMES if c in train_df.columns]
X_train = train_df[feature_cols].values.astype(np.float32)
y_train = train_df["label"].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
y_val = val_df["label"].values.astype(np.float32)

n_pos, n_neg = int(y_train.sum()), len(y_train) - int(y_train.sum())
spw = n_neg / n_pos
print(f"    Train: {n_pos:,} pos, {n_neg:,} neg, spw={spw:.2f}")
print(f"    Features: {len(feature_cols)}")

del train_df
gc.collect()

# Ground truth
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
gt_map = {}
for _, row in gt_df.iterrows():
    s1 = row["source1_entity_id"]
    matched = row["matched_entity_ids"].strip()
    gt_map[s1] = set(x.strip() for x in matched.split(",") if x.strip()) if matched else set()
del gt_df

all_s1 = sorted(val_df["source1_entity_id"].unique())
s1_scores = {}
for s1_id, grp in val_df.groupby("source1_entity_id"):
    s1_scores[s1_id] = {"cands": list(grp["candidate_entity_id"].values), "idx": list(grp.index)}

# --- MODEL A: LightGBM ---
print("\n" + "=" * 50)
print("[3] Training LightGBM...")
t_lgb = time.time()
lgb_model = lgb.LGBMClassifier(
    n_estimators=5000, learning_rate=0.02, num_leaves=127,
    min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
    max_bin=511, random_state=42, n_jobs=-1, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(200, verbose=True), lgb.log_evaluation(500)])
lgb_time = time.time() - t_lgb
lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_probs)
print(f"    LGB: {lgb_time:.0f}s, iter={lgb_model.best_iteration_}, AUC={lgb_auc:.6f}")
lgb_model.booster_.save_model("experiments/lgbm_model_v2.txt")

# --- MODEL B: XGBoost ---
print("\n" + "=" * 50)
print("[4] Training XGBoost...")
t_xgb = time.time()
dtrain = xgb_lib.DMatrix(X_train, label=y_train, feature_names=feature_cols)
dval = xgb_lib.DMatrix(X_val, label=y_val, feature_names=feature_cols)
xgb_params = {
    "max_depth": 8, "learning_rate": 0.02, "subsample": 0.7,
    "colsample_bytree": 0.7, "min_child_weight": 100,
    "scale_pos_weight": spw, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "max_bin": 512, "nthread": -1, "seed": 42}
xgb_model = xgb_lib.train(xgb_params, dtrain, num_boost_round=5000,
    evals=[(dval, "val")], early_stopping_rounds=200, verbose_eval=500)
xgb_time = time.time() - t_xgb
xgb_probs = xgb_model.predict(dval)
xgb_auc = roc_auc_score(y_val, xgb_probs)
print(f"    XGB: {xgb_time:.0f}s, iter={xgb_model.best_iteration}, AUC={xgb_auc:.6f}")
xgb_model.save_model("experiments/xgb_model_v2.json")
del dtrain
gc.collect()

# --- MODEL C: CatBoost ---
print("\n" + "=" * 50)
print("[5] Training CatBoost...")
t_cat = time.time()
cat_model = CatBoostClassifier(
    iterations=5000, learning_rate=0.02, depth=8,
    l2_leaf_reg=3.0, subsample=0.7, auto_class_weights="Balanced",
    random_seed=42, verbose=500, early_stopping_rounds=200)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
cat_time = time.time() - t_cat
cat_probs = cat_model.predict_proba(X_val)[:, 1]
cat_auc = roc_auc_score(y_val, cat_probs)
print(f"    Cat: {cat_time:.0f}s, AUC={cat_auc:.6f}")
cat_model.save_model("experiments/catboost_model_v2.cbm")

del X_train, y_train
gc.collect()

# --- ENSEMBLE WEIGHT OPTIMIZATION ---
print("\n" + "=" * 50)
print("[6] Optimizing ensemble weights + thresholds...")

# Build per-entity score lists
for s1_id in all_s1:
    idxs = s1_scores[s1_id]["idx"]
    s1_scores[s1_id]["lgb"] = [lgb_probs[i] for i in idxs]
    s1_scores[s1_id]["xgb"] = [xgb_probs[i] for i in idxs]
    s1_scores[s1_id]["cat"] = [cat_probs[i] for i in idxs]

weight_options = [
    ("equal",       1/3, 1/3, 1/3),
    ("lgb_heavy",   0.5, 0.25, 0.25),
    ("lgb_heavier", 0.6, 0.2, 0.2),
    ("xgb_heavy",   0.25, 0.5, 0.25),
    ("balanced",    0.4, 0.35, 0.25),
    ("lgb_only",    1.0, 0.0, 0.0),
    ("lgb_xgb",     0.5, 0.5, 0.0),
    ("lgb_cat",     0.5, 0.0, 0.5),
]

best_overall_f05, best_cfg = -1, ""
results = []

for name, w1, w2, w3 in weight_options:
    best_f, best_t, best_met = -1, 0.5, {}
    for th in list(np.arange(0.90, 1.001, 0.005)):
        pred_map = {}
        for s1_id in all_s1:
            sc = s1_scores[s1_id]
            combined = [w1*l + w2*x + w3*c for l, x, c in zip(sc["lgb"], sc["xgb"], sc["cat"])]
            pred_map[s1_id] = {sc["cands"][i] for i, p in enumerate(combined) if p >= th}
        m = evaluate_predictions(gt_map, pred_map, all_s1)
        if m["macro_f05"] > best_f:
            best_f, best_t, best_met = m["macro_f05"], th, m

    print(f"    {name:15s} w=({w1:.2f},{w2:.2f},{w3:.2f}): F0.5={best_f:.4f} @ th={best_t:.3f} "
          f"P={best_met['macro_precision']:.4f} R={best_met['macro_recall']:.4f}")
    results.append((name, w1, w2, w3, best_f, best_t, best_met))

    if best_f > best_overall_f05:
        best_overall_f05 = best_f
        if w2 == 0 and w3 == 0:
            best_cfg = f"lgb_only||{best_t:.3f}"
        else:
            best_cfg = f"weighted|{w1},{w2},{w3}|{best_t:.3f}"
        best_overall_th = best_t
        best_overall_met = best_met
        best_name = name

print(f"\n*** BEST: {best_name}, F0.5={best_overall_f05:.4f}, threshold={best_overall_th:.3f} ***")
print(f"    P={best_overall_met['macro_precision']:.4f}, R={best_overall_met['macro_recall']:.4f}, "
      f"Sing={best_overall_met['singleton_accuracy']:.4f}")

# Save
with open("experiments/ensemble_config.txt", "w") as f:
    f.write(best_cfg + "\n")
with open("experiments/threshold_v2.txt", "w") as f:
    f.write(f"{best_overall_th:.3f}\n")

print(f"\n    Config: {best_cfg}")
print(f"    Saved: ensemble_config.txt, threshold_v2.txt")

# Log
with open("experiments/log.md", "a") as f:
    f.write(f"\n### Ensemble Training — {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"- LGB: {lgb_time:.0f}s AUC={lgb_auc:.6f} | XGB: {xgb_time:.0f}s AUC={xgb_auc:.6f} | Cat: {cat_time:.0f}s AUC={cat_auc:.6f}\n")
    f.write(f"- Best: {best_name} → **Macro F0.5: {best_overall_f05:.4f}** (th={best_overall_th:.3f})\n")
    f.write(f"- P={best_overall_met['macro_precision']:.4f}, R={best_overall_met['macro_recall']:.4f}, Sing={best_overall_met['singleton_accuracy']:.4f}\n")
print("\nEnsemble training complete.")
