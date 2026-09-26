#!/bin/bash
set -e

cd ~/Grokking
export PYTHONPATH=code/business_entity_resolution/src

echo "=== Pipeline Master Started: $(date) ==="

# 1. Wait for US training features (PID 7721) to finish
echo "Waiting for US training features (PID 7721) to complete..."
while kill -0 7721 2>/dev/null; do
    sleep 30
done
echo "US training features completed at $(date)!"

# 2. Merge India and US training features
echo "Merging India and US training features..."
~/venv/bin/python3 code/business_entity_resolution/src/build_features_full.py \
  --merge experiments/train_features_india.parquet experiments/train_features_us.parquet \
  --output experiments/train_features_k100.parquet

# 3. Step 4: Validation feature extraction
echo "Starting Step 4: Validation feature extraction (K=100) at $(date)..."
~/venv/bin/python3 -u code/business_entity_resolution/src/build_features_full.py \
  --split val --k 100

# 4. Step 5: Production model training
echo "Starting Step 5: Production LightGBM Training & Sweep at $(date)..."
~/venv/bin/python3 -u code/business_entity_resolution/src/train_production.py

echo "=== ALL PRODUCTION PIPELINE STEPS COMPLETED SUCCESSFULLY: $(date) ==="
