#!/bin/bash
set -e
cd /home/azureuser/Grokking
export PYTHONPATH=code/business_entity_resolution/src
VENV=/home/azureuser/venv/bin/python3

echo "======================================================================"
echo "=== PHASE C: Starting 3-Model Ensemble Training ==="
echo "======================================================================"
$VENV code/business_entity_resolution/src/train_v2.py

echo "======================================================================"
echo "=== PHASE D: Starting Full Test Inference (candidate_cap=100) ==="
echo "======================================================================"
$VENV code/business_entity_resolution/src/run_inference.py --candidate-cap 100

echo "======================================================================"
echo "=== ALL PHASES (C & D) COMPLETED SUCCESSFULLY: $(date) ==="
echo "======================================================================"
