#!/bin/bash
# MASTER PIPELINE — Runs everything autonomously.
# Produces 2 validated submissions.
# Expected total runtime: ~27 hours.
set -o pipefail
cd /home/azureuser/Grokking
export PYTHONPATH=code/business_entity_resolution/src
PY=/home/azureuser/venv/bin/python3
LOG=experiments/master_log.txt

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "============================================================"
log "MASTER PIPELINE STARTED"
log "Hard deadline: Sep 27 8:00 PM IST"
log "============================================================"

# =============================================================
# PHASE 1: Kill any stuck processes & verify clean state
# =============================================================
log "--- PHASE 1: Cleanup ---"
pkill -f "train_v2.py" 2>/dev/null || true
sleep 2
log "Memory after cleanup:"
free -h | tee -a "$LOG"

# =============================================================
# PHASE 2: Quick Retrain LGB v2 (~1 hour)
# =============================================================
log ""
log "--- PHASE 2: Quick Retrain LGB v2 ---"
$PY scripts/quick_train_v2.py 2>&1 | tee -a "$LOG"
PHASE2_EXIT=$?
if [ $PHASE2_EXIT -ne 0 ]; then
    log "PHASE 2 FAILED (exit $PHASE2_EXIT). Falling back to v1 model."
    cp experiments/lgbm_model_v1.txt experiments/lgbm_model_v2.txt 2>/dev/null
    echo "0.97" > experiments/threshold_v2.txt
    echo "lgb_only||0.97" > experiments/ensemble_config.txt
fi
log "PHASE 2 DONE"

# =============================================================
# PHASE 3: K=100 Test Inference → Submission 1 (~6.5 hours)
# =============================================================
log ""
log "--- PHASE 3: K=100 Inference (Submission 1) ---"
$PY code/business_entity_resolution/src/run_inference.py \
    --candidate-cap 100 2>&1 | tee -a experiments/inference_sub1.log
PHASE3_EXIT=$?
if [ $PHASE3_EXIT -ne 0 ]; then
    log "PHASE 3 FAILED (exit $PHASE3_EXIT). Check inference_sub1.log."
    # Don't exit — continue to build better model
else
    log "PHASE 3 DONE — Submission 1 ready."
    cp output/matching_results.tsv output/matching_results_sub1.tsv
    cp output/candidate_pairs.tsv output/candidate_pairs_sub1.tsv
    log "Submission 1 files:"
    wc -l output/matching_results_sub1.tsv output/candidate_pairs_sub1.tsv | tee -a "$LOG"
fi

# =============================================================
# PHASE 4: Extract 300K More Training Entities (~3.5 hours)
# =============================================================
log ""
log "--- PHASE 4: Extract 300K Additional Training Features ---"
$PY code/business_entity_resolution/src/build_features_full.py \
    --split train --k 100 --num-entities 300000 --seed 99 \
    --output experiments/train_features_extra300k.parquet 2>&1 | tee -a experiments/extraction_extra.log
PHASE4_EXIT=$?
if [ $PHASE4_EXIT -ne 0 ]; then
    log "PHASE 4 FAILED (exit $PHASE4_EXIT). Ensemble will use original 200K only."
else
    log "PHASE 4 DONE"
    ls -lh experiments/train_features_extra300k.parquet | tee -a "$LOG"
fi

# =============================================================
# PHASE 5: Train 3-Model Ensemble (~2 hours)
# =============================================================
log ""
log "--- PHASE 5: Train Ensemble ---"
$PY scripts/train_ensemble.py 2>&1 | tee -a experiments/ensemble_train.log
PHASE5_EXIT=$?
if [ $PHASE5_EXIT -ne 0 ]; then
    log "PHASE 5 FAILED (exit $PHASE5_EXIT). Using v2 LGB-only for K=200."
    echo "lgb_only||$(cat experiments/threshold_v2.txt)" > experiments/ensemble_config.txt
fi
log "PHASE 5 DONE"

# =============================================================
# PHASE 6: K=200 Test Inference with Ensemble → Submission 2 (~12.5h)
# =============================================================
log ""
log "--- PHASE 6: K=200 Inference (Submission 2) ---"
log "Ensemble config:"
cat experiments/ensemble_config.txt | tee -a "$LOG"
$PY code/business_entity_resolution/src/run_inference.py \
    --candidate-cap 200 2>&1 | tee -a experiments/inference_sub2.log
PHASE6_EXIT=$?
if [ $PHASE6_EXIT -ne 0 ]; then
    log "PHASE 6 FAILED with K=200 (exit $PHASE6_EXIT). Retrying K=100..."
    $PY code/business_entity_resolution/src/run_inference.py \
        --candidate-cap 100 2>&1 | tee -a experiments/inference_sub2_fallback.log
    PHASE6_EXIT=$?
fi
if [ $PHASE6_EXIT -eq 0 ]; then
    cp output/matching_results.tsv output/matching_results_sub2.tsv
    cp output/candidate_pairs.tsv output/candidate_pairs_sub2.tsv
    log "PHASE 6 DONE — Submission 2 ready."
    log "Submission 2 files:"
    wc -l output/matching_results_sub2.tsv output/candidate_pairs_sub2.tsv | tee -a "$LOG"
else
    log "PHASE 6 FAILED ALL ATTEMPTS."
fi

# =============================================================
# FINAL SUMMARY
# =============================================================
log ""
log "============================================================"
log "MASTER PIPELINE COMPLETE: $(date)"
log "============================================================"
log ""
log "=== Submission Files ==="
ls -lh output/matching_results_sub1.tsv output/candidate_pairs_sub1.tsv 2>/dev/null | tee -a "$LOG"
ls -lh output/matching_results_sub2.tsv output/candidate_pairs_sub2.tsv 2>/dev/null | tee -a "$LOG"
log ""
log "=== Model Files ==="
ls -lh experiments/lgbm_model_v2.txt experiments/xgb_model_v2.json experiments/catboost_model_v2.cbm 2>/dev/null | tee -a "$LOG"
log ""
log "=== Config ==="
cat experiments/ensemble_config.txt | tee -a "$LOG"
cat experiments/threshold_v2.txt | tee -a "$LOG"
log ""
log "ALL DONE. Ready for upload."
