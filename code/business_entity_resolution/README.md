# Business Entity Resolution Pipeline
Amazon ML Challenge 2026

High-precision, country-partitioned multi-strategy entity resolution pipeline for noisy business entity matching across heterogeneous data sources.

## Pipeline Structure
- `src/normalize.py`: Unicode-preserving text normalization, Indic script protection, and legal suffix extraction.
- `src/split.py`: Stratified held-out validation split preserving base-rate match distributions.
- `src/blocking.py`: Multi-strategy candidate generation (Name tokens, character n-grams, Soundex, address shingles & postal codes).
- `src/evaluate_phase2.py`: Large-scale validation benchmark measuring recall@K and throughput.

## Reproduction Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate stratified validation split
python src/split.py

# 3. Run candidate generation & evaluation benchmark
python src/evaluate_phase2.py
```
