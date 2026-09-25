# Amazon ML Challenge 2026: Business Entity Resolution

This repository contains our end-to-end machine learning solution for the **Amazon ML Challenge 2026** (Business Entity Resolution).

## Objective
Match noisy business entity records (business name and business address) from `train_source1.tsv` / `test_source1.tsv` across two candidate catalogs (`source2` and `source3`), maximizing the competition evaluation metric: **Macro F0.5 Score**.

## Project Architecture
```
Grokking/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalize.py            # Phase 1: Unicode-preserving text normalization
│       │   ├── split.py                # Stratified validation sampling
│       │   ├── blocking.py             # Phase 2: Multi-strategy candidate generation
│       │   └── evaluate_phase2.py      # High-scale validation harness
│       ├── README.md
│       └── requirements.txt
├── experiments/
│   ├── log.md                          # Measured experiment logs and baseline records
│   ├── phase1_examples.md              # Phase 1 inspected normalization samples
│   ├── phase2_blocking_report.md       # Phase 2 20,000-entity blocking evaluation report
│   └── val_split.parquet               # 25,000-entity stratified validation set
├── output/                             # Generated submission files
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
├── references/                         # Guidelines, problem brief, and starter notes
├── utils/
│   └── validate_submission.py          # Official organizer format validator
├── AGENTS.md                           # Strict engineering constraints & protocol
├── AWS_SETUP.md                        # AWS infrastructure instructions
├── Documentation_template.md           # Submission documentation template
├── PROBLEM_BRIEF.md                    # Problem specification & data contract
├── TECHNICAL_APPROACH.md               # Phased build plan
└── requirements.txt                    # Project dependencies
```

## Highlights from Measured Milestones
- **Phase 0 (Setup & EDA):** Analyzed 26.4M records across 7 files. Mapped exact match distributions (5.58% singletons, 5.40% 1-match, 89.02% multi-match). Confirmed 100% hard country partitioning.
- **Phase 1 (Normalization):** Full Unicode preservation of Indic scripts (Devanagari, Tamil, Bengali, Gujarati combining marks) + tightened legal suffix stripping.
- **Phase 2 (Candidate Generation):** Multi-strategy inverted index achieved **99.92% union recall** on 20,000 entities across 103.2 billion comparisons, with single-threaded querying at **1,143 queries/second** (full test set retrieval in ~25.3 minutes).
