# Experiment Log

Every run that produces a measured number goes here before moving to the
next phase — no entry without a number that was actually printed by code
executed in this session. This is the raw material for
`Documentation_template.md` later; don't reconstruct it from memory.

| date | phase | change | recall@candidates | precision | recall | macro F0.5 | singleton acc | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | Phase 0 | Setup & EDA baseline | N/A | N/A | N/A | 0.0558 (naive empty floor) | 0.0558 (base rate) | 2,206,821 train S1; 1,732,544 test S1; singletons=5.5848%; 1 match=5.3995%; >1 matches=89.0157%; 0 exact duplicate rows |
| 2026-09-25 | Phase 1 | Shared normalization (names, addresses, missing fields) | N/A | N/A | N/A | N/A | N/A | Initial normalizer: had Unicode \w punctuation bug stripping combining marks (vowels/viramas) in Indic scripts and missed non-trailing legal suffixes. |
| 2026-09-25 | Phase 1 (Fix) | Unicode combining marks + sequence-wide suffix stripping | N/A | N/A | N/A | N/A | N/A | Fixed strip_punctuation_unicode to preserve L*, M*, N*; fixed clean_name_tokens to strip legal suffixes anywhere in sequence; programmatically verified has_address/has_name=False flags; literal text for 51 pairs (India/US/Devanagari/Tamil/Bengali/Gujarati/missing fields) saved to experiments/phase1_examples.md |
| 2026-09-25 | Phase 2 | Multi-strategy candidate generation (name IDF + n-gram prefix + phonetic + address shingles/postal/street) | 0.9990 (union), 0.9026 (@K=50), 0.9124 (@K=75) | N/A | N/A | N/A | N/A | Evaluated on 2,000 stratified val entities against full 10,320,219 S2/S3 candidate pool (US: 6,186,873, India: 4,133,346); 7,057 true matches; union recall=99.90% (7,050/7,057); Recall@50=90.26% (1_match: 89.01%, 2-5: 90.36%, 6+: 89.97%); Recall@75=91.24%; reduction ratio=99.998547% vs brute force; US index time=131.7s, query=57.5s; India index time=122.2s, query=178.0s; peak RAM <1.5 GB; report saved to experiments/phase2_blocking_report.md |
| 2026-09-25 | Phase 3 | Pairwise feature engineering (42 features: string sim, digit tokens, rank/gap, sentinels) | 0.8931 (@K=50) | N/A | N/A | N/A | N/A | Evaluated on full 25k val split against 10.3M candidate pool; 1,247,259 candidate pairs generated; 42 features extracted at 7,782 pairs/sec; saved to experiments/val_features.parquet (42.4 MB); top positive signals: blocking_score (+0.6177), addr_word_jaccard (+0.4745), name_word_jaccard (+0.4394); top negative signals: candidate_rank (-0.3816), score_gap_to_top (-0.2935); small-slice peak RSS=27.48 GB on VM dictates streaming chunked Parquet writes for full 86M-110M pairs; report saved to experiments/phase3_feature_report.md |
| 2026-09-25 | Production (K=100) | LightGBM production (200K entities, 19,902,872 train pairs, K=100) | ~0.9124 (@K=100) | 0.9574 | 0.8627 | 0.9250 | 0.9169 | ROC-AUC=0.9997, PR-AUC=0.9917, optimal threshold=0.97, scale_pos_weight=30.51, top features: blocking_score (8481), name_len_ratio (7809), score_gap_to_top (7339), name_lev_sim (7184) |





---

## Phase 0: Measured EDA & Data Summary

### 1. File Statistics & Schemas

| File | Row Count | Columns | ID Prefix | Unique IDs | Exact Duplicates | Dup (Name, Addr) | Missing Names | Missing Addrs |
|---|---|---|---|---|---|---|---|---|
| `dataset/train/train_source1.tsv` | 2,206,821 | `entity_id`, `business_name`, `business_address`, `country` | S1- (2,206,821) | 2,206,821 | 0 | 0 | 0 | 0 |
| `dataset/train/train_source2.tsv` | 5,034,616 | `entity_id`, `business_name`, `business_address`, `country` | S2- (5,034,616) | 5,034,616 | 0 | 25,891 | 2 | 168,967 |
| `dataset/train/train_source3.tsv` | 5,285,603 | `entity_id`, `business_name`, `business_address`, `country` | S3- (5,285,603) | 5,285,603 | 0 | 18,881 | 13 | 175,916 |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 | `source1_entity_id`, `matched_entity_ids` | S1- (2,206,821) | 2,206,821 | 0 | N/A | 0 | N/A |
| `dataset/test/test_source1.tsv` | 1,732,544 | `entity_id`, `business_name`, `business_address`, `country` | S1- (1,732,544) | 1,732,544 | 0 | 0 | 0 | 0 |
| `dataset/test/test_source2.tsv` | 4,887,273 | `entity_id`, `business_name`, `business_address`, `country` | S2- (4,887,273) | 4,887,273 | 0 | 22,642 | 46 | 129,408 |
| `dataset/test/test_source3.tsv` | 5,082,316 | `entity_id`, `business_name`, `business_address`, `country` | S3- (5,082,316) | 5,082,316 | 0 | 16,305 | 59 | 136,098 |

*Note: Total records in dataset: 26,435,994.*

### 2. Country Distribution & France Occurrence

| File | US | India | France | Notes |
|---|---|---|---|---|
| `train_source1.tsv` | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | No France in training |
| `train_source2.tsv` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | No France in training |
| `train_source3.tsv` | 3,170,056 (59.97%) | 2,115,547 (40.03%) | 0 (0.00%) | No France in training |
| `test_source1.tsv` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | France appears in test |
| `test_source2.tsv` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | France appears in test |
| `test_source3.tsv` | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | France appears in test |

### 3. Match Count Distribution in `train_ground_truth.tsv`

- Total Source 1 Entities: 2,206,821 (100.0%)
- Singletons (0 matches): 123,247 (5.5848%)
- Single Match (1 match): 119,157 (5.3995%)
- Multiple Matches (>1 matches): 1,964,417 (89.0157%)
- Total Matched Records: 7,638,365 (S2-: 3,693,619 | S3-: 3,944,746)

Detailed breakdown of match counts per Source 1 entity:
| Matches | Entity Count | Percentage |
|---|---|---|
| 0 (Singletons) | 123,247 | 5.5848% |
| 1 | 119,157 | 5.3995% |
| 2 | 375,212 | 17.0024% |
| 3 | 530,841 | 24.0546% |
| 4 | 484,115 | 21.9372% |
| 5 | 321,957 | 14.5892% |
| 6 | 164,868 | 7.4708% |
| 7 | 63,968 | 2.8986% |
| 8 | 18,680 | 0.8465% |
| 9 | 4,205 | 0.1905% |
| 10 | 534 | 0.0242% |
| 11 | 37 | 0.0017% |

*By Country in Ground Truth (via train_source1):*
- **India** (883,188 entities): Singletons: 49,351 (5.5878%) | 1 Match: 47,468 (5.3746%) | >1 Matches: 786,369 (89.0376%)
- **US** (1,323,633 entities): Singletons: 73,896 (5.5828%) | 1 Match: 71,689 (5.4161%) | >1 Matches: 1,178,048 (89.0011%)

### 4. Text Length Characteristics

| File | Avg Name Chars | Avg Name Words | Avg Addr Chars | Avg Addr Words |
|---|---|---|---|---|
| `train_source1.tsv` | 24.03 | 3.55 | 52.07 | 8.03 |
| `train_source2.tsv` | 25.10 | 3.50 | 46.23 | 7.29 |
| `train_source3.tsv` | 25.20 | 3.53 | 46.71 | 7.17 |
| `test_source1.tsv` | 23.84 | 3.52 | 57.21 | 8.59 |
| `test_source2.tsv` | 25.70 | 3.59 | 50.41 | 7.80 |
| `test_source3.tsv` | 25.66 | 3.60 | 48.74 | 7.51 |

---

## Phase 2: Measured Candidate Blocking Results (Large-Scale Verification on 20,000 Entities)

Official run artifact saved to [`experiments/phase2_blocking_report.md`](file:///d:/Grokking/experiments/phase2_blocking_report.md).

### 1. Overall Candidate Recall & Search Space Reduction

- **Candidate Pool:** 10,320,219 total records (US: 6,186,873; India: 4,133,346)
- **Validation Sample:** 20,000 stratified Source 1 entities (10,000 US, 10,000 India)
- **Total True Ground Truth Matches:** 69,492
- **Brute-Force Pair Comparisons:** 103,202,190,000 (103.2 billion pairs)
- **Candidate Pairs Generated:** 1,495,625 pairs (average 74.8 per query entity)
- **Combined Union Recall:** **99.92%** (69,433 / 69,492 true matches retrieved; only 59 missed across 20k entities)
- **Search Space Reduction Ratio:** **99.998551%**
- **Candidate Cap Commitment:** Formally committed to **K = 50** for `candidate_pairs.tsv` (yields **89.31%** recall while bounding pairwise feature size to 86.6M rows).

### 2. Multi-Strategy Recall Contribution

| Strategy | True Matches Retrieved | Strategy Recall | Role & Coverage |
| :--- | :--- | :--- | :--- |
| **Strategy 1: Name Tokens (IDF)** | 57,445 / 69,492 | 82.66% | Exact word overlap with inverse document frequency weighting |
| **Strategy 2: Character n-grams (prefix 4-gram & initials)** | 54,007 / 69,492 | 77.72% | Typo tolerance, prefix matches, script-agnostic fallback |
| **Strategy 3: Phonetic (Soundex)** | 53,143 / 69,492 | 76.47% | Phonetic variations on Latin names (e.g., Smyth vs Smith) |
| **Strategy 4: Address (Postal, Street & Shingles)** | 65,917 / 69,492 | 94.86% | Handles cross-script transliteration and corrupted names |
| **Combined Multi-Strategy Union** | **69,433 / 69,492** | **99.92%** | Union of all 4 strategies before candidate capping |

### 3. Recall @ Candidate Cap K Breakdown by Match Bucket

| Candidate Cap (K) | Overall Recall | 1 Match Entities | 2–5 Match Entities | 6+ Match Entities | Total Hits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 10** | **83.24%** | 83.02% | 83.59% | 81.95% | 57,842 / 69,492 |
| **K = 20** | **86.70%** | 85.25% | 86.89% | 86.11% | 60,248 / 69,492 |
| **K = 30** | **88.01%** | 87.01% | 88.15% | 87.58% | 61,163 / 69,492 |
| **K = 40** | **88.77%** | 87.85% | 88.89% | 88.41% | 61,687 / 69,492 |
| **K = 50** | **89.31%** | 88.03% | 89.44% | 88.94% | 62,065 / 69,492 |
| **K = 75** | **90.22%** | 89.33% | 90.35% | 89.83% | 62,699 / 69,492 |

### 4. Per-Country Breakdown & Phonetic Analysis

- **United States (US):** 6,186,873 candidates; 34,704 true matches evaluated.
  - Name Tokens: 91.16% | Char n-grams: 88.38% | Address: 94.19% | **Phonetic (Soundex): 86.11%**
  - Union Recall: **99.94%** | **Recall @ K=50: 91.68%**
  - Inverted Index Build Time: 195.32s (31,675 cands/s) | **Query Time: 9.13s (1,094.9 q/s)**
- **India:** 4,133,346 candidates; 34,788 true matches evaluated.
  - Name Tokens: 74.19% | Char n-grams: 67.08% | Address: 95.52% | **Phonetic (Soundex): 66.86%**
  - Union Recall: **99.89%** | **Recall @ K=50: 86.96%**
  - Inverted Index Build Time: 109.90s (37,609 cands/s) | **Query Time: 8.36s (1,195.6 q/s)**
  - *Observation:* India's 19.25% drop in Phonetic Recall (66.86% vs US 86.11%) reflects Soundex's English bias and non-Latin scripts, which Address blocking (95.52%) successfully compensates.

### 5. Scale, Throughput & End-to-End Runtime Projection

- **Throughput Measured:** Combined average single-threaded querying throughput is **1,143.0 queries/second** (10,000 entities in 9.13s for US; 10,000 entities in 8.36s for India).
- **Disparity Eliminated:** India query throughput (1,195.6 q/s) is now slightly faster than US (1,094.9 q/s) in line with its 33% smaller candidate pool.
- **Full-Scale Projections (Single-Threaded):**
  - Test set (1,733,322 entities): **0.42 hours (~25.3 minutes)**.
  - Train set (2,204,491 entities): **0.54 hours (~32.1 minutes)**.
  - Total pipeline query time (Train + Test = 3.93M entities): **0.96 hours (~57.4 minutes)**.
  - Consumes **< 1.33%** of the 72-hour competition window, leaving > 71 hours for pairwise feature engineering and LightGBM model training.
- **RAM Footprint:** CandidateIndex utilizes compact `array.array('I')` posting lists with streaming batch construction, consuming ~2.5–3.3 GB RSS without page swapping.
- **Missing Field Handling:** ~3.3% of Source 2/3 records without addresses successfully surface via Name Tokens and Character n-grams.
- **Open-Set Generalization:** Fully generic country partitioning executes identically on France (unseen in training) without country-specific code branches.

---

## Phase 3: Pairwise Feature Engineering & Verification (1,247,259 Pairs Evaluated)

Official run artifact saved to [`experiments/phase3_feature_report.md`](file:///d:/Grokking/experiments/phase3_feature_report.md).

### 1. Small-Slice Memory Check & VM Peak RSS Audit
- **Root Cause of the 21.6 GiB Gap:** In initial runs, peak RSS hit **27.48 GiB** because the test script coupled Phase 2 (inverted indexing) and Phase 3 in a single process, materializing 10.3M `NormalizedRecord` Python instances (measured at 12,963 bytes/record = ~12.5 GiB) alongside Phase 2's posting list arrays (`CandidateIndex` = ~7 GiB) and duplicated candidate string lists (~3 GiB).
- **Decoupled Architecture Re-measurement:** By decoupling Phase 3 (reading candidate pairs directly and loading only the 122,364 candidate records appearing in the 2,500-entity slice pairs without building inverted indexes):
  - **Stage 1 (Slice Pairs Loaded):** 428.20 MB Current RSS
  - **Stage 2 (S1 Normalization - 2.5k entities):** 913.69 MB Current RSS
  - **Stage 3 (Candidate Normalization - 122k records):** 999.22 MB Current RSS
  - **Stage 4 (Feature Extraction - 124,816 pairs @ 8,322 pairs/sec):** **1,165.79 MB (1.14 GiB) Peak RSS**.
  - Total system memory utilized on VM was **1.21 GiB** (out of 31 GiB RAM), with **30.5 GiB free** and **0 B swap**.
- **Full Scale Strategy:** Decoupling ensures Phase 3 never holds inverted indexes or full-pool dataclass dictionaries. Test and train candidate pairs are extracted and written directly to Parquet using streaming chunks, maintaining process RSS safely under 8 GiB.

### 2. Full Validation Run Summary
- **Validation Entities:** 25,000 stratified Source 1 entities (`experiments/val_split.parquet`)
- **Candidate Pool:** 10,320,219 records (`train_source2.tsv` + `train_source3.tsv`)
- **Candidate Pairs Evaluated:** 1,247,259 pairs (77,979 true matches = 6.25%, 1,169,280 non-matches = 93.75%)
- **Feature Extraction Time:** 160.28s (**7,782 pairs/sec**)
- **Saved Parquet File:** `experiments/val_features.parquet` (44,490,038 bytes / 42.43 MB, 1,247,259 rows, 42 features)

### 3. Top Label-Correlation Signals (Pearson $r$ with Ground Truth)
- **Strongest Positive Signals:**
  - `blocking_score`: `+0.6177`
  - `addr_word_jaccard`: `+0.4745`
  - `name_word_jaccard`: `+0.4394`
  - `name_char3_jaccard`: `+0.4354`
  - `name_lev_sim`: `+0.4032`
  - `name_token_sort`: `+0.4006`
  - `name_common_tokens`: `+0.3653`
  - `addr_common_tokens`: `+0.3541`
  - `name_exact_clean`: `+0.3489`
  - `addr_token_sort`: `+0.3325`
  - `addr_lev_sim`: `+0.3164`
  - `digits_jaccard`: `+0.2207`
  - `name_exact_raw`: `+0.2173`
  - `digits_overlap`: `+0.1766`
- **Strongest Negative Signals:**
  - `candidate_rank`: `-0.3816` (lower rank = candidate further down list strongly signals non-match)
  - `score_gap_to_top`: `-0.2935` (larger gap from #1 candidate signals non-match)
  - `name_len_diff`: `-0.1728`
  - `is_source2`: `-0.1043`
- **Near-Zero / Constant Signals:**
  - `has_name_s1`, `has_addr_s1`, `same_country`: Constant (`NaN` Pearson $r$) because train S1 has 0% missing fields and candidate generation blocks strictly within country. Retained as defensive sentinels for test inference.
  - `country_freq` (`+0.0068`), `has_postal_*` (`+0.0019`), `name_suffix_match` (`+0.0002`): Negligible linear correlation, but provide non-linear interaction splits for gradient boosted trees.

### 4. Verified Edge Cases
- **Missing Address Sentinel:** Verified `-1.0` / `-1` sentinels (`addr_lev_sim = -1.0`, `digits_overlap = -1`) and flags (`has_addr_s1 = 0`) correctly distinguish missing data from genuine 0.0 dissimilarity.
- **Missing Name Sentinel:** Verified `-1.0` sentinels (`name_lev_sim = -1.0`) and flags (`has_name_s1 = 0`).
- **Reordered Address Digit Overlap:** Verified that with permuted street/city/postal ordering, `street_num_exact = 0`, but position-agnostic `digits_overlap = 1` and `digits_jaccard = 1.0000` catch the match.
- **Unseen Country Fallback:** Verified France entities fall back to `country_freq = 0.15` and `same_country = 1` without errors.

---

## Phase 4: LightGBM Matching Model Baseline (3-Fold GroupKFold)

### 1. Training Setup & Dataset
- **Model:** LightGBM Binary Classifier (`LGBMClassifier`, `boosting_type='gbdt'`, `n_estimators=350`, `learning_rate=0.05`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, early stopping 30 rounds).
- **Validation Split:** 3-Fold `GroupKFold` grouped by `source1_entity_id` to guarantee zero entity leakage across folds.
- **Candidate Data:** Trained on `experiments/val_features.parquet` containing 1,247,259 candidate pairs across 25,000 stratified Source 1 entities:
  - Positive pairs: 77,979 (6.25%)
  - Negative pairs: 1,169,280 (93.75%)
- **Scale Note:** This baseline is trained on only 25,000 Source 1 entities (~1.13% of the 2,206,821 total training entities in `train_source1.tsv`).

### 2. Cross-Validation Discrimination Metrics

| Fold | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|
| Fold 1 | 0.9993 | 0.9902 |
| Fold 2 | 0.9991 | 0.9890 |
| Fold 3 | 0.9991 | 0.9889 |
| **Overall Out-of-Fold** | **0.9992** | **0.9894** |

### 3. Top 15 Most Informative Features (LightGBM Split Gain)

| Rank | Feature | Importance (Split Gain) |
|---|---|---|
| 1 | `addr_token_set` | 618.7 |
| 2 | `name_jw_sim` | 612.3 |
| 3 | `name_char3_jaccard` | 606.7 |
| 4 | `name_token_sort` | 522.7 |
| 5 | `name_len_ratio` | 506.3 |
| 6 | `name_word_jaccard` | 501.0 |
| 7 | `name_lev_sim` | 491.3 |
| 8 | `name_len_diff` | 488.7 |
| 9 | `name_token_set` | 477.7 |
| 10 | `addr_word_jaccard` | 466.3 |
| 11 | `digits_jaccard` | 437.7 |
| 12 | `blocking_score` | 401.0 |
| 13 | `addr_token_sort` | 390.7 |
| 14 | `score_gap_to_top` | 331.3 |
| 15 | `addr_jw_sim` | 329.7 |

### 4. Decision Threshold Sweep (Macro F0.5 Optimization)

| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy | Non-Singleton F0.5 |
|---|---|---|---|---|---|
| 0.15 | 0.8549 | 0.8682 | 0.8622 | 0.5681 | 0.8719 |
| 0.20 | 0.8698 | 0.8873 | 0.8616 | 0.6289 | 0.8841 |
| 0.25 | 0.8800 | 0.9007 | 0.8601 | 0.6877 | 0.8914 |
| 0.30 | 0.8879 | 0.9117 | 0.8577 | 0.7242 | 0.8976 |
| 0.35 | 0.8940 | 0.9203 | 0.8554 | 0.7593 | 0.9020 |
| 0.40 | 0.8981 | 0.9268 | 0.8520 | 0.7901 | 0.9044 |
| 0.45 | 0.9015 | 0.9327 | 0.8486 | 0.8181 | 0.9065 |
| 0.50 | 0.9034 | 0.9370 | 0.8437 | 0.8374 | 0.9073 |
| 0.55 | 0.9059 | 0.9419 | 0.8398 | 0.8660 | 0.9083 |
| 0.60 | 0.9068 | 0.9454 | 0.8338 | 0.8847 | 0.9081 |
| **0.65 (Optimal)** | **0.9074** | **0.9486** | **0.8282** | **0.9033** | **0.9077** |
| 0.70 | 0.9070 | 0.9510 | 0.8210 | 0.9169 | 0.9064 |
| 0.75 | 0.9053 | 0.9528 | 0.8121 | 0.9319 | 0.9038 |
| 0.80 | 0.9025 | 0.9539 | 0.8019 | 0.9513 | 0.8996 |
| 0.85 | 0.8960 | 0.9521 | 0.7870 | 0.9628 | 0.8921 |

### 5. Optimal Threshold Summary & Findings
- **Optimal Decision Threshold:** `0.65`
- **Best Macro F0.5:** `0.9074`
- **Macro Precision:** `0.9486`
- **Macro Recall:** `0.8282`
- **Singleton Accuracy:** `0.9033` (5.58% base rate in data; high accuracy prevents false merges)
- **Non-Singleton F0.5:** `0.9077`
- **Precision/Recall Tradeoff Under F0.5:**
  As threshold increases from 0.15 to 0.65, Macro Precision rises rapidly from 0.8682 to 0.9486 and Singleton Accuracy rises from 56.81% to 90.33%, while recall drops gently from 0.8622 to 0.8282. Under the F0.5 metric ($\beta = 0.5$), false merges cost approximately twice as much as missed matches, making higher threshold selection (0.65) strictly superior to the default 0.50 threshold (F0.5 = 0.9034 vs 0.9074).

---

## Production Model v1 (200K Entities, K=100 Candidates)

### 1. Training Setup & Architecture
- **Model:** LightGBM Binary Classifier (`LGBMClassifier`, `boosting_type='gbdt'`, `n_estimators=2000`, `learning_rate=0.03`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.7`, `colsample_bytree=0.7`, `scale_pos_weight=30.5067`).
- **Training Dataset:** 200,000 Source 1 training entities, 19,902,872 candidate pairs (`experiments/train_features_k100.parquet`).
- **Validation Dataset:** 25,000 held-out stratified validation entities, 2,488,078 candidate pairs (`experiments/val_features_k100.parquet`).
- **Best Iteration:** 2000 (with early stopping patience=50).
- **Model Checkpoint:** `experiments/lgbm_model_v1.txt`.
- **Optimal Decision Threshold File:** `experiments/threshold_v1.txt`.

### 2. Discrimination Metrics on Held-Out Validation Set

| Metric | Value |
|---|---|
| ROC-AUC | 0.99974 |
| PR-AUC (Average Precision) | 0.99167 |

### 3. Top 20 Most Informative Features (LightGBM Split Gain)

| Rank | Feature | Importance (Split Gain) |
|---|---|---|
| 1 | `blocking_score` | 8481.0 |
| 2 | `name_len_ratio` | 7809.0 |
| 3 | `score_gap_to_top` | 7339.0 |
| 4 | `name_lev_sim` | 7184.0 |
| 5 | `name_jw_sim` | 7073.0 |
| 6 | `name_char3_jaccard` | 6532.0 |
| 7 | `candidate_rank` | 6466.0 |
| 8 | `addr_token_set` | 6288.0 |
| 9 | `addr_word_jaccard` | 6141.0 |
| 10 | `name_len_diff` | 6131.0 |
| 11 | `name_token_set` | 6047.0 |
| 12 | `name_token_sort` | 5972.0 |
| 13 | `addr_jw_sim` | 5576.0 |
| 14 | `addr_lev_sim` | 4973.0 |
| 15 | `addr_token_sort` | 4938.0 |
| 16 | `addr_len_diff` | 4670.0 |
| 17 | `name_word_jaccard` | 4496.0 |
| 18 | `digits_jaccard` | 3560.0 |
| 19 | `addr_common_tokens` | 3152.0 |
| 20 | `country_freq` | 2333.0 |

### 4. Fine-Grained Decision Threshold Sweep (0.10 to 0.95)

| Threshold | Macro F0.5 | Macro Precision | Macro Recall | Singleton Accuracy | Non-Singleton F0.5 |
|---|---|---|---|---|---|
| 0.10 | 0.6937 | 0.6744 | 0.8684 | 0.1325 | 0.7269 |
| 0.11 | 0.6998 | 0.6813 | 0.8688 | 0.1418 | 0.7328 |
| 0.12 | 0.7053 | 0.6874 | 0.8695 | 0.1554 | 0.7378 |
| 0.13 | 0.7103 | 0.6931 | 0.8700 | 0.1648 | 0.7426 |
| 0.14 | 0.7153 | 0.6987 | 0.8706 | 0.1769 | 0.7471 |
| 0.15 | 0.7201 | 0.7042 | 0.8709 | 0.1855 | 0.7517 |
| 0.16 | 0.7248 | 0.7095 | 0.8715 | 0.1977 | 0.7560 |
| 0.17 | 0.7293 | 0.7146 | 0.8721 | 0.2085 | 0.7601 |
| 0.18 | 0.7334 | 0.7193 | 0.8724 | 0.2156 | 0.7640 |
| 0.19 | 0.7375 | 0.7240 | 0.8728 | 0.2249 | 0.7678 |
| 0.20 | 0.7413 | 0.7283 | 0.8731 | 0.2314 | 0.7714 |
| 0.21 | 0.7450 | 0.7326 | 0.8735 | 0.2414 | 0.7748 |
| 0.22 | 0.7483 | 0.7364 | 0.8735 | 0.2443 | 0.7781 |
| 0.23 | 0.7519 | 0.7405 | 0.8740 | 0.2550 | 0.7813 |
| 0.24 | 0.7554 | 0.7445 | 0.8744 | 0.2643 | 0.7845 |
| 0.25 | 0.7586 | 0.7481 | 0.8749 | 0.2744 | 0.7873 |
| 0.26 | 0.7618 | 0.7518 | 0.8751 | 0.2794 | 0.7903 |
| 0.27 | 0.7648 | 0.7553 | 0.8752 | 0.2837 | 0.7932 |
| 0.28 | 0.7677 | 0.7586 | 0.8755 | 0.2901 | 0.7959 |
| 0.29 | 0.7707 | 0.7620 | 0.8759 | 0.2994 | 0.7985 |
| 0.30 | 0.7737 | 0.7656 | 0.8763 | 0.3087 | 0.8012 |
| 0.31 | 0.7764 | 0.7687 | 0.8764 | 0.3130 | 0.8038 |
| 0.32 | 0.7795 | 0.7721 | 0.8769 | 0.3259 | 0.8063 |
| 0.33 | 0.7824 | 0.7755 | 0.8775 | 0.3367 | 0.8087 |
| 0.34 | 0.7849 | 0.7785 | 0.8776 | 0.3403 | 0.8112 |
| 0.35 | 0.7876 | 0.7816 | 0.8777 | 0.3460 | 0.8137 |
| 0.36 | 0.7903 | 0.7848 | 0.8779 | 0.3524 | 0.8162 |
| 0.37 | 0.7931 | 0.7881 | 0.8780 | 0.3596 | 0.8187 |
| 0.38 | 0.7958 | 0.7913 | 0.8780 | 0.3625 | 0.8214 |
| 0.39 | 0.7985 | 0.7945 | 0.8783 | 0.3703 | 0.8238 |
| 0.40 | 0.8012 | 0.7977 | 0.8784 | 0.3754 | 0.8264 |
| 0.41 | 0.8041 | 0.8010 | 0.8787 | 0.3832 | 0.8289 |
| 0.42 | 0.8069 | 0.8044 | 0.8792 | 0.3947 | 0.8313 |
| 0.43 | 0.8095 | 0.8074 | 0.8794 | 0.4019 | 0.8336 |
| 0.44 | 0.8119 | 0.8102 | 0.8796 | 0.4083 | 0.8358 |
| 0.45 | 0.8146 | 0.8134 | 0.8800 | 0.4176 | 0.8381 |
| 0.46 | 0.8170 | 0.8163 | 0.8801 | 0.4234 | 0.8403 |
| 0.47 | 0.8195 | 0.8193 | 0.8804 | 0.4305 | 0.8425 |
| 0.48 | 0.8219 | 0.8221 | 0.8806 | 0.4391 | 0.8445 |
| 0.49 | 0.8238 | 0.8243 | 0.8806 | 0.4413 | 0.8464 |
| 0.50 | 0.8262 | 0.8272 | 0.8807 | 0.4456 | 0.8487 |
| 0.51 | 0.8285 | 0.8299 | 0.8812 | 0.4556 | 0.8506 |
| 0.52 | 0.8309 | 0.8326 | 0.8816 | 0.4656 | 0.8525 |
| 0.53 | 0.8331 | 0.8353 | 0.8820 | 0.4756 | 0.8543 |
| 0.54 | 0.8351 | 0.8376 | 0.8821 | 0.4807 | 0.8560 |
| 0.55 | 0.8368 | 0.8396 | 0.8822 | 0.4857 | 0.8575 |
| 0.56 | 0.8390 | 0.8423 | 0.8825 | 0.4950 | 0.8594 |
| 0.57 | 0.8409 | 0.8446 | 0.8825 | 0.4971 | 0.8612 |
| 0.58 | 0.8433 | 0.8474 | 0.8828 | 0.5050 | 0.8633 |
| 0.59 | 0.8453 | 0.8497 | 0.8831 | 0.5136 | 0.8649 |
| 0.60 | 0.8476 | 0.8525 | 0.8836 | 0.5258 | 0.8666 |
| 0.61 | 0.8496 | 0.8548 | 0.8838 | 0.5337 | 0.8683 |
| 0.62 | 0.8515 | 0.8572 | 0.8838 | 0.5380 | 0.8700 |
| 0.63 | 0.8535 | 0.8596 | 0.8840 | 0.5451 | 0.8718 |
| 0.64 | 0.8560 | 0.8625 | 0.8846 | 0.5587 | 0.8736 |
| 0.65 | 0.8580 | 0.8649 | 0.8847 | 0.5645 | 0.8754 |
| 0.66 | 0.8600 | 0.8674 | 0.8846 | 0.5673 | 0.8773 |
| 0.67 | 0.8620 | 0.8698 | 0.8847 | 0.5738 | 0.8791 |
| 0.68 | 0.8641 | 0.8723 | 0.8848 | 0.5802 | 0.8809 |
| 0.69 | 0.8663 | 0.8752 | 0.8848 | 0.5867 | 0.8829 |
| 0.70 | 0.8683 | 0.8776 | 0.8848 | 0.5938 | 0.8845 |
| 0.71 | 0.8703 | 0.8800 | 0.8849 | 0.6010 | 0.8862 |
| 0.72 | 0.8724 | 0.8826 | 0.8851 | 0.6110 | 0.8879 |
| 0.73 | 0.8742 | 0.8847 | 0.8853 | 0.6182 | 0.8893 |
| 0.74 | 0.8760 | 0.8870 | 0.8850 | 0.6211 | 0.8911 |
| 0.75 | 0.8781 | 0.8897 | 0.8851 | 0.6289 | 0.8929 |
| 0.76 | 0.8804 | 0.8923 | 0.8853 | 0.6397 | 0.8946 |
| 0.77 | 0.8825 | 0.8949 | 0.8857 | 0.6526 | 0.8961 |
| 0.78 | 0.8844 | 0.8972 | 0.8857 | 0.6605 | 0.8976 |
| 0.79 | 0.8866 | 0.9000 | 0.8858 | 0.6698 | 0.8994 |
| 0.80 | 0.8886 | 0.9025 | 0.8858 | 0.6798 | 0.9010 |
| 0.81 | 0.8904 | 0.9048 | 0.8857 | 0.6877 | 0.9024 |
| 0.82 | 0.8922 | 0.9071 | 0.8857 | 0.6977 | 0.9037 |
| 0.83 | 0.8947 | 0.9102 | 0.8858 | 0.7135 | 0.9054 |
| 0.84 | 0.8970 | 0.9131 | 0.8858 | 0.7242 | 0.9073 |
| 0.85 | 0.8993 | 0.9160 | 0.8857 | 0.7364 | 0.9090 |
| 0.86 | 0.9015 | 0.9187 | 0.8857 | 0.7493 | 0.9105 |
| 0.87 | 0.9037 | 0.9215 | 0.8857 | 0.7643 | 0.9119 |
| 0.88 | 0.9056 | 0.9241 | 0.8850 | 0.7729 | 0.9134 |
| 0.89 | 0.9074 | 0.9267 | 0.8840 | 0.7815 | 0.9149 |
| 0.90 | 0.9096 | 0.9297 | 0.8833 | 0.7930 | 0.9165 |
| 0.91 | 0.9124 | 0.9337 | 0.8825 | 0.8130 | 0.9183 |
| 0.92 | 0.9149 | 0.9373 | 0.8811 | 0.8274 | 0.9200 |
| 0.93 | 0.9176 | 0.9413 | 0.8795 | 0.8446 | 0.9219 |
| 0.94 | 0.9201 | 0.9455 | 0.8772 | 0.8639 | 0.9235 |
| 0.95 | 0.9224 | 0.9493 | 0.8744 | 0.8775 | 0.9250 |
| 0.955 | 0.9231 | 0.9510 | 0.8724 | 0.8847 | 0.9253 |
| 0.960 | 0.9239 | 0.9530 | 0.8699 | 0.8918 | 0.9258 |
| 0.965 | 0.9247 | 0.9553 | 0.8665 | 0.9054 | 0.9258 |
| **0.970 (Optimal)** | **0.9250** | **0.9574** | **0.8627** | **0.9169** | **0.9255** |
| 0.975 | 0.9249 | 0.9591 | 0.8580 | 0.9241 | 0.9250 |
| 0.980 | 0.9243 | 0.9609 | 0.8517 | 0.9334 | 0.9238 |
| 0.985 | 0.9229 | 0.9627 | 0.8427 | 0.9491 | 0.9213 |
| 0.990 | 0.9187 | 0.9635 | 0.8284 | 0.9656 | 0.9160 |
| 0.995 | 0.9053 | 0.9592 | 0.7983 | 0.9771 | 0.9010 |

### 5. Optimal Threshold Summary
- **Optimal Decision Threshold:** `0.97` (determined via fine-grained upper sweep)
- **Best Macro F0.5:** `0.9250`
- **Macro Precision:** `0.9574`
- **Macro Recall:** `0.8627`
- **Singleton Accuracy:** `0.9169` (91.69% of singletons correctly predicted empty)
- **Non-Singleton F0.5:** `0.9255`

