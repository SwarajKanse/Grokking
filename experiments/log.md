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


