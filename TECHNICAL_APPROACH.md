# TECHNICAL_APPROACH.md — Build Plan

This is the phased plan to execute. Each phase has a goal and a check —
don't move to the next phase until the check passes, and log the result in
`experiments/log.md` per `AGENTS.md`.

Two tiers: **Tier 1 (Phases 0–9)** gets you a correct, honestly-validated,
competitive baseline — this alone is enough for a strong result. **Tier 2
(Phases 10–15)** is what you layer on afterward to push for the top of the
leaderboard specifically. Do not start Tier 2 until Tier 1 has produced one
real, measured, logged score end-to-end — techniques stacked on an
unverified baseline just produce a more confident-looking wrong answer.

## Why this approach
The task decomposes cleanly into candidate generation (blocking) followed by
pairwise classification — the standard, well-tested pattern for this exact
shape of problem:
- Foursquare ran a near-identical Kaggle competition in 2022 ("Foursquare
  Location Matching"): match POI records across noisy sources by name and
  address, at over a million records, judged on getting duplicates right
  without over-merging. It drew over 22,000 submissions, and the
  high-placing pattern there was multi-strategy candidate generation
  followed by a gradient-boosted-tree classifier over engineered pairwise
  similarity features — essentially this problem with coordinates added.
- On the academic Amazon-Google entity-matching benchmark, a plain Random
  Forest over similarity features scores competitively (around 79 F1) with
  specialized deep entity-matching architectures, and the only approaches
  that clearly beat it are large-LLM few-shot methods — which aren't
  available here, since external API calls are banned and this challenge
  caps model size and license. That's a reasonable signal that a
  well-engineered gradient-boosted-tree pipeline isn't a compromise choice
  for this kind of task, it's a legitimately strong one.
- The general "candidate generation → LightGBM classifier" recipe also won
  other recent, similarly-shaped ranking/matching competitions, reinforcing
  it as the dependable default rather than something exotic.

# Tier 1 — a correct, validated baseline

## Findings from the measured Phase 0 run (2026-09-25) — read before Phase 2
These are cross-checked, internally-consistent numbers from the actual data,
not assumptions — they change specifics in Phases 2, 3, and 6 below.
- **89.02% of train Source 1 entities have >1 true match** (avg 3.67 matches
  per non-singleton entity, up to 11). Only 5.58% are singletons and 5.40%
  have exactly 1 match. Blocking's candidate cap must be sized against this
  real distribution, not an assumed "mostly 1:1" shape — a small top-K would
  silently drop real matches for the majority of entities.
- **Country mix shifts between splits, not just "France is added":** train
  is ~60% US / 40% India; test is ~38% US / 47% India / 15% France. A
  validation split drawn from train's natural mix will misjudge performance
  on test's actual mix even setting France aside.
- **Scale:** ~26.4M records total across all files (test alone is ~11.7M
  across its three files); brute-force Source1×(Source2+Source3) comparison
  on the test side is ~17.3 trillion pairs. This is well over 10x the size
  of the Foursquare competition referenced below — blocking and feature
  computation need real engineering care (vectorized/indexed operations),
  not just a sound algorithm, or Phase 2 won't finish inside the challenge
  window.
- **Running on Azure (`Standard_D16s_v5`, 16 vCPU / 64 GiB) via an
  existing, already-verified Azure Education credit — not the 16GB
  laptop.** See `AZURE_SETUP.md`. The credit pool is large enough (~12,000+
  hours at this VM's rate) that the earlier local-vs-cloud time tradeoff
  no longer applies; size up to `Standard_E16s_v5` (128 GiB) without much
  hesitation if Phase 3's memory check below suggests it's needed, rather
  than fighting it with chunking discipline that local hardware would have
  forced. The Phase 3 memory-check step is still good practice regardless
  of where it runs — it's cheaper to catch a memory problem on a small
  slice than on a multi-hour full run, cloud or not.
- **Missing values are real, not hypothetical:** ~3.3% of Source 2/3
  records are missing `business_address` (comparable rate in train and
  test), and a small number are missing `business_name` outright. Source 1
  has zero missing fields in either split. Normalization, blocking, and
  feature code all need an explicit path for this, not an assumption every
  field is populated.

## Phase 0 — Setup & EDA
- Load all files with `sep="\t"`. Sanity-check row counts, id prefixes, and
  country value counts per split (confirm France only appears in test).
- Compute base rates: singleton fraction, match-count distribution (how
  many Source 1 entities have 1 vs. many matches), country mix, average
  name/address length, duplicate rows.
- **Check:** state, from data, roughly what fraction of train Source 1
  entities are singletons. That number is your naive "predict nothing"
  F0.5 floor and a sanity check against every later prediction.

## Phase 1 — Normalization
Build one shared normalization function used everywhere (feature building,
blocking, and inference) — inconsistency here is a classic silent bug.
- Names: casefold, Unicode NFKC normalize, strip punctuation, expand a
  legal-suffix dictionary (Corp/Corporation, Pvt/Private, Ltd/Limited,
  Inc/Incorporated, `&`/and), collapse whitespace.
- Addresses: casefold, expand common abbreviations (Rd/Road, St/Street,
  Apt/Apartment...), pull out a trailing digit-group as a loose "postal
  code" signal without assuming a fixed length — requiring exactly 5 or 6
  digits breaks the moment you hit an unseen country's format — tokenize.
- Keep both raw and normalized text. Some noise (landmark references,
  transliteration) is easier to catch on raw tokens than after aggressive
  normalization has stripped it away.
- Handle missing fields explicitly: ~3.3% of Source 2/3 records have no
  `business_address`, and a few have no `business_name`. Decide and encode
  a clear convention (e.g. empty normalized string + an explicit
  `has_address` / `has_name` boolean flag) rather than letting nulls
  propagate into string-similarity functions downstream.
- **Do not strip Unicode "non-word" characters with a bare `\w`-based
  regex.** Devanagari, Tamil, and other Indic-script vowel signs and the
  virama are Unicode combining marks (category Mn) — they are not letters
  on their own and get excluded by `\w`, so a naive
  `re.sub(r'[^\w\s]', ' ', text)`-style punctuation strip will blank out
  every vowel sign in a native-script name while leaving the bare
  consonants behind, destroying the string. Verify any punctuation-stripping
  step explicitly against real native-script examples (not just Latin
  accented characters, which behave differently) before trusting it.
- **Check:** eyeball 30–50 known-matching training pairs before/after
  normalization; confirm it isn't erasing the signal that made them match.
  Separately, confirm the pipeline runs cleanly on a sample that includes
  real missing-address and missing-name rows pulled from the actual data.

## Phase 2 — Blocking / candidate generation
This sets the recall ceiling for the entire pipeline — nothing downstream
can recover a match that was never generated as a candidate. Use several
cheap strategies and take the union, then measure combined recall against
ground truth before doing anything else:
1. **Token-overlap inverted index** — index Source 2/3 by normalized name
   tokens (and separately by address tokens); for each Source 1 entity,
   pull records sharing enough tokens.
2. **Character n-gram / MinHash-LSH or TF-IDF cosine kNN** — robust to
   typos and transliteration, and language-agnostic, so it's the piece most
   likely to still work well on French records.
3. **Phonetic key on name** (Soundex / NYSIIS / Metaphone) as a
   supplementary bucket, not the sole strategy — phonetic algorithms are
   English-tuned and will underperform on French names.
4. **Postal / locality token block** — loose digit-group or city-token
   match on address, as another supplementary bucket.
- For the ~3.3% of Source 2/3 records missing `business_address`, the
  address-based buckets (3 and 4) can't fire — make sure those records can
  still surface through the name-based buckets (1 and 2) rather than
  falling out of candidate generation entirely because one field is empty.
- Union all buckets, dedupe, and cap candidates per Source 1 entity. Size
  the cap against the measured match distribution, not a guess — with up to
  11 true matches on some entities and 89% having more than one, a cap
  under roughly 30-50 will start cutting off real matches for a meaningful
  slice of entities. Confirm this concretely in the check below rather than
  picking a round number.
- **Check (the single most important number in the whole pipeline):** on a
  held-out validation split, what fraction of true matches survive into the
  candidate set (recall@candidates)? Compute this both overall and broken
  out by true match-count bucket (singleton / 1 / 2-5 / 6+) — a high
  overall average can hide poor recall specifically on the high-match-count
  entities, which is where a too-small cap would bite first. Push this as
  high as practically possible; every point lost here is unrecoverable
  later. Also record the reduction ratio (candidates vs. brute-force
  Source1 × [Source2+Source3]) for the methodology doc.
- **Check — scale, not just recall:** log wall-clock time and peak memory
  for the full blocking pass on the real train and test files, not a
  sample. At ~26M total records and ~17 trillion possible test-side
  comparisons, a naive per-pair Python loop will not finish. Confirm the
  implementation uses vectorized or indexed operations (inverted index
  lookups, sparse matrix operations, hashed joins) before trusting the
  recall number — a version that would time out on the real submission run
  isn't actually done, regardless of what it measures on a sample.
  Measuring wall-clock time on a validation sample is not enough on its
  own — explicitly project that per-entity rate to the full train (~2.2M)
  and test (~1.7M) entity counts and compare the total against the time
  actually left in the challenge window before treating this phase as
  done. A sample-level number that looks fine can still project to a
  full-scale runtime that doesn't fit — that's the number that matters.
  If per-country (or other partition) rates differ substantially, project
  and report each partition separately rather than one blended average, so
  a slow partition doesn't hide behind a fast one.

## Phase 3 — Pairwise feature engineering
At a candidate cap in the tens per entity and ~1.7-2.2M Source 1 entities
per split, this is a feature table in the tens to low hundreds of millions
of rows. Compute features with vectorized/batched operations (pandas/numpy
vector ops, or a columnar tool if pandas gets too slow), not row-wise
`.apply()` over individual pairs — that will not finish at this scale.

**Memory check, first — before writing the full feature set.** Running
locally on 16GB RAM. Before building out all the features below, run a
small end-to-end slice (e.g. one country partition, or a 5-10% sample of
candidates) with just 3-4 representative features, and measure actual
peak RSS, not an estimate — then project that to the full row/column count
the same way Phase 2's runtime projection was done. If the projection
doesn't comfortably fit in 16GB with headroom for LightGBM training
alongside it, use chunked/streaming construction (process and write
Parquet in batches rather than building one in-memory DataFrame) or
downcast dtypes (float32/int8 where the range allows) before scaling up —
confirm the fix with a re-measurement, not just the fact that it seems
like it should help.

For every (Source 1, candidate) pair surviving Phase 2, compute:
- **Name**: Levenshtein / edit distance (raw + normalized), Jaro-Winkler,
  token-set and token-sort ratio, Jaccard on word tokens and on character
  q-grams, TF-IDF cosine, common-token count, length ratio/diff,
  suffix-stripped exact-match flag. Note from Phase 1 review: the suffix
  dictionary match is exact-string, so a suffix word that itself picked up
  injected accent noise (e.g. "Límited" vs "Limited") won't be recognized,
  making the `clean` field inconsistent between two records that actually
  match. If this feature's importance looks weak, check for this before
  assuming the feature itself is uninformative.
- **Address**: the same string-similarity family on the full normalized
  address; city/locality token overlap. For numeric components, don't rely
  only on position-typed fields (postal vs. street number) — Phase 1's
  examples showed the same digit group (e.g. a flat/plot number) gets
  tagged differently depending on where it lands after address-component
  reordering, which the problem statement explicitly calls out as expected
  noise. Add a position-agnostic feature too: does the *set* of digit-groups
  extracted from each side overlap at all, regardless of which field each
  one landed in. Keep the position-typed exact-match/edit-distance features
  as well — just don't let them be the only numeric signal.
- **Meta**: candidate rank/score within its Source 1 entity's bucket,
  number of candidates for that entity, whether the pair was surfaced by
  more than one blocking strategy (a strong precision signal —
  corroborated candidates are more trustworthy).
- **Country**: same-value-on-both-sides flag, plus country as a categorical
  feature via something that degrades gracefully on an unseen category
  (e.g. frequency encoding with a default fallback), not naive one-hot.
- **Missing fields:** every address- or name-based feature needs a defined
  behavior when one side is missing that field (e.g. a sentinel value plus
  the `has_address`/`has_name` flag from Phase 1 as its own feature) rather
  than letting a null silently become `NaN` or `0` in a way indistinguishable
  from a genuine low-similarity score.
- **Check:** compute feature importances or at least label correlation on a
  validation fold before moving on — drop anything that's pure noise.

## Phase 4 — Matching model
- Baseline: LightGBM (or XGBoost/CatBoost) binary classifier, label = true
  match from ground truth, trained on the Phase 3 features. This is the
  default final model — self-trained, no license question to resolve.
- Negative sampling: candidates from Phase 2 that aren't true matches are
  your negatives; if a Source 1 entity has a large candidate set, consider
  down-weighting or subsampling easy negatives so the model doesn't just
  learn to reject obviously-wrong pairs.
- Optional secondary signal: a small MIT/Apache-2.0 sentence-embedding
  model (≤8B params) run once over normalized name+address strings, with
  cosine similarity added as one more feature into the same GBM. Cheap to
  add — don't build a separate end-to-end neural pipeline around it unless
  the GBM baseline is already solid and there's time left over.
- **Check:** cross-validate with a grouped split (group by Source 1 entity,
  so no candidate pair for the same entity leaks across train/val) and
  report AUC / PR-AUC before touching thresholds.

## Phase 5 — Thresholding & singleton handling
- F0.5 is threshold-sensitive and precision-heavy: sweep the decision
  threshold on a validation split and pick the value that maximizes macro
  F0.5, not accuracy or plain F1.
- For each Source 1 entity: keep candidates above threshold; if none clear
  it, predict empty (singleton). Add explicit "no-match" signal features —
  best score, gap between best and second-best, count of candidates above a
  lower bar — since correctly predicting singletons is worth as much as any
  other entity, and false merges on them are penalized directly.
- Sanity-check threshold behavior isn't secretly tuned only to US/India
  patterns even though the logic itself isn't country-branched.
- **Check:** report validation macro F0.5 split by (a) true singletons and
  (b) true non-singletons separately — a strong overall score can hide a
  weak singleton score under this metric.

## Phase 6 — Local validation harness
- Implement the exact F0.5 formula from `PROBLEM_BRIEF.md` as a local
  scorer over a held-out split of the training ground truth, so changes get
  evaluated without spending a leaderboard submission.
- Stratify by singleton/non-singleton, and by country — but weight the
  reported macro F0.5 toward test's actual country mix (measured: ~38%
  US / 47% India / 15% unseen-country), not train's natural mix (measured:
  ~60% US / 40% India). Either resample the held-out split to that ratio,
  or compute per-country scores and combine them with test's weights
  instead of trusting a flat average over train's proportions — otherwise
  the harness overweights US performance relative to what the leaderboard
  will actually reward. There's no French ground truth to validate the 15%
  directly; treat that portion's expected performance as informed by the
  Phase 7 stress test below, not by this harness.
- Treat this harness as the source of truth for every iteration; only push
  to the real leaderboard to confirm the harness and public leaderboard
  agree, and at deliberate submission checkpoints.

## Phase 7 — Generalizing to an unseen country (France)
- Since France never appears in training, explicitly stress-test: does the
  pipeline run cleanly if a validation fold's country is synthetically
  relabelled to a placeholder unseen string? Any code path that only works
  for `{US, India}` will break silently, or quietly score worse, here.
- Prefer blocking keys and features that are inherently script/language
  agnostic (character n-grams, edit distance) over ones tuned to
  English/Indian conventions (English legal-suffix dictionaries, Soundex).
- Keep any suffix/abbreviation dictionary as a plug-in list, not baked into
  control flow — makes it easy to both extend it and prove the pipeline
  degrades gracefully without a matching entry.

## Phase 8 — Error analysis loop
- Pull the worst false positives (shouldn't have merged) and false
  negatives (should have merged, but blocking missed it or the classifier
  scored it too low) from validation.
- False positives cost the F0.5 score twice as much as false negatives —
  prioritize whichever pattern is actually driving the loss.
- Iterate Phases 2–5 based on what's found; log each change and its effect
  on local F0.5.

## Phase 9 — Output & packaging
- Generate `candidate_pairs.tsv` directly from the exact candidate set fed
  to the Phase 4 model at inference — not an earlier, looser blocking pass.
- Generate `matching_results.tsv` from the thresholded model output.
- Run `utils/validate_submission.py` against both; fix anything it flags
  before treating the run as submission-ready.
- Fill in the organizers' `Documentation_template.md` as you go —
  methodology, blocking strategy plus the recall/reduction numbers from
  Phase 2, model architecture and features from Phase 3–4, and anything
  notable from Phase 7–8 — rather than reconstructing it from memory later.
- Assemble the final zip exactly matching the structure in
  `PROBLEM_BRIEF.md`.

# Tier 2 — pushing for the top of the leaderboard

Only start here once Tier 1 has produced one real, end-to-end, logged score.
Each of these should be validated against the Phase 6 harness and kept only
if it measurably helps — added complexity that doesn't move the number is a
liability, not progress, under the grounding rules in `AGENTS.md`.

## Phase 10 — Ensembling & stacking for lower-variance precision
- Train LightGBM, XGBoost, and CatBoost on the same Phase 3 features with
  different random seeds / bagging; blend their probability outputs
  (simple average, or a small logistic-regression meta-learner) before
  thresholding. Different tree implementations disagree most on borderline
  pairs, which is exactly where F0.5's precision penalty bites hardest, so
  blending reduces variance where it matters most.
- **Check:** the blend must beat every individual model on the Phase 6
  harness before you adopt it.

## Phase 11 — Iterative hard-negative mining
- After the first working classifier, run it over the full Phase 2
  candidate set and pull out the false positives it's most confident
  about — the hard negatives blocking generated but the model wrongly
  trusts.
- Fold them back into training as explicit negative examples and retrain.
  Repeat once or twice; diminishing returns set in fast, so log the harness
  score after each round and stop once it plateaus.
- **Check:** precision should visibly improve round over round. If it
  doesn't, spot-check the mined examples by hand — they may be genuinely
  ambiguous or mislabeled rather than truly hard.

## Phase 12 — Borderline re-ranking (optional, only if time remains)
- Identify candidate pairs whose blended score sits in a narrow band around
  the decision threshold — these are the pairs actually deciding the
  leaderboard score, since anything far from the threshold was already
  going to be classified correctly either way.
- Optionally re-score just that narrow band with a small MIT/Apache-2.0
  cross-encoder or embedding model (≤8B params) as a second opinion, and
  use agreement/disagreement with the GBM as an extra feature or
  tie-breaker.
- **Check:** measure this against the harness restricted to exactly the
  borderline band — that's the only slice this stage can move at all.

## Phase 13 — Calibration & a dedicated singleton meta-classifier
- Calibrate the final blended score (Platt scaling or isotonic regression)
  so the Phase 5 threshold behaves consistently rather than being an
  artifact of one fold.
- Consider a second, small classifier trained purely on per-entity
  aggregate features (best candidate score, score gap to second-best,
  candidate count, how many blocking strategies corroborated the top
  candidate) whose only job is "does this Source 1 entity have any match at
  all." Its decision boundary can be tuned independently of the pairwise
  classifier's — worthwhile because singleton correctness is worth as much
  as everything else combined and deserves its own precision/recall
  trade-off.
- **Check:** report singleton accuracy and non-singleton F0.5 separately
  before and after — it should move singleton accuracy without hurting
  non-singleton performance.

## Phase 14 — Validation discipline under a tight submission budget
- After your first real leaderboard submission, compare the public score
  against what the Phase 6 harness predicted. A close match means the
  harness can be trusted for the rest of the challenge; a meaningful gap
  means investigate before spending more submissions on it.
- Keep a single "current best" artifact (config + score) and require every
  subsequent submission to have beaten it locally first, by a margin larger
  than the harness's own fold-to-fold variance — never submit on noise.
- Reserve at least 2–3 of the 15 total submissions for the final day, after
  ensembling and calibration work is done, rather than spending the whole
  budget early on incremental blocking tweaks.

## Phase 15 — Error taxonomy (sharpens Phase 8)
Bucket false positives/negatives by pattern instead of reviewing them as an
undifferentiated pile, so fixes target the actual failure mode:
- transliteration / script variation missed by blocking
- landmark-only address with no shared tokens
- legal-suffix or DBA name variant scored too low
- two distinct businesses at the same address/building over-merged
- punctuation/abbreviation noise defeating exact-token features
Track the count per bucket per iteration — a bucket that isn't shrinking
after a fix aimed at it means the fix didn't address the real cause.

## Optional — pseudo-labeling for the unseen country
France has zero labelled examples. If time and validation budget allow:
take the highest-confidence test predictions (ensemble members agree, score
far from threshold in either direction), fold them back into training as
pseudo-labels restricted to French records, retrain, and re-validate.
Treat this strictly as an experiment: verify on the Phase 6 harness that it
actually helps before relying on it — pseudo-labeling can just as easily
entrench an existing blind spot as fix one, and a claim that it helped
needs the same measured-artifact standard as everything else in this repo.

## Reference reading (background only, not required)
- Classical toolkits worth knowing the shape of, even if you build custom:
  Python `recordlinkage`, `dedupe`, and the UK Ministry of Justice's
  `splink` library all implement blocking plus Fellegi-Sunter or ML-based
  pairwise scoring for exactly this class of problem — useful as a sanity
  check on your own pipeline's structure, not necessarily as a dependency.
- Academic entity-matching names worth recognizing for extra feature ideas:
  DeepMatcher, Ditto (pretrained-LM entity matching), Magellan /
  py_entitymatching. Their feature families — attribute-level similarity
  plus a learned classifier — are exactly what Phase 3–4 implements by hand.
