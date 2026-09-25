# PROBLEM_BRIEF.md — Business Entity Resolution (condensed spec)

## The task
Given business records from three independent, noisy sources, find — for
every Source 1 entity — all matching records in Source 2 and Source 3. A
Source 1 entity may match 0, 1, or many records. Only `business_name` and
`business_address` are available; there is no shared id across sources.

## Data files
```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```
All tab-separated. Read with `sep="\t"` explicitly — a default `read_csv`
silently collapses each row into one column because both addresses and id
lists contain commas.

### Source files (`*_source{1,2,3}.tsv`)
| column | notes |
|---|---|
| entity_id | prefixed `S1-`, `S2-`, or `S3-`; a record's source is its prefix plus which file it's in — there's no separate source column |
| business_name | abbreviations, legal-suffix variants, typos, transliterations |
| business_address | partial, abbreviated, landmark-based, missing components, reordered |
| country | open string label. Train = {US, India}. **Test adds France, unseen in train.** Never hardcode to {US, India}. |

### Ground truth (`train_ground_truth.tsv`)
| column | notes |
|---|---|
| source1_entity_id | a Source 1 id |
| matched_entity_ids | comma-separated Source 2/3 ids; empty = singleton (no match) |

One row per Source 1 entity — this mirrors exactly what you submit.

## Noise to expect
- **Names**: Corp/Corporation, Pvt/Private, Ltd/Limited, DBA/trade names,
  `&` vs `and`, word-order swaps, typos.
- **Addresses**: Rd/Road, St/Street, transliteration variants, missing
  PIN/state, landmark references ("Near SBI ATM"), municipal numbering,
  component reordering.

## Outputs you produce
Both go in `output/`, tab-separated, no quoting on the comma-joined id lists.

**`matching_results.tsv`** — the only file scored on the leaderboard.
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**`candidate_pairs.tsv`** — not scored, but audits blocking quality; must be
the actual candidate set your matching model scored at inference (the last
stage, not an earlier loose pass). Every id in `matching_results.tsv` must
appear here.
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
```

**Hard rules for both files:** one row per Source 1 test entity (every one
of them, no exceptions), empty string when nothing found, no duplicate ids
within a list, no duplicate `source1_entity_id` rows, ids must exist in the
test set, S2-/S3- ids only (no self-matches back to Source 1).

## Scoring
Macro-averaged **F0.5** per Source 1 entity, then averaged across all
entities:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Precision is weighted 2× recall — a false merge (matching two different
businesses) costs roughly twice what a missed match costs. A correctly
predicted empty list on a true singleton scores 1.0; any predicted match on
a singleton scores 0.0.

Public leaderboard = a subset of the test set, live during the challenge.
Private leaderboard = the remaining portion, revealed after close and used
for final ranking. You submit predictions for the full test set either way
— the split happens during scoring, not in what you upload.

## Constraints that affect design
- No external databases, APIs, or lookups of any kind — pure ML on the
  provided data only.
- Final model: MIT or Apache-2.0 licensed, ≤8B parameters, for any
  pretrained component used.
- Max 5 leaderboard submissions/day, 3 days (25–27 Sept 2026 IST), 15 total.
- Run before every upload:
  ```
  python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
  ```

## Final submission package
```
<team_name>_submission.zip
├── output/matching_results.tsv
├── output/candidate_pairs.tsv
├── code/business_entity_resolution/{src/, README.md, requirements.txt}
└── Documentation_template.md   # methodology: approach, blocking strategy,
                                  model + features, other notes — use the
                                  organizers' own template, don't invent one
```
Top 100 teams' packages get reviewed in detail before final rankings are
confirmed — the code must actually reproduce both output files end-to-end
from the `README.md` instructions, on data alone, with no manual steps left
out.
