"""
Split module: creates a reproducible, stratified held-out validation set
of Source 1 entities based on country and true match-count distribution.
Used consistently across blocking, feature engineering, and model validation.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.csv as pv


def create_validation_split(
    train_dir: str = "dataset/train",
    val_size: int = 25000,
    random_state: int = 42,
    output_path: str = "experiments/val_split.parquet",
) -> pd.DataFrame:
    """
    Creates and saves a stratified held-out validation split of Source 1 entities.
    Stratifies across (country x match_bucket) to preserve the exact base-rate
    distribution of singletons, 1-match, 2-5 matches, and 6+ matches.
    """
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")

    print(f"Loading ground truth and Source 1 for validation split...")
    gt = pv.read_csv(gt_file, parse_options=pv.ParseOptions(delimiter="\t")).to_pandas()
    s1 = pv.read_csv(s1_file, parse_options=pv.ParseOptions(delimiter="\t")).to_pandas()

    merged = gt.merge(s1[["entity_id", "country"]], left_on="source1_entity_id", right_on="entity_id")
    merged = merged.drop(columns=["entity_id"])

    # Compute match counts
    def get_count(m: object) -> int:
        if pd.isna(m) or str(m).strip() == "":
            return 0
        return len(str(m).split(","))

    merged["match_count"] = merged["matched_entity_ids"].apply(get_count)

    # Bin into match_bucket
    def get_bucket(c: int) -> str:
        if c == 0:
            return "singleton"
        elif c == 1:
            return "1_match"
        elif c <= 5:
            return "2-5_matches"
        else:
            return "6+_matches"

    merged["match_bucket"] = merged["match_count"].apply(get_bucket)
    merged["stratum"] = merged["country"] + "_" + merged["match_bucket"]

    # Stratified sampling
    frac = val_size / len(merged)
    sample_ids = []
    for stratum_name, grp in merged.groupby("stratum"):
        n_sample = int(round(len(grp) * frac))
        sample_ids.extend(grp.sample(n=n_sample, random_state=random_state)["source1_entity_id"].tolist())

    val_df = merged[merged["source1_entity_id"].isin(set(sample_ids))].copy()

    # If minor rounding occurred, adjust to exact val_size
    if len(val_df) > val_size:
        val_df = val_df.sample(n=val_size, random_state=random_state)
    elif len(val_df) < val_size:
        diff = val_size - len(val_df)
        remaining = merged[~merged["source1_entity_id"].isin(val_df["source1_entity_id"])]
        fill = remaining.sample(n=diff, random_state=random_state)
        val_df = pd.concat([val_df, fill])

    val_df = val_df.reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    val_df.to_parquet(output_path, index=False)
    print(f"Validation split created with {len(val_df)} entities saved to {output_path}")
    print(val_df["stratum"].value_counts(normalize=True))
    return val_df


if __name__ == "__main__":
    create_validation_split()
