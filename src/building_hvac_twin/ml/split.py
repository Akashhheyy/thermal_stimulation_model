"""Leakage-free grouped splitting by shelter design identity.

Rows of the same ``design_id`` appear under every weather scenario, so a naive
random row split would leak design information between train and test.  This
module shuffles unique design ids instead, then splits the designs 70/15/15
and derives row-level index sets from that grouping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["GroupedSplit", "grouped_design_split", "DEFAULT_SPLIT_FRACTIONS"]

DEFAULT_SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass
class GroupedSplit:
    """Row index sets plus the split summary for reporting."""

    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_index": self.train_index,
            "validation_index": self.validation_index,
            "test_index": self.test_index,
        }


def _split_summary(
    frame: pd.DataFrame, name: str, index: np.ndarray
) -> dict[str, Any]:
    part = frame.iloc[index]
    return {
        "split": name,
        "rows": int(len(index)),
        "unique_designs": int(part["design_id"].nunique()),
        "weather_scenarios": int(part["weather_scenario_id"].nunique()),
    }


def grouped_design_split(
    frame: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = DEFAULT_SPLIT_FRACTIONS["train"],
    validation_fraction: float = DEFAULT_SPLIT_FRACTIONS["validation"],
) -> GroupedSplit:
    """Split by unique design id; a design never spans train and test.

    The same seed always produces the same split.  Every split is guaranteed
    to contain at least one design and at least one row.
    """
    if "design_id" not in frame.columns:
        raise ValueError("frame must contain a design_id column")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be within (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train plus validation fractions must leave room for a test set")

    designs = np.array(sorted(frame["design_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = designs[rng.permutation(len(designs))]

    n_designs = len(shuffled)
    n_train = max(1, int(round(train_fraction * n_designs)))
    n_validation = max(1, int(round(validation_fraction * n_designs)))
    # Guarantee each split keeps at least one design even for tiny frames.
    n_train = min(n_train, n_designs - 2)
    n_validation = min(n_validation, n_designs - n_train - 1)
    if n_train < 1 or n_validation < 1:
        raise ValueError("frame has too few unique designs for a three-way split")

    train_designs = set(shuffled[:n_train])
    validation_designs = set(shuffled[n_train : n_train + n_validation])
    test_designs = set(shuffled[n_train + n_validation :])
    if train_designs & test_designs or train_designs & validation_designs:
        raise RuntimeError("grouped split leaked designs between splits")

    is_train = frame["design_id"].isin(train_designs).to_numpy()
    is_validation = frame["design_id"].isin(validation_designs).to_numpy()
    is_test = frame["design_id"].isin(test_designs).to_numpy()
    train_index = np.flatnonzero(is_train)
    validation_index = np.flatnonzero(is_validation)
    test_index = np.flatnonzero(is_test)
    if len(test_index) == 0:
        raise RuntimeError("test split is empty; refusing to continue")

    summary = {
        "seed": int(seed),
        "method": "grouped by design_id",
        "unique_designs_total": int(n_designs),
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": round(1.0 - train_fraction - validation_fraction, 6),
        },
        "splits": [
            _split_summary(frame, "train", train_index),
            _split_summary(frame, "validation", validation_index),
            _split_summary(frame, "test", test_index),
        ],
        "leakage_check": "no design_id appears in more than one split",
    }
    return GroupedSplit(
        train_index=train_index,
        validation_index=validation_index,
        test_index=test_index,
        summary=summary,
    )
