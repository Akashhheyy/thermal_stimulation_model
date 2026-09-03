"""Deterministic multi-objective ranking of predicted shelter designs.

The ranking is an application-level decision aid.  It is deliberately NOT the
``performance_score`` from ``comparison.py`` (a relative within-batch metric);
it is a transparent weighted score over physical ML targets, documented so
the number can be reproduced and audited.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from .schemas import (
    PHYSICAL_TARGETS,
    RankedRecommendation,
    RecommendationObjective,
)

__all__ = [
    "DEFAULT_OBJECTIVES",
    "rank_designs",
    "score_components",
]


class FastObjectiveImport:
    """Local alias for the FastObjective enum used in objective definitions."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


# A sensible default: maximise comfort time and minimise discomfort degree
# hours.  These are physical targets with trained artifacts only.
DEFAULT_OBJECTIVES: tuple[RecommendationObjective, ...] = (
    RecommendationObjective(
        target="percent_time_comfortable",
        direction=FastObjectiveImport.MAXIMIZE,
        weight=1.0,
    ),
    RecommendationObjective(
        target="degree_hours_below_comfort",
        direction=FastObjectiveImport.MINIMIZE,
        weight=0.5,
    ),
    RecommendationObjective(
        target="degree_hours_above_comfort",
        direction=FastObjectiveImport.MINIMIZE,
        weight=0.5,
    ),
)


def _normalize(values: Sequence[float], direction: str) -> list[float]:
    """Min-max normalise a series so higher is always better.

    A constant series (no spread) maps to 0.5 for every candidate so it does
    not distort the weighted score.  ``minimize`` directions are inverted.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0.0:
        return [0.5] * len(values)
    normalized = [(value - lo) / span for value in values]
    if direction == "minimize":
        normalized = [1.0 - n for n in normalized]
    return normalized


def score_components(
    candidates: Sequence[dict[str, dict[str, object]]],
    objectives: Sequence[RecommendationObjective],
) -> tuple[list[dict[str, float]], list[float]]:
    """Return per-candidate objective components and the weighted 0..100 score.

    ``candidates`` is the list of ``primary_predictions`` dicts (one per
    design).  Missing or non-finite target values raise rather than being
    silently treated as zero.
    """
    if not candidates:
        return [], []
    for objective in objectives:
        if objective.target not in PHYSICAL_TARGETS:
            raise ValueError(f"objective target {objective.target!r} is not a physical ML target")

    components: list[dict[str, float]] = [{} for _ in candidates]
    weighted: list[float] = [0.0 for _ in candidates]
    total_weight = sum(objective.weight for objective in objectives)
    if total_weight <= 0.0:
        raise ValueError("at least one objective must have a positive weight")

    for objective in objectives:
        raw_values = []
        for index, candidate in enumerate(candidates):
            prediction = candidate.get(objective.target)
            if prediction is None or "value" not in prediction:
                raise ValueError(
                    f"candidate {index} is missing a primary prediction for "
                    f"{objective.target!r}"
                )
            value = float(prediction["value"])
            if value != value:  # NaN check
                raise ValueError(
                    f"candidate {index} has a non-finite prediction for "
                    f"{objective.target!r}"
                )
            raw_values.append(value)
        normalized = _normalize(raw_values, objective.direction.value)
        for index, component in enumerate(normalized):
            components[index][objective.target] = component
            weighted[index] += objective.weight * component
    scores = [100.0 * value / total_weight for value in weighted]
    return components, scores


def rank_designs(
    design_ids: Sequence[str],
    candidates: Sequence[dict[str, dict[str, object]]],
    objectives: Sequence[RecommendationObjective] = DEFAULT_OBJECTIVES,
) -> list[RankedRecommendation]:
    """Rank candidate designs deterministically by a weighted decision score.

    Ties are broken by ``design_id`` so the ordering is reproducible.  The
    returned score is an application-level metric in the range 0..100 and is
    never ``performance_score``.
    """
    if len(design_ids) != len(candidates):
        raise ValueError("design_ids and candidates must have the same length")
    components, scores = score_components(candidates, objectives)
    indexed = list(zip(range(len(design_ids)), design_ids, scores, components))
    indexed.sort(key=lambda item: (-item[2], item[1]))
    ranked: list[RankedRecommendation] = []
    for rank, (original_index, design_id, score, component) in enumerate(indexed, start=1):
        ranked.append(
            RankedRecommendation(
                design_id=design_id,
                rank=rank,
                recommendation_score=round(score, 6),
                components=dict(component),
                primary_predictions=dict(candidates[original_index]),
                provenance=(
                    "Weighted decision score over physical ML targets; NOT "
                    "performance_score from comparison.py."
                ),
            )
        )
    return ranked