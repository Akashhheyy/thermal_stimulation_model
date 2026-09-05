"""POST /recommend: rank candidate designs with the existing ranking.

Thin route: the existing deterministic design generator, the existing
predictor and the existing ``rank_designs`` with the default objectives.
No second ranking algorithm exists here.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..deps import get_bundle, get_metadata, persist_result, require_scenario
from ..schemas import RecommendRequest, RecommendResponse
from ...database import recommendation_document

router = APIRouter(tags=["recommendation"])

RANKING_PROVENANCE = (
    "Weighted decision score over physical ML targets computed by the "
    "existing recommendation ranking; NOT performance_score."
)


@router.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest, request: Request) -> RecommendResponse:
    from ...shelter.ml_dataset import build_shelter_config, generate_designs
    from ...recommendation import predict_design
    from ...recommendation.ranking import DEFAULT_OBJECTIVES, rank_designs

    state = request.app.state
    metadata = get_metadata(request)
    scenario = require_scenario(metadata, payload.scenario_id)
    bundle = get_bundle(request)

    designs = generate_designs(count=payload.count, seed=payload.seed)
    outcomes = []
    try:
        for design in designs:
            config = build_shelter_config(design)
            outcomes.append(
                predict_design(
                    bundle,
                    config=config,
                    scenario_id=payload.scenario_id,
                    metadata_path=state.metadata_path,
                )
            )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    ranked = rank_designs(
        [outcome.design_id for outcome in outcomes],
        [outcome.primary_predictions for outcome in outcomes],
        DEFAULT_OBJECTIVES,
    )

    body = {
        "scenario_id": scenario["scenario_id"],
        "count": len(ranked),
        "objectives": [
            {
                "target": objective.target,
                "direction": objective.direction.value,
                "weight": objective.weight,
            }
            for objective in DEFAULT_OBJECTIVES
        ],
        "ranking": [
            {
                "design_id": rec.design_id,
                "rank": rec.rank,
                "recommendation_score": rec.recommendation_score,
                "components": rec.components,
                "primary_predictions": rec.primary_predictions,
            }
            for rec in ranked
        ],
        "provenance": RANKING_PROVENANCE,
    }
    persistence = persist_result(
        state, "save_recommendation", recommendation_document(body)
    )
    return RecommendResponse(**body, persistence=persistence)
