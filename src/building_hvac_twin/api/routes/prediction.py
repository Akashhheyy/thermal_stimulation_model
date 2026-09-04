"""POST /predict: ML surrogate prediction for one design and scenario.

Thin route: validation, then the existing ``predict_design`` from the
recommendation package.  No new targets and no new model logic.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..deps import get_bundle, get_dataset, get_metadata, require_design_id, require_scenario
from ..schemas import PHYSICAL_TARGETS, PredictRequest, PredictResponse
from ...recommendation import predict_design

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    state = request.app.state
    dataset = get_dataset(request)
    metadata = get_metadata(request)
    require_design_id(dataset, payload.design_id)
    require_scenario(metadata, payload.scenario_id)

    try:
        outcome = predict_design(
            get_bundle(request),
            design_id=payload.design_id,
            scenario_id=payload.scenario_id,
            dataset_path=state.dataset_path,
            metadata_path=state.metadata_path,
        )
    except ValueError as exc:
        # The existing predictor raises ValueError for unknown pairs and
        # invalid input; surface it as a client error.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = outcome.to_dict()
    return PredictResponse(
        design_id=result["design_id"],
        scenario_id=result["weather_scenario_id"],
        input_mode=result["input_mode"],
        targets=list(PHYSICAL_TARGETS),
        raw_predictions=result["raw_predictions"],
        primary_predictions=result["primary_predictions"],
        out_of_bounds=result["out_of_bounds"],
        provenance=result["provenance"],
        artifact_info=result["artifact_info"],
    )
