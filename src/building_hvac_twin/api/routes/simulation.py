"""Catalog and ML-vs-physics routes backed by the existing packages.

- ``POST /compare``: the existing ``compare_prediction_with_physics``, which
  runs the same design through the ML surrogate and the real thermal engine;
  the result is persisted best-effort to the ``comparisons`` collection.
- ``GET /scenarios``: the NASA POWER scenario catalog stored in MongoDB by
  the seed command (origin: dataset metadata, no synthetic weather).
- ``GET /designs``: the shelter design catalog stored in MongoDB by the seed
  command (origin: the existing ML dataset, no invented designs).

When MongoDB is not configured, the catalog endpoints answer with a clear
503 and the write endpoints still compute and flag the result as not
persisted.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..deps import (
    get_bundle,
    get_dataset,
    get_metadata,
    get_repositories,
    persist_result,
    require_design_id,
    require_scenario,
)
from ..schemas import (
    ComparisonRow,
    CompareRequest,
    CompareResponse,
    DesignSummary,
    DesignsResponse,
    ScenarioSummary,
    ScenariosResponse,
)
from ...database import comparison_document

router = APIRouter(tags=["simulation"])


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest, request: Request) -> CompareResponse:
    from ...shelter.ml_dataset import build_shelter_config
    from ...recommendation import load_scenario_weather, compare_prediction_with_physics

    state = request.app.state
    dataset = get_dataset(request)
    metadata = get_metadata(request)
    require_design_id(dataset, payload.design_id)
    scenario = require_scenario(metadata, payload.scenario_id)

    row = dataset[dataset["design_id"] == payload.design_id].iloc[0]
    try:
        config = build_shelter_config(row.to_dict(), name=payload.design_id)
        weather = load_scenario_weather(payload.scenario_id)
        result = compare_prediction_with_physics(
            get_bundle(request),
            config,
            weather,
            metadata_path=state.metadata_path,
        )
    except (ValueError, FileNotFoundError) as exc:
        # Missing weather cache or an invalid design both come back as
        # client-visible errors from the existing functions.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    body = {
        "design_id": result["design_id"],
        "scenario_id": scenario["scenario_id"],
        "compared_targets": list(result["compared_targets"]),
        "rows": result["rows"],
        "provenance": result["provenance"],
    }
    persistence = persist_result(
        state, "save_comparison", comparison_document(body)
    )
    return CompareResponse(
        compared_targets=body["compared_targets"],
        rows=[ComparisonRow(**row) for row in body["rows"]],
        **{key: body[key] for key in ("design_id", "scenario_id", "provenance")},
        persistence=persistence,
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios(request: Request) -> ScenariosResponse:
    repositories = get_repositories(request)
    documents = repositories.weather_scenarios.list()
    if not documents:
        raise HTTPException(
            status_code=503,
            detail=(
                "the weather_scenarios collection is empty; seed it with "
                "`python -m building_hvac_twin.database.seed`"
            ),
        )
    first = documents[0]
    return ScenariosResponse(
        count=len(documents),
        location_name=str(first.get("location_name") or ""),
        latitude=float(first.get("latitude") or 0.0),
        longitude=float(first.get("longitude") or 0.0),
        nasa_power_source=str(first.get("nasa_power_source") or "NASA POWER"),
        scenarios=[ScenarioSummary(**document) for document in documents],
    )


@router.get("/designs", response_model=DesignsResponse)
def designs(
    request: Request,
    limit: int | None = None,
    offset: int = 0,
) -> DesignsResponse:
    if limit is not None and limit < 1:
        raise HTTPException(
            status_code=422, detail="limit must be a positive integer"
        )
    if offset < 0:
        raise HTTPException(
            status_code=422, detail="offset must be nonnegative"
        )
    repositories = get_repositories(request)
    if repositories.designs.count() == 0:
        raise HTTPException(
            status_code=503,
            detail=(
                "the designs collection is empty; seed it with "
                "`python -m building_hvac_twin.database.seed`"
            ),
        )
    documents = repositories.designs.list(limit=limit, offset=offset)
    summaries = [
        DesignSummary(
            design_id=str(document["design_id"]),
            design_parameters=document.get("design_parameters", {}),
        )
        for document in documents
    ]
    return DesignsResponse(count=len(summaries), designs=summaries)
