"""Catalog and ML-vs-physics routes backed by the existing packages.

- ``POST /compare``: the existing ``compare_prediction_with_physics``, which
  runs the same design through the ML surrogate and the real thermal engine.
- ``GET /scenarios``: the NASA POWER scenario catalog recorded in the dataset
  metadata during dataset generation.
- ``GET /designs``: the shelter design catalog represented in the existing
  ML dataset.

No weather is fetched or invented here and no design is created here.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..deps import (
    get_bundle,
    get_dataset,
    get_metadata,
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

    return CompareResponse(
        design_id=result["design_id"],
        scenario_id=scenario["scenario_id"],
        compared_targets=list(result["compared_targets"]),
        rows=[ComparisonRow(**row) for row in result["rows"]],
        provenance=result["provenance"],
    )


@router.get("/scenarios", response_model=ScenariosResponse)
def scenarios(request: Request) -> ScenariosResponse:
    metadata = get_metadata(request)
    nasa = metadata.get("nasa_power", {})
    used = metadata.get("weather_scenarios", {}).get("used", [])
    return ScenariosResponse(
        count=len(used),
        location_name=str(nasa.get("location_name", "")),
        latitude=float(nasa.get("latitude", 0.0)),
        longitude=float(nasa.get("longitude", 0.0)),
        nasa_power_source=str(nasa.get("source", "NASA POWER")),
        scenarios=[ScenarioSummary(**entry) for entry in used],
    )


@router.get("/designs", response_model=DesignsResponse)
def designs(
    request: Request,
    limit: int | None = None,
) -> DesignsResponse:
    from ...shelter.ml_dataset import DESIGN_PARAMETER_COLUMNS

    dataset = get_dataset(request)
    # Design parameters are constant across the scenario rows of a design;
    # take the first row per design_id to describe it.
    first_rows = dataset.drop_duplicates(subset="design_id", keep="first")
    first_rows = first_rows.sort_values("design_id")
    if limit is not None:
        if limit < 1:
            raise HTTPException(
                status_code=422, detail="limit must be a positive integer"
            )
        first_rows = first_rows.head(limit)
    summaries = []
    for row in first_rows.itertuples(index=False):
        record = row._asdict()
        summaries.append(
            DesignSummary(
                design_id=str(record["design_id"]),
                design_parameters={
                    column: record[column]
                    for column in DESIGN_PARAMETER_COLUMNS
                    if column in record
                },
            )
        )
    return DesignsResponse(count=len(summaries), designs=summaries)
