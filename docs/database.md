# MongoDB integration

MongoDB is the **application/persistence layer only**. The following stay in
their existing project locations and are never copied into MongoDB:

- the 3,000-row ML dataset (`data/shelter_ml_dataset.csv`)
- trained model artifacts (`models/regression/`)
- NASA POWER raw weather files (`data/nasa_weather_raw/`)
- the thermal simulation source data

MongoDB never runs ML or physics; it stores catalogs and application results.

## 1. Configuration

Copy `.env.example` to `.env` and fill in your local values. Real credentials
are never committed and never hard-coded.

| Variable           | Required | Default                     | Purpose                          |
|--------------------|----------|-----------------------------|----------------------------------|
| `MONGODB_URI`      | yes      | none (no default credentials) | MongoDB connection string     |
| `MONGODB_DATABASE` | no       | `building_energy_hvac_twin` | database name                    |
| `BHVAC_MODELS_DIR`| no | `models/regression` | trained artifact directory |
| `BHVAC_DATASET_PATH` | no | `data/shelter_ml_dataset.csv` | existing ML dataset |
| `BHVAC_METADATA_PATH` | no | `data/shelter_ml_dataset_metadata.json` | dataset metadata |
| `BHVAC_METRICS_REPORT` | no | `reports/ml_metrics.json` | training metrics report |

Start the API with the env file:

```
python -m uvicorn building_hvac_twin.api.main:app --reload --env-file .env
```

(Plain `KEY=VALUE` `.env` files are supported without python-dotenv; real
environment variables always win.)

If `MONGODB_URI` is not set, the API still starts. Database-backed endpoints
(`GET /designs`, `GET /scenarios`) answer with a clear HTTP 503 instead of
fake data, and the write endpoints compute results normally and report
`"persistence": {"saved": false, ...}`.

## 2. Collections

| Collection           | Key            | Content                                                        |
|----------------------|----------------|----------------------------------------------------------------|
| `designs`            | `design_id`    | the existing 300 shelter designs (design parameters only)       |
| `weather_scenarios`  | `scenario_id`  | the existing 10 NASA POWER scenario metadata + provenance       |
| `predictions`        | append-only    | results of `POST /predict` (9 ML targets, primary models, UTC timestamp) |
| `recommendations`    | append-only    | results of `POST /recommend` (objectives, ranked designs, UTC timestamp) |
| `comparisons`        | append-only    | results of `POST /compare` (ML surrogate vs thermal engine rows, UTC timestamp) |

Stored provenance:

- `predictions` documents carry the NASA POWER provenance statement and the
  surrogate-model disclaimer.
- `comparisons` documents preserve the distinction that the ML column is a
  surrogate model output, the physics column is the thermal engine output,
  and neither is a measured building performance.
- `performance_score` is not an ML target and is never stored.

## 3. Where designs and scenarios come from

The seed command reads (never modifies) the existing project artifacts:

- designs: first dataset row per `design_id` in
  `data/shelter_ml_dataset.csv`, restricted to the design parameter columns
  defined by `shelter.ml_dataset.DESIGN_PARAMETER_COLUMNS`;
- scenarios: the `weather_scenarios.used` entries and the `nasa_power`
  provenance block in `data/shelter_ml_dataset_metadata.json`.

No designs, scenarios or weather values are invented by the API or the seed.

## 4. Initialize / seed the database

From the repository root:

```
python -m building_hvac_twin.database.seed
```

Options: `--uri`, `--database`, `--dataset`, `--metadata`. The command is
**idempotent**: designs and scenarios are upserted by their natural ids and
unique indexes are created, so running it twice never creates duplicates.
The database is never seeded automatically at API startup.

## 5. Tests

Repository, document-builder, seed and API tests run against in-memory fakes
(`tests/database/fakes.py`) seeded from the real dataset and metadata; no
MongoDB server or credentials are needed for `python -m pytest -q`.

An optional live integration test runs only when you explicitly point it at
a server:

```
set MONGODB_TEST_URI=mongodb://localhost:27017
python -m pytest tests/database/test_mongo_integration.py -v
```

It seeds a throwaway `building_energy_hvac_twin_test` database and drops it
afterwards; it never touches the real application database.
