"""Repositories: the only place that speaks the MongoDB query language.

Repositories depend on the small surface of ``pymongo.Collection`` they use
(``create_index``, ``update_one``, ``find``, ``count_documents``,
``insert_one``) so tests can substitute in-memory fakes.  The API layer calls
these classes; it never issues raw queries.

``RepositoryBundle`` groups the five repositories and the best-effort
persistence helpers used by the write endpoints: a database problem never
breaks an ML or physics computation, and the outcome is reported honestly in
the response instead of being hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymongo.errors import PyMongoError

from .collections import (
    COMPARISONS_COLLECTION,
    DESIGNS_COLLECTION,
    PREDICTIONS_COLLECTION,
    RECOMMENDATIONS_COLLECTION,
    WEATHER_SCENARIOS_COLLECTION,
)

__all__ = [
    "DesignRepository",
    "ScenarioRepository",
    "PredictionRepository",
    "RecommendationRepository",
    "ComparisonRepository",
    "RepositoryBundle",
    "PersistenceResult",
    "build_repositories",
]


class DesignRepository:
    """Upsert-by-``design_id`` access to the designs collection."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        self._collection.create_index("design_id", unique=True)

    def upsert_many(self, documents: list[dict[str, Any]]) -> int:
        """Upsert documents keyed by ``design_id``; repeat runs are no-ops."""
        for document in documents:
            self._collection.update_one(
                {"design_id": document["design_id"]},
                {"$set": document},
                upsert=True,
            )
        return len(documents)

    def count(self) -> int:
        return int(self._collection.count_documents({}))

    def list(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        cursor = (
            self._collection.find({}, {"_id": False})
            .sort("design_id", 1)
            .skip(offset)
        )
        if limit is not None:
            cursor = cursor.limit(limit)
        return list(cursor)

    def get(self, design_id: str) -> dict[str, Any] | None:
        return self._collection.find_one(
            {"design_id": design_id}, {"_id": False}
        )


class ScenarioRepository:
    """Upsert-by-``scenario_id`` access to the weather_scenarios collection."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        self._collection.create_index("scenario_id", unique=True)

    def upsert_many(self, documents: list[dict[str, Any]]) -> int:
        for document in documents:
            self._collection.update_one(
                {"scenario_id": document["scenario_id"]},
                {"$set": document},
                upsert=True,
            )
        return len(documents)

    def count(self) -> int:
        return int(self._collection.count_documents({}))

    def list(self) -> list[dict[str, Any]]:
        return list(
            self._collection.find({}, {"_id": False}).sort("scenario_id", 1)
        )

    def get(self, scenario_id: str) -> dict[str, Any] | None:
        return self._collection.find_one(
            {"scenario_id": scenario_id}, {"_id": False}
        )


class PredictionRepository:
    """Append-only persistence for application prediction results."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        self._collection.create_index([("design_id", 1), ("scenario_id", 1)])

    def insert(self, document: dict[str, Any]) -> Any:
        result = self._collection.insert_one(document)
        return result.inserted_id


class RecommendationRepository:
    """Append-only persistence for application recommendation results."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        self._collection.create_index("scenario_id")

    def insert(self, document: dict[str, Any]) -> Any:
        result = self._collection.insert_one(document)
        return result.inserted_id


class ComparisonRepository:
    """Append-only persistence for ML-vs-physics comparison results."""

    def __init__(self, collection: Any) -> None:
        self._collection = collection

    def ensure_indexes(self) -> None:
        self._collection.create_index([("design_id", 1), ("scenario_id", 1)])

    def insert(self, document: dict[str, Any]) -> Any:
        result = self._collection.insert_one(document)
        return result.inserted_id


@dataclass(frozen=True)
class PersistenceResult:
    """Honest report of a best-effort persistence attempt."""

    saved: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"saved": self.saved, "detail": self.detail}


@dataclass
class RepositoryBundle:
    """The five repositories plus the connection handle and database name."""

    designs: DesignRepository
    weather_scenarios: ScenarioRepository
    predictions: PredictionRepository
    recommendations: RecommendationRepository
    comparisons: ComparisonRepository
    database_name: str
    client: Any = None

    def ensure_indexes(self) -> None:
        for repository in (
            self.designs,
            self.weather_scenarios,
            self.predictions,
            self.recommendations,
            self.comparisons,
        ):
            repository.ensure_indexes()

    def ping(self) -> bool:
        """Cheap connectivity probe used by the health endpoint."""
        try:
            self.client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    # -- best-effort persistence helpers (used by the write endpoints) ----

    def save_prediction(self, document: dict[str, Any]) -> PersistenceResult:
        return self._try_insert(self.predictions, document)

    def save_recommendation(self, document: dict[str, Any]) -> PersistenceResult:
        return self._try_insert(self.recommendations, document)

    def save_comparison(self, document: dict[str, Any]) -> PersistenceResult:
        return self._try_insert(self.comparisons, document)

    def _try_insert(
        self,
        repository: Any,
        document: dict[str, Any],
    ) -> PersistenceResult:
        try:
            repository.insert(document)
            return PersistenceResult(saved=True)
        except PyMongoError as exc:
            # The computation already succeeded; report the persistence
            # failure instead of failing the request or hiding the problem.
            return PersistenceResult(saved=False, detail=str(exc))


def build_repositories(
    client: Any,
    database: Any,
    database_name: str,
) -> RepositoryBundle:
    """Wrap a connected pymongo database in the repository bundle."""
    return RepositoryBundle(
        designs=DesignRepository(database[DESIGNS_COLLECTION]),
        weather_scenarios=ScenarioRepository(database[WEATHER_SCENARIOS_COLLECTION]),
        predictions=PredictionRepository(database[PREDICTIONS_COLLECTION]),
        recommendations=RecommendationRepository(database[RECOMMENDATIONS_COLLECTION]),
        comparisons=ComparisonRepository(database[COMPARISONS_COLLECTION]),
        database_name=database_name,
        client=client,
    )
