"""In-memory fakes that mimic the pymongo surface used by the repositories.

The fakes implement only what ``database.repositories`` actually calls:
``create_index``, ``update_one`` (upsert), ``find`` with ``sort``/``skip``/
``limit``, ``find_one``, ``count_documents`` and ``insert_one``.  They let
repository, seed and API tests run without MongoDB or any credentials.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from building_hvac_twin.database.repositories import RepositoryBundle, build_repositories

__all__ = ["FakeCollection", "FakeClient", "make_fake_bundle"]


class FakeInsertOneResult:
    def __init__(self, inserted_id: int) -> None:
        self.inserted_id = inserted_id


class FakeCursor:
    """Chainable cursor over an in-memory list, like pymongo's."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents
        self._sort: list[tuple[str, int]] = []
        self._skip = 0
        self._limit: int | None = None

    def sort(self, key_or_list: Any, direction: int = 1) -> "FakeCursor":
        if isinstance(key_or_list, str):
            self._sort.append((key_or_list, direction))
        else:
            self._sort.extend(key_or_list)
        return self

    def skip(self, count: int) -> "FakeCursor":
        self._skip = count
        return self

    def limit(self, count: int) -> "FakeCursor":
        self._limit = count
        return self

    def __iter__(self):
        return iter(self._resolve())

    def _resolve(self) -> list[dict[str, Any]]:
        documents = list(self._documents)
        for key, direction in reversed(self._sort):
            documents.sort(key=lambda doc: doc.get(key), reverse=direction < 0)
        if self._skip:
            documents = documents[self._skip :]
        if self._limit is not None:
            documents = documents[: self._limit]
        return [deepcopy(document) for document in documents]


class FakeCollection:
    """Minimal pymongo.Collection stand-in over a list of documents."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.documents: list[dict[str, Any]] = []
        self.indexes: list[Any] = []

    # -- indexes -----------------------------------------------------------
    def create_index(self, key_or_list: Any, **_kwargs: Any) -> str:
        self.indexes.append(key_or_list)
        return str(key_or_list)

    # -- writes ------------------------------------------------------------
    def _matches(self, document: dict[str, Any], filter: dict[str, Any]) -> bool:
        return all(document.get(key) == value for key, value in filter.items())

    def update_one(
        self,
        filter: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
    ) -> None:
        for document in self.documents:
            if self._matches(document, filter):
                document.update(deepcopy(update.get("$set", {})))
                return
        if upsert:
            new_document = {**deepcopy(filter), **deepcopy(update.get("$set", {}))}
            self.documents.append(new_document)

    def insert_one(self, document: dict[str, Any]) -> FakeInsertOneResult:
        self.documents.append(deepcopy(document))
        return FakeInsertOneResult(inserted_id=len(self.documents))

    # -- reads -------------------------------------------------------------
    def find(
        self,
        filter: dict[str, Any] | None = None,
        _projection: Any = None,
    ) -> FakeCursor:
        matching = [
            deepcopy(document)
            for document in self.documents
            if self._matches(document, filter or {})
        ]
        return FakeCursor(matching)

    def find_one(
        self,
        filter: dict[str, Any],
        _projection: Any = None,
    ) -> dict[str, Any] | None:
        for document in self.documents:
            if self._matches(document, filter):
                return deepcopy(document)
        return None

    def count_documents(self, filter: dict[str, Any] | None = None) -> int:
        return sum(
            1
            for document in self.documents
            if self._matches(document, filter or {})
        )


class FakeAdmin:
    def command(self, command: str) -> dict[str, Any]:
        return {"ok": 1.0, command: 1.0}


class FakeClient:
    """Client stand-in whose only used surface is the health ping."""

    def __init__(self) -> None:
        self.admin = FakeAdmin()

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def make_fake_bundle(database_name: str = "fake_db") -> RepositoryBundle:
    """Build a RepositoryBundle wired to in-memory fake collections."""
    collections = {name: FakeCollection(name) for name in (
        "designs",
        "weather_scenarios",
        "predictions",
        "recommendations",
        "comparisons",
    )}
    client = FakeClient()
    database = {
        name: collection for name, collection in collections.items()
    }
    bundle = build_repositories(client, database, database_name)
    bundle.collections = collections  # exposed for test assertions
    return bundle
