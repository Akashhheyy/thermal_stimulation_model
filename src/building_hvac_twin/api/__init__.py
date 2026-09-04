"""FastAPI API layer over the existing prediction, recommendation and
thermal simulation packages.

Run locally from the repository root with::

    python -m uvicorn building_hvac_twin.api.main:app --reload
"""
from .main import app, create_app

__all__ = ["app", "create_app"]
