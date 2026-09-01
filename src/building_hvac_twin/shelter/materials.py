"""Extendable REFERENCE / DEMONSTRATION material library.

Nothing in this module is a measured Ladakh material property.  Every bundled
value is a clearly labelled reference value chosen for demonstration, and the
library exists so verified measurements can replace them later without
changing any model code.  Use :func:`material_library` to obtain a copy, and
:meth:`MaterialLibrary.register` or :meth:`MaterialLibrary.update` to extend it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import DataCategory, Material

__all__ = [
    "MaterialLibrary",
    "REFERENCE_MATERIALS",
    "REFERENCE_LIBRARY_VALUES",
    "material_library",
]

REFERENCE_LIBRARY_VALUES: dict[str, dict[str, Any]] = {
    "concrete": {
        "thermal_conductivity_w_mk": 1.7,
        "density_kg_m3": 2300.0,
        "specific_heat_j_kgk": 880.0,
    },
    "insulation": {
        "thermal_conductivity_w_mk": 0.035,
        "density_kg_m3": 30.0,
        "specific_heat_j_kgk": 1400.0,
    },
    "brick_masonry": {
        "thermal_conductivity_w_mk": 0.7,
        "density_kg_m3": 1800.0,
        "specific_heat_j_kgk": 840.0,
    },
    "earth_adobe": {
        "thermal_conductivity_w_mk": 0.6,
        "density_kg_m3": 1600.0,
        "specific_heat_j_kgk": 1000.0,
    },
    "glass": {
        "thermal_conductivity_w_mk": 1.0,
        "density_kg_m3": 2500.0,
        "specific_heat_j_kgk": 840.0,
    },
    "timber": {
        "thermal_conductivity_w_mk": 0.14,
        "density_kg_m3": 550.0,
        "specific_heat_j_kgk": 1600.0,
    },
    "water": {
        "thermal_conductivity_w_mk": 0.6,
        "density_kg_m3": 1000.0,
        "specific_heat_j_kgk": 4186.0,
    },
    "stone": {
        "thermal_conductivity_w_mk": 2.5,
        "density_kg_m3": 2600.0,
        "specific_heat_j_kgk": 900.0,
    },
}


@dataclass
class MaterialLibrary:
    """Name-keyed collection of :class:`Material` records.

    ``category`` marks provenance for the whole library; individual materials
    keep their own category as well so nothing can lose its label.
    """

    materials: dict[str, Material]
    category: DataCategory = DataCategory.REFERENCE
    label: str = "REFERENCE / DEMONSTRATION VALUES, not measured site data"

    def __post_init__(self) -> None:
        if not isinstance(self.materials, dict):
            raise ValueError("MaterialLibrary.materials must be a dict of Material")
        for name, material in self.materials.items():
            if not isinstance(material, Material):
                raise ValueError(f"MaterialLibrary entry {name!r} is not a Material")

    def __contains__(self, name: str) -> bool:
        return name in self.materials

    def __len__(self) -> int:
        return len(self.materials)

    def get(self, name: str) -> Material:
        """Return the named material or raise a clear ValueError."""
        key = name.strip().lower()
        if key not in self.materials:
            raise ValueError(
                f"Unknown material {name!r}. Available: {sorted(self.materials)}"
            )
        return self.materials[key]

    def register(self, material: Material, overwrite: bool = False) -> Material:
        """Add one material to the library."""
        if not isinstance(material, Material):
            raise ValueError("register expects a Material")
        key = material.name.strip().lower()
        if key in self.materials and not overwrite:
            raise ValueError(f"Material {material.name!r} already registered; pass overwrite=True to replace")
        self.materials[key] = material
        return material

    def update(self, others: Iterable[Material], overwrite: bool = False) -> None:
        for material in others:
            self.register(material, overwrite=overwrite)

    def names(self) -> list[str]:
        return sorted(self.materials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category.value,
            "materials": {name: m.to_dict() for name, m in sorted(self.materials.items())},
        }


def material_library(
    values: Mapping[str, Mapping[str, float]] | None = None,
    category: DataCategory = DataCategory.REFERENCE,
    label: str = "REFERENCE / DEMONSTRATION VALUES, not measured site data",
) -> MaterialLibrary:
    """Build a library from explicit property dicts (defaults to the bundled set)."""
    source = dict(REFERENCE_LIBRARY_VALUES) if values is None else dict(values)
    materials: dict[str, Material] = {}
    for name, props in source.items():
        materials[name.strip().lower()] = Material(
            name=name.strip().lower(),
            thermal_conductivity_w_mk=float(props["thermal_conductivity_w_mk"]),
            density_kg_m3=float(props["density_kg_m3"]),
            specific_heat_j_kgk=float(props["specific_heat_j_kgk"]),
            data_category=category,
            source=label,
        )
    return MaterialLibrary(materials=materials, category=category, label=label)


REFERENCE_MATERIALS: MaterialLibrary = material_library()
