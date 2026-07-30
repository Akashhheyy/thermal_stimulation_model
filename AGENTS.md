# Agent Instructions

1. Read `AGENT_PLAYBOOK.md`, `docs/model-card.md`, `docs/data-contract.md`, and `.agents/skills/building-energy-hvac-digital-twin/SKILL.md` before scientific changes.
2. Run `pytest -q` and `python tests/smoke_test.py` before editing model behavior.
3. Never overwrite original uploaded data. Put adaptations under `projects/<name>/` or `case_studies/<name>/`.
4. Never infer units, timezone, interval meaning, missing labels, equipment semantics, or sensor provenance.
5. Keep calibration and held-out validation periods separate. Do not tune on validation results.
6. Print metrics and inspect residuals. A successful run is not scientific validation.
7. Call bundled results synthetic reference results. Never claim production readiness without measured validation evidence.
8. Preserve vendor neutrality in core source. Put vendor-specific curves and mappings in case studies.
9. Add tests for valid inputs, invalid inputs, physical sanity, and any new feature.
10. Never use the em dash character in repository text.
