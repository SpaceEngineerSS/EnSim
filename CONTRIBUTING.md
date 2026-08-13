# Contributing to EnSim

Contributions are welcome when they preserve EnSim's explicit model boundaries,
reproducibility and scientific traceability.

## Development setup

```bash
git clone https://github.com/SpaceEngineerSS/EnSim.git
cd EnSim
python -m venv .venv
python -m pip install -e ".[dev,docs]"
python -m pytest
```

Before opening a pull request, run:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest
mkdocs build --strict
```

## Scientific changes

A model change must state its governing equations, units, assumptions and valid
domain. Cite a primary source where one exists. Add tests for invariants, invalid
inputs and at least one independent reference value; update `docs/THEORY.md`,
`docs/VALIDATION.md` or `docs/MODEL_LIMITATIONS.md` as appropriate. Do not hide a
failed solve behind a nominal value, undocumented efficiency or generic
correlation.

External comparisons must distinguish verification, cross-comparison and
experimental validation. Include the exact source version, input case, extraction
method and tolerances needed to reproduce the evidence.

## Code changes

- Keep numerical models independent of Qt.
- Preserve SI internally and label every public unit boundary.
- Prefer immutable input/result records for numerical APIs.
- Avoid broad exception handling in core calculations.
- Add only comments that explain a non-obvious decision or source.
- Keep commits focused and use a clear imperative subject.

Use the issue templates for defects, feature proposals or scientific challenges.
Security reports follow [SECURITY.md](SECURITY.md).
