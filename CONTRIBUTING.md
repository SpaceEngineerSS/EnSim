# Contributing

Thank you for helping. Bug reports, corrections to the physics and small,
focused pull requests are all welcome.

## Setting up

```console
$ git clone https://github.com/SpaceEngineerSS/EnSim.git
$ cd EnSim
$ python -m venv .venv
$ pip install -e ".[all]"
$ pytest
```

`[all]` installs the desktop application, the test tools and the
documentation tools. To work on the library alone, `pip install -e ".[dev]"` is
enough; the interface tests are skipped without PyQt6.

## Before a pull request

```console
$ ruff check src tests scripts
$ ruff format --check src tests scripts
$ pytest
$ mkdocs build --strict
```

CI runs the same, plus the core tests on Windows, macOS and Linux without Qt
and a wheel build.

## Changes to the physics

- Say which equation changes and cite the source.
- Keep SI inside the library and state the unit of every public argument.
- Add a test of the equation or invariant, a test of an invalid input, and,
  when an independent reference exists, a comparison in `tests/validation`
  with the reference data in `tests/reference` and its provenance.
- Update the matching page in `docs/theory`, and `docs/model-limitations.md`
  if the model's reach changes.
- Never hide a failure: raise rather than return a nominal value, and do not
  add an efficiency or margin that is not an input.

`scripts/generate_cea_references.py` regenerates the CEA reference set with
NASA's `cea` package; `scripts/make_doc_figures.py` redraws the validation
figure from it.

## Changes to the application

- Calculations belong in `ensim.core`; the interface only collects inputs and
  shows results.
- Anything that takes longer than a frame goes through `TaskRunner` and
  accepts cancellation.
- Colours, spacing and fonts come from `ensim/ui/theme.py`; icons are SVG in
  `assets/icons`, drawn on a 20 px grid with a 1.5 px stroke and
  `currentColor`.
- Labels are short and in sentence case. Numbers use the mono font.
- If you change what a page looks like, run
  `python scripts/capture_screenshots.py` and commit the new images.

## Commits

One change per commit, with a short subject that says what changed, prefixed
by the area (`core:`, `ui:`, `docs:`, `tests:`).

## Reporting

Use the issue templates. For a physics question, give the full inputs with
units and the result you expected, with its source. Security problems go
through private vulnerability reporting as described in
[SECURITY.md](SECURITY.md), not to a public issue.
