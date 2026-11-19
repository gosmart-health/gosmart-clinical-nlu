# gosmart-clinical-nlu

Clinical NLU library shared across peer projects (Python, managed with [uv](https://docs.astral.sh/uv/)).

## Layout

Uses the standard Python `src` layout, which keeps the importable package
separated from project tooling and guarantees tests run against the installed
package rather than the working directory:

```
.
├── src/
│   └── gosmart_clinical_nlu/   # the importable package (underscores — required by Python)
│       ├── __init__.py
│       ├── core.py
│       └── __main__.py
├── tests/                      # test suite (pytest)
├── pyproject.toml
└── uv.lock
```

The distribution name is `gosmart-clinical-nlu` (hyphens are allowed in
distribution names), but Python module names cannot contain hyphens, so the
import name is `gosmart_clinical_nlu`.

## Development

```bash
uv sync                                # create/update .venv, install the package (editable) + dev deps
uv run pytest                          # run the test suite
uv run gosmart-clinical-nlu            # run the console entry point
uv run python -m gosmart_clinical_nlu  # same thing, via module execution
uv build                               # build sdist + wheel for distribution
```

## Using from a peer project

Install from a Git remote or a local path:

```bash
uv add "gosmart-clinical-nlu @ git+https://<host>/<org>/gosmart-clinical-nlu.git"
# or, locally:
uv add --editable ../gosmart-clinical-nlu
```

Then import it:

```python
from gosmart_clinical_nlu import print_hi, __version__
```
