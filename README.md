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

Add the package straight from GitHub (no PyPI release required):

```bash
# from the default branch (main)
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git"

# from the dev branch
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git@dev"

# pinned to a tag/release
uv add "gosmart-clinical-nlu @ git+https://github.com/gosmart-health/gosmart-clinical-nlu.git@v0.1.0"

# or, while iterating locally on a sibling clone
uv add --editable ../gosmart-clinical-nlu
```

### Importing the package

The distribution name is `gosmart-clinical-nlu`, but Python module names
cannot contain hyphens, so the import name is `gosmart_clinical_nlu`:

```python
import gosmart_clinical_nlu

gosmart_clinical_nlu.__version__          # '0.1.0'

# The public API is re-exported at the package root:
from gosmart_clinical_nlu import print_hi

print_hi("peer project")                  # Hi, peer project
```

### Running it

After `uv sync` inside a clone of this repo:

```bash
uv run gosmart-clinical-nlu               # console entry point
uv run python -m gosmart_clinical_nlu     # equivalent, via module execution
```
