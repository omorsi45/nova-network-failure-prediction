## Contributing

Thanks for taking the time to contribute.

### Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running checks

```bash
ruff check .
pytest -q
```

### Pull requests

- Keep changes focused and include a short rationale in the PR description.
- Add or update tests when changing behavior.
- Ensure `ruff` and `pytest` pass before requesting review.

