FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

CMD ["python", "scripts/08_simulate_failures.py", "--out", "data/raw", "--days", "2", "--nodes", "8"]
