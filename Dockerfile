FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY routecalib ./routecalib
COPY tests ./tests

RUN pip install --no-cache-dir numpy matplotlib && pip install --no-cache-dir --no-deps -e .

EXPOSE 8080
CMD ["python3", "-m", "routecalib.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
