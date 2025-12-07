# Dockerfile dla Animownik - Pipeline Anonimizacji
FROM python:3.11-slim

# Ustaw zmienne środowiskowe
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Zainstaluj zależności systemowe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Utwórz katalog roboczy
WORKDIR /app

# Skopiuj pliki konfiguracyjne projektu
COPY pyproject.toml ./

# Zainstaluj zależności projektu
# Najpierw spróbuj z uv (jeśli dostępne), potem fallback do pip
RUN pip install --no-cache-dir uv && \
    (uv pip install --system -e . 2>/dev/null || pip install --no-cache-dir -e .) || \
    pip install --no-cache-dir -e .

# Zainstaluj model spaCy dla polskiego
RUN python -m spacy download pl_core_news_sm

# Skopiuj kod źródłowy
# COPY src/ ./src/
# COPY data/ ./data/ 2>/dev/null || true
# COPY models/ ./models/ 2>/dev/null || true

# Utwórz katalogi na dane wyjściowe
RUN mkdir -p /app/outputs /app/data

# Ustaw punkt wejścia
ENTRYPOINT ["python", "-m", "src.scripts.run_pipeline"]

# Domyślne argumenty (można nadpisać)
CMD ["--help"]

