FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

RUN uv sync --frozen --no-cache

# Stage 2: Final minimal runner
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

EXPOSE 8000
EXPOSE 7860

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
