FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5.21 /uv /uvx /bin/

# Copy project files
COPY . .

# Install the library and dependencies using uv
RUN uv sync --frozen --no-cache

EXPOSE 8000
EXPOSE 7860

CMD ["/app/.venv/bin/uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
