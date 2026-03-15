FROM python:3.12-slim

# Install system dependencies for audio, image, and Git (for HF)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Install the library and dependencies
RUN pip install --no-cache-dir .

# Expose ports for API (8000) and Gradio (7860)
EXPOSE 8000
EXPOSE 7860

# Default to running the API, can be overridden to run 'serve' or 'gradio'
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
