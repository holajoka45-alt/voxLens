FROM python:3.11-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY pyproject.toml .
COPY voxlens/ voxlens/

# Install dependencies
RUN pip install --no-cache-dir -e ".[gpu]"

# Copy remaining files
COPY . .

# Default command
CMD ["pytest", "tests/", "-v", "-m", "not gpu"]
