FROM python:3.12-slim

# Install system dependencies for LightGBM and other ML libraries
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the PWD
WORKDIR /app

# Copy the Code
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY ./src ./src
COPY ./models ./models
RUN uv sync --frozen --no-cache

# Setting the ENV Path
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app"]

# Run the application
CMD ["--host", "0.0.0.0", "--port", "8000"]