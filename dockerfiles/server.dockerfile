FROM python:3.12-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the PWD
WORKDIR /app

# Copy the Code
COPY ./pyproject.toml .
COPY ./uv.lock .
COPY . .
RUN uv sync --frozen --no-cache

# Setting the ENV Path
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["uv", "run", "uvicorn", "src.main:app"]

# Run the application
CMD ["--host", "0.0.0.0", "--port", "8000"]