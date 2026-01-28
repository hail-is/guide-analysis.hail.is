# use uv for installing packages
# use astral's debian-slim image for runtime
FROM ghcr.io/astral-sh/uv:debian-slim

# Set working directory
WORKDIR /app

ENV UV_NO_CACHE=1
ENV UV_NO_DEV=1
ENV UV_PYTHON=3.13

RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . .

EXPOSE 8000

CMD ["uv", "run", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
