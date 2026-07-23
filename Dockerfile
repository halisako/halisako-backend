# Halisako Chess2Fight backend — Render Web Service

FROM python:3.12-slim

WORKDIR /app

# System deps for building any C-extension wheels (e.g. pydantic-core
# has prebuilt wheels for this base image in practice, but keeping
# build-essential keeps this resilient if that ever changes).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets $PORT at runtime; default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

# Shell form (no brackets) so $PORT is expanded by the shell — Render
# assigns this dynamically and the app must bind to it.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
