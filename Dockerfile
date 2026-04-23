# Use Playwright's official image — comes with Chromium preinstalled
FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY infinite_dom/ ./infinite_dom/
COPY inference.py client.py openenv.yaml ./
COPY scripts/ ./scripts/

EXPOSE 8000 9000

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_HEADLESS=true \
    INFINITE_DOM_HOST=0.0.0.0 \
    INFINITE_DOM_PORT=8000

CMD ["uvicorn", "infinite_dom.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
