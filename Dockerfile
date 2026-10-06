FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/model_bundle.joblib

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src ./src
COPY configs ./configs
# The trained bundle is produced by `python -m inactivity all` and committed to the repo.
COPY artifacts ./artifacts

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
# Hosting services set PORT; locally it defaults to 8000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT','8000')).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn inactivity.service.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
