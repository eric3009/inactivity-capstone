# Hosting the app

The container listens on `$PORT` (default 8000), serves the web interface at `/`, the API at `/score` and
`/score/transactions`, and a health check at `/health`. It needs no database, secrets or outside services, and
holds only public data (UCI Online Retail II, CC BY 4.0) and the trained model, about 3.4 MB.

There is no login, so anyone with the address can use it. That is fine for this public data; do not put
private customer data behind it without adding authentication.

## Option A: Render (simplest, free tier)

1. Push this repository to GitHub (the account and repository are yours to create).
2. On render.com choose New, then Blueprint, and select the repository. `render.yaml` does the rest.
3. Wait for the first build (about 10 minutes). The address will look like `https://inactivity-api.onrender.com`.
4. The free tier sleeps after about 15 minutes idle, and the first request after that takes 30 to 60 seconds.
   Open the address a few minutes before presenting.

## Option B: Google Cloud Run (scales to zero, one command)

With the `gcloud` CLI installed and a project with billing enabled:

```bash
gcloud run deploy inactivity-api --source . --region europe-west1 --allow-unauthenticated --port 8000
```

## Option C: Hugging Face Spaces (Docker)

Create a Space with the Docker SDK, add this front matter at the top of that Space's README, and push the repository:

```yaml
---
title: Customer inactivity risk
sdk: docker
app_port: 8000
---
```

## After deploying

Check `<address>/health` returns `{"status": "ok", ...}`, then open `<address>/` and click through the pages.
Run `python scripts/smoke_test_api.py <address>` to score a customer through it.
