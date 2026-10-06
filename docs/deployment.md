# Deployment

## Docker Compose

Stop any standalone container already using ports 6333 or 8501, then run:

```powershell
docker compose up --build
```

Services:

```text
Qdrant   http://localhost:6333
API      http://localhost:8000
API docs http://localhost:8000/docs
UI       http://localhost:8501
```

Stop:

```powershell
docker compose down
```

Delete the local vector volume only when you intentionally want to remove indexed data:

```powershell
docker compose down -v
```

## Production Checklist

- Replace demo API keys with random values.
- Replace demo users with a real identity provider.
- Use Qdrant Cloud or a managed Postgres/pgvector service.
- Add TLS through a reverse proxy such as Caddy or Nginx.
- Configure Langfuse keys and cost dashboards.
- Restrict upload size and supported MIME types.
- Back up the Qdrant volume.
- Add log retention and rate limiting.
