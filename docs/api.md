# API Service

Start the FastAPI service:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

API keys are configured in `.env`:

```env
DEMO_API_KEY=demo-api-key
ANALYST_API_KEY=analyst-api-key
```

For production, replace these values with random secrets.

## Upload a document

```bash
curl -X POST http://127.0.0.1:8000/v1/documents \
  -H "X-API-Key: demo-api-key" \
  -F "file=@samples/company_policy.txt"
```

## Chat

```bash
curl -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key" \
  -d "{\"question\":\"正式员工入职满三年后，每年有多少天带薪年假？\"}"
```

## Streaming chat

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-api-key" \
  -d "{\"question\":\"正式员工入职满三年后，每年有多少天带薪年假？\"}"
```

Each API key is bound to one workspace. Retrieval always filters by that workspace.
