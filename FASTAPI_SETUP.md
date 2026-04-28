# FastAPI Setup

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the API server from the repository root:

```powershell
python -m uvicorn src.app.api:app --host 127.0.0.1 --port 8000 --reload
```

Useful URLs after startup:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/openapi.json
```

Optional environment variables:

```powershell
$env:AEGRA_RUNTIME_STORE_BACKEND='file'
$env:AEGRA_RUNTIME_STORE_DIR='D:\Aegra\var\runtime'
$env:AEGRA_CONTROL_API_TITLE='Aegra Control API'
$env:AEGRA_CONTROL_API_VERSION='0.1.0'
```
