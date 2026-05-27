FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    AEGRA_RUNTIME_STORE_BACKEND=file \
    AEGRA_RUNTIME_STORE_DIR=/app/var/runtime \
    AEGRA_AUDIT_DIR=/app/var/audit

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        iputils-ping \
        nmap \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
