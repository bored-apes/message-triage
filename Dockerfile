FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY triage ./triage
COPY data ./data
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
# Batch:  docker run --rm -e ANTHROPIC_API_KEY -v "$PWD/out:/app/out" triage python -m triage data/messages.json
# Server: docker run --rm -e ANTHROPIC_API_KEY -p 8000:8000 triage
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
