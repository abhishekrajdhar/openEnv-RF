FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml /app/
COPY support_queue_env /app/support_queue_env
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 7860

CMD ["uvicorn", "support_queue_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
