# Minimal Dockerfile for LLM-first bot (no DB)
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv sync
COPY . .
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
EXPOSE 8080
CMD ["python", "fixed_bot.py"]
