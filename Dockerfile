FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.lock pyproject.toml README.md ./
RUN pip install -r requirements.lock
COPY app app
COPY agents agents
COPY artifacts artifacts
COPY config config
COPY graph graph
COPY prompts prompts
COPY schemas schemas
COPY services services
COPY tools tools
RUN pip install --no-deps . && useradd --uid 10001 --create-home agentops && mkdir -p /data && chown agentops /data
USER agentops
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "60"]
