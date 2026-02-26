FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY proto/ proto/

RUN pip install --no-cache-dir .

ENTRYPOINT ["python", "-m", "privateboost.grpc"]
