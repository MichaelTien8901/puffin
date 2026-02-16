FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[all]" || true

COPY . .

CMD ["bash"]
