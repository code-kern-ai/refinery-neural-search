ARG PARENT_IMAGE=python:3.11-slim
FROM ${PARENT_IMAGE}

WORKDIR /app

VOLUME ["/app"]

USER root

COPY requirements*.txt .

RUN pip3 install --no-cache-dir -r requirements-dev.txt

COPY / .

CMD ["/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app", "--reload"]
