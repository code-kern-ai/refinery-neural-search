ARG PARENT_IMAGE=registry.dev.kern.ai/code-kern-ai/refinery-parent-images:dev-common
FROM ${PARENT_IMAGE}

WORKDIR /app

VOLUME ["/app"]

USER root

COPY requirements*.txt .

RUN pip3 install --no-cache-dir -r requirements-dev.txt

COPY / .

USER 65532:65532

CMD ["/usr/local/bin/uvicorn", "--host", "0.0.0.0", "--port", "80", "app:app", "--reload"]
