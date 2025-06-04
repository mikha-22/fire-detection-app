# src/ml_model/Dockerfile
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

LABEL maintainer="Wildfire Detection Team <dev@example.com>"
LABEL version="0.1.0"
LABEL description="TorchServe container for Wildfire Detection AI Model (PyTorch 2.7.0)"

ENV PATH="/opt/conda/bin:${PATH}"
ENV HOME="/home/model-server"
ENV PYTHONUNBUFFERED=TRUE
ENV TEMP_DIR_PATH="/tmp"

WORKDIR ${HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-11-jdk-headless curl wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY src/ml_model/requirements.txt .

RUN pip install --no-cache-dir --default-timeout=120 --retries=3 \
    "torchserve==0.12.0" \
    "torch-model-archiver==0.12.0" \
    -r requirements.txt

COPY src/ml_model/model.mar ./model_store/model.mar

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
  CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["torchserve"]
CMD ["--start", "--model-store", "/home/model-server/model_store", "--models", "model=model.mar", "--disable-token-authorization"]
