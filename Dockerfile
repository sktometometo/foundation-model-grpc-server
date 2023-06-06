FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update \
    && apt-get dist-upgrade -q -y \
    && apt-get install -q -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace/foundation-model-grpc-server
RUN pip install --no-cache-dir -e /workspace/foundation-model-grpc-server/
RUN ["download_model_cache"]

CMD ["run_lavis_server"]
