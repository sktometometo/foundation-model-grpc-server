FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update \
    && apt-get dist-upgrade -q -y \
    && apt-get install -q -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace/LAVIS-grpc-server
RUN pip install --no-cache-dir -e /workspace/LAVIS-grpc-server/
RUN ["/workspace/LAVIS-grpc-server/scripts/download_model_cache.py"]

CMD ["/workspace/LAVIS-grpc-server/scripts/run_lavis_server.py"]
