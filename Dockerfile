FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG SKIP_MODEL_DOWNLOAD

RUN apt-get update \
    && apt-get dist-upgrade -q -y \
    && apt-get install -q -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace/foundation-model-grpc-server
RUN git clone https://github.com/sktometometo/LLaMA-Adapter.git /workspace/LLaMA-Adapter/ -b v2_pacakged
RUN pip install --no-cache-dir -r /workspace/LLaMA-Adapter/requirements.txt
RUN pip install --no-cache-dir -r /workspace/LLaMA-Adapter/llama_adapter_v2_multimodal/requirements.txt
RUN pip install --no-cache-dir -e /workspace/LLaMA-Adapter/llama_adapter_v2_multimodal/
RUN pip install --no-cache-dir -e /workspace/foundation-model-grpc-server/
RUN cd /workspace/foundation-model-grpc-server && make
RUN if [ -z "${SKIP_MODEL_DOWNLOAD}" ]; then download_lavis_model; download_llama_adapter_server_model; fi

CMD ["run_lavis_server"]
