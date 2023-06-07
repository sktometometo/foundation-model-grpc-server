FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update \
    && apt-get dist-upgrade -q -y \
    && apt-get install -q -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY . /workspace/foundation-model-grpc-server
RUN git clone https://github.com/sktometometo/LLaMA-Adapter.git LLama -b v2_pacakged
RUN pip install --no-cache-dir /workspace/LLaMA-Adapter/requirements.txt
RUN pip install --no-cache-dir /workspace/LLaMA-Adapter/llama_adapter_v2_multimodal/requirements.txt
RUN pip install --no-cache-dir -e /workspace/LLaMA-Adapter/llama_adapter_v2_multimodal/
RUN pip install --no-cache-dir -e /workspace/foundation-model-grpc-server/
RUN ["download_lavis_model"]
RUN ["download_llama_adapter_server_model"]

CMD ["run_lavis_server"]
