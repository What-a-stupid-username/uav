ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV UAV_MODEL_DIR=/models PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TORCH_EXTENSIONS_DIR=/workspace/tmp/extensions TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_COMPILE_THREADS=4 USER=uav \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    PYTHONPATH=/workspace:/workspace/third_party/sam3
WORKDIR /opt/uav-env
RUN apt-get -o Acquire::http::Proxy=$HTTP_PROXY -o Acquire::https::Proxy=$HTTPS_PROXY update \
    && apt-get -o Acquire::http::Proxy=$HTTP_PROXY -o Acquire::https::Proxy=$HTTPS_PROXY install -y --no-install-recommends build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
# Application/model source is supplied by a read-only repository mount.
# Only dependency changes require rebuilding this image.
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "uav.cli"]
CMD ["--help"]
