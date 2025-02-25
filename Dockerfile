## Build time
# Use the specified Python runtime as a parent image
FROM docker.io/nvidia/cuda:12.1.0-devel-ubuntu22.04@sha256:e3a8f7b933e77ecee74731198a2a5483e965b585cea2660675cf4bb152237e9b AS build

# Set the working directory in the container
WORKDIR /usr/src/app
COPY packages.txt .

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# Delete nvidia apt list and Install required packages
RUN DEBIAN_FRONTEND=noninteractive \
    && rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    && apt-get -yq update \
    && apt-get install --no-install-recommends -yq \
        apt-transport-https \
        ca-certificates \
        $(cat packages.txt) \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual env
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
COPY gradio_image_prompter-0.1.0-py3-none-any.whl .
RUN --mount=type=cache,id=pip,target=/root/.cache \
    pip install -r requirements.txt

# Copy grounding dino ops
WORKDIR /usr/src/app/models/GroundingDINO/ops
COPY models/GroundingDINO/ops .

# Run the setup script and the test script
RUN CC=/usr/bin/gcc-11 python3 setup.py build && \
    pip install .

## Runtime
# Use the specified Python runtime as a parent image
FROM ubuntu:22.04

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/home/user/.local/bin:$PATH" \
    HOME=/home/user

RUN DEBIAN_FRONTEND=noninteractive apt-get -yq update && apt-get install --no-install-recommends -yq \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 1000 user && chown -R user /app

COPY --from=build --chown=user /opt/venv /opt/venv

COPY --chown=user checkpoints checkpoints
COPY --chown=user checkpoint_best_regular.pth .
COPY --chown=user *.jpg *.JPG ./
COPY --chown=user datasets datasets
COPY --chown=user groundingdino groundingdino
COPY --chown=user models models
COPY --chown=user util util
COPY --chown=user app.py cfg_app.py ./

USER user
# Expose the port Gradio will run on
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
# Default command to run the Gradio app
CMD ["/opt/venv/bin/python3", "app.py"]
