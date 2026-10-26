# syntax=docker/dockerfile:1.7

# NVIDIA's PyTorch 25.04 iGPU image is the JetPack 6.2 ARM64 build closest to
# OpenPI's torch 2.7.1 requirement.  The digest pins the verified arm64 manifest.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:d4a2f7f873bf0bbb72c2b15dc07860369c760dfda2348859826947bc71a88248
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off \
    OPENPI_VENV=/opt/openpi-venv \
    OPENPI_DATA_HOME=/openpi_assets \
    OPENPI_DISABLE_TORCH_COMPILE=1 \
    JAX_PLATFORMS=cpu \
    IS_DOCKER=true \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

# The default PyPI route is unreliable on this deployment network. Keep the
# mirror after the apt layer so system-package caching is unaffected.
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# The venv can see the NVIDIA image's CUDA-enabled torch.  A generated
# constraint file prevents dependency resolution from replacing that stack
# with generic PyPI CUDA packages.
RUN python3 -m venv --system-site-packages "${OPENPI_VENV}"
ENV PATH="${OPENPI_VENV}/bin:${PATH}"

COPY scripts/docker/requirements.orin.txt /tmp/requirements.orin.txt

RUN python - <<'PY' > /tmp/nvidia-stack.constraints.txt
from importlib import metadata

for package in ("torch", "torchvision"):
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        raise SystemExit(f"NVIDIA base image is missing required package: {package}")
try:
    print(f"torchaudio=={metadata.version('torchaudio')}")
except metadata.PackageNotFoundError:
    pass
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade "pip<27" setuptools wheel \
    && python -m pip install \
         --constraint /tmp/nvidia-stack.constraints.txt \
         --requirement /tmp/requirements.orin.txt

COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src
RUN python -m pip install --no-deps ./packages/openpi-client

COPY pyproject.toml README.md LICENSE LICENSE_GEMMA.txt ./
COPY src src
COPY scripts scripts
COPY examples examples

# Do not resolve OpenPI's original torch==2.7.1 / jax[cuda12] pins here: those
# are x86-oriented and would overwrite the validated Jetson GPU stack.
RUN python - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
replacements = {
    '"jax[cuda12]==0.5.3",': '"jax==0.5.3",',
    '"opencv-python>=4.10.0.84",': '"opencv-python-headless>=4.10.0.84",',
    '    "torch==2.7.1",\n': '',
}
for old, new in replacements.items():
    if text.count(old) != 1:
        raise SystemExit(f"unexpected OpenPI dependency declaration: {old!r}")
    text = text.replace(old, new)
path.write_text(text)
PY

RUN python -m pip install --no-deps --editable .

# OpenPI's PyTorch implementation requires these exact Transformers patches.
RUN python - <<'PY'
from pathlib import Path
import shutil
import transformers

source = Path("/app/src/openpi/models_pytorch/transformers_replace")
target = Path(transformers.__file__).resolve().parent
if not source.is_dir():
    raise SystemExit(f"missing OpenPI Transformers patches: {source}")
for item in source.iterdir():
    destination = target / item.name
    if item.is_dir():
        shutil.copytree(item, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(item, destination)
PY

# Build-time checks do not require a mounted GPU, but they ensure that pip did
# not replace NVIDIA's CUDA-enabled Torch and that JAX remains CPU-only.
RUN python - <<'PY'
import platform
from importlib import metadata
import jax
import lerobot.datasets.lerobot_dataset
import torch
import torchvision
import transformers
from torchvision.ops import nms

print("machine:", platform.machine())
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("jax:", jax.__version__, "backend:", jax.default_backend())
print("transformers:", transformers.__version__)
if platform.machine() not in {"aarch64", "arm64"}:
    raise SystemExit("this image must be built for ARM64")
if torch.version.cuda is None:
    raise SystemExit("generic CPU Torch replaced the NVIDIA Jetson build")
if transformers.__version__ != "4.53.2":
    raise SystemExit("OpenPI requires transformers==4.53.2")
if jax.default_backend() != "cpu":
    raise SystemExit("the Orin policy-server image expects CPU-only JAX plumbing")
nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5)
expected = {}
for line in open("/tmp/nvidia-stack.constraints.txt"):
    name, version = line.strip().split("==", 1)
    expected[name] = version
for name, version in expected.items():
    if metadata.version(name) != version:
        raise SystemExit(f"NVIDIA package was replaced: {name} {version} -> {metadata.version(name)}")
PY

RUN python -m pip check

EXPOSE 8002

ENTRYPOINT ["python", "scripts/serve_policy.py"]
CMD ["--port", "8002", "policy:checkpoint", "--policy.config=pi05_tb6r5", "--policy.dir=/models/pi05_tb6r5"]
