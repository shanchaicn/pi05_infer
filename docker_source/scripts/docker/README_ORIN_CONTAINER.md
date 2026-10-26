# TB6R5 pi0.5 on AGX Orin

This deployment serves the existing converted PyTorch checkpoint as an OpenPI
WebSocket policy server. With host networking, OpenPI binds `0.0.0.0:8002`;
the local robot-side client connects to `127.0.0.1:8002` and already reads that
port from `/home/ai/pi0.5/pi05_infer/config.env`. Restrict TCP/8002 with the
host firewall if other LAN hosts must not reach the policy server.

## Host paths

- OpenPI build context: `/home/ai/pi0.5/pi05_infer/runtime/openpi`
- Inference checkpoint: `/home/ai/pi0.5/pi05_infer/policy`
- OpenPI/tokenizer cache: `/home/test/.cache/openpi`
- Compose environment: `scripts/docker/.env.orin.tb6r5`

The checkpoint is bind-mounted read-only.  The 13 GB training optimizer is not
needed and is not copied into the image.

## Why the Huazhong freeze is not installed directly

The source snapshot is Ubuntu 24.04/x86_64 and contains generic CUDA wheels,
`torch==2.7.1`, and `jax-cuda12` packages.  Installing it verbatim on Jetson
would replace NVIDIA's ARM64 CUDA stack.  `requirements.orin.txt` keeps the
application versions but deliberately omits Torch, TorchVision, CUDA, cuDNN,
and CUDA-JAX packages.

The image uses NVIDIA's JetPack 6.2-compatible ARM64 base:

```text
nvcr.io/nvidia/pytorch:25.04-py3-igpu
```

It retains NVIDIA CUDA Torch for BF16 pi0.5 inference and uses CPU JAX only for
OpenPI plumbing.  `torch.compile` is disabled for the first stable deployment.

## Build and start

The `test` account must be able to access `/var/run/docker.sock`.  If preflight
reports a permission failure, an administrator should run this once and the
SSH session must then be reconnected:

```bash
sudo usermod -aG docker test
```

Run on the Orin:

```bash
cd /home/ai/pi0.5/pi05_infer/runtime/openpi
scripts/docker/preflight.orin.sh
docker compose \
  --env-file scripts/docker/.env.orin.tb6r5 \
  -f scripts/docker/compose.orin.tb6r5.yml \
  build --pull
docker compose \
  --env-file scripts/docker/.env.orin.tb6r5 \
  -f scripts/docker/compose.orin.tb6r5.yml \
  up -d --wait --wait-timeout 600
scripts/docker/verify.orin.sh
```

This deployment intentionally provides only the Docker policy server. The
preflight script reports a failure when port `8002` is already occupied and
never stops an existing container or process automatically.

Useful operations:

```bash
docker compose --env-file scripts/docker/.env.orin.tb6r5 -f scripts/docker/compose.orin.tb6r5.yml logs -f
docker compose --env-file scripts/docker/.env.orin.tb6r5 -f scripts/docker/compose.orin.tb6r5.yml down
```
