# pi0.5 在 TB6-R5 上的推理

本仓库提供 `pi05_pp_white_red_h50` 的 JAX / PyTorch 推理服务、真机客户端、RTC 三段任务脚本和 LeRobot 录制入口。当前模型预测 50 步，训练数据的动作表示为 8 维 TCP 位姿及夹爪距离。真机运行前先检查相机、机器人状态和动作方向。

## 代码与第三方依赖

| 组件 | 本仓库位置或外部位置 | 用途 |
| --- | --- | --- |
| OpenPI RTC 源码 | `rtc_stage/openpi/` | JAX / PyTorch 模型及 RTC 采样；JAX Docker 挂载到 `/app/src/openpi` |
| OpenPI WebSocket 客户端 | `docker_source/packages/openpi-client/src/openpi_client/` | 策略请求与消息序列化；Docker 挂载到 `/app/packages/openpi-client/src/openpi_client` |
| 策略服务入口 | `jax_infer.py`、`docker_source/scripts/serve_policy.py` | 分别启动 JAX、PyTorch 服务 |
| TB6-R5 机器人 SDK | 宿主机相邻目录 `/home/ai/pi0.5/tb6r5_policy_infer/` | Topic 状态、RPC 运动、HTTP 相机辅助代码；真机客户端需要，策略服务容器不需要 |
| 模型权重 | `/home/ai/pi0.5/model/` | 不随 Git 仓库分发 |

OpenPI 上游项目：[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。机器人 SDK 接口示例：[臻识控制器 Python SDK](https://gitee.com/zhenshi-controller-api-data/sdk_send_recevice_demo_py)。本仓库的 RTC 实现与配置经过本地改动，不能直接用上游最新版替换。`app/` 是本机另外一份 OpenPI 工作树，已被 Git 忽略；JAX Docker Compose 使用仓库内已跟踪的 `rtc_stage/openpi/`。机器人 SDK 的二进制扩展不在仓库中，部署新机器时需准备与宿主机 Python、架构匹配的版本。

## 当前模型和端口

| 后端 | 服务端口 | checkpoint | 客户端入口 |
| --- | --- | --- | --- |
| PyTorch | `8002` | `/home/ai/pi0.5/model/pytorch/pp_white_red/19999` | `run_pytorch.sh` |
| JAX GPU | `8003` | `/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999` | `run_jax.sh`、`run_jax_rtc.sh` |
| PyTorch RTC | `8012` | 同上 PyTorch checkpoint | `run_pytorch_rtc.sh` |

四个入口默认用 `ROBOT_IP=192.168.2.151` 拼出两路 HTTP 相机 URL，端口为 `6888`。机器人状态 Topic 用 `19091`，运动 RPC 用 `5868`。`CAMERA_MODE=sdk` 对应的 CameraUpdatePb / `giant_frame_data` 仍是实验路径，当前真机默认不使用；详情见 [真机与录制说明](PI05_RTC_LEROBOT_使用说明.md)。

## JAX RTC：宿主机 GPU 服务

151 上 `/home/ai/condaenv/openpi/bin/python` 已能识别 JAX CUDA。终端一先启动策略服务；它只加载模型，不会连接运动 RPC：

```bash
cd /home/ai/pi0.5/pi05_infer
export OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/rtc_stage
export PYTHONPATH=/home/ai/pi0.5/pi05_infer/rtc_stage:/home/ai/pi0.5/pi05_infer/docker_source/packages/openpi-client/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false

/home/ai/condaenv/openpi/bin/python jax_infer.py serve \
  --checkpoint /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999 \
  --config pi05_pp_white_red_h50 \
  --asset-id . \
  --device gpu \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8003 \
  --allow-hardware
```

如果需要后台运行，使用同一命令加 `nohup`、输出日志和 `&`；进程 PID 应记录到本地 `.pid` 文件（Git 已忽略）。不要在已有服务占用 `8003` 时重复启动。

终端二检查服务并先做 dry-run：

```bash
cd /home/ai/pi0.5/pi05_infer
curl --noproxy '*' --fail http://127.0.0.1:8003/healthz
./run_jax_rtc.sh
```

确认相机映射、真实状态、动作和安全阈值后，执行三段红—白—红任务：

```bash
REAL=1 ./run_jax_rtc_red_white_red.sh
```

脚本先对 JAX RTC 做预热/编译，再以 `v60` 回到 `act_vla_0`，然后依次执行三段任务。每段通过真实 TCP 位置与夹爪反馈判断完成。默认 30 FPS 控制、50 步预测；`RTC_NUM_STEPS=4`、`RTC_MIN_EXECUTION_STEPS=1`、`RTC_GUIDANCE_BETA=10.0`。首次 JAX 编译可能明显慢于后续调用；预热阶段在机械臂动作之前。普通非 RTC 三段任务可用 `REAL=1 ./run_jax_red_white_red.sh`，其默认执行 horizon 为 25。

## JAX RTC：Docker GPU 服务

151 上已存在 `jax-orin:0.5.3-cuda` 镜像，并验证容器内 JAX 能识别 `CudaDevice(id=0)`。`compose.jax.yml` 默认使用这个镜像、JAX h50 checkpoint 和仓库内 RTC OpenPI / WebSocket 客户端代码。它监听 `8003`，与上面的宿主机服务二选一。镜像通过 `jax_gpu_build/Dockerfile.runtime` 在目标 Jetson 上构建；该构建目录包含本地轮子，没有纳入 Git，迁移到其他机器需要单独准备兼容的镜像。

如果 `docker ps` 显示手工创建的 `pi05-jax-rtc` 正在使用 `8003`，它已经是可用的 GPU JAX 服务；无需再次执行 Compose。确实要切换为 Compose 管理时，先结束正在使用该服务的客户端，再停止并移除旧容器，然后执行下面的 `up` 命令。

```bash
cd /home/ai/pi0.5/pi05_infer
docker compose -f compose.jax.yml config
docker compose -f compose.jax.yml up -d --wait
docker compose -f compose.jax.yml logs -f tb6r5_jax_policy_server
```

另开终端验证，再运行客户端：

```bash
curl --noproxy '*' --fail http://127.0.0.1:8003/healthz
./run_jax_rtc.sh
REAL=1 ./run_jax_rtc_red_white_red.sh
```

停止 Docker 服务：

```bash
docker compose -f compose.jax.yml down
```

使用其他 GPU 镜像或 checkpoint 时，设置 `JAX_IMAGE`、`JAX_CHECKPOINT_DIR`。`POLICY_CONFIG` 默认 `pi05_pp_white_red_h50`，`JAX_NUM_STEPS` 默认 10，`JAX_POLICY_PORT` 默认 8003。修改端口时，客户端也要用相同的 `POLICY_PORT`。`--allow-hardware` 只表示服务允许已确认的真机客户端连接；运动仍由客户端的 `REAL=1` 决定。

## PyTorch Docker 服务

当前 PyTorch 服务使用 `runtime/openpi/scripts/docker/compose.orin.tb6r5.yml`，模型在 `8002`。它与 JAX 服务是独立进程，启动 JAX 客户端不会自动启动 PyTorch 容器。

```bash
cd /home/ai/pi0.5/pi05_infer/runtime/openpi/scripts/docker
docker compose --env-file .env.orin.tb6r5 -f compose.orin.tb6r5.yml up -d --wait
curl --noproxy '*' --fail http://127.0.0.1:8002/healthz
```

PyTorch 普通推理入口是 `./run_pytorch.sh`；PyTorch RTC 使用 `rtc_stage/compose.rtc.yml` 的 `8012` 服务和 `./run_pytorch_rtc.sh`。这些入口默认是 dry-run；设置 `REAL=1` 才会进入真机控制。

## 录制与排查

独立录制一个 LeRobot episode：在一个终端运行 `./record_one_episode.sh`，另一终端运行推理；结束时在录制终端按一次 `Ctrl+C`，等待打印 `SAVED`。录制与推理的帧率设置、三段任务完成条件、参数含义和故障排查见 [PI05_RTC_LEROBOT_使用说明.md](PI05_RTC_LEROBOT_使用说明.md)。

查看 JAX 日志或进程：

```bash
ss -ltnp | grep ':8003'
docker ps --filter name=pi05-jax
```

如果 `8003` 无监听，先启动上述宿主机或 Docker GPU 服务；如果两种方式同时启动，后启动的一方会因端口冲突失败。
