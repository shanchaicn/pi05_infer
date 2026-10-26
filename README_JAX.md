# pi05_infer 独立 JAX 推理

`jax_infer.py` 复用仓库里的 OpenPI 模型、JIT 和数据变换，提供
`check`、`infer`、`serve`、`client` 四个命令。未设置 `OPENPI_ROOT` 时，模型代码
来自 `rtc_stage/`（JAX 与 PyTorch 共用的 RTC 实现），客户端来自
`docker_source/packages/openpi-client`；若设置了 `OPENPI_ROOT`，则使用该目录下的
`src/openpi`。它直接从 `params/` 恢复 JAX 权重，并通过 `Policy(..., is_pytorch=False)`
执行推理。同目录即使有 `model.safetensors`，也不会切换到 PyTorch。服务元数据里的
`rtc_supported` 在 JAX `Pi0.sample_actions_rtc` 可用时为 true，主机可以用 `--rtc`
连到这个服务。

流程：原始 observation → 训练配置的数据输入变换 → Normalize → 模型输入变换 →
JAX JIT/sample_actions → 模型输出变换 → Unnormalize → 训练配置的数据输出变换。
状态维度 `S` 和动作维度 `A` 分别从 checkpoint 的 `state.mean`、`actions.mean`
读取，输出为 `(T, A)`，不固定为 7 或 8 维。脚本不解释关节、位姿或夹爪语义，
训练配置中的 delta/absolute action 变换会保留。

该入口用于模型推理和 WebSocket 通信，不包含机器人状态采集、实时相机采集、
RPC 下发或硬件控制循环。已有 `run_pi05_tb6r5.py` 引用的 `openpi_runner.py`
在当前工作区缺失；运行这里的命令不依赖该旧入口。

## 运行前提

需要与训练配置匹配的 **JAX checkpoint**，目录例如：

```text
/absolute/path/pi05_tb6r5_jax/2000/
├── params/                 # OpenPI/Orbax JAX 参数，包含元数据和权重分片
└── assets/
    └── [训练配置的 asset_id/]
        └── norm_stats.json
```

当前 `model/2000` 是 PyTorch checkpoint，没有 `params/`，不能直接用于该入口。
不要通过改文件名、创建空 `params/` 或删除 `model.safetensors` 来转换权重。
应使用 JAX 训练导出的 checkpoint，或另行完成经过验证的权重转换。

必须用 `--config` 指定训练时的配置，不再默认选择旧的关节配置。
统计量默认从训练配置的 `asset_id` 对应目录读取，也可通过 `--asset-id` 指定。
`check` 未指定 `--asset-id` 时自动查找唯一的 `assets/**/norm_stats.json`；
存在多份统计量时需要指定 asset，避免选错。

你新模型的表示是 **xyz(3) + 四元数(4) + 夹爪距离(1)，共 8 维**；
xyz 和夹爪距离使用米，四元数的分量顺序必须与训练一致。
脚本不会重排或归一化四元数，也不会把夹爪距离乘以 1000。
8 维数据与旧的 7 维关节数据是不同的表示，不能套用旧配置的关节 delta mask。
源码现有的 `pi05_tb6r5` 仍使用旧的关节变换；请指定新模型实际使用的训练配置。
脚本不会为适配输出而截断或补齐维度；这类变换属于训练配置，输出维度不匹配时会报错。

运行环境需要 OpenPI 的依赖。虽然执行 JAX 推理，这份 OpenPI 源码仍会导入 Torch，
因此 Torch 也必须能正常导入。现有 Orin 镜像含可用的 OpenPI 依赖，但 JAX 是 CPU 版；
将 `JAX_PLATFORMS` 改为 `gpu` 不会自动安装 CUDA 后端。
GPU 推理需要另备兼容机器架构、JetPack/CUDA、JAX 版本的运行环境或镜像。

## 本机直接推理

在可正常导入 OpenPI 的 Python 环境中运行：

```bash
cd /home/ai/pi0.5/pi05_infer
python jax_infer.py check \
  --checkpoint /absolute/path/pi05_tb6r5_jax/2000 --device gpu

# 黑图和统计量中的 mean state 只验证模型调用，输出不用于机器人控制。
python jax_infer.py infer \
  --checkpoint /absolute/path/pi05_tb6r5_jax/2000 \
  --device gpu --config YOUR_TRAINING_CONFIG --num-steps 10 \
  --smoke-test --task "Pick up the yellow ring." \
  --output actions.npz
```

`check` 只检查目录、统计量和 JAX 设备，不加载整个模型；权重树与模型结构是否匹配
会在 `infer` 或 `serve` 加载时检查。首轮推理包含 JIT 编译，可能明显慢于后续推理。
使用 `--device cpu` 可在 CPU JAX 环境中验证，但完整模型会消耗较多内存和时间。

两张图片的便捷输入使用 `observation/image`、`observation/wrist_image`、
`observation/state` 和 `prompt`。仅在训练配置接受这些字段时使用此模式；图片转为 RGB，
缩放由模型变换处理。以下 state **仅以四元数顺序 xyzw 为例**，实际必须遵循训练顺序：

```bash
python jax_infer.py infer \
  --checkpoint /absolute/path/pi05_tb6r5_jax/2000 --device gpu \
  --config YOUR_TRAINING_CONFIG \
  --state="0.3,-0.1,0.4,0,0,0,1,0.04" \
  --base-image /absolute/path/realsense_0.png \
  --wrist-image /absolute/path/realsense_1.png \
  --task "Pick up the yellow ring."
```

上面的 state 只是格式示例。state 和返回动作使用训练时的表示及单位，程序不会
把它们当作经过硬件安全校验的控制指令。

任意训练数据字段可通过 `--observation` 传入 `.npz` 文件，字段名和值原样传给
训练配置的数据输入变换。例如，训练变换接受自定义 state 和 camera 字段时：

```python
import numpy as np

np.savez("observation.npz", **{
    "my_state": np.array([0.3, -0.1, 0.4, 0, 0, 0, 1, 0.04], dtype=np.float32),
    "my_camera": np.zeros((224, 224, 3), dtype=np.uint8),
    "prompt": "Pick up the yellow ring.",
})
```

```bash
python jax_infer.py infer \
  --checkpoint /absolute/path/pi05_tb6r5_jax/2000 --device gpu \
  --config YOUR_TRAINING_CONFIG --observation observation.npz
```

替换上述 `my_state`、`my_camera` 为训练变换实际使用的字段名。`--task` 可覆盖
文件中的 `prompt`。文件采用 NumPy 非 pickle 格式。脚本在训练输入变换完成后
检查 state 维度，在训练输出变换完成后检查动作维度，不绑定某个机器人数据配置。

## WebSocket 服务

新服务默认端口是 **8003**，可与原有 8002 端口的 PyTorch 服务并存。

```bash
python jax_infer.py serve \
  --checkpoint /absolute/path/pi05_tb6r5_jax/2000 \
  --config YOUR_TRAINING_CONFIG --device gpu --port 8003

# 在另一个终端运行；客户端只需 requirements.host.txt 中的基础依赖。
python jax_infer.py client --policy-port 8003 \
  --timeout 300 --smoke-test --task "Pick up the yellow ring."
```

服务端使用 OpenPI 原有 NumPy/msgpack WebSocket 协议和 `/healthz` 健康检查。
客户端会核实服务元数据中的 `backend=jax`，避免误连旧的 PyTorch 服务。
真实图片的客户端参数与直接推理相同。服务加载完模型后才监听；首个请求仍需等待 JIT。
客户端的维度和 smoke state 从服务元数据读取，不会自行假设 7 或 8 维。

## Docker

独立配置位于 `compose.jax.yml`，通过挂载本地代码运行，不使用旧 `manage.sh`。

```bash
cd /home/ai/pi0.5/pi05_infer
export JAX_CHECKPOINT_DIR=/absolute/path/pi05_tb6r5_jax/2000
export POLICY_CONFIG=YOUR_TRAINING_CONFIG

# 现有 Orin 镜像：CPU JAX 验证。
JAX_DEVICE=cpu docker compose -f compose.jax.yml up -d
docker compose -f compose.jax.yml logs -f
```

GPU 模式需要已经安装了 JAX CUDA 后端的兼容镜像，配置中的 `JAX_IMAGE` 是该镜像名，
不会自动下载、安装依赖或将现有 CPU 镜像转换为 GPU 镜像：

```bash
JAX_IMAGE=your-compatible-jax-image:tag JAX_DEVICE=gpu \
  docker compose -f compose.jax.yml up -d
```

必须设置 `POLICY_CONFIG`。可设置 `JAX_POLICY_PORT`（默认 8003）和
`JAX_NUM_STEPS`（默认 10）。GPU 初始化失败时进程会报错退出，不会退回 CPU 或 PyTorch。

## 验证

基础输入输出、checkpoint 格式和真实 WebSocket 通信检查：

```bash
python -m unittest discover -s . -p test_jax_infer.py -v
```

在完整 OpenPI 环境中额外运行 JAX 集成检查：

```bash
JAX_PLATFORMS=cpu PI05_TEST_JAX=1 \
  python -m unittest discover -s . -p test_jax_infer.py -v
```

集成检查创建临时的微型模型，验证真实 Orbax 参数恢复、OpenPI 数据变换、JIT 和
动作输出，并验证混有 PyTorch 文件时仍只加载 JAX 参数。它不等价于完整 pi0.5
checkpoint 的 GPU 推理验证。
