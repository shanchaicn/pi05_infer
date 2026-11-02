# π0.5 在 TB6-R5 上的 PyTorch / JAX / RTC / LeRobot 使用说明

本文按 Jetson 设备 `test@192.168.2.151` 上当前代码编写，项目目录为：

```text
/home/ai/pi0.5/pi05_infer
```

## h50 真机快速运行

当前四个快捷脚本位于项目根目录：

```text
run_pytorch.sh       PyTorch 普通推理
run_pytorch_rtc.sh   PyTorch RTC 推理
run_jax.sh           JAX 普通推理
run_jax_rtc.sh       JAX RTC 推理
```

脚本默认使用当前 h50 配置和 checkpoint，并且默认启用 `--dry-run`。四个脚本默认使用由 `ROBOT_IP` 拼装的 HTTP MJPEG 相机 URL。开始前先检查两路 URL 并检查模型：

```bash
cd /home/ai/pi0.5/pi05_infer

curl --noproxy '*' -I --max-time 3 \
  http://192.168.2.151:6888/RsCameraSensor/0/0/color
curl --noproxy '*' -I --max-time 3 \
  http://192.168.2.151:6888/RsCameraSensor/1/0/color

find /home/ai/pi0.5/model/pytorch/pp_white_red/19999 -maxdepth 2 -type f
find /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999 -maxdepth 2 -type f
```

两条请求应返回 `HTTP/1.1 200 OK` 和 `multipart/x-mixed-replace`。

### 15.1 PyTorch 普通

启动服务并确认 checkpoint：

```bash
./manage.sh down
./manage.sh start
./manage.sh status
curl --noproxy '*' http://127.0.0.1:8002/healthz
```

先做只读 dry-run：

```bash
./run_pytorch.sh
```

确认输出、相机和动作映射后，才允许真机运行：

```bash
REAL=1 ./run_pytorch.sh
```

### 15.2 PyTorch RTC

启动 RTC 服务：

```bash
docker compose -f rtc_stage/compose.rtc.yml down
docker compose -f rtc_stage/compose.rtc.yml up -d --wait
curl --noproxy '*' http://127.0.0.1:8012/healthz
```

先做 dry-run：

```bash
./run_pytorch_rtc.sh
```

真机运行：

```bash
REAL=1 ./run_pytorch_rtc.sh
```

`run_jax_rtc.sh` 针对当前 h50 JAX GPU 服务默认使用 `fps=30`、
`rtc-num-steps=4`、`rtc-min-execution-steps=1`、`rtc-guidance-beta=10.0`、
`rtc-latency-margin-steps=2`、`rtc-latency-floor-s=0.5`。
控制与旁路 LeRobot 录制均按30 FPS 工作。可通过同名环境变量覆盖。

`rtc-min-execution-steps` 是下限。runner 会根据预热后的稳定推理延迟和30 Hz控制频率自动选择并固定
`request_step` 和 `delay_budget`，整次运行复用同一组 JAX 编译参数。`rtc-latency-margin-steps=2`
会在实测延迟之外再保留2个控制周期。当前约 `0.5s` 的推理延迟下，通常会得到约
`delay_budget=18`、`request_step=20`；以启动日志的实际值为准。
0.5秒延迟下限会让预热直接使用18/20，避免先编译较小的15/17组合再重新编译。
普通策略先做一次冷调用，再从新观测生成实际初始动作；guided sampler也固定调用两次，第二次用于测量
编译后的稳定延迟。缓存命中时这四次调用约需2秒。
启动时如果遇到尚未编译过的参数组合，预热可能持续几十秒；预热在连接机械臂之前完成。
看到下面这行后才会打开真机命令通道：

```text
[RTC] Fixed session parameters: request_step=20, delay_budget=18, horizon=50
```

2026-10-31 的30 FPS 单 episode（1036帧，红/白/红三段）中，54次在线推理平均耗时
`0.517s`、P95 `0.541s`，原 `delay_budget=16` 的每次推理都实际跳过17或18步。
因此默认增加2步延迟余量；当前三阶段实验默认使用 guidance beta 10。
`RTC_NUM_STEPS=4` 保持不变，避免额外增加推理时间。

机械臂 SDK 打印的 `No valid pending response for seqID` 不代表 RTC 策略失败；判断 RTC 是否
正常应查看是否持续出现 `Requested chunk` 和紧随其后的 `chunk N: ... skipped ...`。

### 15.3 JAX 普通和 JAX RTC

JAX 服务必须使用 h50 checkpoint 和配置：

```text
checkpoint=/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999
config=pi05_pp_white_red_h50
```

JAX 普通服务使用 `/home/ai/pi0.5/pi05_infer/app` 源码，JAX RTC 服务使用
`/home/ai/pi0.5/pi05_infer/rtc_stage` 源码；启动命令仍按第 9.2 和第 9.3 节执行，
但必须将 `--checkpoint` 和 `--config` 替换为上面的 h50 值。服务健康后分别运行：

```bash
./run_jax.sh
./run_jax_rtc.sh
```

确认 dry-run 后，真机调用方式分别为：

```bash
REAL=1 ./run_jax.sh

REAL=1 ./run_jax_rtc.sh
```

### 15.4 真机前置条件

h50 模型的 `action_dim=32`，而当前 TB6-R5 控制桥接的安全动作接口是 8 维末端位姿。
因此在完成并验证 32→8 维动作映射前，不能用 `REAL=1`。必须先确认：

1. policy metadata 和 runner 都接受 32 维动作；
2. 训练时的 32 维动作中哪些维度对应位置、姿态和夹爪；
3. runner 的安全检查与训练单位、四元数顺序一致；
4. dry-run 连续输出稳定，且动作没有越界或突跳。

任何时刻只能运行一个真实机器人 runner。停止时使用 `Ctrl+C`，确认 RPC 已断开后再启动另一种模式。

当前模型配置和 checkpoint：

```text
训练配置名：pi05_pp_white_red_h50
PyTorch：/home/ai/pi0.5/model/pytorch/pp_white_red/19999
JAX：    /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999
```

当前 h50 checkpoint 的模型动作维度为 32，具体动作语义必须以训练时的数据变换为准。旧的 8 维末端位姿说明只适用于 `pi05_pp_white_red_fft`，不能直接套用到 h50：

```text
[x, y, z, qx, qy, qz, qw, gripper_distance_m]
```

## 1. 固定地址、端口和相机映射

### 1.1 网络地址

| 用途 | 地址或端口 |
|---|---|
| 机器人 IP | `192.168.2.151` |
| 机器人运动 RPC | `192.168.2.151:5868` |
| 机器人 Topic 状态 | `tcp://192.168.2.151:19091` |
| 相机 HTTP 服务 | `${ROBOT_IP}:6888`（可用 `CAMERA_IP`/`CAMERA_PORT` 覆盖） |
| PyTorch 普通策略服务 | `8002` |
| JAX 策略服务 | `8003` |
| PyTorch RTC 策略服务 | `8012` |

策略服务端口可以同时存在，但任何时刻只能运行一个真实机器人控制 runner，避免多个进程同时连接 `5868` 下发动作。

### 1.2 相机模式与映射

四个快捷脚本默认使用 HTTP：

```text
CAMERA_MODE=http
realsense_0=http://${ROBOT_IP}:6888/RsCameraSensor/0/0/color
realsense_1=http://${ROBOT_IP}:6888/RsCameraSensor/1/0/color
```

正常运行仍保持原来的简单命令：

```bash
./run_pytorch.sh
./run_pytorch_rtc.sh
./run_jax.sh
./run_jax_rtc.sh
```

HTTP 地址从 `ROBOT_IP` 生成；需要相机服务位于其他主机时才设置 `CAMERA_IP`：

```text
realsense_0=http://${CAMERA_IP:-$ROBOT_IP}:6888/RsCameraSensor/0/0/color
realsense_1=http://${CAMERA_IP:-$ROBOT_IP}:6888/RsCameraSensor/1/0/color
```

`CAMERA_MODE=sdk`（`CameraUpdatePb` / `giant_frame_data` Topic）目前**尚未作为受支持的推理相机路径**。虽然已验证控制器可发布 PNG 帧、SDK 可接收并解码，但帧率约 4.2 Hz、跨机器 Wi-Fi 接收更低，且厂商原生扩展退出稳定性尚未完成验证。因此不要在真机推理命令中设置 `CAMERA_MODE=sdk`；若后续继续开发，应先用独立只读测试验证 Topic，再启用该实验模式。

## 2. 推荐的终端和环境

进入项目：

```bash
ssh test@192.168.2.151
cd /home/ai/pi0.5/pi05_infer
```

运行机器人 runner 时使用 Python 3.10 环境：

```text
/home/ai/condaenv/tb6r5/bin/python
```

该环境与 ARM RPC/Topic SDK 兼容。`openpi_orin` 是 Python 3.12，虽然部分命令可以运行，但真机 RPC 曾出现连接兼容问题，因此本文的真机命令统一使用 `tb6r5` 环境。

先做本地依赖检查：

```bash
/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --local-check --no-camera
```

仅在使用 `CAMERA_MODE=http` 或 HTTP 回退时检查相机 HTTP 服务：

```bash
curl --noproxy '*' -I --max-time 3 \
  http://192.168.11.11:6888/RsCameraSensor/0/0/color

curl --noproxy '*' -I --max-time 3 \
  http://192.168.11.11:6888/RsCameraSensor/1/0/color
```

两条命令应返回 `HTTP/1.1 200 OK` 和 `multipart/x-mixed-replace`。

runner 默认对每路 HTTP 相机最多尝试 3 次，每次等待 5 秒：

```bash
--camera-timeout-s 5 \
--camera-retries 3
```

短暂网络抖动时会打印 `[CAMERA] ... retrying` 并自动重试。当 HTTP multipart/JPEG 流在一帧尚未收完时断开，Python 可能报 `IncompleteRead(0 bytes read)`；当前 root 和 RTC runner 已将它与其他 `HTTPException` 一起按相机短断线处理并重试。这类错误是相机 HTTP 传输中断，不是 checkpoint 或归一化错误。

如果 6888 端口可以连接但持续收不到任何字节，说明机器人侧相机 HTTP 发布服务没有输出画面；增加重试次数不能恢复服务，需要先恢复相机发布程序。

## 四个简化运行脚本

项目根目录提供四个脚本，默认使用 `dry-run`，不会向机器人发送运动 RPC：

```bash
cd /home/ai/pi0.5/pi05_infer

./run_pytorch.sh       # PyTorch 普通推理，连接 8002
./run_pytorch_rtc.sh   # PyTorch RTC，连接 8012
./run_jax.sh           # JAX 普通推理，连接 8003
./run_jax_rtc.sh       # JAX RTC，连接 8003
```

脚本默认使用：

```text
配置：pi05_pp_white_red_h50
JAX checkpoint：/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999
PyTorch checkpoint：/home/ai/pi0.5/model/pytorch/pp_white_red/19999
```

需要先启动对应的策略服务。脚本只负责启动 runner；服务启动命令见第 4、5、9 节。
可以通过环境变量覆盖任务、推理次数、端口和 checkpoint，例如：

```bash
TASK='Pick up the blue block' MAX_INFERENCE_CALLS=10 ./run_pytorch.sh
```

只有明确设置 `REAL=1` 并传入硬件确认参数时，脚本才会移除 `--dry-run`：

```bash
REAL=1 ./run_pytorch.sh
```

当前 h50 模型是 32 维动作。运行真机前必须确认 runner 的动作映射与训练配置一致；旧的 8 维动作安全说明不能直接用于该模型。

## 3. 动作安全阈值

本文的真机示例统一放宽为：

```bash
--joint-step-max-rad 0.1 \
--ee-step-max-m 0.1 \
--rotation-step-max-rad 0.3
```

参数含义：

| 参数 | 单位 | 作用 |
|---|---:|---|
| `--joint-step-max-rad 0.1` | rad | 关节空间相邻动作最大变化 |
| `--ee-step-max-m 0.1` | m | 末端位置相邻动作最大变化，`0.1` 即 10 cm |
| `--rotation-step-max-rad 0.3` | rad | 相邻动作的末端旋转最大变化 |

当前模型使用 `--action-space ee_pose`，所以位置主要由 `--ee-step-max-m` 检查，姿态由 `--rotation-step-max-rad` 检查。`--joint-step-max-rad` 只在 `--action-space joints` 时生效。

这些参数只放宽单步跳变检查，不会关闭以下检查：

- state/action 的位置和夹爪是否落在 checkpoint 训练统计范围附近；
- 四元数是否合法；
- Topic 反馈是否存在或过期；
- RTC 推理是否来得及覆盖当前动作块；
- RPC 是否成功。

训练数据的夹爪动作约为 0–70 mm；当前真机夹爪上限按实际配置设为 65 mm。192.168.2.151 上普通和 RTC runner 的默认值为：

```bash
--gripper-min-distance 0 \
--gripper-max-distance 65
```

即使命令行省略上述两个参数，也会使用 0–65 mm。重跑 VLA 前仍应完成夹爪 65→40→10 mm 的独立长持测试。详细证据和测试顺序见同目录的 `PI05_真机失败排查报告.md`。

## 4. 启动 PyTorch 普通策略服务（8002）

8002 用于普通分块推理，不支持当前 RTC 协议。其配置来自：

```text
runtime/openpi/scripts/docker/.env.orin.tb6r5
```

当前关键值为：

```text
POLICY_CONFIG=pi05_pp_white_red_h50
POLICY_PORT=8002
CHECKPOINT_HOST_DIR=/home/ai/pi0.5/model/pytorch/pp_white_red/19999
```

启动并等待健康检查：

```bash
cd /home/ai/pi0.5/pi05_infer
./manage.sh start
```

查看状态：

```bash
./manage.sh status
curl --noproxy '*' http://127.0.0.1:8002/healthz
```

查看日志：

```bash
./manage.sh logs
```

停止：

```bash
./manage.sh stop
```

确认 8002 加载的模型：

```bash
docker inspect pi05-infer-tb6r5_policy_server-1 \
  --format '{{json .Config.Cmd}}'

docker inspect pi05-infer-tb6r5_policy_server-1 \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

应看到配置 `pi05_pp_white_red_h50`，以及：

```text
/home/ai/pi0.5/model/pytorch/pp_white_red/19999 -> /models/pi05_tb6r5
```

## 5. 启动 PyTorch RTC 服务（8012）

RTC PyTorch 服务使用包含 `sample_actions_rtc` 的源码：

```text
/home/ai/pi0.5/pi05_infer/rtc_stage/openpi
```

启动：

```bash
cd /home/ai/pi0.5/pi05_infer
docker compose -f rtc_stage/compose.rtc.yml up -d --wait
```

检查：

```bash
curl --noproxy '*' http://127.0.0.1:8012/healthz
docker compose -f rtc_stage/compose.rtc.yml ps
```

查看日志：

```bash
docker compose -f rtc_stage/compose.rtc.yml logs -f policy
```

停止：

```bash
docker compose -f rtc_stage/compose.rtc.yml down
```

当前 8012 服务元数据包含 `rtc_supported=True`。如果 runner 报：

```text
Policy server does not advertise RTC support
```

通常是连到了 8002，或者服务没有挂载 `rtc_stage/openpi`。

## 6. PyTorch 普通推理

### 6.1 只读 dry-run：真实相机和真实状态，不发送 RPC

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --dry-run \
  --robot-ip 192.168.2.151 \
  --policy-port 8002 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --n-action-steps 10 \
  --max-inference-calls 3 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

`--print-rpc` 在 dry-run 中只打印 `would send`，不会建立运动 RPC 连接。

### 6.2 真机持续运行

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 \
  --policy-port 8002 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --fps 30 \
  --n-action-steps 10 \
  --max-inference-calls 0 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

`--max-inference-calls 0` 表示持续运行，直到 `Ctrl+C`、安全检查失败或服务断开。

## 7. PyTorch RTC 推理

RTC 使用 10 步动作块。runner 在执行当前动作块时并行请求下一块，并根据已过去的控制 tick 跳过过期动作。

RTC 关键参数：

| 参数 | 当前默认值 | 说明 |
|---|---:|---|
| `--rtc` | 关闭 | 启用 RTC |
| `--rtc-num-steps` | 5 | RTC 采样/去噪步数 |
| `--rtc-min-execution-steps` | 2 | 当前块至少执行多少步后请求下一块 |
| `--rtc-guidance-beta` | 5.0 | RTC guidance 强度 |
| `--fps` | 30 | 控制 tick；RTC 不建议直接使用默认 30 Hz |

当前 Jetson 上 PyTorch 单次推理约 1～2 秒，推荐先从 `--fps 1` 开始。程序按以下方式估算延迟：

```text
delay_steps = ceil(最近推理延迟 × fps) + 1
```

如果需要跳过的步数超过当前动作块剩余步数，程序会报 `RTC latency needs ...` 并停止。出现此错误时优先降低 `--fps`，其次减少 `--rtc-num-steps`。

RTC 模式不使用 `--n-action-steps` 控制动作块消费数量；它按 RTC overlap 逻辑运行。

### 7.1 PyTorch RTC dry-run

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --dry-run \
  --rtc \
  --rtc-num-steps 5 \
  --rtc-min-execution-steps 2 \
  --rtc-guidance-beta 5.0 \
  --fps 1 \
  --robot-ip 192.168.2.151 \
  --policy-port 8012 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --max-inference-calls 3 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

### 7.2 PyTorch RTC 真机持续运行

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --rtc \
  --rtc-num-steps 5 \
  --rtc-min-execution-steps 2 \
  --rtc-guidance-beta 5.0 \
  --fps 1 \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 \
  --policy-port 8012 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --max-inference-calls 0 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

## 8. LeRobot v3 录制

录制功能适用于普通推理和 RTC。每一帧包含：

```text
observation.images.front    480×640 RGB 视频
observation.images.wrist    480×640 RGB 视频
observation.state           8D，动作下发前状态
action                      8D，发送/拟发送的目标动作
task                        文本任务
```

真机模式只在 RPC 发送成功后记录该动作；dry-run 保存的是模型预测但未执行的动作，因此 dry-run 数据不能当作实际执行示教轨迹。

每次必须使用新的空目录：

```bash
OUT="/home/ai/pi0.5/data/lerobotdata/pi05_run_$(date +%Y%m%d_%H%M%S)"
```

录制参数：

```bash
--record-dataset "$OUT" \
--record-fps 1 \
--record-episode-frames 1000
```

`--record-fps` 是写入数据集的 FPS 元数据，并不会对采集进行抽帧。建议设置成与 runner 的 `--fps` 一致。

当前录制器使用一个容量为 32 帧的异步内存队列。达到 `--record-episode-frames` 时，LeRobot 会执行 `Map` 和 H.264 编码；编码期间如果控制循环继续快速产生超过 32 帧，程序会报：

```text
Recording queue full: writer cannot keep up
```

并执行安全退出。为了减少运行过程中触发编码，短时测试推荐将 `--record-episode-frames` 设置为明显大于预计总帧数，例如 `1000`。退出时仍会保存剩余帧，因此按 `Ctrl+C` 后需要等待 `[RECORD] SAVED ...`，不要立即关闭终端或强杀进程。

### 8.1 PyTorch RTC + 真机 + LeRobot 录制

```bash
cd /home/ai/pi0.5/pi05_infer
OUT="/home/ai/pi0.5/data/lerobotdata/pi05_pt_rtc_$(date +%Y%m%d_%H%M%S)"

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --rtc \
  --rtc-num-steps 5 \
  --rtc-min-execution-steps 2 \
  --rtc-guidance-beta 5.0 \
  --fps 1 \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 \
  --policy-port 8012 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --max-inference-calls 0 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --record-dataset "$OUT" \
  --record-fps 1 \
  --record-episode-frames 1000 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

### 8.2 普通 PyTorch + 真机 + LeRobot 录制

普通模式如果 `--fps 30 --n-action-steps 10`，应将 `--record-fps` 设为 `30`：

```bash
cd /home/ai/pi0.5/pi05_infer
OUT="/home/ai/pi0.5/data/lerobotdata/pi05_pt_normal_$(date +%Y%m%d_%H%M%S)"

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 \
  --policy-port 8002 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --fps 30 \
  --n-action-steps 10 \
  --max-inference-calls 0 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --record-dataset "$OUT" \
  --record-fps 30 \
  --record-episode-frames 1000 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```



### 8.3 检查录制结果

```bash
/home/ai/condaenv/tb6r5/bin/python - "$OUT" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
print("codebase_version:", info["codebase_version"])
print("total_episodes:", info["total_episodes"])
print("total_frames:", info["total_frames"])
print("fps:", info["fps"])
print("features:", sorted(info["features"]))
PY
```

## 9. JAX 服务和 JAX RTC

### 9.1 当前 Jetson 上的实际状态

JAX checkpoint 已存在：

```text
/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999/params
/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999/assets/norm_stats.json
```

宿主机环境已经验证可以使用 Orin GPU：

```text
Python：/home/ai/condaenv/openpi/bin/python
JAX：0.5.3
设备：CudaDevice(id=0)
默认 backend：gpu
```

这是 JAX 推理的推荐环境。另有两套本地 Docker 镜像：

```text
openpi_tb6r5_policy_server:orin-jp62
openpi-jax-builder:0.5.3-orin
```

在 2026-10-26 实际检查时，这两套 Docker 镜像中的 `jax.devices()` 都只返回 `CpuDevice(id=0)`。因此：

- 宿主机 `/home/ai/condaenv/openpi` 可以运行 JAX GPU；
- 当前 `compose.jax.yml` 使用的 Docker 镜像只能做 JAX CPU 验证；
- JAX GPU 和 JAX RTC GPU 优先使用下面的宿主机启动方式。

### 9.2 宿主机启动 JAX GPU：不带 RTC

不带 RTC 时使用普通源码：

```text
OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/app
```

启动命令：

```bash
cd /home/ai/pi0.5/pi05_infer

export PY=/home/ai/condaenv/openpi/bin/python
export OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/app
export PYTHONPATH=/home/ai/pi0.5/pi05_infer/app/src:/home/ai/pi0.5/pi05_infer/app/packages/openpi-client/src
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

nohup "$PY" jax_infer.py serve \
  --checkpoint /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999 \
  --config pi05_pp_white_red_h50 \
  --asset-id . \
  --device gpu \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8003 \
  --allow-hardware \
  >jax_infer.log 2>&1 &

echo $! > jax_infer.pid
```

这条命令适合 JAX 普通推理和 dry-run。若需要让 8003 服务允许真实机器人 runner，再加入：

```bash
--allow-hardware
```

例如把命令尾部改为：

```bash
  --port 8003 \
  --allow-hardware \
  >jax_infer.log 2>&1 &
```

检查启动状态：

```bash
tail -f /home/ai/pi0.5/pi05_infer/jax_infer.log
curl --noproxy '*' http://127.0.0.1:8003/healthz
```

确认 GPU：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
export JAX_PLATFORMS=cuda
/home/ai/condaenv/openpi/bin/python -c \
  'import jax; print(jax.__version__); print(jax.devices()); print(jax.default_backend())'
```

停止该服务：

```bash
kill "$(cat /home/ai/pi0.5/pi05_infer/jax_infer.pid)"
rm /home/ai/pi0.5/pi05_infer/jax_infer.pid
```

如果 PID 文件不存在，可以先定位进程：

```bash
pgrep -af 'jax_infer.py serve'
```

不要同时启动两个占用 8003 的 JAX 服务。

### 9.3 宿主机启动 JAX GPU：带 RTC

带 RTC 时必须使用 RTC 源码：

```text
OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/rtc_stage
```

如果仍设置为 `/home/ai/pi0.5/pi05_infer/app`，服务会加载普通 `openpi`，runner 将收到 `Policy server does not advertise RTC support`。

启动 JAX RTC GPU 服务：

```bash
cd /home/ai/pi0.5/pi05_infer

export PY=/home/ai/condaenv/openpi/bin/python
export OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/rtc_stage
export PYTHONPATH=/home/ai/pi0.5/pi05_infer/rtc_stage:/home/ai/pi0.5/pi05_infer/app/packages/openpi-client/src
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu/tegra:${LD_LIBRARY_PATH:-}
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

nohup "$PY" jax_infer.py serve \
  --checkpoint /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999 \
  --config pi05_pp_white_red_h50 \
  --asset-id . \
  --device gpu \
  --num-steps 10 \
  --host 0.0.0.0 \
  --port 8003 \
  --allow-hardware \
  >jax_rtc_infer.log 2>&1 &

echo $! > jax_rtc_infer.pid
```

检查：

```bash
tail -f /home/ai/pi0.5/pi05_infer/jax_rtc_infer.log
curl --noproxy '*' http://127.0.0.1:8003/healthz
```

启动后，连接 8003 的策略元数据必须包含：

```text
backend=jax
device=gpu
rtc_supported=True
hardware_allowed=True
```

停止：

```bash
kill "$(cat /home/ai/pi0.5/pi05_infer/jax_rtc_infer.pid)"
rm /home/ai/pi0.5/pi05_infer/jax_rtc_infer.pid
```

### 9.4 Docker 启动普通 JAX CPU 服务（可选）

当前 `compose.jax.yml` 可以启动普通 JAX 服务：

```bash
cd /home/ai/pi0.5/pi05_infer

export JAX_CHECKPOINT_DIR=/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999
export POLICY_CONFIG=pi05_pp_white_red_h50
export JAX_POLICY_PORT=8003
export JAX_NUM_STEPS=10

JAX_DEVICE=cpu docker compose -f compose.jax.yml up -d
docker compose -f compose.jax.yml logs -f tb6r5_jax_policy_server
```

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8003/healthz
```

停止：

```bash
docker compose -f compose.jax.yml down
```

当前 compose 挂载的是 `app/src` 普通源码，而且没有给 `jax_infer.py serve` 传 `--allow-hardware`，所以它适合 smoke-test/dry-run，不适合当前 JAX RTC 真机。

### 9.5 Docker 启动带 RTC 源码的 JAX CPU 服务（可选）

以下命令明确挂载 `rtc_stage/openpi`，并增加 `--allow-hardware`。当前镜像仍只有 CPU；在获得真正支持 Jetson CUDA 的 JAX 镜像前，先用于功能验证。

```bash
cd /home/ai/pi0.5/pi05_infer

docker rm -f pi05-jax-rtc 2>/dev/null || true

docker run -d --name pi05-jax-rtc \
  --runtime=nvidia \
  --network=host \
  --shm-size=8g \
  -e OPENPI_ROOT=/app \
  -e OPENPI_DATA_HOME=/openpi_assets \
  -e JAX_PLATFORMS=cpu \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999:/models/pi05_tb6r5:ro \
  -v /home/ai/pi0.5/pi05_infer/jax_infer.py:/opt/tb6r5/jax_infer.py:ro \
  -v /home/ai/pi0.5/pi05_infer/rtc_stage/openpi:/app/src/openpi:ro \
  -v /home/ai/pi0.5/pi05_infer/docker_source/packages/openpi-client/src:/app/packages/openpi-client/src:ro \
  -v /home/ai/pi0.5/pi05_infer/cache/openpi:/openpi_assets \
  --entrypoint python \
  openpi_tb6r5_policy_server:orin-jp62 \
  /opt/tb6r5/jax_infer.py serve \
  --checkpoint /models/pi05_tb6r5 \
  --config pi05_pp_white_red_h50 \
  --device cpu \
  --host 0.0.0.0 \
  --port 8003 \
  --num-steps 10 \
  --allow-hardware
```

查看加载日志：

```bash
docker logs -f pi05-jax-rtc
```

停止：

```bash
docker stop pi05-jax-rtc
docker rm pi05-jax-rtc
```

当以后有可用的 JAX CUDA 镜像时，需要同时满足：

```text
jax.devices() 中出现 GPU/CUDA 设备
JAX_PLATFORMS=cuda
jax_infer.py 参数使用 --device gpu
```

只把 `cpu` 改成 `gpu` 而镜像没有 CUDA backend，会直接启动失败。

### 9.6 普通 JAX 的 dry-run 和真机调用

普通 JAX 服务启动后，runner 不加 `--rtc`，并连接 8003。先执行 dry-run：

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --dry-run \
  --robot-ip 192.168.2.151 \
  --policy-port 8003 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --fps 30 \
  --n-action-steps 10 \
  --max-inference-calls 3 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

要运行真机，必须用 9.2 节中带 `--allow-hardware` 的服务启动方式，然后将上面命令中的：

```text
--dry-run
--max-inference-calls 3
```

替换为：

```text
--confirm-hardware TB6R5_REAL_INFERENCE
--max-inference-calls 0
```

普通 JAX 录制使用第 8 节相同的三个参数；`--record-fps` 与 runner 的 `--fps` 保持一致：

```bash
--record-dataset "$OUT" \
--record-fps 30 \
--record-episode-frames 1000
```

### 9.7 JAX RTC dry-run

确认 8003 服务加载完成后：

```bash
cd /home/ai/pi0.5/pi05_infer

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --dry-run \
  --rtc \
  --rtc-num-steps 5 \
  --rtc-min-execution-steps 2 \
  --rtc-guidance-beta 5.0 \
  --fps 1 \
  --robot-ip 192.168.2.151 \
  --policy-port 8003 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --max-inference-calls 2 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

使用宿主机 GPU 服务时，先观察日志中的实际推理延迟。使用 Docker CPU 服务时，推理延迟很可能超过 10 步动作块能覆盖的时间，因此 dry-run 仍可能触发 `RTC latency needs ...`。这说明当前后端达不到 RTC 时序，并不代表相机或机器人连接失败。

### 9.8 JAX RTC 真机和录制命令模板

仅在 8003 元数据包含 `rtc_supported=True`、服务使用 `--allow-hardware`，并且实际推理延迟满足 RTC 时序后运行：

```bash
cd /home/ai/pi0.5/pi05_infer
OUT="/home/ai/pi0.5/data/lerobotdata/pi05_jax_rtc_$(date +%Y%m%d_%H%M%S)"

/home/ai/condaenv/tb6r5/bin/python run_pi05_tb6r5.py \
  --rtc \
  --rtc-num-steps 5 \
  --rtc-min-execution-steps 2 \
  --rtc-guidance-beta 5.0 \
  --fps 1 \
  --confirm-hardware TB6R5_REAL_INFERENCE \
  --robot-ip 192.168.2.151 \
  --policy-port 8003 \
  --camera-topic-sensors 'realsense_0=RsCameraSensor#0#0,realsense_1=RsCameraSensor#1#0' \
  --camera-topic-port 19091 \
  --no-show-camera \
  --max-inference-calls 0 \
  --joint-step-max-rad 0.1 \
  --ee-step-max-m 0.1 \
  --rotation-step-max-rad 0.3 \
  --record-dataset "$OUT" \
  --record-fps 1 \
  --record-episode-frames 1000 \
  --print-rpc \
  --task 'Pick up the red block and place it in the box on the left'
```

运行这一真机模板前，推荐使用 9.3 节的宿主机 JAX GPU RTC 服务，并先完成 9.7 节的 dry-run。

## 10. 端口修改方法

因为所有容器使用 `network_mode: host`，策略端口必须互不冲突。

### PyTorch 普通服务

默认值在：

```text
runtime/openpi/scripts/docker/.env.orin.tb6r5
POLICY_PORT=8002
```

修改后，runner 的 `--policy-port` 必须同步修改。`manage.sh status` 当前固定检查 8002；使用自定义端口时改用：

```bash
curl --noproxy '*' http://127.0.0.1:你的端口/healthz
```

### PyTorch RTC 服务

`rtc_stage/compose.rtc.yml` 当前将 8012 写在 `command` 和 `healthcheck` 中。修改端口时这两处必须一起改，runner 也使用相同的 `--policy-port`。

### JAX 服务

宿主机启动时修改 `jax_infer.py serve` 的：

```text
--port 8003
```

compose 启动时设置：

```bash
export JAX_POLICY_PORT=8003
```

直接 `docker run` 时同时修改：

```text
--port 8003
```

以及 runner：

```text
--policy-port 8003
```

检查端口占用：

```bash
ss -ltn | grep -E ':(8002|8003|8012) '
```

## 11. 推荐启动顺序

以 PyTorch RTC 真机录制为例：

1. 启动机器人控制器，确认机械臂已经处于正常可控制状态。
2. 检查两路相机 URL 返回 200。
3. 启动 8012 RTC Docker 服务。
4. 等待 `curl http://127.0.0.1:8012/healthz` 返回 `OK`。
5. 先运行 `--dry-run --max-inference-calls 2`。
6. 检查输出动作、推理耗时、RTC delay 和相机映射。
7. 创建新的 `OUT` 路径。
8. 去掉 `--dry-run`，加入 `--confirm-hardware TB6R5_REAL_INFERENCE` 后运行真机。
9. 使用 `Ctrl+C` 停止，并等待 `[RECORD] SAVED ...`。
10. 检查数据集的 `meta/info.json` 和帧数。

## 12. 常见错误

### `Hardware requires --confirm-hardware TB6R5_REAL_INFERENCE`

真机命令缺少：

```bash
--confirm-hardware TB6R5_REAL_INFERENCE
```

### `Policy server does not advertise RTC support`

连接到了普通服务，或服务未加载 RTC 源码。PyTorch RTC 应连接 8012；JAX RTC 必须挂载 `rtc_stage/openpi`。

### `Unsafe TCP jump at action ...`

确认命令包含：

```bash
--ee-step-max-m 0.1 --rotation-step-max-rad 0.3
```

### `state outside checkpoint training envelope`

当前状态的位置或夹爪超出 checkpoint 统计范围及代码内固定 margin。`--ee-step-max-m` 不会放宽训练 envelope。四元数不再做逐分量 envelope 比较，因为 `q`/`-q` 等价且四个分量相互耦合；程序仍检查四元数范数和相邻动作旋转角。

### `RTC latency needs ...` 或 `active chunk exhausted`

推理速度赶不上动作块。降低 `--fps`，减少 `--rtc-num-steps`，或者使用更快且真正启用 GPU 的后端。

### `Recording directory is not empty`

每次重新生成：

```bash
OUT="/home/ai/pi0.5/data/lerobotdata/pi05_run_$(date +%Y%m%d_%H%M%S)"
```

### `Recording queue full: writer cannot keep up`

当前异步录制器在 LeRobot `Map`/视频编码期间积压超过 32 帧。短时运行可把 `--record-episode-frames` 提高到 `1000`，减少运行中编码次数。不要把 `--record-fps` 当作抽帧开关。

### 出现 `Map: 100%` 和 `libx264`

这是 LeRobot 保存 episode、整理表格和编码双路视频，不是机械臂指令。退出时出现属于保存剩余帧。运行中频繁出现说明 `--record-episode-frames` 较小。

### `No valid pending response for seqID`

这是 RPC 请求/响应同步异常提示。如果随后出现控制器错误、RPC 返回失败或掉使能，需要结合机器人控制器日志判断；不要只看录制日志。

### `TimeoutError: timed out` 出现在新一轮 inference 之前

普通推理循环会先读取两路相机，成功后才打印下一条 `[OPENPI] state`。如果一个动作块结束后直接出现裸 `TimeoutError`，而没有下一条 state，通常是 HTTP 相机读取超时。新版 runner 会显示具体 URL，并默认重试 3 次：

```text
[CAMERA] http://... attempt 1/3 failed: TimeoutError: timed out; retrying
```

分别检查两路 URL：

```bash
curl --noproxy '*' -v --max-time 4 -o /dev/null \
  http://192.168.11.11:6888/RsCameraSensor/0/0/color

curl --noproxy '*' -v --max-time 4 -o /dev/null \
  http://192.168.11.11:6888/RsCameraSensor/1/0/color
```

如果 TCP 显示 `Connected`，但最后是 `0 bytes received`，机器人侧 6888 服务仍在监听却没有发布 MJPEG 帧。此时应恢复或重启机器人侧相机发布程序，再启动 runner。

## 13. 停止服务并释放内存

Jetson 的 CPU 和 GPU 共用系统内存。JAX/PyTorch 模型占用的内存会在对应服务进程或容器停止后释放；只退出机器人 runner 不会释放策略服务中的模型。

### 13.1 停止机器人 runner

在运行 runner 的终端按一次：

```text
Ctrl+C
```

如果启用了 LeRobot 录制，等待下面的信息出现后再关闭终端：

```text
[RECORD] Draining background writer ...
[RECORD] SAVED ...
```

runner 随后会断开 RPC/Topic 和相机。录制保存期间不要使用 `kill -9`，否则当前 episode 可能不完整。

检查是否还有 runner：

```bash
pgrep -af 'run_pi05_tb6r5.py'
```

### 13.2 停止宿主机 JAX GPU 服务（8003）

JAX RTC 服务使用：

```bash
cd /home/ai/pi0.5/pi05_infer

if [ -f jax_rtc_infer.pid ]; then
  PID="$(cat jax_rtc_infer.pid)"
  kill "$PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
  rm -f jax_rtc_infer.pid
fi
```

普通 JAX 服务如果使用 `jax_infer.pid`，执行：

```bash
cd /home/ai/pi0.5/pi05_infer

if [ -f jax_infer.pid ]; then
  PID="$(cat jax_infer.pid)"
  kill "$PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
  rm -f jax_infer.pid
fi
```

PID 文件缺失时，先查找实际进程：

```bash
pgrep -af 'jax_infer.py serve'
```

确认命令行中的 checkpoint 和端口后，对查到的 PID 执行：

```bash
kill PID
```

不要直接执行不加确认的全局 `pkill python`，它可能同时停止机器人和相机程序。

### 13.3 停止 PyTorch Docker 服务

普通 PyTorch 8002：

```bash
cd /home/ai/pi0.5/pi05_infer
./manage.sh stop
```

PyTorch RTC 8012：

```bash
cd /home/ai/pi0.5/pi05_infer
docker compose -f rtc_stage/compose.rtc.yml down
```

手工启动的 JAX RTC CPU 容器：

```bash
docker rm -f pi05-jax-rtc
```

`docker compose down` 或 `docker rm -f` 会释放容器占用的 RAM/GPU 内存，但不会删除模型、镜像或录制的数据集。

### 13.4 确认端口和内存已经释放

检查三个策略端口：

```bash
ss -ltnp | grep -E ':(8002|8003|8012) ' || echo 'policy ports released'
```

检查策略进程和容器：

```bash
pgrep -af 'jax_infer.py serve|serve_policy|run_pi05_tb6r5.py' || true
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

查看 Jetson 统一内存：

```bash
free -h
timeout 3 tegrastats --interval 1000
```

JAX/PyTorch 可能在进程运行期间保留已申请的内存，这是正常的；结束对应进程或容器才会完整释放。无需执行 `docker system prune`，也无需清理 Linux page cache。

### 13.5 一次关闭本文中的全部策略服务

先在 runner 终端按 `Ctrl+C` 并等待录制保存完成，然后执行：

```bash
cd /home/ai/pi0.5/pi05_infer

./manage.sh stop 2>/dev/null || true
docker compose -f rtc_stage/compose.rtc.yml down 2>/dev/null || true
docker rm -f pi05-jax-rtc 2>/dev/null || true

for PID_FILE in jax_infer.pid jax_rtc_infer.pid; do
  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    kill "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
  fi
done

sleep 2
ss -ltnp | grep -E ':(8002|8003|8012) ' || echo 'all policy ports released'
free -h
```

## 14. 快速命令索引

```bash
# PyTorch 普通服务
./manage.sh start
./manage.sh status
./manage.sh logs
./manage.sh stop

# PyTorch RTC 服务
docker compose -f rtc_stage/compose.rtc.yml up -d --wait
docker compose -f rtc_stage/compose.rtc.yml logs -f policy
docker compose -f rtc_stage/compose.rtc.yml down

# JAX GPU 普通/RTC 服务使用第 9.2/9.3 节宿主机命令
# 普通：OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/app
# RTC： OPENPI_ROOT=/home/ai/pi0.5/pi05_infer/rtc_stage，并传 --allow-hardware

# JAX Docker 普通服务（当前仅 CPU）
JAX_CHECKPOINT_DIR=/home/ai/pi0.5/model/jax/pi05_pp_white_red_h50/19999 \
POLICY_CONFIG=pi05_pp_white_red_h50 \
JAX_DEVICE=cpu JAX_POLICY_PORT=8003 \
docker compose -f compose.jax.yml up -d

# 健康检查
curl --noproxy '*' http://127.0.0.1:8002/healthz
curl --noproxy '*' http://127.0.0.1:8003/healthz
curl --noproxy '*' http://127.0.0.1:8012/healthz

# 查看服务和端口
docker ps --format 'table {{.Names}}\t{{.Status}}'
ss -ltn | grep -E ':(8002|8003|8012) '
```



## 独立录制一个 LeRobot episode

推理和录制是两个独立进程。推理命令不需要添加 `--record-dataset`。runner 会检测本机 sidecar socket；没有启动录制器时不会复制或编码图像。

终端一运行真机推理：

```bash
cd /home/ai/pi0.5/pi05_infer
REAL=1 ./run_jax_rtc_red_white_red.sh
```

终端二在想开始录制时运行：

```bash
cd /home/ai/pi0.5/pi05_infer
./record_one_episode.sh
```

脚本会自动创建类似下面的目录：

```text
/home/ai/pi0.5/data/lerobotdata/pi05_episode_YYYYMMDD_HHMMSS
```

看到下面两行后表示正在接收推理帧：

```text
[RECORD] waiting for inference ...
[RECORD] inference connected; press Ctrl+C to save this episode
```

录够后在录制终端按一次 `Ctrl+C`。录制器保存双相机视频、实际 TCP/夹爪状态、实际下发 action 和任务文本，并打印 `SAVED ...`。不要在编码期间再次按 `Ctrl+C` 或使用 `kill -9`。阶段切换时录制器会保留已有帧并等待下一阶段连接；直到你按 `Ctrl+C`，才把期间收到的三段内容一起保存为一个 episode。

录制 FPS 必须与推理 FPS 一致。两个脚本默认均为30 FPS。如果推理使用8 FPS：

```bash
# 推理终端
FPS=8 REAL=1 ./run_jax_rtc_red_white_red.sh

# 录制终端
FPS=8 ./record_one_episode.sh
```

也可以指定输出目录：

```bash
./record_one_episode.sh /home/ai/pi0.5/data/lerobotdata/my_rtc_episode
```

真机三阶段脚本还会自动保存 RTC 控制日志到：

```text
/home/ai/pi0.5/pi05_infer/logs/
```

后续调整 RTC 参数时，需要提供 episode 目录和对应日志文件。日志包含 inference latency、`request_step`、`delay_budget`、skipped actions、TCP/夹爪完成判断和安全保护信息。

## JAX RTC 红—白—红三阶段自动运行

新增脚本：

```text
run_jax_rtc_red_white_red.sh
```

它依次执行红块、白块、红块三个独立 RTC 阶段，任务文本均为把对应颜色方块放进左侧盒子。
启动第一阶段前以及每阶段检测成功后，都会先回到相同的训练起点，避免上一次中断遗留在盒子附近的
TCP直接触发完成判断。回位命令为：

```text
{MoveAbsJ --jointtarget=act_vla_0 --speed=v60||NotRunExecute}
```

### 自动完成条件

完成判断使用真实 Topic 反馈，不使用模型预测值。必须同时满足：

1. TCP 到左侧盒子中心的三维距离不超过 `0.060 m`；
2. 反馈夹爪开度严格大于 `0.040 m`；
3. 上述位置和开度连续保持 5 个控制周期；
4. 至少已经执行 30 个控制步。

最后一个动作发出后，runner 还会保留 30 个只读反馈周期，等待夹爪实际张开并满足连续保持条件。

默认盒子中心来自两次完整 JAX RTC 记录的释放末帧：

```text
TASK_COMPLETE_TCP=-0.190,-0.651,0.230
```

两次末帧 TCP 分别约为 `(-0.188,-0.658,0.210)` 和 `(-0.193,-0.644,0.240)` 米。现场盒子移动后应覆盖中心或半径：

```bash
TASK_COMPLETE_TCP='-0.190,-0.651,0.230' \
TASK_COMPLETE_RADIUS_M=0.060 \
TASK_COMPLETE_GRIPPER_OPEN_M=0.040 \
TASK_COMPLETE_HOLD_STEPS=5 \
TASK_COMPLETE_GRACE_STEPS=30 \
REAL=1 ./run_jax_rtc_red_white_red.sh
```

日志连续出现以下内容并最终打印 `[TASK] COMPLETE`，才会进入回位和下一阶段：

```text
[TASK] completion candidate: tcp_distance=..., gripper=..., hold=.../5
[TASK] COMPLETE after ...
```

### 夹爪命令

三阶段脚本直接使用模型预测的连续夹爪值，不再把预测值按 `0.035 m` 阈值强制改成 `0.000 m` 或 `0.065 m`。夹爪动作由原有执行器逻辑发送和平滑。

每一阶段只在以下实际反馈连续满足指定周期数后判定完成：

- 当前 TCP 进入 `TASK_COMPLETE_TCP` 附近；
- 机械臂反馈的实际夹爪开度大于 `TASK_COMPLETE_GRIPPER_OPEN_M`，默认 `0.040 m`。

完成判断不再要求此前检测到夹爪闭合。

### 运行

先做 dry-run。dry-run 不运动，因此只按较小的 inference 上限完成流程测试，不启用真实完成判断：

```bash
cd /home/ai/pi0.5/pi05_infer
STAGE_MAX_INFERENCE_CALLS=3 ./run_jax_rtc_red_white_red.sh
```

真机：

```bash
REAL=1 ./run_jax_rtc_red_white_red.sh
```

`STAGE_MAX_INFERENCE_CALLS` 现在是任务完成的超时上限，默认 120。真机若达到上限仍未满足 TCP 和夹爪条件，阶段会以 `SafetyAbort` 停止，不会回位，也不会继续下一阶段。可以分别设置：

```bash
REAL=1 \
STAGE_1_MAX_INFERENCE_CALLS=120 \
STAGE_2_MAX_INFERENCE_CALLS=120 \
STAGE_3_MAX_INFERENCE_CALLS=120 \
./run_jax_rtc_red_white_red.sh
```

若 RTC 阶段触发安全保护、策略服务断开或返回非零状态，脚本同样立即停止。回位参数可通过 `HOME_JOINTTARGET`、`HOME_SPEED` 和 `HOME_TIMEOUT_S` 覆盖。
默认 `INITIAL_HOME=1`；只有已经通过其他方式确认机械臂位于 `act_vla_0` 时，才可用
`INITIAL_HOME=0` 跳过第一阶段前的回位。
真机序列默认 `PRECOMPILE_RTC=1`：先以 dry-run warmup-only 模式编译最终的
`beta/num_steps/request_step/delay_steps` 组合，成功后才执行第一次回位。三个真机阶段仍各自进行约2秒的
缓存校验，但不会在机械臂运动期间触发JAX编译。已经确定缓存存在时可用 `PRECOMPILE_RTC=0` 跳过该步骤。






### 三阶段之间保持机械臂使能

`run_jax_rtc_red_white_red.sh` 和 `run_jax_red_white_red.sh` 在真机模式下默认设置：

```text
KEEP_ENABLED_BETWEEN_STAGES=1
```

第一阶段正常执行一次完整的 `Disable → Enable → Start` 初始化。任务完成后，runner 关闭 RPC 连接但不发送 `Disable`；回位脚本复用已使能的控制器状态，前两次回位后继续保持使能，第二、三阶段也直接复用。第三阶段的最终回位完成后才统一发送 `Disable`。这样减少阶段切换中的重复初始化；模型推理耗时不变。

只有已确认任务完成或回位完成时才允许保持使能。`SafetyAbort`、RPC 失败、回位失败和其他异常仍会执行 `Disable` 并停止序列。

需要恢复原来的每个进程都重新下使能/上使能行为时：

```bash
KEEP_ENABLED_BETWEEN_STAGES=0 REAL=1 ./run_jax_rtc_red_white_red.sh
```

### Topic SDK 退出码 134

厂商 Topic Python 扩展只提供 `start_subscriber`，没有停止订阅接口。旧版在任务已经完成、机器人已经断开后，可能于 Python 解释器退出时出现：

```text
free(): corrupted unsorted chunks
Aborted (core dumped)
```

runner 现在会在显式关闭 RPC、相机和策略连接后直接结束进程，跳过有缺陷的原生扩展析构。三阶段脚本还会实时保存阶段日志；只有返回码为 134，并且日志同时包含任务完成和 `TB6-R5 disconnected.` 时才继续回位和下一阶段。任何 `SafetyAbort`、未完成崩溃或其他非零返回值仍会立即停止。

## JAX 非 RTC 红—白—红三阶段自动运行（默认 25 步）

脚本：

```text
run_jax_red_white_red.sh
```

它执行与 RTC 版本相同的红块、白块、红块三个阶段，使用相同的 TCP 与实际夹爪反馈完成条件，并在每阶段完成后回到 `act_vla_0`。脚本不会传入 `--rtc`。

默认 `ACTION_HORIZON=25`，表示每次普通 JAX 推理返回 50 步后执行其中 25 步，再采集新观测并重新推理。模型本身仍是 `pi05_pp_white_red_h50`，其输出 horizon 保持 50。

先确保端口 8003 的 JAX 策略服务已按“宿主机启动 JAX GPU：不带 RTC”启动，并包含 `--allow-hardware`。然后运行：

```bash
cd /home/ai/pi0.5/pi05_infer
CAMERA_TOPIC_FALLBACK_HTTP=1 REAL=1 ./run_jax_red_white_red.sh
```

需要调整每次执行的步数时：

```bash
ACTION_HORIZON=20 REAL=1 ./run_jax_red_white_red.sh
```

允许范围为 `1..50`。`STAGE_MAX_INFERENCE_CALLS`、三个阶段独立上限、任务文本、完成位置、夹爪完成阈值和回位参数与 RTC 三阶段脚本相同。
