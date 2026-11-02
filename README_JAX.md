# JAX GPU 与 RTC 补充说明

先看仓库根目录的 [README.md](README.md)：其中列出了 151 上的 h50 checkpoint、GPU Docker Compose、宿主机服务启动方式、端口和真机入口。本文件说明 JAX 服务与 RTC 客户端之间的数据流。

## 模型与客户端

`jax_infer.py serve` 只负责加载 JAX checkpoint 的 `params/`、训练时的 `pi05_pp_white_red_h50` 配置与 `assets/./norm_stats.json`。它通过 WebSocket 接受观测、返回动作块，并在元数据中报告 `backend=jax`、`rtc_supported` 和是否允许真机。它不直接控制机器人。

`run_jax_rtc.sh` 使用宿主机 Python 3.10 运行 `rtc_stage/run_pi05_tb6r5.py`，再进入 `rtc_stage/openpi_runner.py`。客户端读取两路 HTTP 相机和机器人 Topic 状态，经 `8003` 请求模型，然后在 `REAL=1` 时通过机器人 RPC 发送经过安全校验的动作。`run_jax_rtc_red_white_red.sh` 在此基础上做预编译、初始回位、红白红三段任务和各段完成判断。

## RTC 的动作衔接

模型的动作块长度为 50。RTC 在当前动作块执行时异步请求下一块，将尚未执行的旧动作作为引导，收到新块后跳过已过去的控制时刻。客户端会检查反馈新鲜度、动作维度、训练统计范围、相邻 TCP 位姿跳变和推理时延。首次请求可能包含 JAX 编译，因此三段任务脚本会在机械臂移动前做预热。

当前三段任务默认值为：

```text
FPS=30
RTC_NUM_STEPS=4
RTC_MIN_EXECUTION_STEPS=1
RTC_GUIDANCE_BETA=10.0
HOME_SPEED=v60
MODEL_CONFIG=pi05_pp_white_red_h50
```

这些值来自当前 h50 模型与已有真机试验，修改推理步数、控制频率或引导参数后，应先在 `REAL=0` 下验证耗时和动作。

## GPU 服务的两种启动方式

宿主机使用 `/home/ai/condaenv/openpi/bin/python jax_infer.py serve`。Docker 使用 `jax-orin:0.5.3-cuda` 和 `compose.jax.yml`。两者都监听 `8003`，一次只运行一种；完整命令及健康检查见 [README.md](README.md)。

Docker 镜像中的 JAX CUDA 与本机 JetPack 配套；`jax_gpu_build/Dockerfile.runtime` 是本地构建来源，构建时用到的轮子未随 Git 上传。容器里的 OpenPI RTC 代码和 `openpi_client` 则从本仓库只读挂载，路径在 `compose.jax.yml` 中列明。机器人 SDK 仅供宿主机客户端使用，不需挂载进策略容器。

普通 JAX 推理：`./run_jax.sh`。RTC：`./run_jax_rtc.sh`。红白红真机任务：`REAL=1 ./run_jax_rtc_red_white_red.sh`。默认相机模式为 `CAMERA_MODE=http`；`CameraUpdatePb` 的 SDK 相机模式尚未作为受支持的真机推理路径。
