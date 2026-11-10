# TB6-R5 真机 SDK 的下载与平台支持

`pi05_infer` 的策略服务只加载模型。真机客户端通过相邻目录的
`tb6r5_policy_infer` 包读取 Topic 状态、发送 RPC 指令；默认相机通过机器人
`6888` 端口的 HTTP URL 读取。这三项功能中，RPC 和 Topic 需要下面的原生 SDK。

| 功能 | 真机客户端使用的接口 | 来源 |
| --- | --- | --- |
| 机器人命令（端口 5868） | `rpc.CPPClient`、`rpc.Msg` | `send_commend_py` |
| 状态反馈（端口 19091） | `topic` 订阅及状态结构 | `get_status_py` |
| 默认两路相机 | HTTP URL | 不需要 RPC/Topic SDK 的相机扩展 |

## 部署本项目的 Linux 真机客户端

`pi05_infer` 在运行时查找同级的 `tb6r5_policy_infer` 目录。这是项目使用的
Python 适配层，包含经过本项目验证的 RPC/Topic 动态库和加载代码。它在
**单独的 GitHub 仓库**中；`pi05_infer` 本身没有复制这些二进制文件。

```bash
cd /home/ai/pi0.5
git clone https://github.com/shanchaicn/tb6r5_policy_infer.git
git -C tb6r5_policy_infer checkout 9b68c81ce1c4c64c543bd5ec60f6d249bf3d01e6
```

若目录放在其他位置，运行客户端前设置
`TB6R5_PACKAGE_ROOT=/实际路径/tb6r5_policy_infer`。使用 **CPython 3.10**
运行客户端，例如 151 上的 `/home/ai/condaenv/tb6r5/bin/python`。
只检查 SDK 文件和动态库加载、不连接机器人：

```bash
TB6R5_PACKAGE_ROOT=/home/ai/pi0.5/tb6r5_policy_infer \
/home/ai/condaenv/tb6r5/bin/python - <<'PY'
import os
import sys
sys.path.insert(0, os.environ['TB6R5_PACKAGE_ROOT'])
from tb6r5_policy_infer.hardware.sdk_paths import dependencies_root, rpc_py_all_root
from tb6r5_policy_infer.hardware.topic_feedback import validate_topic_sdk
validate_topic_sdk()
sys.path.insert(0, rpc_py_all_root())
import rpc_client  # 加载 rpc.so
import rpc
print('RPC/Topic SDK OK:', dependencies_root())
PY
```

这个检查通过后，还需分别检查机器人 IP、Topic 数据和 RPC 连接；导入成功
本身不证明机器人在线。151 当前的真机运行环境是 Linux ARM64 + CPython 3.10。

## 下载厂商原版 SDK（Linux / Windows）

厂商的 [RPC + Topic 综合示例](https://gitee.com/zhenshi-controller-api-data/sdk_send_recevice_demo_py)
可独立下载；[厂商组织主页](https://gitee.com/zhenshi-controller-api-data)
也提供分开的 [RPC SDK](https://gitee.com/zhenshi-controller-api-data/send_commend_py)
和 [Topic SDK](https://gitee.com/zhenshi-controller-api-data/get_status_py)。
综合示例的目录结构与 `tb6r5_policy_infer/vendor` 不同，不能把下载目录
直接指定为 `TB6R5_DEPS_ROOT`；要运行本项目，使用上一节的配套仓库。

Linux 或 Windows 安装 Git 和 CPython 3.10 后，在终端执行：

```bash
git clone https://gitee.com/zhenshi-controller-api-data/sdk_send_recevice_demo_py.git
git -C sdk_send_recevice_demo_py checkout 96e1677c0374b432cddb35a32c212c8f5fe536f6
cd sdk_send_recevice_demo_py/Base_SDK_py
```

Linux 的无机器人导入检查：

```bash
python3.10 -c 'from robot_command import rpc_client; from robot_state.platform_loader import get_topic_module; print("RPC/Topic SDK OK", get_topic_module().__file__)'
```

Windows PowerShell 的无机器人导入检查：

```powershell
py -3.10 -c 'from robot_command import rpc_client; from robot_state.platform_loader import get_topic_module; print("RPC/Topic SDK OK", get_topic_module().__file__)'
```

这些检查只导入扩展，不运行示例 `main.py`；后者会连接控制器并可能发送命令。

| 平台 | 厂商综合示例中的二进制 | 状态 |
| --- | --- | --- |
| Linux x86_64 | `rpc.so`、`topic.so`，各在 `cp310` 目录 | 已在 x86_64 + CPython 3.10 验证导入 |
| Linux ARM64 / aarch64 | 同上，在 `arm/cp310` 目录 | 已在 151 的 ARM64 + CPython 3.10 验证导入 |
| Windows x64 | `rpc.pyd`、`topic.pyd`，附带所需 DLL | 厂商提供并声明支持；本项目没有 Windows 机器做现场验证 |
| Windows ARM64 或 32 位 Windows | 无对应预编译扩展 | 当前不能保证可用；需要厂商提供对应构建 |

厂商这些预编译扩展面向 **CPython 3.10**。Python 3.11、3.12 等版本
应取得匹配 ABI 的扩展，不能只改文件名。厂商文档注明 Linux 库以
Ubuntu 20.04 为 ABI 基线；其他发行版仍需现场验证系统库兼容性。
Windows 上可独立使用厂商 SDK；本仓库的 `run_*.sh` 真机推理入口及
`tb6r5_policy_infer` 集成当前按 Linux 部署，不能据此承诺整套推理在
Windows 上直接运行。
