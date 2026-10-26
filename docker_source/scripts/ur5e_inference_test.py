#!/usr/bin/env python3
"""
UR5e 零样本推理测试脚本 — 不需要微调，直接用预训练 base 模型试效果。

用法:
    # 测试模型能否正常加载和推理（用随机数据）
    uv run scripts/ur5e_inference_test.py

    # 启动策略服务器，供机器人客户端连接
    uv run scripts/ur5e_inference_test.py --serve --port 8000

说明:
    - 使用 pi0_base 或 pi05_base 预训练权重
    - 加载 UR5e 归一化统计量（asset_id="ur5e"）
    - base 模型零样本效果可能较差，但可以先跑通流程
"""

import dataclasses
import logging
import time

import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import ur5e_policy

logger = logging.getLogger(__name__)


# ============================================================
# 选择你要用的 base 模型
# ============================================================
# 选项1: pi0 base (flow matching)
# BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi0_base/params"
# CONFIG_NAME = "pi0_ur5e_base"

# 选项2: pi0.5 base (推荐，更新更强的模型)
BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base/params"
CONFIG_NAME = "pi05_ur5e_base"

# UR5e 归一化统计量所在位置（预训练 checkpoint 中已包含 ur5e 的统计量）
NORM_STATS_ASSETS_DIR = "gs://openpi-assets/checkpoints/pi0_base/assets"
NORM_STATS_ASSET_ID = "ur5e"


def create_ur5e_base_policy():
    """创建基于预训练 base 模型的 UR5e 策略（零样本，未微调）"""

    # 获取训练配置
    config = _config.get_config(CONFIG_NAME)

    # 下载 base 模型 checkpoint
    checkpoint_dir = download.maybe_download(BASE_CHECKPOINT)
    logger.info(f"Checkpoint downloaded to: {checkpoint_dir}")

    # 下载 UR5e 归一化统计量
    norm_stats_dir = download.maybe_download(f"{NORM_STATS_ASSETS_DIR}/{NORM_STATS_ASSET_ID}")
    logger.info(f"Norm stats downloaded to: {norm_stats_dir}")

    # 加载归一化统计量
    from openpi.shared import normalize as _normalize
    norm_stats = _normalize.load(norm_stats_dir)
    logger.info(f"Loaded norm stats with keys: {list(norm_stats.keys()) if norm_stats else 'None'}")

    # 创建策略
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        norm_stats=norm_stats,
    )
    logger.info("Policy created successfully!")
    return policy


def test_inference(policy):
    """用随机观测测试推理是否正常"""
    logger.info("Testing inference with random observations...")

    for i in range(5):
        example = ur5e_policy.make_ur5e_example()
        # 也可以自定义 prompt
        example["prompt"] = "pick up the red block"

        start = time.time()
        result = policy.infer(example)
        elapsed = (time.time() - start) * 1000

        actions = result["actions"]
        logger.info(
            f"  Step {i+1}: actions shape={actions.shape}, "
            f"action_range=[{actions.min():.4f}, {actions.max():.4f}], "
            f"latency={elapsed:.1f}ms"
        )

    logger.info("Inference test passed!")


def serve_policy(policy, port=8000):
    """启动 WebSocket 策略服务器"""
    from openpi.serving import websocket_policy_server

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(f"Starting policy server on {local_ip}:{port}")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=port,
        metadata=policy.metadata,
    )
    server.serve_forever()


def main():
    import tyro

    @dataclasses.dataclass
    class Args:
        """启动参数"""
        # 是否启动策略服务器（否则只做推理测试）
        serve: bool = False
        # 服务器端口
        port: int = 8000

    args = tyro.cli(Args)

    policy = create_ur5e_base_policy()

    if args.serve:
        serve_policy(policy, args.port)
    else:
        test_inference(policy)
        print("\n✅ 推理测试通过！如果要在真实 UR5 上试，运行:")
        print(f"   uv run scripts/ur5e_inference_test.py --serve --port 8000")
        print(f"\n   然后在机器人端运行: uv run scripts/ur5e_robot_client.py --host <GPU服务器IP> --port 8000")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
