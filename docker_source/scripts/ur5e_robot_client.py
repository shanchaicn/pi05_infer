#!/usr/bin/env python3
"""
UR5e 机器人客户端 — 连接远程策略服务器，在真实 UR5 上执行动作。

用法:
    uv run scripts/ur5e_robot_client.py --host <GPU服务器IP> --port 8000

前提:
    1. 先在 GPU 服务器上启动策略服务器:
       uv run scripts/ur5e_inference_test.py --serve --port 8000
    2. 本机需要安装 openpi-client:
       cd packages/openpi-client && pip install -e .

注意:
    - 本脚本提供了与 UR5 机器人通信的框架代码
    - 你需要根据你的 UR5 SDK/驱动替换 get_observation() 和 execute_action() 中的实现
    - 目前提供了两种模式：模拟模式（不需要真实机器人）和真实机器人模式
"""

import dataclasses
import logging
import time

import numpy as np
from openpi_client import image_tools, websocket_client_policy

logger = logging.getLogger(__name__)


# ============================================================
# UR5 机器人接口 — 根据你的实际驱动修改这里
# ============================================================

class UR5RobotInterface:
    """UR5 机器人接口抽象类，子类实现具体的机器人通信。"""

    def get_observation(self) -> dict:
        """读取机器人当前状态，返回观测字典。"""
        raise NotImplementedError

    def execute_action(self, action: np.ndarray) -> None:
        """执行一个动作 (7维: 6关节 + 1夹爪)。"""
        raise NotImplementedError

    def reset(self) -> None:
        """复位机器人到初始位置。"""
        raise NotImplementedError


class SimulatedUR5(UR5RobotInterface):
    """模拟 UR5 — 不需要真实机器人，用随机数据测试推理管线。"""

    def get_observation(self) -> dict:
        # 模拟观测：随机关节角度 + 随机图像
        return {
            "observation/state": np.random.rand(7).astype(np.float32),  # 6 joints + gripper
            "observation/image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            # 如果有腕部相机，取消下面的注释:
            # "observation/wrist_image": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            "prompt": "pick up the block",
        }

    def execute_action(self, action: np.ndarray) -> None:
        logger.info(f"  Executing action: joints={action[:6].round(4)}, gripper={action[6]:.4f}")

    def reset(self) -> None:
        logger.info("  Resetting to home position...")


class RealUR5(UR5RobotInterface):
    """
    真实 UR5 机器人接口。

    你需要根据使用的 UR5 SDK 填充以下方法。
    常用选项:
      - urx / urx2 (Python UR5 驱动): pip install urx
      - rtde (UR RTDE 接口): pip install ur-rtde
      - polyscope / ur_robot_driver (ROS2)
    """

    def __init__(self, robot_ip: str = "192.168.1.100"):
        self.robot_ip = robot_ip
        # TODO: 初始化你的 UR5 连接
        # 例如用 urx:
        # import urx
        # self.robot = urx.Robot(robot_ip)
        #
        # 例如用 rtde:
        # import rtde_control, rtde_receive
        # self.ctrl = rtde_control.RTDEControlInterface(robot_ip)
        # self.recv = rtde_receive.RTDEReceiveInterface(robot_ip)
        logger.info(f"Connected to UR5 at {robot_ip} (placeholder)")

    def get_observation(self) -> dict:
        # TODO: 从真实 UR5 读取数据
        #
        # 1. 读取关节角度 (6维, 弧度制)
        # joints = np.array(self.robot.getj())  # urx
        # joints = np.array(self.recv.getActualQ())  # rtde
        joints = np.zeros(6, dtype=np.float32)

        # 2. 读取夹爪状态 (0.0=全开, 1.0=全闭)
        # gripper = np.array([self.gripper.get_position()])  # 根据你的夹爪驱动
        gripper = np.array([0.0], dtype=np.float32)

        # 3. 拼接 state
        state = np.concatenate([joints, gripper])

        # 4. 读取相机图像
        # base_image = your_camera_capture()  # 返回 (H, W, 3) uint8
        base_image = np.zeros((480, 640, 3), dtype=np.uint8)  # placeholder

        # 5. (可选) 腕部相机
        # wrist_image = your_wrist_camera_capture()
        wrist_image = None

        obs = {
            "observation/state": state,
            "observation/image": base_image,
            "prompt": "pick up the block",  # 由用户输入或预设
        }
        if wrist_image is not None:
            obs["observation/wrist_image"] = wrist_image

        return obs

    def execute_action(self, action: np.ndarray) -> None:
        # TODO: 发送动作到真实 UR5
        joint_targets = action[:6]
        gripper_target = action[6]

        # urx:
        # self.robot.movex("movel", joint_targets, vel=0.5, acc=0.3)
        # rtde:
        # self.ctrl.moveJ(joint_targets, speed=0.5, acceleration=0.3)

        # 夹爪控制 (根据你的夹爪类型)
        # Robotiq 2F-85:
        # if gripper_target > 0.5:
        #     self.gripper.close()
        # else:
        #     self.gripper.open()

        logger.info(f"  Action: joints={joint_targets.round(4)}, gripper={gripper_target:.4f}")

    def reset(self) -> None:
        # TODO: 回到初始位置
        # home_joints = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        # self.robot.movej(home_joints, vel=0.5, acc=0.3)
        logger.info("  Resetting to home position...")


# ============================================================
# 主循环
# ============================================================

def run_episode(
    robot: UR5RobotInterface,
    client: websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    max_steps: int = 200,
    action_exec_steps: int = 4,
):
    """
    运行一个完整的 episode。

    Args:
        robot: UR5 机器人接口
        client: 策略客户端
        prompt: 语言指令
        max_steps: 最大执行步数
        action_exec_steps: 每次推理后执行多少步开环动作
            (策略返回 action_horizon 个动作，但通常只执行前几步就再次推理)
    """
    robot.reset()

    for step in range(max_steps):
        # 1. 读取观测
        obs = robot.get_observation()
        obs["prompt"] = prompt  # 覆盖为当前任务指令

        # 2. 预处理图像 (resize 到 224x224, 转 uint8)
        processed_obs = {
            "observation/state": obs["observation/state"],
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["observation/image"], 224, 224)
            ),
            "prompt": obs["prompt"],
        }
        if "observation/wrist_image" in obs:
            processed_obs["observation/wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["observation/wrist_image"], 224, 224)
            )

        # 3. 推理
        t0 = time.time()
        result = client.infer(processed_obs)
        inference_ms = (time.time() - t0) * 1000

        action_chunk = result["actions"]  # shape: (action_horizon, 7)
        logger.info(f"Step {step}: inference={inference_ms:.0f}ms, chunk_shape={action_chunk.shape}")

        # 4. 开环执行动作 (只执行前 action_exec_steps 步)
        steps_to_exec = min(action_exec_steps, len(action_chunk))
        for i in range(steps_to_exec):
            robot.execute_action(action_chunk[i])
            time.sleep(0.05)  # 20Hz 控制频率


def main():
    import tyro

    @dataclasses.dataclass
    class Args:
        """启动参数"""
        # 策略服务器地址
        host: str = "localhost"
        # 策略服务器端口
        port: int = 8000
        # 模拟模式 (不需要真实机器人)
        sim: bool = True
        # UR5 机器人 IP (仅在非模拟模式下使用)
        robot_ip: str = "192.168.1.100"
        # 语言指令
        prompt: str = "pick up the block"
        # 最大执行步数
        max_steps: int = 50
        # 每次推理后执行几个开环动作步
        action_exec_steps: int = 4

    args = tyro.cli(Args)

    # 创建机器人接口
    if args.sim:
        robot = SimulatedUR5()
        logger.info("Running in SIMULATION mode (no real robot needed)")
    else:
        robot = RealUR5(args.robot_ip)
        logger.info(f"Running with REAL UR5 at {args.robot_ip}")

    # 连接策略服务器
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Connected to policy server at {args.host}:{args.port}")
    logger.info(f"Server metadata: {client.get_server_metadata()}")

    # 运行 episode
    logger.info(f"Starting episode with prompt: '{args.prompt}'")
    run_episode(
        robot=robot,
        client=client,
        prompt=args.prompt,
        max_steps=args.max_steps,
        action_exec_steps=args.action_exec_steps,
    )
    logger.info("Episode finished!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
