#!/usr/bin/env python3

import time
import traceback
import sys

from array import array
from pathlib import Path

import numpy as np
import torch

import rclpy

from ament_index_python.packages import (
    get_package_share_directory,
)
from rclpy.node import Node

from laser_msgs.srv import ComputePpoShield
from rl_policy_c2 import PPOPolicy
from unified_world import UnifiedGenesisWorld
from Wnmpc_shield import shield_action


NUM_UAVS = 2
OBSERVATION_SIZE = 29
ACTION_SIZE = 6

ZERO_OBSERVATION_EXPECTED_ACTION = np.array(
    [
        0.993177474,
        -0.999955893,
        -0.257466048,
        -0.995697916,
        0.999998093,
        -1.000000000,
    ],
    dtype=np.float32,
)


class PpoShieldNode(Node):
    def __init__(self):
        super().__init__(
            "ppo_shield_node"
        )

        package_share = Path(
            get_package_share_directory(
                "laser_uav_multi_drones"
            )
        )

        default_checkpoint = (
            package_share
            / "models"
            / "unified_centralized_2uav_best.pt"
        )

        self._declare_parameters(
            default_checkpoint
        )

        self._read_parameters()

        torch.set_num_threads(
            max(
                1,
                self.torch_num_threads,
            )
        )

        self.device = (
            self._select_device(
                self.requested_device
            )
        )

        self.policy = (
            self._load_policy()
        )

        self._validate_policy()

        self.shadow_world = (
            self._create_shadow_world()
        )

        self.previous_safe_action = None

        self.service = self.create_service(
            ComputePpoShield,
            "compute_ppo_shield",
            self.compute_callback,
        )

        self.get_logger().info(
            "PPO actor/critic and shield "
            "service ready."
        )

        self.get_logger().info(
            f"Checkpoint: "
            f"{self.checkpoint_path}"
        )

        self.get_logger().info(
            f"Device: {self.device}"
        )

        self.get_logger().info(
            "Shield configuration: "
            f"enabled={self.shield_enabled}, "
            f"shadow_mode={self.shadow_mode}, "
            f"samples={self.shield_num_samples}, "
            f"horizon={self.shield_horizon}"
        )

    def _declare_parameters(
        self,
        default_checkpoint: Path,
    ):
        self.declare_parameter(
            "checkpoint_path",
            str(default_checkpoint),
        )

        self.declare_parameter(
            "device",
            "cpu",
        )

        self.declare_parameter(
            "torch_num_threads",
            1,
        )

        self.declare_parameter(
            "shield.enabled",
            True,
        )

        self.declare_parameter(
            "shield.shadow_mode",
            True,
        )

        self.declare_parameter(
            "shield.num_samples",
            96,
        )

        self.declare_parameter(
            "shield.horizon",
            6,
        )

        self.declare_parameter(
            "shield.curriculum_level",
            1.0,
        )

        self.declare_parameter(
            "shield.physics_level",
            0,
        )

        self.declare_parameter(
            "shield.env_level",
            0,
        )

        self.declare_parameter(
            "world.cable_length",
            1.0,
        )

        self.declare_parameter(
            "world.cable_k",
            200.0,
        )

        self.declare_parameter(
            "world.cable_d",
            60.0,
        )

        self.declare_parameter(
            "world.payload_mass",
            1.0,
        )

    def _read_parameters(self):
        self.checkpoint_path = Path(
            self.get_parameter(
                "checkpoint_path"
            ).value
        ).expanduser().resolve()

        self.requested_device = str(
            self.get_parameter(
                "device"
            ).value
        )

        self.torch_num_threads = int(
            self.get_parameter(
                "torch_num_threads"
            ).value
        )

        self.shield_enabled = bool(
            self.get_parameter(
                "shield.enabled"
            ).value
        )

        self.shadow_mode = bool(
            self.get_parameter(
                "shield.shadow_mode"
            ).value
        )

        self.shield_num_samples = int(
            self.get_parameter(
                "shield.num_samples"
            ).value
        )

        self.shield_horizon = int(
            self.get_parameter(
                "shield.horizon"
            ).value
        )

        self.curriculum_level = float(
            self.get_parameter(
                "shield.curriculum_level"
            ).value
        )

        self.physics_level = int(
            self.get_parameter(
                "shield.physics_level"
            ).value
        )

        self.env_level = int(
            self.get_parameter(
                "shield.env_level"
            ).value
        )

        self.cable_length = float(
            self.get_parameter(
                "world.cable_length"
            ).value
        )

        self.cable_k = float(
            self.get_parameter(
                "world.cable_k"
            ).value
        )

        self.cable_d = float(
            self.get_parameter(
                "world.cable_d"
            ).value
        )

        self.payload_mass = float(
            self.get_parameter(
                "world.payload_mass"
            ).value
        )

        if self.shield_num_samples < 2:
            raise ValueError(
                "shield.num_samples must "
                "be at least 2."
            )

        if not 1 <= self.shield_horizon <= 10:
            raise ValueError(
                "shield.horizon must be "
                "between 1 and 10."
            )

        if self.cable_length <= 0.0:
            raise ValueError(
                "world.cable_length must "
                "be positive."
            )

        if self.payload_mass <= 0.0:
            raise ValueError(
                "world.payload_mass must "
                "be positive."
            )

    def _select_device(
        self,
        requested_device: str,
    ) -> torch.device:
        requested_device = (
            requested_device
            .strip()
            .lower()
        )

        if requested_device == "cuda":
            if torch.cuda.is_available():
                return torch.device(
                    "cuda"
                )

            self.get_logger().warning(
                "CUDA requested but unavailable. "
                "Using CPU."
            )

        return torch.device(
            "cpu"
        )

    def _load_policy(
        self,
    ) -> PPOPolicy:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "PPO checkpoint not found: "
                f"{self.checkpoint_path}"
            )

        self.get_logger().info(
            "Loading PPO actor and critic from: "
            f"{self.checkpoint_path}"
        )

        # The checkpoint was produced locally and
        # contains NumPy metadata.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):
            raise RuntimeError(
                "The checkpoint must contain "
                "a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        policy = PPOPolicy().to(
            self.device
        )

        policy.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        policy.eval()

        self.get_logger().info(
            "PPO checkpoint loaded successfully. "
            f"Episode="
            f"{checkpoint.get('episode')}, "
            f"stage="
            f"{checkpoint.get('training_stage')}, "
            f"alpha="
            f"{checkpoint.get('curriculum_alpha')}"
        )

        return policy

    def _run_policy(
        self,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if observation.shape != (
            OBSERVATION_SIZE,
        ):
            raise ValueError(
                "Invalid observation shape: "
                f"expected "
                f"({OBSERVATION_SIZE},), "
                f"received "
                f"{observation.shape}."
            )

        if not np.isfinite(
            observation
        ).all():
            raise ValueError(
                "Observation contains "
                "NaN or Inf."
            )

        observation_tensor = (
            torch.from_numpy(
                observation
            )
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
        )

        with torch.no_grad():
            distribution, critic_value = (
                self.policy(
                    observation_tensor
                )
            )

            action_tensor = (
                distribution.mean
                .squeeze(0)
                .clamp(
                    -1.0,
                    1.0,
                )
            )

        if tuple(
            action_tensor.shape
        ) != (ACTION_SIZE,):
            raise RuntimeError(
                "Invalid actor output shape: "
                f"{tuple(action_tensor.shape)}"
            )

        if not torch.isfinite(
            action_tensor
        ).all():
            raise RuntimeError(
                "Actor output contains "
                "NaN or Inf."
            )

        action = (
            action_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        value = float(
            critic_value
            .squeeze()
            .detach()
            .cpu()
            .item()
        )

        return action, value

    def _validate_policy(self):
        zero_observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

        action, value = self._run_policy(
            zero_observation
        )

        maximum_difference = float(
            np.max(
                np.abs(
                    action
                    - ZERO_OBSERVATION_EXPECTED_ACTION
                )
            )
        )

        self.get_logger().info(
            "PPO zero-observation action: "
            f"{action.tolist()}"
        )

        self.get_logger().info(
            "PPO zero-observation critic value: "
            f"{value:.9f}"
        )

        self.get_logger().info(
            "PPO validation maximum difference: "
            f"{maximum_difference:.9e}"
        )

        if maximum_difference > 1.0e-5:
            raise RuntimeError(
                "PPO numerical validation failed."
            )

        self.get_logger().info(
            "PPO numerical validation passed."
        )

    def _create_shadow_world(
        self,
    ) -> UnifiedGenesisWorld:
        random_state = (
            np.random.get_state()
        )

        try:
            np.random.seed(0)

            world = UnifiedGenesisWorld(
                num_uavs=NUM_UAVS,
                execution_mode=(
                    "centralized"
                ),
                scene=None,
                stage="free",
                visualize=False,
                start_at_target=True,
            )

            world.reset(
                target_radius=0.2,
                physics_level=(
                    self.physics_level
                ),
                env_level=(
                    self.env_level
                ),
            )

        finally:
            np.random.set_state(
                random_state
            )

        # Override properties randomized during
        # reset with deployment parameters.
        world.cable_length = (
            self.cable_length
        )

        world.cable_k = (
            self.cable_k
        )

        world.cable_d = (
            self.cable_d
        )

        world.payload_mass = (
            self.payload_mass
        )

        world.execution_mode = (
            "centralized"
        )

        world.training_stage = (
            "nmpc_active_shield"
        )

        world.curriculum_level = (
            self.curriculum_level
        )

        world.physics_level = (
            self.physics_level
        )

        world.env_level = (
            self.env_level
        )

        if hasattr(
            world,
            "use_state_estimator",
        ):
            world.use_state_estimator = False

        self.get_logger().info(
            "Predictive shadow world created. "
            f"cable_length="
            f"{world.cable_length}, "
            f"cable_k={world.cable_k}, "
            f"cable_d={world.cable_d}, "
            f"payload_mass="
            f"{world.payload_mass}"
        )

        return world

    def _sync_shadow_world(
        self,
        observation: np.ndarray,
        raw_action: np.ndarray,
    ):
        world = self.shadow_world

        for uav_index in range(
            NUM_UAVS
        ):
            state_start = (
                6 * uav_index
            )

            position = np.array(
                observation[
                    state_start:
                    state_start + 3
                ] * 10.0,
                dtype=np.float32,
                copy=True,
            )

            velocity = np.array(
                observation[
                    state_start + 3:
                    state_start + 6
                ] * 5.0,
                dtype=np.float32,
                copy=True,
            )

            setattr(
                world,
                f"uav{uav_index}_pos",
                position,
            )

            setattr(
                world,
                f"uav{uav_index}_vel",
                velocity,
            )

            acceleration_name = (
                f"uav{uav_index}_acc"
            )

            if hasattr(
                world,
                acceleration_name,
            ):
                setattr(
                    world,
                    acceleration_name,
                    np.zeros(
                        3,
                        dtype=np.float32,
                    ),
                )

        shared_start = (
            6 * NUM_UAVS
        )

        world.payload_pos = np.array(
            observation[
                shared_start:
                shared_start + 3
            ] * 10.0,
            dtype=np.float32,
            copy=True,
        )

        world.payload_vel = np.array(
            observation[
                shared_start + 3:
                shared_start + 6
            ] * 5.0,
            dtype=np.float32,
            copy=True,
        )

        tracking_error = np.array(
            observation[
                shared_start + 6:
                shared_start + 9
            ] * 5.0,
            dtype=np.float32,
            copy=True,
        )

        world.target = (
            world.payload_pos
            + tracking_error
        ).astype(np.float32)

        world.target_vel = np.array(
            observation[
                shared_start + 9:
                shared_start + 12
            ] * 5.0,
            dtype=np.float32,
            copy=True,
        )

        world.theta_x = float(
            observation[
                shared_start + 12
            ] * np.pi
        )

        world.theta_y = float(
            observation[
                shared_start + 13
            ] * np.pi
        )

        world.prev_theta_x = (
            world.theta_x
        )

        world.prev_theta_y = (
            world.theta_y
        )

        world.tension = float(
            observation[
                shared_start + 14
            ] * 20.0
        )

        world.wind_strength = float(
            observation[
                shared_start + 15
            ]
        )

        world.wind_enabled = (
            abs(world.wind_strength)
            > 1.0e-9
        )

        if (
            not world.wind_enabled
            and hasattr(
                world,
                "wind_vector",
            )
        ):
            world.wind_vector = np.zeros(
                3,
                dtype=np.float32,
            )

        world.time = max(
            0.0,
            float(
                observation[
                    shared_start + 16
                ] * 10.0
            ),
        )

        if hasattr(
            world,
            "control_time",
        ):
            world.control_time = 0.0

        if hasattr(
            world,
            "done",
        ):
            world.done = False

        if hasattr(
            world,
            "payload_acc",
        ):
            world.payload_acc = np.zeros(
                3,
                dtype=np.float32,
            )

        if self.previous_safe_action is None:
            world.prev_action = (
                raw_action.copy()
            )

        else:
            world.prev_action = (
                self.previous_safe_action
                .copy()
            )

    def _run_shield(
        self,
        observation: np.ndarray,
        raw_action: np.ndarray,
    ) -> tuple[np.ndarray, bool, float]:
        self._sync_shadow_world(
            observation,
            raw_action,
        )

        shield_start = (
            time.perf_counter()
        )

        safe_action = shield_action(
            world=self.shadow_world,
            policy_action=(
                raw_action.copy()
            ),
            curriculum_level=(
                self.curriculum_level
            ),
            physics_level=(
                self.physics_level
            ),
            env_level=(
                self.env_level
            ),
            num_samples=(
                self.shield_num_samples
            ),
            policy=self.policy,
            device=self.device,
            horizon_override=(
                self.shield_horizon
            ),
            is_training=False,
        )

        shield_time_ms = (
            time.perf_counter()
            - shield_start
        ) * 1000.0

        safe_action = np.asarray(
            safe_action,
            dtype=np.float32,
        ).reshape(-1)

        if safe_action.shape != (
            ACTION_SIZE,
        ):
            raise RuntimeError(
                "Invalid shield action shape: "
                f"{safe_action.shape}"
            )

        if not np.isfinite(
            safe_action
        ).all():
            raise RuntimeError(
                "Shield action contains "
                "NaN or Inf."
            )

        safe_action = np.clip(
            safe_action,
            -1.0,
            1.0,
        )

        intervention_norm = float(
            np.linalg.norm(
                safe_action
                - raw_action
            )
        )

        intervened = (
            intervention_norm
            > 1.0e-5
        )

        self.previous_safe_action = (
            safe_action.copy()
        )

        return (
            safe_action,
            intervened,
            shield_time_ms,
        )

    def compute_callback(
        self,
        request: ComputePpoShield.Request,
        response: ComputePpoShield.Response,
    ) -> ComputePpoShield.Response:
        start_time = (
            time.perf_counter()
        )

        response.sequence = (
            request.sequence
        )

        response.success = False
        response.shield_applied = False
        response.shield_intervened = False

        try:
            observation = np.asarray(
                request.observation,
                dtype=np.float32,
            )

            raw_action, critic_value = (
                self._run_policy(
                    observation
                )
            )

            shield_time_ms = 0.0

            if self.shield_enabled:
                (
                    safe_action,
                    intervened,
                    shield_time_ms,
                ) = self._run_shield(
                    observation,
                    raw_action,
                )

                response.shield_intervened = (
                    intervened
                )

                # A false value prevents C++ from
                # publishing while shadow mode is
                # active.
                response.shield_applied = (
                    not self.shadow_mode
                )

                mode_name = (
                    "shield_shadow_mode"
                    if self.shadow_mode
                    else "shield_active"
                )

                response.status = (
                    f"{mode_name}; "
                    f"intervened={intervened}; "
                    f"shield_ms="
                    f"{shield_time_ms:.3f}"
                )

            else:
                safe_action = (
                    raw_action.copy()
                )

                response.status = (
                    "shield_disabled"
                )

            response.raw_action = array(
                "d",
                raw_action.astype(
                    np.float64
                ).tolist(),
            )

            response.safe_action = array(
                "d",
                safe_action.astype(
                    np.float64
                ).tolist(),
            )

            response.critic_value = (
                critic_value
            )

            response.success = True

        except Exception as error:
            response.raw_action = array(
                "d",
                [0.0] * ACTION_SIZE,
            )

            response.safe_action = array(
                "d",
                [0.0] * ACTION_SIZE,
            )

            response.critic_value = 0.0
            response.shield_applied = False
            response.shield_intervened = False
            response.status = str(error)

            self.get_logger().error(
                "PPO/shield request failed: "
                f"{error}\n"
                f"{traceback.format_exc()}"
            )

        response.computation_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return response


def main(args=None):
    rclpy.init(
        args=args
    )

    node = None

    try:
        node = PpoShieldNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
