#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np
from reachy2_sdk import ReachySDK
from typing import Any

# from stretch_body.gamepad_teleop import GamePadTeleop
# from stretch_body.robot import Robot as StretchAPI
# from stretch_body.robot_params import RobotParams

from lerobot.cameras.utils import make_cameras_from_configs

from ..robot import Robot
from .configuration_reachy2 import Reachy2RobotConfig

# {lerobot_keys: reachy2_sdk_keys}
REACHY2_JOINTS = {
    "neck_yaw.pos": "head.neck.yaw",
    "neck_pitch.pos": "head.neck.pitch",
    "neck_roll.pos": "head.neck.roll",
    "r_shoulder_pitch.pos": "r_arm.shoulder.pitch",
    "r_shoulder_roll.pos": "r_arm.shoulder.roll",
    "r_elbow_yaw.pos": "r_arm.elbow.yaw",
    "r_elbow_pitch.pos": "r_arm.elbow.pitch",
    "r_wrist_roll.pos": "r_arm.wrist.roll",
    "r_wrist_pitch.pos": "r_arm.wrist.pitch",
    "r_wrist_yaw.pos": "r_arm.wrist.yaw",
    "r_gripper.pos": "r_arm.gripper",
    "l_shoulder_pitch.pos": "l_arm.shoulder.pitch",
    "l_shoulder_roll.pos": "l_arm.shoulder.roll",
    "l_elbow_yaw.pos": "l_arm.elbow.yaw",
    "l_elbow_pitch.pos": "l_arm.elbow.pitch",
    "l_wrist_roll.pos": "l_arm.wrist.roll",
    "l_wrist_pitch.pos": "l_arm.wrist.pitch",
    "l_wrist_yaw.pos": "l_arm.wrist.yaw",
    "l_gripper.pos": "l_arm.gripper",
    "l_antenna.pos": "head.l_antenna",
    "r_antenna.pos": "head.r_antenna",
}

REACHY2_VEL = {
    "mobile_base.vx": "vx",
    "mobile_base.vy": "vy",
    "mobile_base.vtheta": "vtheta",
}


class Reachy2Robot(Robot):
    """
    [Reachy 2](https://www.pollen-robotics.com/reachy/), by Pollen Robotics.
    """

    config_class = Reachy2RobotConfig
    name = "reachy2"

    def __init__(self, config: Reachy2RobotConfig):
        super().__init__(config)

        self.config = config
        self.robot_type = self.config.type
        self.use_external_commands = self.config.use_external_commands

        self.reachy: None | ReachySDK = None
        self.cameras = make_cameras_from_configs(config.cameras)

        self.logs = {}

    @property
    def observation_features(self) -> dict:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> dict:
        return self.motors_features

    @property
    def camera_features(self) -> dict[str, dict]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def motors_features(self) -> dict:
        return {**dict.fromkeys(
            REACHY2_JOINTS.keys(),
            float,
        ), **dict.fromkeys(
            REACHY2_VEL.keys(),
            float,
        )}

    @property
    def is_connected(self) -> bool:
        return self.reachy.is_connected() if self.reachy is not None else False

    def connect(self) -> None:
        self.reachy = ReachySDK(self.config.ip_address)
        if not self.is_connected:
            print("Error connecting to Reachy 2.")
            raise ConnectionError()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def configure(self) -> None:
        self.reachy.turn_on()
        self.reachy.reset_default_limits()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def _get_state(self) -> dict:
        pos_dict = {k: self.reachy.joints[v].present_position for k, v in REACHY2_JOINTS.items()}
        vel_dict = {k: self.reachy.mobile_base.odometry[v] for k, v in REACHY2_VEL.items()}
        return {**pos_dict, **vel_dict}

    def get_observation(self) -> dict[str, np.ndarray]:
        obs_dict = {}

        # Read Reachy 2 state
        before_read_t = time.perf_counter()
        obs_dict = self._get_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError()

        before_write_t = time.perf_counter()

        vel = {}
        for key, val in action.items():
            if key not in REACHY2_JOINTS:
                if key not in REACHY2_VEL:
                    raise KeyError(f"Key '{key}' is not a valid motor key in Reachy 2.")
                else:
                    vel[REACHY2_VEL[key]] = val
            else:
                self.reachy.joints[REACHY2_JOINTS[key]].goal_position = val
        self.reachy.mobile_base.set_goal_speed(vel["vx"], vel["vy"], vel["vtheta"])

        # We don't send the goal positions if we control Reachy 2 externally
        if not self.use_external_commands:
            self.reachy.send_goal_positions()
            self.reachy.mobile_base.send_speed_command()

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t
        return action

    def disconnect(self) -> None:
        self.reachy.turn_off_smoothly()
        self.reachy.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
