""" some examples of action space configs
"""
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from porscheai.environment.configs.abstract_classes import (
    ActionSpaceConfigs,
    DriverPhysicsParameter,
)


class OneForBrakeAndGearActionSpace(ActionSpaceConfigs):
    def __init__(self, action_space_bounds: Tuple[float, float]) -> None:
        super().__init__()
        self.action_space_bounds = action_space_bounds

    def create_action_space(self) -> Box:
        _action_space_length: int = 1
        return Box(
            low=np.repeat(self.action_space_bounds[0], _action_space_length),
            high=np.repeat(self.action_space_bounds[1], _action_space_length),
            shape=(_action_space_length,),
            dtype=np.float32,
        )
