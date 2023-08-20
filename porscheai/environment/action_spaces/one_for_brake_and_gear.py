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
    def __init__(self, action_space_bounds: Tuple[float, float] = (-1.0, 1.0)) -> None:
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

    def get_brake_from_action(self) -> float:
        return super().get_brake_from_action()

    def get_throttle_from_action(self) -> float:
        return super().get_throttle_from_action()

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return super().unnormalize_action(action=action)


__all__ = [OneForBrakeAndGearActionSpace.__name__]
