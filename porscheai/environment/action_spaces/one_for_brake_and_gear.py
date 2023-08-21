""" some examples of action space configs
"""
from typing import Tuple
from gymnasium.spaces import Box
import numpy as np
from porscheai.environment.configs.abstract_classes import (
    ActionSpaceConfigs,
)
from porscheai.environment.configs import PhysicConfigs


default_physics = PhysicConfigs()


class OneForBrakeAndGearActionSpace(ActionSpaceConfigs):
    """action space based on the idea to use one action for brake and gear to prevent
    invalid actions
    """

    def __init__(
        self,
        action_space_bounds: Tuple[float, float] = (-1.0, 1.0),
        physics_configs: PhysicConfigs = default_physics,
    ) -> None:
        super().__init__()
        self.action_space_bounds = action_space_bounds
        self.throttle_bounds = physics_configs.throttle_bounds
        self.brake_bounds = physics_configs.brake_bounds

    def create_action_space(self) -> Box:
        _action_space_length: int = 1
        return Box(
            low=np.repeat(self.action_space_bounds[0], _action_space_length),
            high=np.repeat(self.action_space_bounds[1], _action_space_length),
            shape=(_action_space_length,),
            dtype=np.float32,
        )

    def get_brake_from_action(self, action: float) -> float:
        """get brake value from played action

        Args:
            action (float): action to play

        Returns:
            float: brake value for physics
        """
        _unnormalized_brake = -min(0.0, action)
        return self.unnormalize_action_brake(_unnormalized_brake)

    def get_throttle_from_action(self, action: float) -> float:
        """get throttle value from played action

        Args:
            action (float): action to play

        Returns:
            float: throttle value for physics
        """
        _unnormalized_throttle = max(0.0, action)
        return self.unnormalize_action_throttle(_unnormalized_throttle)

    def unnormalize_action_throttle(self, throttle_norm: float) -> float:
        """unnormlize throttle value from action space

        Args:
            throttle_norm (float): normed throttle value from action

        Returns:
            float: unnormalized throttle value
        """
        y_range = self.action_space_bounds[1] - 0.0
        x_range = self.throttle_bounds[1] - self.throttle_bounds[0]
        return (throttle_norm * x_range) / y_range + self.throttle_bounds[0]

    def unnormalize_action_brake(self, brake_norm: float) -> float:
        """unnormlize brake value from action space

        Args:
            brake_norm (float): normed brake value from action

        Returns:
            float: unnormalized brake value for pyhsics
        """
        y_range = 0.0 - self.action_space_bounds[0]
        x_range = self.brake_bounds[1] - self.brake_bounds[0]
        return (brake_norm * x_range) / y_range + self.brake_bounds[0]


__all__ = [OneForBrakeAndGearActionSpace.__name__]
