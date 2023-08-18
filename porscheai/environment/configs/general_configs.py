""" base environment for driver gym environment
"""
from dataclasses import dataclass
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
from strenum import StrEnum
import numpy as np

@dataclass
class GeneralGameconfigs:
    """gameconfigs including bounds for reward, observatio and action space as well as
    bounds for throttle and brake
    """

    rewardscale: float = 1.0  # factor scaling the rewards
    obs_bounds: Tuple[float, float] = (-1.0, 1.0)  # bounds for observation space
    action_space_bounds: Tuple[float] = (-1.0, 1.0)  # bounds for action space
    outlook_length: int = 1  # number of timesteps to look in the future
    velocity_ms_addition_upper_bound: float = (
        20.0  # upper bound for possible velocities
    )


class ObservationSpaceType(StrEnum):
    OUTLOOK = "outlook"


@dataclass
class ObservationSpaceConfigs:
    obs_type: ObservationSpaceType = ObservationSpaceType.OUTLOOK

    def create_action_space(self, game_configs: GeneralGameconfigs) -> gym.Space:
        """create action space

        Args:
            game_configs (GeneralGameconfigs): game configs

        Returns:
            gym.Space: action space
        """
        return Box(
            low=game_configs.action_space_bounds[0],
            high=game_configs.action_space_bounds[1],
            shape=(1,),
            dtype=np.float32,
        )


__all__ = [GeneralGameconfigs.__name__]
