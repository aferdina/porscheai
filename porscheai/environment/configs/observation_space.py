from abc import ABC, abstractclassmethod
import gymnasium as gym
from gymnasium.spaces import Box
from strenum import StrEnum
import numpy as np

class ObservationSpaceType(StrEnum):
    OUTLOOK = "outlook"

class ObservationSpaceConfigs(ABC):
    obs_type: ObservationSpaceType = ObservationSpaceType.OUTLOOK

    @abstractclassmethod
    def create_observation_space(self, game_configs: GeneralGameconfigs) -> gym.Space:
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
    
    @abstractclassmethod
    def get_observation(self, game_configs: GeneralGameconfigs) -> np.ndarray:
        """get observation

        Args:
            game_configs (GeneralGameconfigs): game configs

        Returns:
            np.ndarray: observation
        """
        pass