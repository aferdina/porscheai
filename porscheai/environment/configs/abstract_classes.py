"""observation space configs"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class DriverPhysicsParameter:
    """dataclass to store relevant game physics during to retrieve actions and observations"""

    velocity_ms: float = 0.0
    current_time_step: int = 0


class ObservationSpaceConfigs(ABC):
    """abstract class for observation space configs"""

    @abstractmethod
    def create_observation_space(self) -> gym.Space:
        """create observation space

        Returns:
            gym.Space: observation space
        """

    @abstractmethod
    def get_observation(
        self, driver_physics_params: DriverPhysicsParameter
    ) -> np.ndarray:
        """get observation

        Args:
            game_configs (GeneralGameconfigs): game configs

        Returns:
            np.ndarray: observation
        """

    @abstractmethod
    def get_reward(self, observation: np.ndarray) -> float:
        """get reward for specific observation

        Args:
            observation (np.ndarray): observation

        Returns:
            float: reward
        """


class ActionSpaceConfigs(ABC):
    """abstract class for action space configs"""

    @abstractmethod
    def create_action_space(self) -> gym.Space:
        """create action space
        Returns:
            gym.Space: action space
        """

    @abstractmethod
    def get_brake_from_action(self, action: Any) -> float:
        """get action

        Args:
            action (Any): action from action space
        Returns:
            float: brake value as float
        """

    @abstractmethod
    def get_throttle_from_action(self, action: Any) -> float:
        """get action

        Args:
            action (Any): action from action space
        Returns:
            float: throttle value as float
        """


__all__ = [
    DriverPhysicsParameter.__name__,
    ObservationSpaceConfigs.__name__,
    ActionSpaceConfigs.__name__,
]
