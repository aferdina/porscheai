""" some examples of observation space configs
"""
from typing import Tuple, Callable
from gymnasium.spaces import Box
import numpy as np
from porscheai.environment.configs import (
    ObservationSpaceConfigs,
    DriverPhysicsParameter,
    ReferenceTrajectory,
    create_reference_trajecotry_ms,
)


class OutlookObservationSpace(ObservationSpaceConfigs):
    """observation space for outlook observation space"""

    def __init__(
        self,
        reference_trajectory: ReferenceTrajectory,
        outlook_length: int = 1,
        obs_bounds: Tuple[float, float] = (-1.0, 1.0),
        reward_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.outlook_length = outlook_length
        self.obs_bounds = obs_bounds
        self.velocity_ms_addition_upper_bound: float = (
            20.0  # upper bound for possible velocities
        )
        self.total_no_timesteps = reference_trajectory.total_timesteps
        self.velocity_ms_normalisation = self._get_velocity_ms_normalisation(
            traj_config=reference_trajectory
        )
        _target_velocity_traj_ms = create_reference_trajecotry_ms(
            reference_traj_conf=reference_trajectory
        )
        self.target_velocity_traj_ms_normalized = self.velocity_ms_normalisation(
            _target_velocity_traj_ms.copy()
        )
        self.last_target_velocity_ms = self.target_velocity_traj_ms_normalized[-1]
        self.reward_scaling = reward_scaling

    def create_observation_space(self) -> Box:
        """create observation space

        Args:
            game_configs (GeneralGameconfigs): game configs

        Returns:
            gym.Space: observation space
        """
        _obs_space_length = self.outlook_length + 1
        return Box(
            low=np.repeat(self.obs_bounds[0], _obs_space_length),
            high=np.repeat(self.obs_bounds[1], _obs_space_length),
            shape=(_obs_space_length,),
            dtype=np.float32,
        )

    def get_observation(
        self, driver_physics_params: DriverPhysicsParameter
    ) -> np.ndarray:
        # normalize current speed
        _normalized_velocity = self.velocity_ms_normalisation(
            driver_physics_params.velocity_ms
        )

        # Get target speed vector, normalize and calculate deviation
        len_outlook_iteration = np.min(
            [
                self.total_no_timesteps - (driver_physics_params.current_time_step + 1),
                self.outlook_length,
            ]
        )  # get future reference trajectory, shorted if we are at end of episode
        _velocity_target_ms_norm = self.target_velocity_traj_ms_normalized[
            driver_physics_params.current_time_step : (
                driver_physics_params.current_time_step + len_outlook_iteration + 1
            )
        ]
        # if len outlook of this episode is shorter than total outlook length, then append values to
        # make it the same length, pay attention to the fact that the last value is repeated and
        # should be set meaningful
        if len_outlook_iteration < self.outlook_length:
            _velocity_target_ms_norm = self._append_short_trajectory_ms_values(
                trajectory=_velocity_target_ms_norm
            )
        # divide by 2 to ensure the sum of two normalized numbers is in the normalization range
        deviation = (_normalized_velocity - float(_velocity_target_ms_norm[0])) / 2
        obs = np.append(_velocity_target_ms_norm[1:], deviation).astype(np.float32)
        return obs

    # pylint: disable=E0202
    def velocity_ms_normalisation(
        self, value: float | np.ndarray, unnorm: bool = False
    ) -> float | np.ndarray:
        """create normalized value for velocity based on configurations

        Args:
            value (float | np.ndarray): velocity in m/s to be normalized

        Returns:
            float | np.ndarray: normalized velocity for environment
        """
        if not unnorm:
            return value
        return -value

    def _append_short_trajectory_ms_values(self, trajectory: np.ndarray) -> np.ndarray:
        """method to update the outlook velocity trajectory if the episode is
        shorter than the total outlook length

        Args:
            trajectory (np.ndarray): trajectory to be updated

        Returns:
            np.ndarray: updated trajectory
        """
        appended_trajectory = np.pad(
            trajectory,
            (0, self.outlook_length + 1 - len(trajectory)),
            mode="constant",
            constant_values=self.last_target_velocity_ms,
        )
        return appended_trajectory

    def _get_velocity_ms_normalisation(
        self,
        traj_config: ReferenceTrajectory,
    ) -> Callable[[float | np.ndarray, bool], float | np.ndarray]:
        """create normalisation function based on configurations

        Args:
            traj_config (ReferenceTrajectory): trajectory configurations
            general_config (GeneralGameconfigs): general game configurations

        Returns:
            Callable[[float | np.ndarray], float | np.ndarray]: function to normalise values
        """
        # lower velocity bound should be equal to zero
        x_range = (
            traj_config.velocity_bounds_ms[1] + self.velocity_ms_addition_upper_bound
        )
        y_range = self.obs_bounds[1] - self.obs_bounds[0]

        def _dummy_normalisation(
            value: float | np.ndarray, unnorm: bool = False
        ) -> float | np.ndarray:
            if not unnorm:
                return value * y_range / x_range + self.obs_bounds[0]
            return value * x_range / y_range

        return _dummy_normalisation

    def get_reward(self, observation: np.ndarray) -> float:
        """get reward based on observation

        Args:
            observation (np.ndarray): observation

        Returns:
            float: reward
        """
        deviation = observation[-1]
        return -abs(deviation) * self.reward_scaling


__all__ = [OutlookObservationSpace.__name__]
