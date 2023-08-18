""" base environment for driver gym environment
"""
from typing import Any, Dict, Tuple, Callable
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from .helpclasses import (
    GeneralGameconfigs,
    ReferenceTrajectory,
    PhysicConfigs,
    create_reference_trajecotry_ms,
)

df_general_game_configs = GeneralGameconfigs()
df_trajectory_configs = ReferenceTrajectory()
df_physic_configs = PhysicConfigs()


@dataclass
class DriverPhysicsParameter:
    """dataclass to store relevant game physics during the game"""

    velocity_ms: float = 0.0


class SimpleDriver(gym.Env):
    """gym environment to simulate a simple car driver"""

    def __init__(
        self,
        game_configs: GeneralGameconfigs = df_general_game_configs,
        traj_configs: ReferenceTrajectory = df_trajectory_configs,
        physic_configs: PhysicConfigs = df_physic_configs,
    ):
        """

        Args:
            game_configs (GeneralGameconfigs, optional): _description_.
            Defaults to genral_game_configs.
            trajectory_configs (ReferenceTrajectory, optional): _description_.
            Defaults to trajectory_configs.
            physic_configs (PhysicConfigs, optional): _description_.
            Defaults to physic_configs.
        """

        self.physics = physic_configs
        # needed information for game
        self.total_no_timesteps: int = traj_configs.total_timesteps
        self.time_step_size_s: float = traj_configs.simulation_frequency_s
        self.start_velocity_ms: float = self.get_start_velocity_ms(
            traj_conifgs=traj_configs
        )
        self.velocity_ms_normalisation = self._get_velocity_ms_normalisation(
            traj_config=traj_configs, general_config=game_configs
        )
        _target_velocity_traj_ms = create_reference_trajecotry_ms(
            reference_traj_conf=traj_configs
        )
        self.target_velocity_traj_ms_normalized = self.velocity_ms_normalisation(
            _target_velocity_traj_ms.copy()
        )
        self.current_time_step: int = 0
        # observation space dependent information
        self.outlook_length: int = game_configs.outlook_length

        # reward configs
        self.reward_scaling = game_configs.rewardscale

        self.game_physics_params: DriverPhysicsParameter = DriverPhysicsParameter(
            velocity_ms=self.start_velocity_ms
        )

        # Gym setup
        # current state space consist of future target velocities and current deviation
        _obs_space_length: int = game_configs.outlook_length + 1
        self.observation_space = Box(
            low=np.repeat(game_configs.obs_bounds[0], _obs_space_length),
            high=np.repeat(game_configs.obs_bounds[1], _obs_space_length),
            shape=(_obs_space_length,),
            dtype=np.float32,
        )

        # adapt to multiple types of action spaces
        _action_space_length: int = 1
        self.action_space = Box(
            low=np.repeat(game_configs.action_space_bounds[0], _action_space_length),
            high=np.repeat(game_configs.action_space_bounds[1], _action_space_length),
            shape=(_action_space_length,),
            dtype=np.float32,
        )

    def get_start_velocity_ms(self, traj_conifgs: ReferenceTrajectory) -> float:
        """get start velocity of car based on configurations

        Args:
            traj_conifgs (ReferenceTrajectory): needed configs to get starting velocity

        Returns:
            float: starting velocity in m/s
        """
        start_velocity_ms = self.physics.car_configs.start_velocity_ms
        start_velocity_ms: float = (
            start_velocity_ms
            if start_velocity_ms is not None
            else traj_conifgs.velocities_ms[0]
        )
        return start_velocity_ms

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

    def _get_velocity_ms_normalisation(
        self, traj_config: ReferenceTrajectory, general_config: GeneralGameconfigs
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
            traj_config.velocity_bounds_ms[1]
            + general_config.velocity_ms_addition_upper_bound
        )
        y_range = general_config.obs_bounds[1] - general_config.obs_bounds[0]

        def _dummy_normalisation(
            value: float | np.ndarray, unnorm: bool = False
        ) -> float | np.ndarray:
            if not unnorm:
                return value * y_range / x_range + general_config.obs_bounds[0]
            return value * x_range / y_range

        return _dummy_normalisation

    def _reshape_to_length_n(self, array: np.array, length: int) -> np.array:
        """
        Stretches a 1d numpy array to length n.
        """
        len_array = len(array)
        array = np.append(array, np.repeat(array[-1], length - len_array))
        return array

    def get_throttle(self, action: np.ndarray) -> float:
        """get throttle value from actions

        Args:
            action (np.ndarray): action to play

        Returns:
            float: throttle value
        """
        return action

    def get_brake(self, action: np.ndarray) -> float:
        """get brake value from action

        Args:
            action (np.ndarray): action to play

        Returns:
            float: brake value
        """
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """doing one step in car simulation environment

        Args:
            action (np.ndarray): action to be performed

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: next state, reward, bool if done,
            bool if truncated, info if done
        """

        # do all physics for the game
        if action is not None:
            # unnormalize the actions
            throttle = self.get_throttle(action=action)
            brake = self.get_brake(action=action)
            new_velocity = self.physics.get_velocity(
                throttle=throttle,
                brake=brake,
                velocity_ms=self.game_physics_params.velocity_ms,
                time_step_s=self.time_step_size_s,
            )
        else:
            new_velocity = self.start_velocity_ms

        self.game_physics_params.velocity_ms = new_velocity

        done = False

        # normalize current speed
        _normalized_velocity = self.velocity_ms_normalisation(new_velocity)

        # Get target speed vector, normalize and calculate deviation
        len_outlook_iteration = np.min(
            [self.total_no_timesteps - self.current_time_step, self.outlook_length]
        )  # get future reference trajectory, shorted if we are at end of episode
        _velocity_target_ms_norm = self.target_velocity_traj_ms_normalized[
            self.current_time_step : (self.current_time_step + len_outlook_iteration)
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
        observation = np.append(_velocity_target_ms_norm, deviation)
        # TODO: outsource reward
        # Calculate Reward
        reward = -abs(deviation)
        reward = reward * self.reward_scaling

        # update time step
        self.current_time_step += 1
        # Check if Round is done
        if self.current_time_step == self.total_no_timesteps:
            done = True

        return observation, reward, done, False, {}

    def _append_short_trajectory_ms_values(self, trajectory: np.ndarray) -> np.ndarray:
        """method to update the outlook velocity trajectory if the episode is
        shorter than the total outlook length

        Args:
            trajectory (np.ndarray): trajectory to be updated

        Returns:
            np.ndarray: updated trajectory
        """
        return trajectory

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset the environment

        Args:
            seed (int | None, optional): seed from parent. Defaults to None.
            options (dict[str, Any] | None, optional): options from parent. Defaults to None.

        Returns:
            tuple[np.ndarray, dict[str, Any]]: first state and info dictionary
        """
        super().reset(seed=seed, options=options)
        self.current_time_step = 0
        return self.step(action=None)[0], {}

    def render(self, mode="human"):
        pass
