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
    FACTOR_KMH_MS,
)

genral_game_configs = GeneralGameconfigs()
trajectory_configs = ReferenceTrajectory()
physic_configs = PhysicConfigs()


@dataclass
class DriverPhysicsParameter:
    """dataclass to store relevant game physics during the game"""

    velocity_ms: float = 0.0


class SimpleDriver(gym.Env):
    """gym environment to simulate a simple car driver"""

    def __init__(
        self,
        game_configs: GeneralGameconfigs = genral_game_configs,
        traj_configs: ReferenceTrajectory = trajectory_configs,
        physic_configs: PhysicConfigs = physic_configs,
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
        # adding information from configs
        self.total_no_timesteps: int = traj_configs.total_timesteps
        self.outlook_length: int = game_configs.outlook_length
        self.time_step_size_s: float = traj_configs.simulation_frequency_s
        # Initalization for Integration
        self.vehicle_distance_m: float = 0.0  # distance of vehicle from starting point

        # reward configs
        self.reward_scaling = game_configs.rewardscale

        self.velocity_kmh_normalisation = self._get_velocity_ms_normalisation(
            traj_config=traj_configs, general_config=genral_game_configs
        )
        # init start velocity
        _start_velocity_ms = physic_configs.car_configs.start_velocity_ms
        start_velocity_ms: float = (
            _start_velocity_ms
            if _start_velocity_ms is not None
            else traj_configs.velocities_kmh[0] / FACTOR_KMH_MS
        )
        self.game_physics_params: DriverPhysicsParameter = DriverPhysicsParameter(
            velocity_ms=start_velocity_ms
        )

        self.target_velocity_traj_ms = create_reference_trajecotry_ms(
            reference_traj_conf=traj_configs
        )
        
        # Gym setup
        # current state space consist of future target velocities and current deviation
        obs_space_length: int = game_configs.outlook_length + 1
        self.observation_space = Box(
            low=np.repeat(game_configs.obs_bounds[0], obs_space_length),
            high=np.repeat(game_configs.obs_bounds[1], obs_space_length),
            shape=(obs_space_length,),
            dtype=np.float32,
        )

        # adapt to multiple types of action spaces
        action_space_length: int = 1
        self.action_space = Box(
            low=np.repeat(game_configs.action_space_bounds[0], action_space_length),
            high=np.repeat(game_configs.action_space_bounds[1], action_space_length),
            shape=(action_space_length,),
            dtype=np.float32,
        )

        self.target_velocity_traj_ms_normalized = self.velocity_ms_normalisation(
            self.target_velocity_traj_ms.copy()
        )
        # get normalized velocity for complete trajectory
        v_target = self.target_velocity_traj_kmh[: self.outlook_length]
        v_target_norm = self._normalize_2(
            value=v_target,
            min_val=self.v_Car_min,
            max_val=self.v_Car_max,
            min_norm=-1,
            max_norm=1,
        )
        deviation = (
            v_Car_norm - v_target_norm
        ) / 2  # divide by 2 to ensure the sum of two normalized
        # numbers is in the normalization range
        self.first_state = np.append(v_Car_norm, deviation)
        self.state = self.first_state

        self.current_time_step: int = 0
        self.info: Dict[str, Any] = {}

    def velocity_ms_normalisation(
        self, value: float | np.ndarray
    ) -> float | np.ndarray:
        """create normalized value for velocity based on configurations

        Args:
            value (float | np.ndarray): velocity in m/s to be normalized

        Returns:
            float | np.ndarray: normalized velocity for environment
        """
        return value

    def _get_velocity_ms_normalisation(
        self, traj_config: ReferenceTrajectory, general_config: GeneralGameconfigs
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
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

        def _dummy_normalisation(value: float | np.ndarray) -> float | np.ndarray:
            return (value) * y_range / x_range + general_config.obs_bounds[0]

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
            Tuple[np.ndarray, float, bool, Dict]: next state, reward, bool if done, bool if truncated, info if done
        """
        done = False

        # unnormalize the actions
        throttle = self.get_throttle(action=action)
        brake = self.get_brake(action=action)
        _new_velocity = self.physics.get_velocity(
            throttle=throttle,
            brake=brake,
            velocity_ms=self.game_physics_params.velocity_ms,
            time_step_s=self.time_step_size_s,
        )

        # TODO: update game state
        # Get current speed and normalize
        _normalized_velocity = self.velocity_ms_normalisation(_new_velocity)

        # Get traget speed vector, normalize and calculate deviation
        len_outlook_iteration = np.min(
            [self.episode_length - self.current_time_step, self.len_outlook]
        )  # get future reference trajectory, shorted if we are at end of episode
        v_target = self.v_Soll_all[
            self.current_time_step : (self.current_time_step + len_outlook_iteration)
        ]
        if len(v_target) != self.len_outlook:
            v_target = self._reshape_to_length_n(
                array=v_target, length=self.len_outlook
            )
        v_target_norm = self._normalize_2(
            value=v_target,
            min_val=self.v_Car_min,
            max_val=self.v_Car_max,
            min_norm=-1,
            max_norm=1,
        )

        deviation = (
            v_target_norm - v_Car_norm
        ) / 2  # divide by 2 to ensure the sum of two normalized numbers is in the normalization range
        obs = np.append(v_Car_norm, deviation)

        # Calculate Reward
        reward = -(np.abs(v_Car_norm - v_target_norm[0]).astype(float))
        reward = reward * self.reward_scaling

        # update time step
        self.current_time_step += 1
        # Check if Round is done
        if self.current_time_step == self.total_no_timesteps:
            done = True

        return obs, reward, done, False, self.info

    def _update_vehicle_distance(self, vehicle_velocity_ms: float) -> None:
        """update vehicle distance based on vehicle velocity

        Args:
            vehicle_velocity_ms (float): vehicle velocity to use for update in meters/second
        """
        self.vehicle_distance_m = (
            self.vehicle_distance_m + self.time_step_size_s * vehicle_velocity_ms
        )

    def _un_normalize_2(
        self,
        value: float,
        min_val: float,
        max_val: float,
        min_norm: float,
        max_norm: float,
    ) -> float:
        """
        Un_normalizes a value from the [min_norm, max_norm] normalization
        Inverse of _normalize function

        Args:
        - value: the value to un_normalize
        - min_val: the minimum value in the allowed range of the value
        - max_val: the maximum value in the allowed range of the value
        - min_norm: lower boundary of the normalization range
        - max_norm: upper boundary of the normalization range
        """
        unnorm_value = (value - min_norm) * (max_val - min_val) / (
            max_norm - min_norm
        ) + min_val

        return unnorm_value

    def reset(self):
        # Reset Environment
        self.state = self.first_state
        self.current_time_step = 0
        self.vehicle_distance_m = 0.0

        return self.state

    def render(self, mode="human"):
        pass
