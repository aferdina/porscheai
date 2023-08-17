""" base environment for driver gym environment
"""
from typing import Any, Dict, Tuple, Callable
from dataclasses import dataclass
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
import math
from .helpclasses import (
    GeneralGameconfigs,
    ReferenceTrajectory,
    PhysicConfigs,
    create_reference_trajecotry,
    FACTOR_KMH_MS,
)

genral_game_configs = GeneralGameconfigs()
trajectory_configs = ReferenceTrajectory()
physic_configs = PhysicConfigs()


@dataclass
class DriverPhysicsParameter:
    velocity_ms: float = 0.0
    brake: float = 0.0
    throttle: float = 0.0


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
            game_configs (GeneralGameconfigs, optional): _description_. Defaults to genral_game_configs.
            trajectory_configs (ReferenceTrajectory, optional): _description_. Defaults to trajectory_configs.
            physic_configs (PhysicConfigs, optional): _description_. Defaults to physic_configs.
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

        self.velocity_kmh_normalisation = self._get_velocity_kmh_normalisation(
            traj_config=traj_configs, general_config=genral_game_configs
        )
        # init start velocity
        _start_velocity_ms = physic_configs.car_configs.start_velocity_ms
        self.velocity_car_ms: float = (
            _start_velocity_ms
            if _start_velocity_ms is not None
            else traj_configs.velocities_kmh[0] / FACTOR_KMH_MS
        )

        self.target_velocity_traj_kmh = create_reference_trajecotry(
            reference_traj_conf=traj_configs
        )
        self.target_velocity_traj_kmh_normalized = self.velocity_kmh_normalisation(
            self.target_velocity_traj_kmh.copy()
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

        # Add the first State
        v_Car_norm = self._normalize_2(
            value=self.v_Car_ms * 3.6,
            min_val=self.v_Car_min,
            max_val=self.v_Car_max,
            min_norm=-1,
            max_norm=1,
        )
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
        ) / 2  # divide by 2 to ensure the sum of two normalized numbers is in the normalization range
        self.first_state = np.append(v_Car_norm, deviation)
        self.state = self.first_state

        self.current_time_step: int = 0
        self.info: Dict[str, Any] = {}

    def velocity_kmh_normalisation(value: float | np.ndarray) -> float | np.ndarray:
        """create normalized value for velocity based on configurations

        Args:
            value (float | np.ndarray): _description_

        Returns:
            float | np.ndarray: _description_
        """
        return value

    def _get_velocity_kmh_normalisation(
        self, traj_config: ReferenceTrajectory, general_config: GeneralGameconfigs
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """create normalisation function based on configurations

        Args:
            traj_config (ReferenceTrajectory): trajectory configurations
            general_config (GeneralGameconfigs): general game configurations

        Returns:
            Callable[[float | np.ndarray], float | np.ndarray]: function to normalise values
        """
        x_range = traj_config.velocity_bounds[1] - traj_config.velocity_bounds[0]
        y_range = general_config.obs_bounds[1] - general_config.obs_bounds[0]

        def _dummy_normalisation(value: float | np.ndarray) -> float | np.ndarray:
            return (
                value - traj_config.velocity_bounds[0]
            ) * y_range / x_range + general_config.obs_bounds[0]

        return _dummy_normalisation

    def _reshape_to_length_n(self, array: np.array, length: int) -> np.array:
        """
        Stretches a 1d numpy array to length n.
        """
        len_array = len(array)
        array = np.append(array, np.repeat(array[-1], length - len_array))
        return array

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """doing one step in car simulation environment

        Args:
            action (np.ndarray): action to be performed

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: next state, reward, bool if done, bool if truncated, info if done
        """
        done = False

        # unnormalize the actions
        self.throttle = self._un_normalize_2(
            value=action[0],
            min_val=self.DK_Soll_min,
            max_val=self.DK_Soll_max,
            min_norm=-1,
            max_norm=1,
        )
        self.brake = self._un_normalize_2(
            value=action[1],
            min_val=self.Bremse_S_min,
            max_val=self.Bremse_S_max,
            min_norm=-1,
            max_norm=1,
        )

        # Take simple model step to action to calculate new car speed
        self._Car(self.throttle, self.brake)

        # Get current speed and normalize
        v_Car_norm = self._normalize_2(
            value=self.v_Car_kmh,
            min_val=self.v_Car_min,
            max_val=self.v_Car_max,
            min_norm=-1,
            max_norm=1,
        )

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
        self.vehicle_distance = (
            self.vehicle_distance + self.time_step_size_s * vehicle_velocity_ms
        )

    def _update_speed(self, throttle, brake) -> None:
        # Air & Rolling Resistance in Newton
        air_resistance = self.physics.calculate_air_resistance_n(
            velocity_ms=self.v_Car_ms
        )
        rolling_resistance = self.physics.calculate_rolling_resistance_n(
            velocity_ms=self.v_Car_ms
        )

        # Propulsion
        engine_speed = self.physics.calculate_engine_speed(velocity_ms=self.v_Car_ms)
        engine_torque = self.physics.get_engine_torque_from_speed_in_Nm(
            engine_speed=engine_speed, throttle=throttle
        )

        engine_force = self.physics.get_engine_force(engine_torque_nm=engine_torque)

        out_acceleration = self.physics(
            brake=brake,
            air_resistance=air_resistance,
            rolling_resistance=rolling_resistance,
            engine_force=engine_force,
        )
        if (self.v_Car_ms <= 0.1) and (out_acceleration < 0):
            out_acceleration = 0.0

        self.v_Car_ms = self.physics.calculate_velocity(
            old_velocity_ms=self.v_Car_ms,
            acceleration=out_acceleration,
            time_step=self.time_step_size_s,
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

        return self.state
