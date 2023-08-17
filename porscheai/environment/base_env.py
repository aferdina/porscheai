""" base environment for driver gym environment
"""
from dataclasses import dataclass
from typing import List, Tuple
from strenum import StrEnum
import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete
import numpy as np
import math
from .helpclasses import GeneralGameconfigs, ReferenceTrajectoryConfigs, PhysicConfigs

genral_game_configs = GeneralGameconfigs()
trajectory_configs = ReferenceTrajectoryConfigs()
physic_configs = PhysicConfigs()


class SimpleDriver(gym.Env):
    """gym environment to simulate a simple car driver"""

    def __init__(
        self,
        game_configs: GeneralGameconfigs = genral_game_configs,
        trajectory_configs: ReferenceTrajectoryConfigs = trajectory_configs,
        physic_configs: PhysicConfigs = physic_configs,
    ):
        """

        Args:
            game_configs (GeneralGameconfigs, optional): _description_. Defaults to genral_game_configs.
            trajectory_configs (ReferenceTrajectory, optional): _description_. Defaults to trajectory_configs.
            physic_configs (PhysicConfigs, optional): _description_. Defaults to physic_configs.
        """

        # Initializations
        self.ref_secs = ref_secs
        self.ref_speeds = ref_speeds
        self.len_outlook = len_outlook

        # Normalization Bounds

        # Initalization for Integration
        self.sim_stepsize = 0.01  # s
        self.vehicle_distance = 0  # m
        if v_Car_Init is None:
            self.v_Car_ms = self.ref_speeds[0] / 3.6
        else:
            self.v_Car_ms = v_Car_Init / 3.6

        # Episode Length
        self.dtype = np.float32
        self.sim_duration = ref_secs[-1]
        self.episode_length = int(self.sim_duration / self.sim_stepsize)

        # load target velocities
        self.v_Soll_all = self._load_ref_trajectory()

        # Gym setup
        num_obs = self.len_outlook + 1
        self.observation_space = Box(
            low=np.repeat(self.min_norm_obs, num_obs),
            high=np.repeat(self.max_norm_obs, num_obs),
            shape=(num_obs,),
            dtype=self.dtype,
        )

        # inputs: [DK_SOLL, Bremse], _normalized to [-1,1]
        self.action_space = Box(
            low=np.repeat(self.min_norm_act, num_act),
            high=np.repeat(self.max_norm_act, num_act),
            shape=(num_act,),
            dtype=self.dtype,
        )

        # Add the first State
        v_Car_norm = self._normalize_2(
            value=self.v_Car_ms * 3.6,
            min_val=self.v_Car_min,
            max_val=self.v_Car_max,
            min_norm=-1,
            max_norm=1,
        )
        v_target = self.v_Soll_all[: self.len_outlook]
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

        self.current_iteration = 0
        self.info = {}

    def _reshape_to_length_n(self, array: np.array, length: int) -> np.array:
        """
        Stretches a 1d numpy array to length n.
        """
        len_array = len(array)
        array = np.append(array, np.repeat(array[-1], length - len_array))
        return array

    def _load_ref_trajectory(self) -> np.array:
        """
        Loads a reference trajectory. See the notebook make_ref_trajectory for an example.
        """
        ref_trajectory = np.array([])

        for idx in range(len(self.ref_secs) - 1):
            sect_dur = self.ref_secs[idx + 1] - self.ref_secs[idx]
            curr_speed = self.ref_speeds[idx]
            next_speed = self.ref_speeds[idx + 1]
            num_steps = int(self.episode_length * sect_dur / self.sim_duration)

            section = np.linspace(curr_speed, next_speed, num_steps)
            ref_trajectory = np.append(ref_trajectory, section)

        return ref_trajectory

    def step(self, action):
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
            [self.episode_length - self.current_iteration, self.len_outlook]
        )  # get future reference trajectory, shorted if we are at end of episode
        v_target = self.v_Soll_all[
            self.current_iteration : (self.current_iteration + len_outlook_iteration)
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
        reward = reward * REWARDSCALE

        self.current_iteration += 1

        # Check if Round is done
        if self.current_iteration == self.episode_length:
            done = True

        return obs, reward, done, self.info

    def _Car(self, throttle, brake) -> None:
        # parameters for the car
        vehicle_weight = 2500  # kg
        engine_power = 400000  # W
        engine_n_max_power = 6000  # upm
        engine_n_max = 22000  # upm
        gearbox_ratio = 8  # -
        tire_radius = 0.3  # m
        rho_air = 1.2  # kg/m³
        vehicle_cW = 0.25  # -
        vehicle_A = 2.2  # m²
        tire_fR = 0.01  # -
        g = 9.81  # m/s²

        # Air & Rolling Resistance in Newton
        F_AirR = (
            1
            / 2
            * rho_air
            * vehicle_cW
            * vehicle_A
            * self.v_Car_ms**2
            * np.sign(self.v_Car_ms)
        )
        F_RR = vehicle_weight * tire_fR * g * np.sign(self.v_Car_ms)

        # Propulsion
        engine_speed = self.v_Car_ms / tire_radius / 2 / math.pi * 60 * gearbox_ratio
        if engine_speed < 0:
            engine_torque_ = 0
        elif (engine_speed >= 0) and (engine_speed < engine_n_max_power):
            engine_torque_ = (
                throttle / 100 * engine_power / engine_n_max_power * 60 / 2 / math.pi
            )
        elif (engine_speed > engine_n_max_power) and (engine_speed < engine_n_max):
            engine_torque_ = (
                throttle / 100 * engine_power / engine_speed * 60 / 2 / math.pi
            )
        elif engine_speed > engine_n_max:
            engine_torque_ = 0

        F_Prop = engine_torque_ * gearbox_ratio / tire_radius

        # Newton
        out_acceleration = ((F_Prop - F_RR - F_AirR) / vehicle_weight) - min(
            self.Bremse_S_max, brake
        )
        if (self.v_Car_ms <= 0.1) and (out_acceleration < 0):
            out_acceleration = 0

        # Integrate
        self.v_Car_ms = self.v_Car_ms + self.sim_stepsize * out_acceleration
        self.v_Car_kmh = self.v_Car_ms * 3.6
        self.vehicle_distance = (
            self.vehicle_distance + self.sim_stepsize * self.v_Car_ms
        )

    def _normalize_2(
        self,
        value: float,
        min_val: float,
        max_val: float,
        min_norm: float,
        max_norm: float,
    ) -> float:
        """
        normalizes a value to the [min_norm, max_norm] range
        Inverse of _normalize function

        Args:
        - value: the value to un_normalize
        - min_val: the minimum value in the allowed range of the value
        - max_val: the maximum value in the allowed range of the value
        - min_norm: lower boundary of the normalization range
        - max_norm: upper boundary of the normalization range
        """
        val_norm = (value - min_val) * (max_norm - min_norm) / (
            max_val - min_val
        ) + min_norm

        return val_norm

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
        self.current_iteration = 0

        return self.state
