""" base environment for driver gym environment
"""
from typing import Any, Dict, Tuple
import gymnasium as gym
import numpy as np
from porscheai.environment.configs import (
    ReferenceTrajectory,
    PhysicConfigs,
    DriverPhysicsParameter,
    ObservationSpaceConfigs,
    ActionSpaceConfigs,
)
from porscheai.environment.observation_spaces import OutlookObservationSpace
from porscheai.environment.action_spaces import OneForBrakeAndGearActionSpace


df_trajectory_configs = ReferenceTrajectory()
df_physic_configs = PhysicConfigs()
df_observations_sapce_configs = OutlookObservationSpace(
    reference_trajectory=df_trajectory_configs
)
df_action_space_configs = OneForBrakeAndGearActionSpace(
    physics_configs=df_physic_configs
)


class SimpleDriver(gym.Env):
    """gym environment to simulate a simple car driver"""

    def __init__(
        self,
        traj_configs: ReferenceTrajectory = df_trajectory_configs,
        physic_configs: PhysicConfigs = df_physic_configs,
        observation_space_configs: ObservationSpaceConfigs = df_observations_sapce_configs,
        action_space_configs: ActionSpaceConfigs = df_action_space_configs,
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
        self.traj_configs = traj_configs
        # needed information for game
        self.total_no_timesteps: int = traj_configs.total_timesteps
        self.time_step_size_s: float = traj_configs.simulation_frequency_s
        self.start_velocity_ms: float = self.get_start_velocity_ms(
            traj_conifgs=traj_configs
        )
        # observation space dependent information

        # Outsource to reward configs afterwards
        # reward configs

        self.game_physics_params: DriverPhysicsParameter = DriverPhysicsParameter(
            velocity_ms=self.start_velocity_ms, current_time_step=0
        )

        # Gym setup
        # current state space consist of future target velocities and current deviation
        self.observation_space = observation_space_configs.create_observation_space()
        self.observation_space_configs = observation_space_configs
        # adapt to multiple types of action spaces
        self.action_space_configs = action_space_configs
        self.action_space = action_space_configs.create_action_space()

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
            throttle = self.action_space_configs.get_throttle_from_action(action=action)
            brake = self.action_space_configs.get_brake_from_action(action=action)
            new_velocity = self.physics.get_velocity(
                throttle=throttle,
                brake=brake,
                velocity_ms=self.game_physics_params.velocity_ms,
                time_step_s=self.time_step_size_s,
            )
        else:
            new_velocity = self.start_velocity_ms

        # update velocity and time
        self.game_physics_params.velocity_ms = new_velocity
        self.game_physics_params.current_time_step = self.current_time_step + 1
        # game specifics for observation space and actions
        done = False

        observation = self.observation_space_configs.get_observation(
            driver_physics_params=self.game_physics_params
        )

        # Calculate Reward
        reward = self.observation_space_configs.get_reward(observation)

        # Check if Round is done
        if self.game_physics_params.current_time_step == self.total_no_timesteps:
            done = True

        return observation, reward, done, False, {}

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
        self.game_physics_params.velocity_ms = self.start_velocity_ms
        self.game_physics_params.current_time_step = 0
        return self.step(action=None)[0], {}

    def render(self, mode="human"):
        pass
