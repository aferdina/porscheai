""" init configurations for tests
"""
import pytest
import numpy as np
from porscheai.environment.base_env import SimpleDriver
from porscheai.environment.configs import (
    ReferenceTrajectory,
    PhysicConfigs,
    CarConfigs,
)
from porscheai.environment.observation_spaces import OutlookObservationSpace
from porscheai.environment.action_spaces import OneForBrakeAndGearActionSpace


@pytest.fixture(scope="module")
def car_configs() -> CarConfigs:
    return CarConfigs(
        start_velocity_kmh=None,
        vehicleweight_kg=2500,
        engine_power_w=400000,
        engine_n_max_power=6000,
        engine_n_max=22000,
        gearbox_ratio=8,
        tire_radius=0.3,
        vehicle_w=0.25,
        vehicle_a=2.2,
        tire_fr=0.01,
    )


@pytest.fixture(scope="module")
def physics_configs() -> PhysicConfigs:
    return PhysicConfigs(
        throttle_bounds=(0.0, 100.0),
        brake_bounds=(0.0, 10.0),
        car_configs=CarConfigs(
            start_velocity_kmh=None,
            vehicleweight_kg=2500,
            engine_power_w=400000,
            engine_n_max_power=6000,
            engine_n_max=22000,
            gearbox_ratio=8,
            tire_radius=0.3,
            vehicle_w=0.25,
            vehicle_a=2.2,
            tire_fr=0.01,
        ),
    )


@pytest.fixture(scope="module")
def reference_trajectory() -> ReferenceTrajectory:
    return ReferenceTrajectory(
        seconds_markers_s=np.array([0.0, 2.0, 4.0, 5.0, 7.0, 10.0], dtype=np.float32),
        velocities_kmh=np.array([0.0, 10.0, 20.0, 15.0, 20.0, 15.0], dtype=np.float32),
        simulation_frequency_s=0.01,
    )


@pytest.fixture(scope="module")
def observation_space_configs_outlook() -> OutlookObservationSpace:
    return OutlookObservationSpace(
        reference_trajectory=ReferenceTrajectory(
            seconds_markers_s=np.array(
                [0.0, 2.0, 4.0, 5.0, 7.0, 10.0], dtype=np.float32
            ),
            velocities_kmh=np.array(
                [0.0, 10.0, 20.0, 15.0, 20.0, 15.0], dtype=np.float32
            ),
            simulation_frequency_s=0.01,
        ),
        outlook_length=1,
        obs_bounds=(-1.0, 1.0),
        reward_scaling=1.0,
    )


@pytest.fixture(scope="module")
def action_space_one_breakgear() -> OneForBrakeAndGearActionSpace:
    return OneForBrakeAndGearActionSpace(
        action_space_bounds=(-1.0, 1.0),
        physics_configs=PhysicConfigs(
            throttle_bounds=(0.0, 100.0),
            brake_bounds=(0.0, 10.0),
            car_configs=CarConfigs(
                start_velocity_kmh=None,
                vehicleweight_kg=2500,
                engine_power_w=400000,
                engine_n_max_power=6000,
                engine_n_max=22000,
                gearbox_ratio=8,
                tire_radius=0.3,
                vehicle_w=0.25,
                vehicle_a=2.2,
                tire_fr=0.01,
            ),
        ),
    )


@pytest.fixture(scope="module")
def easy_general_game_config(
    reference_trajectory: ReferenceTrajectory,
    physics_configs: PhysicConfigs,
    observation_space_configs_outlook: OutlookObservationSpace,
    action_space_one_breakgear: OneForBrakeAndGearActionSpace,
) -> SimpleDriver:
    return SimpleDriver(
        traj_configs=reference_trajectory,
        physic_configs=physics_configs,
        observation_space_configs=observation_space_configs_outlook,
        action_space_configs=action_space_one_breakgear,
    )
