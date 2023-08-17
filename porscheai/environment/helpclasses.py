""" base environment for driver gym environment
"""
from dataclasses import dataclass
from typing import Tuple
from strenum import StrEnum
import numpy as np
import matplotlib.pyplot as plt
from porscheai.utils import is_float32_array, is_stricly_increasing

# some physics constants
GRAVITY = 9.81
RHO_AIR = 1.2  # density of air in kg/m^3 at around 20 degrees


@dataclass
class PhysicConfigs:
    """configuration class for physics"""

    vehicleweight: int = 2500  # weight of vehicle in kg
    engine_power: int = 400000  # power of engine in W
    engine_n_max_power: int = 6000
    engine_n_max: int = 22000
    gearbox_ratio: int = 8
    tire_radius: float = 0.3
    vehicle_w: float = 0.25
    vehcile_a: float = 2.2
    tire_fr: float = 0.01
    throttle_bounds: Tuple[float, float] = (0.0, 100.0)  # bounds for throttle values
    brake_bounds: Tuple[float, float] = (0.0, 10.0)  # bounds for brake
    start_velocity: float = 0.0


class TrajectoryType(StrEnum):
    """different types for creating trajectory"""

    LINEAR_INTERPOLATION = "linear_interpolation"


@dataclass
class ReferenceTrajectory:
    """stores information about a reference trajectory for driving the vehicle"""

    seconds_markers: np.ndarray = np.array(
        [0.0, 2.0, 4.0, 5.0, 7.0, 10.0], dtype=np.float32
    )
    velocities: np.ndarray = np.array(
        [0.0, 10.0, 20.0, 15.0, 20.0, 15.0], dtype=np.float32
    )
    simulation_frequency: float = 0.01  # frequency of simulation time points
    traj_type: TrajectoryType = TrajectoryType.LINEAR_INTERPOLATION
    velocity_bounds: Tuple[float, float] | None = None
    last_time_step: float | None = None
    first_time_step: float | None = None
    total_duration: float | None = None
    total_timesteps: int | None = None

    def __post_init__(self):
        assert is_float32_array(self.seconds_markers) and is_float32_array(
            self.velocities
        ), "second marker and velocities should be numpy arrays"
        assert (
            self.seconds_markers.shape == self.velocities.shape
        ), "markers and velocity shhould be of same length"
        assert is_stricly_increasing(
            self.seconds_markers
        ), "second marker must increasing"
        self.velocity_bounds = (min(self.velocities), max(self.velocities))
        self.last_time_step = np.max(self.seconds_markers)
        self.first_time_step = np.min(self.seconds_markers)
        self.total_duration = self.last_time_step - self.first_time_step
        self.total_timesteps = int(self.total_duration / self.simulation_frequency)


def create_reference_trajecotry(reference_traj_conf: ReferenceTrajectory) -> np.ndarray:
    """create complete trajectory from configs

    Args:
        reference_traj_conf (ReferenceTrajectoryConfigs): configs to read data from

    Returns:
        np.ndarray: _description_
    """
    if reference_traj_conf.traj_type == TrajectoryType.LINEAR_INTERPOLATION:
        return np.interp(
            np.linspace(
                start=reference_traj_conf.first_time_step,
                stop=reference_traj_conf.last_time_step,
                num=reference_traj_conf.total_timesteps,
                endpoint=True,
            ),
            reference_traj_conf.seconds_markers,
            reference_traj_conf.velocities,
        )
    raise ValueError("Trajectory type is unknown")


@dataclass
class GeneralGameconfigs:
    """gameconfigs including bounds for reward, observatio and action space as well as
    bounds for throttle and brake
    """

    rewardscale: float = 1.0  # factor scaling the rewards
    obs_bounds: Tuple[float, float] = (-1.0, 1.0)  # bounds for observation space
    action_space_bounds: Tuple[float] = (-1.0, 1.0)  # bounds for action space
    outlook_length: int = 1  # number of timesteps to look in the future


TRAJ_X_LABEL = "Time (Sec.)"
TRAJ_Y_LABEL = "Speed (m/s)"
PLOT_TITLE = "Reference Trajectory"


def plot_reference_trajectory(
    traj_configs: ReferenceTrajectory, save_path: str | None = None
) -> None:
    """plot/ save trajectory created from configuration

    Args:
        traj_configs (ReferenceTrajectory): configurations for trajectory
        save_path (str| None): path to save trajectory, if None trajectory is not saved.
        Defaults to None
    """
    time_array = create_reference_trajecotry(
        reference_traj_conf=traj_configs,
    )
    x_ticks = np.linspace(
        traj_configs.first_time_step,
        traj_configs.last_time_step,
        num=traj_configs.total_timesteps,
    )
    plt.plot(x_ticks, time_array)
    plt.xlabel(TRAJ_X_LABEL)
    plt.ylabel(TRAJ_Y_LABEL)
    plt.title(PLOT_TITLE)
    plt.grid()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


__all__ = [
    GeneralGameconfigs.__name__,
    ReferenceTrajectory.__name__,
    PhysicConfigs.__name__,
    create_reference_trajecotry.__name__,
]


if __name__ == "__main__":
    reference_configs = ReferenceTrajectory()
    plot_reference_trajectory(traj_configs=reference_configs)
#     reference_configs = ReferenceTrajectory(
#         seconds_markers=np.array([0, 2, 4, 5, 7, 10], dtype=np.float32),
#         velocities=np.array([5, 2, 15, 15, 10, 8], dtype=np.float32),
#     )
#   plot_reference_trajectory(traj_configs=reference_configs)
