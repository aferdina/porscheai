"""
This module contains the configs for the reference trajectory
"""
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from strenum import StrEnum

from porscheai.utils import (FACTOR_KMH_MS, is_float32_array,
                             is_stricly_increasing)

# some plotting constants
TRAJ_X_LABEL = "Time (Sec.)"
TRAJ_Y_LABEL = "Speed (km/h)"
PLOT_TITLE = "Reference Trajectory"


class TrajectoryType(StrEnum):
    """different types for creating trajectory"""

    LINEAR_INTERPOLATION = "linear_interpolation"


# pylint: disable=R0902
@dataclass
class ReferenceTrajectory:
    """stores information about a reference trajectory for driving the vehicle"""

    seconds_markers_s: np.ndarray = np.array(
        [0.0, 2.0, 4.0, 5.0, 7.0, 10.0], dtype=np.float32
    )
    velocities_kmh: np.ndarray = np.array(
        [0.0, 10.0, 20.0, 15.0, 20.0, 15.0], dtype=np.float32
    )
    simulation_frequency_s: float = 0.01  # frequency of simulation time points in ms
    velocities_ms: np.ndarray | None = None
    traj_type: TrajectoryType = TrajectoryType.LINEAR_INTERPOLATION
    velocity_bounds_kmh: Tuple[float, float] | None = None
    velocity_bounds_ms: Tuple[float, float] | None = None
    last_time_step: float | None = None
    first_time_step: float | None = None
    total_duration: float | None = None
    total_timesteps: int | None = None

    def __post_init__(self):
        assert is_float32_array(self.seconds_markers_s) and is_float32_array(
            self.velocities_kmh
        ), "second marker and velocities should be numpy arrays"
        assert (
            self.seconds_markers_s.shape == self.velocities_kmh.shape
        ), "markers and velocity shhould be of same length"
        assert is_stricly_increasing(
            self.seconds_markers_s
        ), "second marker must increasing"
        self.velocities_ms = self.velocities_kmh / FACTOR_KMH_MS
        self.velocity_bounds_kmh = (min(self.velocities_kmh), max(self.velocities_kmh))
        self.velocity_bounds_ms = (
            self.velocity_bounds_kmh[0] / FACTOR_KMH_MS,
            self.velocity_bounds_kmh[1] / FACTOR_KMH_MS,
        )
        self.last_time_step = np.max(self.seconds_markers_s)
        self.first_time_step = np.min(self.seconds_markers_s)
        self.total_duration = self.last_time_step - self.first_time_step
        self.total_timesteps = int(self.total_duration / self.simulation_frequency_s)


def create_reference_trajecotry_ms(
    reference_traj_conf: ReferenceTrajectory,
) -> np.ndarray:
    """create complete velocitiy trajectory from configs in ms

    Args:
        reference_traj_conf (ReferenceTrajectoryConfigs): configs to read data from

    Returns:
        np.ndarray: velocity trajectory in ms
    """
    if reference_traj_conf.traj_type == TrajectoryType.LINEAR_INTERPOLATION:
        return np.interp(
            np.linspace(
                start=reference_traj_conf.first_time_step,
                stop=reference_traj_conf.last_time_step,
                num=reference_traj_conf.total_timesteps,
                endpoint=True,
            ),
            reference_traj_conf.seconds_markers_s,
            reference_traj_conf.velocities_ms,
        )
    raise ValueError("Trajectory type is unknown")


def plot_reference_trajectory(
    traj_configs: ReferenceTrajectory, save_path: str | None = None
) -> None:
    """plot/ save trajectory created from configuration

    Args:
        traj_configs (ReferenceTrajectory): configurations for trajectory
        save_path (str| None): path to save trajectory, if None trajectory is not saved.
        Defaults to None
    """
    time_array = create_reference_trajecotry_ms(
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
    plot_reference_trajectory.__name__,
    ReferenceTrajectory.__name__,
    create_reference_trajecotry_ms.__name__,
]
