""" base environment for driver gym environment
"""
import math
from dataclasses import dataclass, field
from typing import Tuple

from porscheai.utils import FACTOR_KMH_MS, GRAVITY_MS, RHO_AIR_KGM3


# TODO: clean up car configs
@dataclass
class CarConfigs:
    """config class to store car configurations"""

    start_velocity_kmh: float | None = None  # starting velocity in km/h
    start_velocity_ms: float | None = None  # starting velocity in m/s
    vehicleweight_kg: int = 2500  # weight of vehicle in kg
    engine_power_w: int = 400000  # power of engine in W
    engine_n_max_power: int = 6000
    engine_n_max: int = 22000
    gearbox_ratio: int = 8
    tire_radius: float = 0.3
    vehicle_w: float = 0.25
    vehicle_a: float = 2.2
    tire_fr: float = 0.01

    def __post_init__(self):
        if self.start_velocity_kmh is not None:
            self.start_velocity_ms = self.start_velocity_kmh * FACTOR_KMH_MS


@dataclass
class PhysicConfigs:
    """configuration class for physics"""

    throttle_bounds: Tuple[float, float] = (0.0, 100.0)  # bounds for throttle values
    brake_bounds: Tuple[float, float] = (0.0, 10.0)  # bounds for brake
    car_configs: CarConfigs = field(default_factory=lambda: CarConfigs)
    _air_resistance_factor: float | None = None
    _engine_speed_factor: float | None = None

    def __post_init__(self):
        self._air_resistance_factor = (
            1
            / 2
            * RHO_AIR_KGM3
            * self.car_configs.vehicle_w
            * self.car_configs.vehicle_a
        )
        self._engine_speed_factor = (
            self.car_configs.tire_radius
            / 2
            / math.pi
            * 60
            * self.car_configs.gearbox_ratio
        )

    def calculate_air_resistance_n(self, velocity_ms: float) -> float:
        """calculate air resistance in newton based on velocity in meter per second

        Args:
            velocity_ms (float): velocity

        Raises:
            ValueError: _description_

        Returns:
            float: _description_
        """
        return (
            self._air_resistance_factor
            * velocity_ms**2
            * math.copysign(1, velocity_ms)
        )

    def calculate_rolling_resistance_n(self, velocity_ms: float) -> float:
        """calculate rolling resistance in newton based on velocity in meter per second

        Args:
            velocity_ms (float): velocity in meter per second
        Returns:
            float: rolling resistance in newton
        """
        return (
            self.car_configs.vehicleweight_kg
            * self.car_configs.tire_fr
            * GRAVITY_MS
            * math.copysign(1, velocity_ms)
        )

    def calculate_engine_speed(self, velocity_ms: float) -> float:
        """calculate engine speed in newton based on velocity in meter per second

        Args:
            velocity_ms (float): velocity in meter per second
        Returns:
            float: engine speed in newton
        """
        return velocity_ms / self._engine_speed_factor

    def calculate_velocity(
        self, old_velocity_ms: float, acceleration: float, time_step: float
    ) -> float:
        """calculate new velocity based on old velocity and acceleration as well as time step

        Args:
            old_velocity_ms (float): old velocity in meter per second
            acceleration (float): acceleration in meter per second^2
            time_step (float): time step to use acceleration on old velocity
        Returns:
            float: new velocity in meter per second
        """
        return old_velocity_ms + time_step * acceleration

    def get_engine_torque_from_speed_in_nm(
        self, engine_speed: float, throttle: float
    ) -> float:
        """calculate torque in Newton meter from engine speed

        Args:
            speed (float): Engine speed in W
        Returns:
            float: engine torque in Newton meter
        """
        if engine_speed < 0:
            engine_torque = 0
        elif 0 <= engine_speed < self.car_configs.engine_n_max_power:
            engine_torque = (
                throttle
                / 100
                * self.car_configs.engine_power_w
                / self.car_configs.engine_n_max_power
                * 60
                / 2
                / math.pi
            )
        elif (
            self.car_configs.engine_n_max
            > engine_speed
            > self.car_configs.engine_n_max_power
        ):
            engine_torque = (
                throttle
                / 100
                * self.car_configs.engine_power_w
                / engine_speed
                * 60
                / 2
                / math.pi
            )
        elif engine_speed > self.car_configs.engine_n_max:
            engine_torque = 0
        return engine_torque

    def get_car_acceleration(
        self,
        brake: float,
        air_resistance: float,
        rolling_resistance: float,
        engine_force: float,
    ) -> float:
        """calculate car acceleration based on given parameters

        Args:
            brake (float): value of brake
            air_resistance (float): air resistance force
            rolling_resistance (float): rolling resistance force
            engine_force (float): force of engine

        Returns:
            float: acceleration in meter pers second per second
        """
        return (
            (engine_force - rolling_resistance - air_resistance)
            / self.car_configs.vehicleweight_kg
        ) - min(self.brake_bounds[1], brake)

    def get_engine_force(self, engine_torque_nm: float) -> float:
        """get engine force from engine torque

        Args:
            engine_torque_nm (float): engine torque in Newton meter

        Returns:
            float: engine force in Newton
        """
        return (
            engine_torque_nm
            * self.car_configs.gearbox_ratio
            / self.car_configs.tire_radius
        )

    def get_velocity(
        self, throttle: float, brake: float, velocity_ms: float, time_step_s: float
    ) -> float:
        """calculate speed given throttle, brake and current velocity

        Args:
            throttle (float): value for throttle
            brake (float): value for brake
            velocity_ms (float): velocity in meter per seconds
            time_step_s (float): time step in seconds

        Returns:
            float: return velocity in meter per seconds
        """
        # Air & Rolling Resistance in Newton
        air_resistance = self.calculate_air_resistance_n(velocity_ms=velocity_ms)
        rolling_resistance = self.calculate_rolling_resistance_n(
            velocity_ms=velocity_ms
        )

        # Propulsion
        engine_speed = self.calculate_engine_speed(velocity_ms=velocity_ms)
        engine_torque = self.get_engine_torque_from_speed_in_nm(
            engine_speed=engine_speed, throttle=throttle
        )

        engine_force = self.get_engine_force(engine_torque_nm=engine_torque)

        out_acceleration = self.get_car_acceleration(
            brake=brake,
            air_resistance=air_resistance,
            rolling_resistance=rolling_resistance,
            engine_force=engine_force,
        )
        if (velocity_ms <= 0.1) and (out_acceleration < 0):
            out_acceleration = 0.0

        velocity_ms = self.calculate_velocity(
            old_velocity_ms=velocity_ms,
            acceleration=out_acceleration,
            time_step=time_step_s,
        )
        return velocity_ms


__all__ = [CarConfigs.__name__, PhysicConfigs.__name__]
