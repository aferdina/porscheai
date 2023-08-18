""" base environment for driver gym environment
"""
from dataclasses import dataclass


@dataclass
class GeneralGameconfigs:
    """gameconfigs including bounds for reward, observatio and action space as well as
    bounds for throttle and brake
    """

    rewardscale: float = 1.0  # factor scaling the rewards
