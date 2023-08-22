""" rendering wrapper method for Logistic game
"""
from typing import Any, List, OrderedDict, Tuple

import gymnasium as gym
import pygame

from porscheai.environment.base_env import SimpleDriver
from porscheai.environment.configs import create_reference_trajecotry_ms
from porscheai.environment.rendering.render_classes import (BaseSpriteConfig,
                                                            Colors,
                                                            HistorySprite,
                                                            RectPositions)

# Constants for rendering
METADATA = {"render_modes": ["human"], "render_fps": 60}
WINDOW_W = 1800
WINDOW_H = 1200


class RenderWrapper(gym.Wrapper):
    """add rendering method to logistic game"""

    def __init__(self, env: SimpleDriver):
        self.env: SimpleDriver
        gym.Wrapper.__init__(self, env=env)

        # init game statistics
        self.brake_history: List[int] = []
        self.throttle_history: List[int] = []
        self.velocity_ms_history: List[float] = []
        self.reward_history: List[float] = []
        self.target_trajectory_velocity_ms: List[
            float
        ] = create_reference_trajecotry_ms(self.env.traj_configs)
        self._init_statistics()

        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.isopen: bool = True

    def _init_statistics(self) -> None:
        self.brake_history = []
        self.reward_history = []
        self.throttle_history = []
        self.velocity_ms_history = []
        self.action_history = []

    def render(self, mode: str = "human", close: bool = False) -> None:
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        Args:
            mode (str): the mode to render with
        """
        assert mode in METADATA["render_modes"]

        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        # creating manufacturing group
        velocity_history_group = pygame.sprite.Group()
        velocity_data = {
            "target_velocity": self.target_trajectory_velocity_ms[
                : self.env.game_physics_params.current_time_step
            ],
            "agent_velocity": self.velocity_ms_history,
        }
        velocity_sprite = HistorySprite(
            state_history=velocity_data,
            spriteconfig=BaseSpriteConfig(
                x_pos=0,
                y_pos=0,
                width=400,
                height=400,
                rect_position=RectPositions.TOPLEFT,
            ),
            title="Velocity in m/s",
        )
        velocity_history_group.add(velocity_sprite)
        brake_throttle_group = pygame.sprite.Group()
        brake_throttle_sprite = HistorySprite(
            state_history={
                "throttle": self.throttle_history,
                "brake": self.brake_history,
            },
            spriteconfig=BaseSpriteConfig(
                x_pos=WINDOW_W,
                y_pos=0,
                width=400,
                height=400,
                rect_position=RectPositions.TOPRIGHT,
            ),
            title="Throttle and Brake",
        )
        brake_throttle_group.add(brake_throttle_sprite)

        # plot images
        self.screen.fill(Colors.YELLOW.value)
        velocity_history_group.draw(self.screen)
        brake_throttle_group.draw(self.screen)

        # rendering the current game period
        font = pygame.font.Font(None, 24)
        count_text = font.render(
            f"Count: {self.env.game_physics_params.current_time_step}",
            True,
            Colors.PURPLE.value,
        )
        text_rect = count_text.get_rect()
        text_rect.bottomright = [WINDOW_W, WINDOW_H]
        self.screen.blit(count_text, text_rect)

    def close(self) -> None:
        """close the game window"""
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

    def step(
        self, action: float
    ) -> Tuple[OrderedDict[str, Any], float, bool, bool, dict]:
        """interaction with the game environment by playing action `step_action`
        and overload method by updating statistics

        Args:
            action (float): action to play

        Returns:
            Tuple[OrderedDict[str, Any], float, bool, bool, dict]: game state,
                reward, termination bool, truncated bool, info dictionary
        """
        self.action_history.append(action)
        _brake = self.env.action_space_configs.get_brake_from_action(action)
        _throttle = self.env.action_space_configs.get_throttle_from_action(action)
        self.brake_history.append(float(_brake))
        self.throttle_history.append(float(_throttle))
        # get throttle and brake value
        # get velocitity value
        # get acceleration value
        observation, reward, done, truncated, info = self.env.step(action)
        self.velocity_ms_history.append(self.env.game_physics_params.velocity_ms)
        self.reward_history.append(reward)
        return observation, reward, done, truncated, info
