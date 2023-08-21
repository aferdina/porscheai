""" add classes for rendering game content in pygame
"""
from typing import Dict
from dataclasses import dataclass
from enum import Enum
from strenum import StrEnum
import numpy as np
import matplotlib.pyplot as plt
import pygame
from pygame.sprite import Sprite
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Colors(Enum):
    """Sprite to define used colors in sprites"""

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 50)
    PURPLE = (80, 50, 145)


class RectPositions(StrEnum):
    """Enum class to define rect positions for sprites"""

    TOPLEFT = "topleft"
    MIDLEFT = "midleft"
    BOTTOMLEFT = "bottomleft"
    CENTER = "center"
    MIDBOTTOM = "midbottom"
    MIDTOP = "midtop"
    TOPRIGHT = "topright"
    MIDRIGHT = "midright"
    BOTTOMRIGHT = "bottomright"


@dataclass
class BaseSpriteConfig:
    """Base class to set Base Sprite configs"""

    x_pos: int
    y_pos: int
    width: int
    height: int
    rect_position: RectPositions


class BaseSprite(Sprite):
    """Base Sprite information to render data"""

    def __init__(self, spriteconfig: BaseSpriteConfig, background_color: Colors):
        super().__init__()
        self.image = pygame.Surface([spriteconfig.width, spriteconfig.height])
        self.image.fill(background_color)
        self.rect = self.image.get_rect()
        setattr(
            self.rect,
            spriteconfig.rect_position,
            [spriteconfig.x_pos, spriteconfig.y_pos],
        )


class LINECOLORS(Enum):
    """Enum class to define colors for line charts"""

    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)


class HistorySprite(BaseSprite):
    """history base sprite class to render historic data"""

    def __init__(
        self,
        state_history: Dict[str, list[float]],
        title: str,
        spriteconfig: BaseSpriteConfig,
    ) -> None:
        """inititalize sprite class

        Args:
            state_history (Dict[str, list[float]]): data to plot in the sprite
            title (str): title of the image
            spriteconfig (BaseSpriteConfig): Basic information like position, shape,...
        """
        super().__init__(
            spriteconfig=spriteconfig, background_color=Colors.YELLOW.value
        )
        self.state_history = state_history
        # Convert pixel values to inches

        # Create a Matplotlib figure and axis
        fig, axis = plt.subplots()
        fig.set_facecolor(color=tuple(value / 255 for value in Colors.YELLOW.value))
        # Plot the data from the dictionary
        for key, data in self.state_history.items():
            axis.plot(
                data,
                label=key,
            )
        font_color_matplotlib = tuple(value / 255 for value in Colors.PURPLE.value)
        # Add legend
        axis.legend(labelcolor=font_color_matplotlib)
        axis.set_title(title, color=font_color_matplotlib)
        # Change the color of the x-axis and y-axis
        axis.tick_params(axis="x", colors=font_color_matplotlib)
        axis.tick_params(axis="y", colors=font_color_matplotlib)
        # Create a Matplotlib canvas and render the figure onto it
        canvas: FigureCanvas = FigureCanvas(fig)
        canvas.draw()
        # Convert the Matplotlib canvas to a Pygame surface
        size = canvas.get_width_height()
        image_surface = pygame.image.fromstring(
            canvas.get_renderer().tostring_rgb(), size, "RGB"
        )
        rescaled_image = pygame.transform.scale(
            image_surface, (spriteconfig.width, spriteconfig.height)
        )
        image_rect = rescaled_image.get_rect(topleft=(0, 0))
        self.image.blit(rescaled_image, image_rect)
        plt.close()


__all__ = [
    Colors.__name__,
    RectPositions.__name__,
    BaseSpriteConfig.__name__,
    BaseSprite.__name__,
    LINECOLORS.__name__,
    HistorySprite.__name__,
]
