""" render trained models with pygame
"""
import os
from typing import OrderedDict

import gymnasium as gym
import pygame
import yaml
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from strenum import StrEnum

from porscheai.environment.base_env import SimpleDriver
from porscheai.environment.wrapper_classes.render_wrapper import WINDOW_H, WINDOW_W
from porscheai.training.utils import ALGOS, get_wrapper_class

RENDERWRAPPER = "porscheai.environment.wrapper_classes.render_wrapper.RenderWrapper"


class FileNames(StrEnum):
    """Enum class to look up file names"""

    CONFIGFILE = "Simple-Driver/config.yml"
    MODELFILE = "best_model.zip"
    GAMEPATH = "Simple-Driver"


class RenderTrainedAgent:
    """Class to render a trained model on a game environment"""

    def __init__(
        self,
        trained_model_path: str,
        wrapper_classes: list[gym.Wrapper],
        algo_type: str = "sac",
    ):
        """inintialize class

        Args:
            trained_model_path (str): path to all required output files for algorithm
            wrapper_classes (list[gym.Wrapper]): list of wrapper classes used for plotting algorithm
            algo_type (str, optional): Used algorith type. Defaults to "sac".
        """
        self.config_path = os.path.join(trained_model_path, FileNames.CONFIGFILE)
        self.configs: OrderedDict[str, str] = self.load_yml_file(
            path_to_yml=self.config_path
        )
        self.game_env = self._load_env(
            game_path=os.path.join(trained_model_path, FileNames.GAMEPATH),
            used_wrapper_classes=wrapper_classes,
        )
        self.model = self._load_agent(
            model_path=os.path.join(trained_model_path, FileNames.MODELFILE),
            algo_type=algo_type,
        )

    @staticmethod
    def load_yml_file(path_to_yml: str) -> OrderedDict[str, str]:
        """load configs from autohpo

        Args:
            path_to_yml (str): path to the stored yml config file
        Returns:
            OrderedDict[str, str]: configuration stored in yml file
        """

        # Custom constructor function
        def construct_ordered_dict(loader: yaml.Loader, node: yaml.Node):
            # data = loader.construct_yaml_seq(node.value[0])
            loaded_data = []
            for item in node.value[0].value:
                key = item.value[0].value
                value = item.value[1].value
                loaded_data.append((key, value))
            ordered_dict = OrderedDict(loaded_data)
            return ordered_dict

        # Add the constructor to SafeLoader
        yaml.add_constructor(
            "tag:yaml.org,2002:python/object/apply:collections.OrderedDict",
            construct_ordered_dict,
            yaml.SafeLoader,
        )

        with open(path_to_yml, encoding="utf-8") as file:
            configs = yaml.safe_load(file)
        return configs

    def _load_env(
        self, game_path: str, used_wrapper_classes: list[gym.Wrapper]
    ) -> VecEnv:
        """load game environment from path to game statistics file

        Args:
            game_path (str): path to the game statistics

        Returns:
            VecEnv: Vectorized environment
        """

        used_wrapper = get_wrapper_class({"env_wrapper": used_wrapper_classes})
        loaded_env = make_vec_env(
            make_env,
            n_envs=1,
            seed=None,
            wrapper_class=used_wrapper,
        )
        _path = os.path.join(game_path, "vecnormalize.pkl")
        if os.path.exists(_path):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(_path, venv=loaded_env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False
            return env

        return loaded_env

    def _maybe_normalize(self, env: VecEnv, norm_path: str | None) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.

        :param env:
        :param eval_env:
        :return:
        """
        # Pretrained model, load normalization

        if os.path.exists(norm_path):
            print("Loading saved VecNormalize stats")
            env = VecNormalize.load(norm_path, venv=env)
            # Deactivate training and reward normalization
            env.training = False
            env.norm_reward = False

        return env

    @staticmethod
    def _load_agent(model_path: str, algo_type: str) -> BaseAlgorithm:
        """load an reinforcement learning model from a zip file

        Args:
            model_path (str): path to model information
            algo_type (str): type of reinforcement learning algorithm

        Returns:
            BaseAlgorithm: loaded game model
        """
        loaded_model = ALGOS[algo_type].load(model_path)
        return loaded_model

    def run_game_trained(self) -> None:
        """render a trained model on the given environment"""
        # Initialize Pygame
        pygame.init()
        self.game_env.render_mode = "human"
        # Set up the Pygame window
        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        clock = pygame.time.Clock()

        # Reset the environment
        observation = self.game_env.reset()
        done = False
        # Main game loop
        running = True
        action = None
        truncuated = None
        event: pygame.event.Event
        while running:
            action = None
            done = None
            truncuated = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        repeat = 1
                    elif event.key == pygame.K_RIGHT:
                        repeat = 10
                    elif event.key == pygame.K_TAB:
                        repeat = 100
                    else:
                        repeat = 0
                    for _ in range(repeat):
                        action, _ = self.model.predict(observation=observation)
                        if action is not None:
                            # Attention: Vec Normalized environment returns only four elements,
                            # truncated information is in _infos variable
                            observation, _reward, done, _infos = self.game_env.step(
                                action
                            )

            # Render the environment
            screen.fill((255, 255, 255))  # Clear the screen
            self.game_env.render(mode="human")  # Render the Gym environment

            # Update the display
            pygame.display.flip()
            clock.tick(60)
            # Check if the episode is done
            if done:
                print("Episode finished.")
                break
            if truncuated:
                print("Episode truncated.")
                break
        # Quit Pygame
        pygame.quit()


def make_env() -> SimpleDriver:
    """create instance of Verbund class for vecnormalize environment

    Returns:
        Verbund:
    """
    game_env = SimpleDriver()
    game_env.render_mode = "human"
    return game_env


if __name__ == "__main__":
    TRAINEDMODEL = "logs/trpo/Simple-Driver_1"
    Render_Game = RenderTrainedAgent(
        trained_model_path=TRAINEDMODEL,
        wrapper_classes=[
            RENDERWRAPPER,
        ],
        algo_type="trpo",
    )
    Render_Game.run_game_trained()
