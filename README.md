# Reinforcement Learning for self-driving cars

Welcome to the README guide for training an autonomous driving agent using reinforcement learning! This guide will walk you through the necessary steps to train an AI agent capable of driving a car in a simulated environment using reinforcement learning techniques.

## Introduction

Autonomous driving is a challenging task that involves training an agent to navigate complex road environments while adhering to traffic rules and ensuring passenger safety. Reinforcement learning is a powerful approach to address this problem by enabling the agent to learn from its interactions with the environment and make decisions that maximize a cumulative reward signal.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.10.4
- Reinforcement learning libraries (e.g., gymnasium, PyTorch)

## Setup

1. Install pyenv and as described [here](https://github.com/aferdina/IntroductionRL/blob/main/01_Firststeps.md) and [here](https://github.com/aferdina/IntroductionRL/blob/main/03_Codingguidelines.md)
2. Install python 3.10.4 by using

    ```sh
    pyenv install 3.10.0
    ```

3. Install Dependencies: Set up a virtual environment and install the required packages using

    ```sh
    poetry install
    ```

## Training Process

1. **State Space Definition:** The repository provides an interface to implement different observation spaces. The abstract class can be found [here](porscheai/environment/configs/abstract_classes.py).

   ```python
   class ObservationSpaceConfigs(ABC):
    """abstract class for observation space configs"""

    @abstractmethod
    def create_observation_space(self) -> gym.Space:

    @abstractmethod
    def get_observation(
        self, driver_physics_params: DriverPhysicsParameter
    ) -> np.ndarray:

    @abstractmethod
    def get_reward(self, observation: np.ndarray) -> float:
   ```

    An example of an observation space is the class `OutlookObservationSpace`, which contains the current velocity deviation and an outlook of `outlook_length` steps in the future.

2. **Action Space Definition:** Specify the possible actions the agent can take, such as accelerating, braking, and steering. Similar to the observation space an interface to implement different action spaces is provided. The abstract class can be found [here](porscheai/environment/configs/abstract_classes.py).

    ```python
   class ActionSpaceConfigs(ABC):
    """abstract class for action space configs"""

    @abstractmethod
    def create_action_space(self) -> gym.Space:

    @abstractmethod
    def get_brake_from_action(self, action: Any) -> float:

    @abstractmethod
    def get_throttle_from_action(self, action: Any) -> float:
   ```

    An example of an action space is the class `OneForBrakeAndGearActionSpace`, which used one action for brake and throttle value to avoid the simultanious use of brake and throttle.

3. **Reward Function:** The reward function can be modified in the observation space.

4. **Reinforcement Learning Algorithm:** To use a reinforcment learning algorithm create a yaml file in the [folder](porscheai/training/hyperparams). Examples for `SAC` and `PPO` are provided.

5. Training Loop: To start a training process run

   ```sh
   python porscheai/training/train.py
   ```

    with the desired configurations.

## Evaluation

1. **Testing:** During the training a tensorboard callback is provided. Use

    ```sh
    tensorboard --logdir monitor_tmp
    ```

    to watch the trainings process.
2. **Metrics:** Further metrices can be used in the tensorboard by adding it to the callback [here](porscheai/training/callbacks.py).

## Fine-Tuning and Experimentation

Hyperparameter Tuning (not fully tested yet): Hyperparameter Optimizing can be used by setting the flag `--optimize`

```sh
python porscheai/training/train.py --optimize
```

Model Architecture: Explore different neural network architectures for the agent's policy and value functions by adapting the hyperparameters file.

Reward Shaping: Refine the reward function to guide the agent towards desired behaviors more effectively.

## Run trained model

Run

```sh
python porscheai/run_trained_model.py
```

to run a trained model in pygame. The path `TRAINEDMODEL` to the stored agent should be edited.

## Conclusion

## References

