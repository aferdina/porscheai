from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self
    , env_dict: dict = None, learn_dict: dict = None, other_dict: dict = None
    ):
        self.env_dict = env_dict
        self.learn_dict = learn_dict
        self.other_dict = other_dict
        self.metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }

    def _on_training_start(self) -> None:



        self.logger.record(
            "hparams",
            HParam(self.metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True