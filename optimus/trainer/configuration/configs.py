import dataclasses
import json
import os
from time import strftime

from optimus.trainer.configuration.dataset import DatasetConfig
from optimus.trainer.configuration.distributed import DistributedConfig
from optimus.trainer.configuration.model import ModelConfig
from optimus.trainer.configuration.system import SystemConfig
from optimus.trainer.configuration.train import TrainConfig


class Config:
    def __init__(self, **kwargs):
        """
        Initialize config object with default configurations, can be updated with provided arguments.

        Args:
            **kwargs: Keyword arguments containing parameter names and their new values.
        """
        config = {}
        if "reload_checkpoint" in kwargs:
            with open(f"{kwargs['reload_checkpoint']}/config.json", "r") as file:
                config = json.load(file)

        self.train = TrainConfig(**config.get("train", {}))
        self.distributed = DistributedConfig(**config.get("distributed", {}))
        self.model = ModelConfig(**config.get("model", {}))
        self.data = DatasetConfig(**config.get("data", {}))
        self.system = SystemConfig(**config.get("system", {}))

        self.update_config(**kwargs)

        assert self.system.num_nodes > 0, "Number of nodes must be greater than 0."

    def update_config(self, **kwargs):
        """
        Update configuration parameters recursively with arguments passed.
        Args:
            **kwargs: Keyword arguments containing parameter names and their new values.
        """
        for _, config_obj in self.__dict__.items():
            if dataclasses.is_dataclass(config_obj):
                config_dict = dataclasses.asdict(config_obj)
                for k, v in kwargs.items():
                    if k in config_dict:
                        setattr(config_obj, k, v)

    def save(self, folder_path: str):
        """
        Save the configuration object to a file.
        """
        os.makedirs(folder_path, exist_ok=True)
        with open(f"{folder_path}/config.json", "w") as file:
            config = {
                "train": dataclasses.asdict(self.train),
                "distributed": dataclasses.asdict(self.distributed),
                "model": dataclasses.asdict(self.model),
                "data": dataclasses.asdict(self.data),
                "system": dataclasses.asdict(self.system),
            }
            json.dump(config, file, indent=2)

    def log_print(self, *args, main_only=True, force_print=False, **kwargs):
        if (not main_only or self.system.rank == 0) and (self.verbose or force_print):
            print(f'[{strftime("%Y-%m-%d %H:%M:%S")}]', *args, flush=True, **kwargs)

    @property
    def verbose(self):
        return self.system.verbose

    @property
    def use_fsdp(self):
        return self.train.fsdp

    @property
    def is_main_process(self):
        return True if self.system.rank == 0 else False
