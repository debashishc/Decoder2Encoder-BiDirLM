import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.state_dict as dcp_sd
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
)

from optimus.trainer.configuration.configs import Config
from optimus.trainer.model.tools import ModelTools


class Distributed:
    """Distributed training utilities for PyTorch."""

    def __init__(self, config: Config):
        """Initialize the process group for distributed training, set GPU devices and update config file.
        Args:
           verbose (bool, optional): Verbose mode. Defaults to False.
        """
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        self.config = config
        self.distributed_config = config.distributed

        dist.init_process_group(
            backend="nccl",
        )
        self.main_process = True if self.rank == 0 else False

        dist.barrier()
        if config.verbose and self.main_process:
            config.log_print("All ranks initialized global process group.")

        torch.cuda.set_device(self.local_rank)
        ModelTools.clear_gpu_cache()

        # Update config with distributed information.
        config.update_config(**self.get_information())
        config.log_print("Distributed interface initialized successfully.")

    def fsdp_setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Set up model for FSDP sharding.
        Args:
            model (torch.nn.Module): Model to shard.
        Returns:
            torch.nn.Module: Sharded model.
        """
        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=self.distributed_config.wrap_policy,
            sharding_strategy=self.distributed_config.sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
            mixed_precision=self.distributed_config.mixed_precision,
            sync_module_states=True,
            use_orig_params=True,
        )
        dist.barrier()

        self.config.log_print(
            f"All rank set up model for FSDP sharding successfully in {self.distributed_config._mixed_precision}."
        )
        return model

    def save_fsdp_model_optimizer(
        self,
        model: FullyShardedDataParallel,
        optimizer: torch.optim.Optimizer,
        step_dir: str,
    ):
        model_sd, optimizer_sd = dcp_sd.get_state_dict(model, optimizer)
        sd = {
            "model": model_sd,
            "optimizer": optimizer_sd,
        }

        if self.config.train.lora_finetuning:
            sd["model"] = {
            k: v for k, v in model_sd.items() 
            if "lora" in k or "bias" in k or "lm_head" in k or "embed_tokens" in k
        }
            if dist.get_rank() == 0:
                base_model = model.module if hasattr(model, "module") else model
                if hasattr(base_model, "peft_config"):
                    adapter_config = base_model.peft_config.get("default")
                    if adapter_config:
                        adapter_config.save_pretrained(step_dir)

        dcp.save(sd, checkpoint_id=step_dir)

    def load_fsdp_model_optimizer(
        self,
        model: FullyShardedDataParallel,
        optimizer: torch.optim.Optimizer,
        step_dir: str,
    ):
        model_sd, optimizer_sd = dcp_sd.get_state_dict(model, optimizer)
        states = {
            "model": model_sd,
            "optimizer": optimizer_sd,
        }
        dcp.load(states, checkpoint_id=step_dir)
        dcp_sd.set_state_dict(
            model, optimizer, model_state_dict=model_sd, optim_state_dict=optimizer_sd
        )

    def get_information(self) -> dict[str, int]:
        """Get the rank and local rank of the current process and the world size."""
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": dist.get_world_size(),
            "gpu_per_node": torch.cuda.device_count(),
            "num_nodes": dist.get_world_size() // torch.cuda.device_count(),
        }

    def cleanup(self):
        """Destroy process group, and deinitialize the distributed package."""
        dist.destroy_process_group()
        self.config.log_print("Distributed training cleaned up successfully.")
