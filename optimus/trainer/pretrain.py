import asyncio
import contextlib
import os
import time
from contextlib import nullcontext
from typing import Generator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from optimus.trainer.configuration.configs import Config
from optimus.trainer.data import Data
from optimus.trainer.distributed import Distributed
from optimus.trainer.model.load import compile_model
from optimus.trainer.model.tools import ModelTools
from optimus.trainer.script.cache import Cache
from optimus.trainer.script.distillation.knowledge_distillation import (
    KnowledgeDistillation,
)
from optimus.trainer.script.warmup_stable_decay_lr import WarmupStableDecayLR

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class Pretrain:
    """Pretrain class to train the model."""

    def __init__(
        self, model: nn.Module, data: Data, distributed: Distributed, config: Config
    ) -> None:
        """
        Args:
            model: Model to train.
            data: Data class object.
            config: Config class object.
        """
        self.model = model
        self.data = data
        self.distributed = distributed
        self.config = config
        self.train_config = config.train
        self.system_config = config.system

        self.main_process = config.is_main_process
        self.tokens_per_step = (
            config.data.batch_size
            * config.train.gradient_accumulation_steps
            * config.data.length
            * config.system.world_size
        )

        if (
            self.main_process
            and self.train_config.tensorboard
            and (
                self.train_config.reload_checkpoint is None
                or self.config.train.skip_reload_tensorboard
            )
        ):
            os.makedirs(
                f"{self.train_config.output_dir}/{self.train_config.project_name}/tensorboard",
                exist_ok=True,
            )
            self.writer = SummaryWriter(
                f"{self.train_config.output_dir}/{self.train_config.project_name}/tensorboard"
            )

        self.steps_per_epoch = int(
            len(self.data.train_dataloader)
            / self.train_config.gradient_accumulation_steps
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
            betas=(self.train_config.beta1, self.train_config.beta2),
            eps=self.train_config.eps,
            fused=self.train_config.fused,
        )

        self.scheduler = self.get_scheduler(self.train_config.lr_scheduler)
        self.step = 0

        # Knowledge Distillation setup
        if self.train_config.knowledge_distillation:
            self.asyncio_loop = asyncio.get_event_loop()
            self.knowledge_distillation = KnowledgeDistillation(
                train_config=self.train_config,
                dataset_config=self.config.data,
                logger=self.config.log_print,
            )

        # Resume training if a checkpoint is provided
        if self.train_config.reload_checkpoint:
            self.resume()

        # Compile model for training
        if self.config.train.compile_model:
            self.model = compile_model(self.model, config)
            if distributed:
                dist.barrier()
            self.config.log_print("Model compiled.")

        # Init cache for model if required
        self.cache = Cache()
        dtype = torch.bfloat16 if self.train_config.mixed_bfloat16 else torch.float32
        for mod in self.model.modules():
            if hasattr(mod, "init_cache"):
                mod.init_cache(self.cache, dtype=dtype, device=self.model.device)

        # Clear GPU cache before start training
        ModelTools.clear_gpu_cache()

    def train(self) -> None:
        """
        Launch the training loop.
        """
        # Set up profiling and mixed precision context managers
        profiler = (
            self.profiler()
            if self.train_config.profile and self.main_process
            else nullcontext()
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (self.train_config.mixed_bfloat16 and not self.train_config.fsdp)
            else nullcontext()
        )

        # Initialize training parameters
        total_loss = 0
        skip_threshold = (
            self.config.data.step_to_skip
            * self.config.train.gradient_accumulation_steps
        )

        if self.distributed:
            dist.barrier()
        self.config.log_print(
            "Ready to start training (first iteration may take some time due to MosaicML indexing)."
        )
        start_time = time.time()

        with profiler as prof:
            for i, batch in enumerate(self.data.train_dataloader, start=1):
                # First batch processing
                if self.pre_batch_step(i, skip_threshold):
                    continue

                # No sync context manager for gradient accumulation
                no_sync = (
                    self.model.no_sync()
                    if i % self.config.train.gradient_accumulation_steps != 0
                    and i != len(self.data.train_dataloader)
                    else nullcontext()
                )

                with no_sync:
                    with autocast:
                        if self.config.model.huggingface_id:
                            loss = self.model(**batch)[0]
                        elif self.train_config.knowledge_distillation:
                            # Parallelize teacher and student forward passes with asynchrounous teacher call
                            teacher_forward = (self.knowledge_distillation.get_teacher_forward(**batch))
                            teacher_task = self.asyncio_loop.create_task(teacher_forward)
                            logits, ce_loss = self.model(**batch, cache=self.cache)

                            # Ensure the teacher task is complete
                            if not teacher_task.done():
                                self.asyncio_loop.run_until_complete(teacher_task)
                            target_token_ids, target_logprobs, target_mask = (
                                teacher_task.result()
                            )

                            kl_loss = self.knowledge_distillation.loss(
                                student_logits=logits,
                                target_token_ids=target_token_ids.to(device=logits.device),
                                target_logprobs=target_logprobs.to(device=logits.device),
                                target_mask=target_mask,
                            )
                            loss = (
                                self.train_config.kd_alpha * kl_loss
                                + (1 - self.train_config.kd_alpha) * ce_loss
                            )
                        else:
                            _, loss = self.model(**batch, cache=self.cache)
                        loss = loss / self.config.train.gradient_accumulation_steps
                    loss.backward()
                total_loss += loss.detach().item()

                # Training step
                if i % self.train_config.gradient_accumulation_steps == 0 or i == len(
                    self.data.train_dataloader
                ):
                    grad_norm = self.clip_grad_norm_(self.train_config.clip_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    end_time = time.time()
                    if self.main_process:
                        self.config.log_print(
                            f"Step: {self.step}",
                            f"Loss: {total_loss:.4f}",
                            *(
                                [
                                    f"KL Div: {kl_loss:.4f}",
                                    f"CE Loss: {ce_loss:.4f}"
                                ]
                                if self.train_config.knowledge_distillation else []
                            ),
                            f"Time/step (s): {(end_time - start_time):.2f}",
                            f"Learning rate: {self.scheduler.get_last_lr()[0]}",
                            f"Grad norm: {grad_norm:.4f}",
                        )

                    if (
                        self.train_config.tensorboard
                        and self.main_process
                        and self.step % self.train_config.log_every_n_steps == 0
                    ):
                        self.writer.add_scalar("Loss/train", total_loss, self.step)
                        if self.train_config.knowledge_distillation:
                            self.writer.add_scalar("Loss/KL_divergence", kl_loss.detach().item(), self.step)
                            self.writer.add_scalar("Loss/cross_entropy", ce_loss.detach().item(), self.step)
                        self.writer.add_scalar("Gradient norm", grad_norm, self.step)
                        self.writer.add_scalar(
                            "Learning rate", self.scheduler.get_last_lr()[0], self.step
                        )
                        self.writer.add_scalar(
                            "Time/step in seconds", end_time - start_time, self.step
                        )
                        self.writer.add_scalar(
                            "Tokens seen", self.tokens_per_step * self.step, self.step
                        )
                        self.writer.add_scalar(
                            "Tokens seen/second",
                            self.tokens_per_step / (end_time - start_time),
                            self.step,
                        )

                    # Validation
                    if self.train_config.run_validation and (
                        self.step % self.train_config.validation_step == 0
                    ):
                        if self.data.eval_dataloader is not None:
                            self.eval()

                    # Save model and other states
                    if (
                        self.step % self.train_config.save_step == 0 and self.step != 0
                    ) or i == len(self.data.train_dataloader):
                        self.save()
                        self.config.log_print(
                            f"Remaining steps: {(self.steps_per_epoch * self.train_config.num_epochs) - (self.step)}"
                        )

                    # Profiling
                    if isinstance(prof, torch.profiler.profile) and self.main_process:
                        prof.step()
                        if prof.step_num == 20 and self.train_config.exit_end_profiling:
                            break

                    self.step += 1
                    start_time = end_time
                    total_loss = 0

    def eval(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.train_config.mixed_bfloat16 and not self.train_config.fsdp
            else nullcontext()
        )
        self.model.eval()

        total_loss = 0
        for batch in self.data.eval_dataloader:
            input_ids = batch["input_ids"].to(device=self.model.device).contiguous()
            labels = batch["labels"].to(device=self.model.device).contiguous()

            with torch.no_grad():
                with autocast:
                    _, loss = self.model(input_ids, labels=labels, cache=self.cache)
            total_loss += loss.item()

        loss = total_loss / len(self.data.eval_dataloader)
        if self.train_config.fsdp:
            loss = torch.tensor(loss, device=self.model.device)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            loss = loss.item()

        if self.train_config.tensorboard and self.main_process:
            self.writer.add_scalar("Loss/eval", loss, self.step)
            self.config.log_print(f"Validation loss: {loss}")
        self.model.train()

    # ----------------------
    # Tool functions
    # ----------------------

    def save(self) -> None:
        """
        Save the model, optimizer, scheduler, dataloader state dicts and config.
        """
        path = f"{self.train_config.output_dir}/{self.train_config.project_name}/checkpoints/{self.step}/"
        os.makedirs(path, exist_ok=True)
        self.config.log_print(f"Saving checkpoint at steps: {self.step}.")
        self.config.log_print(f"Saving checkpoint at path: {path}")

        if self.train_config.save_model:
            if self.train_config.fsdp:
                assert self.train_config.save_optimizer, (
                    "FSDP requires saving the optimizer with the model."
                )
                self.distributed.save_fsdp_model_optimizer(
                    self.model, self.optimizer, path
                )
            elif self.main_process:
                if self.train_config.lora_finetuning:
                    self.model.save_pretrained(path)
                else:
                    torch.save(self.model.state_dict(), path + "model.pt")
                self.config.log_print("Model saved.")

        # Save optimizer handled with model saving for FSDP
        if (
            not self.train_config.fsdp
            and self.train_config.save_optimizer
            and self.main_process
        ):
            torch.save(self.optimizer.state_dict(), path + "optimizer.pt")
            os.remove(path + "/README.md") if os.path.exists(path + "/README.md") else None
            self.config.log_print("Optimizer saved.")

        if self.train_config.save_scheduler and self.main_process:
            torch.save(self.scheduler.state_dict(), path + "scheduler.pt")
            self.config.log_print("Scheduler saved.")

        if self.train_config.save_data_loader and self.main_process:
            torch.save(
                self.data.train_dataloader.state_dict(), path + "train_dataloader.pt"
            )
            self.config.log_print("Train dataloader saved.")

        if self.train_config.save_config and self.main_process:
            self.config.save(path)
            self.config.log_print("Config saved.")

    def resume(self) -> None:
        """
        Load the optimizer, scheduler and dataloader state dicts to resume training.
        Nb: The configuration is reloaded during the init Config object.
        """
        checkpoint_path = self.train_config.reload_checkpoint
        self.step = int(checkpoint_path.split("/")[-1])
        self.config.log_print(f"Reloading checkpoint from steps: {self.step}.")

        model_path = os.path.join(checkpoint_path, "model.pt")
        optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
        dataloader_path = os.path.join(checkpoint_path, "train_dataloader.pt")
        tensorboard_path = os.path.join(
            self.train_config.output_dir, self.train_config.project_name, "tensorboard"
        )

        # Model reloading
        if self.train_config.fsdp:
            self.distributed.load_fsdp_model_optimizer(
                self.model, self.optimizer, checkpoint_path
            )
        else:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.config.log_print("Model reloaded.")

        # Optimizer reloading
        if not self.train_config.fsdp:
            optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.config.log_print("Optimizer reloaded.")

        # Scheduler reloading
        if self.config.train.skip_reload_scheduler:
            self.config.log_print("Skipping reloading the scheduler.")
            self.config.train.skip_reload_scheduler = False
        else:
            scheduler_state_dict = torch.load(scheduler_path, map_location="cpu")
            self.scheduler.load_state_dict(scheduler_state_dict)
            if self.train_config.lr_scheduler == "WarmupStableDecayLR":
                if self.scheduler.target_lr != self.train_config.lr:
                    self.scheduler.target_lr = self.train_config.lr
                    self.config.log_print(
                        f"Scheduler LR updated to {self.train_config.lr}."
                    )
            self.config.log_print("Scheduler reloaded.")

        # Dataloader reloading
        if self.config.train.skip_reload_dataloader:
            self.config.log_print("Skipping reloading the dataloader.")
            self.config.train.skip_reload_dataloader = False
        else:
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu")
            self.data.train_dataloader.load_state_dict(dataloader_state_dict)
            self.config.log_print("Train dataloader reloaded.")

        # Tensorboard reloading
        if self.config.train.skip_reload_tensorboard:
            self.config.log_print("Skipping reloading the tensorboard.")
            self.config.train.skip_reload_tensorboard = False
        elif self.train_config.tensorboard and self.main_process:
            self.writer = SummaryWriter(log_dir=tensorboard_path, purge_step=self.step)
            self.config.log_print("Tensorboard reloaded.")

        if self.distributed:
            dist.barrier()
        self.config.log_print("Training reloaded.")

    def clip_grad_norm_(self, max_norm: float) -> torch.Tensor:
        """
        Clip the norm of the gradients for all parameters.
        """
        if hasattr(self.model, "clip_grad_norm_"):
            return self.model.clip_grad_norm_(max_norm)
        else:
            return nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def get_scheduler(
        self, lr_scheduler: str = "OneCycleLR"
    ) -> torch.optim.lr_scheduler:
        """
        Get the scheduler.
        Args:
            lr_scheduler: The name of the learning rate scheduler.
        Returns:
            torch.optim.lr_scheduler: The learning rate scheduler.
        """
        if lr_scheduler == "WarmupStableDecayLR":
            return WarmupStableDecayLR(
                self.optimizer,
                max_lr=self.train_config.lr,
                warmup_iters=self.train_config.pct_start,
                initial_div_factor=self.train_config.div_factor,
                decay_iters=self.train_config.end_start,
                final_div_factor=self.train_config.final_div_factor,
                epochs=self.train_config.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
            )
        elif lr_scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.steps_per_epoch * self.train_config.num_epochs,
            )
        else:
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.train_config.lr,
                epochs=self.train_config.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=self.train_config.pct_start,
                div_factor=self.train_config.div_factor,
                final_div_factor=self.train_config.final_div_factor,
            )

    # ----------------------
    # Pre-batch functions
    # ----------------------

    def pre_batch_step(self, iter: int, skip_threshold: int) -> bool:
        """
        Handles operations prior to the main training step, such as processing
        the first batch and skipping specified steps.

        Args:
            iter (int): Current iteration number.
            skip_threshold (int): Number of steps to skip during training.

        Returns:
            bool: Whether to skip the current step.
        """
        if iter > skip_threshold and iter != 1:
            return False

        if iter == 1:
            self.config.log_print(
                f"Rank {self.system_config.rank} get the first batch.", main_only=False
            )
            if self.distributed:
                dist.barrier()
            self.config.log_print("All ranks with first batch, training will start.")
            self.config.log_print(
                f"Remaining steps: {(self.steps_per_epoch * self.train_config.num_epochs) - self.step}"
            )

        if iter <= skip_threshold:
            self.config.log_print(
                f"Skipping {self.config.data.step_to_skip} steps, (iter {iter}/{skip_threshold})."
            )
            if iter == skip_threshold:
                if self.distributed:
                    self.config.log_print("Waiting all rank...")
                    dist.barrier()
                self.config.log_print(
                    f"Completed skipping {self.config.data.step_to_skip} steps."
                )
            return True

        return False

    # ----------------------
    # Profiling functions
    # ----------------------

    @contextlib.contextmanager
    def profiler(self) -> Generator[Optional[torch.profiler.profile], None, None]:
        profile_dir = (
            f"{self.train_config.output_dir}/{self.train_config.project_name}/profiler"
        )
        os.makedirs(profile_dir, exist_ok=True)

        if self.train_config.profiler_output == "chrome":

            def trace_handler(p):
                p.export_chrome_trace(f"{profile_dir}/trace_{p.step_num}.json")
        elif self.train_config.profiler_output == "tensorboard":
            trace_handler = torch.profiler.tensorboard_trace_handler(profile_dir)
        else:
            raise ValueError(
                f"Profiler type {self.train_config.profiler_output} not supported."
            )

        self.config.log_print(f"Profiling data will be saved in {profile_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=12, warmup=6, active=4, repeat=1),
            on_trace_ready=trace_handler,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            yield prof
