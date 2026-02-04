from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TrainConfig:
    project_name: str = "training"
    reload_checkpoint: Optional[str] = None
    output_dir: str = "output"

    lr: float = 1e-4
    num_epochs: int = 1
    clip_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    optimizer: str = "AdamW"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    fused: bool | None = False

    lr_scheduler: str = "OneCycleLR"
    pct_start: float = 0.01
    div_factor: int = 0
    end_start: float = 0
    final_div_factor: int = 0

    # Compilation configurations. These configurations are described in the
    # torch.compile documentation.
    # https://pytorch.org/docs/stable/generated/torch.compile.html
    compile_model: bool = False  # Compile model
    compile_mode: str | None = None  # Compilation mode
    compile_options: dict | None = None  # Compilation options
    compile_dynamic: bool | None = None  # Dynamic compilation

    # Validation configurations
    run_validation: bool = True
    validation_step: int = 5000

    # Save configurations
    save_step: int = 5000
    save_model: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_data_loader: bool = True
    save_config: bool = True

    # Masking configurations
    mlm_probability: float = 0.3
    mask_probability: float = 1.0
    random_probability: float = 0.0
    original_probability: float = 0.0
    mntp_objective: bool = False

    # Reloading configurations
    skip_reload_scheduler: bool = False
    skip_reload_dataloader: bool = False
    skip_reload_tensorboard: bool = False

    # Other configurations
    fsdp: bool = False
    mixed_bfloat16: bool = True
    seed: int = 42
    tensorboard: bool = True
    profile: bool = False
    exit_end_profiling: bool = True
    profiler_output: Literal["chrome", "tensorboard"] = "chrome"
    log_every_n_steps: int = 10

    # Knowledge Distillation configurations
    knowledge_distillation: bool = False
    kd_num_logprobs: int = 32
    kd_num_output_chunks: int = 8
    kd_temperature: float = 1.0
    kd_alpha: float = 0.5
    kd_teacher_skip_first_token: bool = False
    kd_base_url: str = "http://localhost:8000/"
    kd_server_timeout: int = 600

    # LoRA configurations
    lora_finetuning: bool = False
    lora_r: int = 128
    lora_target_modules: list[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head")
    lora_alpha: int = 256
    lora_dropout: float = 0