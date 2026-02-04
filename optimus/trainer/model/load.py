from dataclasses import asdict, dataclass

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    logging,
)
import warnings

from optimus.trainer.configuration.configs import Config
from optimus.trainer.model.encoder.bert import Bert, bert_config
from optimus.trainer.model.encoder.eurobert import EuroBERT, eurobert_config
from optimus.trainer.model.encoder.biqwen import Qwen3ForMaskedLM
from optimus.trainer.model.encoder.bigemma import Gemma3ForCausalLM
from optimus.trainer.model.tools import ModelTools

try:
    from peft import get_peft_model, LoraConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

def update_config(config: dataclass, config_dict: dict) -> dict:
    """
    Update the configuration object with the given dictionary.
    Args:
        config (dataclass): Configuration object.
        config_dict (dict): Configuration dictionary.
    Returns: dict: Updated configuration dictionary.
    """
    config_as_dict = asdict(config)
    filtered_config_dict = {
        key: value
        for key, value in config_as_dict.items()
        if key in config_dict and config_as_dict[key] is not None
    }
    config_dict.update(filtered_config_dict)
    return config_dict


def load_tokenizer(config: Config) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load the tokenizer from the given name or path.
    Args:
        config (config): Configuration object.
    Returns: PreTrainedTokenizer | PreTrainedTokenizerFast: Tokenizer object.
    """
    logging.set_verbosity_error()
    tokenizer_name = (
        config.model.huggingface_id
        if config.model.huggingface_id
        else config.model.tokenizer_path_or_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.mask_token is None:
        assert (
            config.model.mask_token_id is not None
        ), "Mask token id to use is not provided (config.model.mask_token_id)."
        tokenizer.mask_token = "[MASK]"
        tokenizer.mask_token_id = config.model.mask_token_id
    if config.data.add_bos_token and tokenizer.bos_token is None:
        assert (
            config.model.bos_token_id is not None
        ), "bos_token id is not provided (config.model.bos_token_id)."
        tokenizer.bos_token = tokenizer.convert_ids_to_tokens(config.model.bos_token_id)
        tokenizer.bos_token_id = config.model.bos_token_id
    if config.data.add_eos_token and tokenizer.eos_token is None:
        assert (
            config.model.eos_token_id is not None
        ), "eos_token id is not provided (config.model.eos_token_id)."
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(config.model.eos_token_id)
        tokenizer.eos_token_id = config.model.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.verbose:
        config.log_print("Tokenizer loaded successfully.")
        config.log_print(f"Tokenizer name or path: {tokenizer_name}")
    return tokenizer


def load_model(config: Config):
    """Load model based on a config.model architecture or from a huggingface id.
    Args:
        config (config): Configuration object.
    Returns: Model object.
    """
    if config.model.huggingface_id:
        model = AutoModelForMaskedLM.from_pretrained(
            config.model.huggingface_id, return_dict=False, trust_remote_code=True,
            attn_implementation="flash_attention_2" if config.model.attn_impl == "flash" else None,
        )
    else:
        if config.model.model_name == "bert":
            dict_config_model = update_config(
                config.model, bert_config[config.model.model_size]
            )
            model = Bert(dict_config_model)
        elif config.model.model_name == "eurobert":
            dict_config_model = update_config(
                config.model, eurobert_config[config.model.model_size]
            )
            model = EuroBERT(dict_config_model)
        elif "qwen3" in config.model.model_name.lower():
            logging.set_verbosity_error()
            model = Qwen3ForMaskedLM.from_pretrained(
                config.model.model_name,
                attn_implementation="flash_attention_2" if config.model.attn_impl == "flash" else None,
                fused_cross_entropy=config.model.fused_cross_entropy,
            )
            dict_config_model = asdict(config.model)
        elif "gemma3" in config.model.model_name.lower():
            logging.set_verbosity_error()
            model = Gemma3ForCausalLM.from_pretrained(
                config.model.model_name,
                use_bidirectional_attention = True,
                attn_implementation="flash_attention_2" if config.model.attn_impl == "flash" else None,
                fused_cross_entropy=config.model.fused_cross_entropy,
            )
            model.config.sliding_window = model.config.sliding_window // 2
            dict_config_model = asdict(config.model)
        else:
            raise ValueError(f"Model name {config.model.model_name} is not supported.")
        config.update_config(**dict_config_model)

    # Lora tuning
    if config.train.lora_finetuning:
        assert PEFT_AVAILABLE, "Please install the 'peft' library to use LoRA finetuning: pip install peft"
        warnings.filterwarnings("ignore", message=".*Setting `save_embedding_layers` to `True`.*")

        lora_config = LoraConfig(
            r=config.train.lora_r,
            target_modules=config.train.lora_target_modules,
            lora_alpha=config.train.lora_alpha,
            lora_dropout=config.train.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
        if config.verbose:
            config.log_print(f"LoRA finetuning is enabled: {lora_config}")

    # Move model to GPU if available
    if torch.cuda.is_available() and config.model.gpu:
        model = model.to(f"cuda:{config.system.local_rank}")
    elif (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and config.model.gpu
    ):
        model = model.to("mps")
    else:
        config.log_print(
            "WARNING: Model is loaded on CPU. Training will be slow.", force_print=True
        )

    if config.verbose and config.is_main_process:
        ModelTools.model_summary(model, model_layers=True)
    return model


def compile_model(model: torch.nn.Module, config: Config):
    return torch.compile(
        model,
        backend="inductor",
        dynamic=config.train.compile_dynamic,
        mode=config.train.compile_mode,
        options=config.train.compile_options,
    )
