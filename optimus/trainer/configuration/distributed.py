import functools
from dataclasses import dataclass
from typing import Callable, Union

import torch
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, wrap
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from optimus.trainer.model.encoder.biqwen import Qwen3EncoderLayer
from optimus.trainer.model.encoder.bigemma import Gemma3DecoderLayer
from optimus.trainer.model.model import Block

@dataclass
class DistributedConfig:
    _sharding_strategy: Union[ShardingStrategy, str] = "HYBRID_SHARD"

    @property
    def sharding_strategy(self) -> ShardingStrategy:
        if isinstance(self._sharding_strategy, ShardingStrategy):
            return self._sharding_strategy
        elif isinstance(self._sharding_strategy, str):
            return ShardingStrategy[self._sharding_strategy.upper()]

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: Union[ShardingStrategy, str]):
        self._sharding_strategy = sharding_strategy

    _mixed_precision: str = "bfloat16"

    @property
    def mixed_precision(self) -> MixedPrecision:
        if self._mixed_precision == "float32":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        elif self._mixed_precision == "float16":
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif self._mixed_precision == "bfloat16":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self._mixed_precision == "mixed_float16":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif self._mixed_precision == "mixed_bfloat16":
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif self._mixed_precision == "bfloat16_reduce_32":
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )

    @mixed_precision.setter
    def mixed_precision(self, mixed_precision: Union[MixedPrecision, str]):
        self._mixed_precision = mixed_precision

    _wrap_policy: str = "transformer_auto_wrap_policy"

    @property
    def wrap_policy(self) -> Union[Callable, wrap.ModuleWrapPolicy, wrap.CustomPolicy]:
        if self._wrap_policy == "size_based_auto_wrap_policy":
            return functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=20000,
                exclude_wrap_modules={}
                )
        elif self._wrap_policy == "transformer_auto_wrap_policy":
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block, Qwen3EncoderLayer, Gemma3DecoderLayer},
            )
