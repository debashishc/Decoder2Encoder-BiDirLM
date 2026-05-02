import json
from types import MethodType
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from streaming import Stream, StreamingDataLoader, StreamingDataset
from torch.nn.utils.rnn import pad_sequence

from optimus.trainer.configuration.configs import Config


class Data:
    """Manages dataset loading, dataloader creation, and state management."""

    def __init__(self, config: Config, tokenizer):
        self.data_config = config.data
        self.system_config = config.system
        self.train_config = config.train
        self.main_process = config.is_main_process
        self.mntp_objective = config.train.mntp_objective
        self.knowledge_distillation = config.train.knowledge_distillation
        self.tokenizer = tokenizer
        self.hf_model = config.model.huggingface_id is not None

        assert (
            config.train.mask_probability
            + config.train.random_probability
            + config.train.original_probability
            == 1.0
        ), "The sum of masking probabilities must be equal to 1.0."

        self.num_canonical_nodes = (
            config.system.num_nodes
            if config.data.num_canonical_nodes <= 0
            else config.data.num_canonical_nodes
        )
        self.mlm_probability = config.train.mlm_probability
        self.mask_probability = config.train.mask_probability
        self.random_probability = (
            config.train.random_probability / (1 - self.mask_probability)
            if self.mask_probability < 1.0
            else 0.0
        )

        if config.data.num_canonical_nodes <= 0:
            config.update_config(num_canonical_nodes=self.num_canonical_nodes)

        self.train_streams = self._load_data_mix(f"{self.data_config.data_mix_path}/train.json")
        self.train_dataset = self._create_dataset(self.train_streams)
        self.train_dataloader = self._create_dataloader(self.train_dataset)
        self.eval_dataloader = None

        config.log_print(f"Number of canonical nodes: {self.num_canonical_nodes}")
        config.log_print("Train dataset created successfully:", len(self.train_dataset))
        config.log_print("Train dataloader created successfully:", len(self.train_dataloader))
        config.log_print(
            f"Masking probabilities: MLM={config.train.mlm_probability}, "
            f"Mask={config.train.mask_probability}, Random={config.train.random_probability}, "
            f"Original={config.train.original_probability}, "
            f"Add BOS token: {self.data_config.add_bos_token}, Add EOS token: {self.data_config.add_eos_token}"
        )

    def _load_data_mix(self, path: str) -> list[Stream]:
        with open(path, "r") as f:
            return [Stream(**item) for item in json.load(f)]

    def _create_dataset(self, streams: list[Stream], eval: bool = False) -> StreamingDataset:
        return MaskingDataset(
            streams=streams,
            shuffle=not eval and self.data_config.shuffle,
            shuffle_seed=9176 if eval else self.data_config.seed,
            batch_size=self.data_config.batch_size,
            num_canonical_nodes=self.num_canonical_nodes,
            shuffle_block_size=int(max(4_000_000 // self.num_canonical_nodes, 1 << 18)),
            predownload=self.data_config.predownload * self.data_config.batch_size,
            mlm_probability=self.mlm_probability,
            mask_probability=self.mask_probability,
            random_probability=self.random_probability,
            tokenizer=self.tokenizer,
            mntp_objective=self.mntp_objective,
            add_bos_token=self.data_config.add_bos_token,
            add_eos_token=self.data_config.add_eos_token,
            knowledge_distillation=self.knowledge_distillation,
        )

    def _create_dataloader(self, dataset: StreamingDataset) -> StreamingDataLoader:
        collate_fn = (
            self.to_torch_collate_HF_pad_fn if self.hf_model
            else self.to_torch_collate_var_len_fn_with_KD if self.knowledge_distillation
            else self.to_torch_collate_var_len_fn
        )

        dataloader = StreamingDataLoader(
            dataset,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor or None,
            collate_fn=collate_fn,
            pin_memory=self.data_config.pin_memory,
            drop_last=True,
        )

        dataloader._get_batch_size = MethodType(_get_batch_size, dataloader)
        return dataloader

    def to_torch_collate_var_len_fn(self, batch):
        input_seqs, label_seqs, cu_seqlens = zip(*batch)

        x = torch.cat([torch.as_tensor(seq, dtype=torch.long) for seq in input_seqs])
        y = torch.cat([torch.as_tensor(seq, dtype=torch.long) for seq in label_seqs])

        parts = [torch.zeros(1, dtype=torch.int32)]
        offset = 0
        max_seqlen = 0
        for seq, cu_seq in zip(input_seqs, cu_seqlens):
            parts.append(torch.as_tensor(cu_seq[1:], dtype=torch.int32) + offset)
            offset += cu_seq[-1]
            max_seqlen = max(max_seqlen, len(seq))

        return {
            "x": x,
            "labels": y,
            "cu_seqlens": torch.cat(parts),
            "max_seqlen": max_seqlen,
        }

    def to_torch_collate_var_len_fn_with_KD(self, batch):
        prompts = [item.pop() for item in batch]
        result = self.to_torch_collate_var_len_fn(batch)
        result["prompts"] = prompts
        return result

    def to_torch_collate_HF_pad_fn(self, batch):
        input_seqs, label_seqs = zip(*batch)

        input_tensors = [torch.tensor(seq, dtype=torch.long) for seq in input_seqs]
        label_tensors = [torch.tensor(seq, dtype=torch.long) for seq in label_seqs]

        padded_inputs = pad_sequence(input_tensors, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_inputs,
            "attention_mask": (padded_inputs != 0).long(),
            "labels": padded_labels,
        }


class MaskingDataset(StreamingDataset):
    def __init__(
        self,
        mlm_probability: float,
        mask_probability: float,
        random_probability: float,
        tokenizer,
        mntp_objective: bool = False,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        knowledge_distillation: bool = False,
        *args,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability
        self.mntp_objective = mntp_objective
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.knowledge_distillation = knowledge_distillation
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        
        inputs, cu_seqlens = self._add_special_tokens(item["tokens"], item.get("cu_seqlens"))
        prompts = inputs.tolist() if self.knowledge_distillation else None
        inputs, labels, cu_seqlens = self._apply_masking(inputs, cu_seqlens)

        result = [inputs, labels]
        if cu_seqlens is not None:
            result.append(cu_seqlens)
        if prompts is not None:
            result.append(prompts)
        return result

    def _add_special_tokens(self, tokens: NDArray, cu_seqlens: NDArray | None = None):
        if self.add_bos_token:
            tokens, cu_seqlens = self._insert_token(tokens, cu_seqlens, self.tokenizer.bos_token_id, at_start=True)
        if self.add_eos_token:
            tokens, cu_seqlens = self._insert_token(tokens, cu_seqlens, self.tokenizer.eos_token_id, at_start=False)
        return tokens, cu_seqlens

    def _insert_token(self, tokens: NDArray, cu_seqlens: NDArray | None, token_id: int, at_start: bool):
        if cu_seqlens is not None:
            num_seqs = len(cu_seqlens) - 1
            total_len = len(tokens) + num_seqs
            insert_pos = (cu_seqlens[:-1] if at_start else cu_seqlens[1:]) + np.arange(num_seqs)

            new_tokens = np.empty(total_len, dtype=tokens.dtype)
            new_tokens[insert_pos] = token_id
            mask = np.ones(total_len, dtype=bool)
            mask[insert_pos] = False
            new_tokens[mask] = tokens

            return new_tokens, cu_seqlens + np.arange(len(cu_seqlens))
        else:
            return (
                np.concatenate(([token_id], tokens)) if at_start
                else np.concatenate((tokens, [token_id]))
            ), None

    def _apply_masking(self, tokens: NDArray, cu_seqlens: NDArray | None = None):
        inputs = np.copy(tokens)
        labels = np.copy(tokens)

        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = np.array(
            self.tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True),
            dtype=bool,
        )
        probability_matrix[special_tokens_mask] = 0.0

        masked_indices = np.random.rand(*probability_matrix.shape) < probability_matrix
        labels[~masked_indices] = -100

        indices_replaced = (np.random.rand(*labels.shape) < self.mask_probability) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
            (np.random.rand(*labels.shape) < self.random_probability)
            & masked_indices
            & ~indices_replaced
        )
        # Use vocab_size, not len(tokenizer): added tokens (e.g. Gemma3's
        # <image_soft_token> at id 262144) sit beyond the model's embedding
        # rows and trigger a device-side assert when looked up.
        inputs[indices_random] = np.random.randint(0, self.tokenizer.vocab_size, size=labels.shape)[indices_random]

        if self.mntp_objective:
            cu_seqlens = np.copy(cu_seqlens)
            inputs, labels, cu_seqlens = self._apply_mntp_shift(inputs, labels, cu_seqlens)

        return inputs, labels, cu_seqlens

    @staticmethod
    def _apply_mntp_shift(inputs: NDArray, labels: NDArray, cu_seqlens: NDArray):
        num_seqs = len(cu_seqlens) - 1

        if cu_seqlens is not None and num_seqs > 1:
            input_mask = np.ones(len(inputs), dtype=bool)
            input_mask[cu_seqlens[1:] - 1] = False
            inputs = inputs[input_mask]

            label_mask = np.ones(len(labels), dtype=bool)
            label_mask[cu_seqlens[:-1]] = False
            labels = labels[label_mask]

            return inputs, labels, cu_seqlens - np.arange(num_seqs + 1)
        else:
            cu_seqlens[1] -= 1
            return inputs[:-1], labels[1:], cu_seqlens


def _get_batch_size(self, batch: Any) -> int:
    return self.batch_size


def patch_spanner():
    from streaming.base import spanner
    spanner.Spanner.__init__ = SpannerPatch.__init__
    spanner.Spanner.__getitem__ = SpannerPatch.__getitem__


class SpannerPatch:
    """Patches the large memory allocation in the original Spanner initialization.

    This implementation was taken from: https://github.com/mosaicml/streaming/pull/773
    """

    def __init__(self, shard_sizes: NDArray[np.int64], span_size: int = 1 << 10) -> None:
        self.shard_sizes = shard_sizes
        self.span_size = span_size
        self.num_samples = sum(shard_sizes)
        self.shard_bounds = np.concatenate([np.zeros(1, np.int64), shard_sizes.cumsum()])

        overflow = self.num_samples % span_size
        underflow = span_size - overflow if overflow else 0
        self.shard_sizes[-1] += underflow

        n_shards = len(shard_sizes)
        current_shard = 0
        current_position_in_shard = 0

        span_lowest_shards = []
        span_highest_shards = []

        while current_shard < n_shards:
            span_min_shard = current_shard
            span_max_shard = current_shard
            remaining_span_size = span_size

            while remaining_span_size > 0 and current_shard < n_shards:
                available_in_current_shard = shard_sizes[current_shard] - current_position_in_shard

                if remaining_span_size >= available_in_current_shard:
                    remaining_span_size -= available_in_current_shard
                    current_shard += 1
                    current_position_in_shard = 0
                else:
                    current_position_in_shard += remaining_span_size
                    remaining_span_size = 0

                if current_shard < n_shards:
                    span_max_shard = current_shard

            span_lowest_shards.append(span_min_shard)
            span_highest_shards.append(span_max_shard)

        self.spans = [np.arange(low, high + 1) for low, high in zip(span_lowest_shards, span_highest_shards)]
        self.shard_sizes[-1] -= underflow

    def __getitem__(self, index: int) -> tuple[int, int]:
        if not (0 <= index < self.num_samples):
            raise IndexError(f"Invalid sample index `{index}`: 0 <= {index} < {self.num_samples}")

        span = index // self.span_size
        for shard in self.spans[span]:
            shard_start = self.shard_bounds[shard]
            shard_stop = self.shard_bounds[shard + 1]
            if shard_start <= index < shard_stop:
                return shard, int(index - shard_start.item())  # pyright: ignore

        raise RuntimeError("Internal error: shards were indexed incorrectly")
