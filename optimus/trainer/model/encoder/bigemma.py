import copy
from typing import Optional

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
    LIGER_KERNEL_AVAILABLE = True
except ImportError:
    LIGER_KERNEL_AVAILABLE = False


class Gemma3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = self.config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
        window_size: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(0, 1)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(0, 1)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.config._attn_implementation == "flash_attention_2" and FLASH_ATTN_AVAILABLE:
            attn_output = flash_attn.flash_attn_varlen_func(
                query_states.transpose(0, 1),
                key_states.transpose(0, 1),
                value_states.transpose(0, 1),
                cu_seqlens,
                cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.scaling,
                causal=not self.config.use_bidirectional_attention,
                window_size=window_size
            )
        else:
            attn_output, _ = sdpa_attention_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    dropout=self.attention_dropout if self.training else 0.0,
                    softcap=self.attn_logit_softcapping,
                )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
    

def sdpa_attention_forward(
    q, k, v, 
    attention_mask,
    scaling, 
    dropout: float = 0.0, 
    softcap: Optional[float] = None,
    ):
    attn_weights = torch.matmul(q, k.transpose(1, 2)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout)

    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous()

    return attn_output, attn_weights


def create_packed_seqs_mask(
    cu_seqlens: torch.Tensor,
    causal: bool = True,
    device: torch.device = torch.device("cpu"),
    window_size: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Builds a block-diagonal attention mask for packed sequences.
    Returns shape [total_len, total_len] with 0.0 for attention and -inf for masked.
    """
    total_len = cu_seqlens[-1]
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    
    seq_ids = torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device),
        seq_lengths
    )
    
    mask = seq_ids.unsqueeze(0) == seq_ids.unsqueeze(1)

    if causal:
        mask &= torch.tril(torch.ones(total_len, total_len, device=device, dtype=torch.bool))

    if window_size is not None:
        left, right = window_size
        start_indices = torch.repeat_interleave(cu_seqlens[:-1].to(device), seq_lengths)
        relative_pos = torch.arange(total_len, device=device) - start_indices

        distance = relative_pos.unsqueeze(0) - relative_pos.unsqueeze(1)

        if left >= 0:
            mask &= (distance >= -left)
        if right >= 0:
            mask &= (distance <= right)

    attn_mask = torch.full((total_len, total_len), float('-inf'), device=device)
    attn_mask.masked_fill_(mask, 0.0)
    
    return attn_mask


class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        window_size: Optional[tuple[int, int]] = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            window_size=window_size,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3PreTrainedModel(PreTrainedModel):
    config_class = Gemma3Config
    config: Gemma3Config
    _supports_flash_attn = True
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        super()._init_weights(module)
        # if isinstance(module, Gemma3MultiModalProjector):
        #     module.mm_input_projection_weight.data.zero_()
        # # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        # elif "RMSNorm" in module.__class__.__name__:
        #     module.weight.data.zero_()
        if "RMSNorm" in module.__class__.__name__:
            module.weight.data.zero_()

class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return self.weight[input_ids, :] * self.embed_scale.to(self.weight.dtype)


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[:, None].float().to(x.device)
        position_ids_expanded = position_ids[None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(0, 1)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=0):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


class Gemma3TextModel(Gemma3PreTrainedModel):
    config: Gemma3TextConfig

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> BaseModelOutputWithPast:
        hidden_states = self.embed_tokens(input_ids)

        position_ids = torch.arange(len(input_ids), device=hidden_states.device)
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        window_size = (
            (
                self.config.sliding_window,
                self.config.sliding_window if self.config.use_bidirectional_attention else 0
            )
            if self.config.sliding_window is not None
            else None
        )
        mask_mapping = {
            "full_attention": create_packed_seqs_mask(cu_seqlens, causal=not self.config.use_bidirectional_attention, device=hidden_states.device),
            "sliding_attention": create_packed_seqs_mask(cu_seqlens, causal=not self.config.use_bidirectional_attention, device=hidden_states.device, window_size=window_size)
        }

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=mask_mapping[decoder_layer.attention_type],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                window_size=window_size if decoder_layer.attention_type == "sliding_attention" else (-1, -1),
            )

        hidden_states = self.norm((hidden_states))
        return hidden_states


class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    config: Gemma3TextConfig

    def __init__(self, config, fused_cross_entropy: bool = False):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.fused_cross_entropy = fused_cross_entropy
        if self.fused_cross_entropy:
            assert LIGER_KERNEL_AVAILABLE, "Liger kernel is not available."
            self.ligerCrossEntropy = LigerCrossEntropyLoss()

        self.post_init()

    def forward(
        self,
        x: torch.LongTensor,
        *,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        hidden_states = self.model(
            input_ids=x,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            if self.fused_cross_entropy:
                loss = self.ligerCrossEntropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
            else:
                loss = self.loss_function(
                    logits, labels, vocab_size=self.config.vocab_size
                )

        return logits, loss





# MultiModal
# class Gemma3Model(Gemma3PreTrainedModel):
#     _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
#     # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
#     accepts_loss_kwargs = False

#     def __init__(self, config: Gemma3Config):
#         super().__init__(config)
#         self.vision_tower = AutoModel.from_config(config=config.vision_config)
#         self.multi_modal_projector = Gemma3MultiModalProjector(config)
#         self.vocab_size = config.text_config.vocab_size

#         language_model = AutoModel.from_config(config=config.text_config)
#         self.language_model = language_model

#         self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.language_model.get_input_embeddings()

#     def set_input_embeddings(self, value):
#         self.language_model.set_input_embeddings(value)

#     def set_decoder(self, decoder):
#         self.language_model = decoder

#     def get_decoder(self):
#         return self.language_model

#     def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
#         """
#         Projects the last hidden state from the vision model into language model space.

#         Args:
#             pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
#                The tensors corresponding to the input images.
#         Returns:
#             image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
#         """
#         vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
#         image_features = self.multi_modal_projector(vision_outputs)
#         return image_features

#     def get_placeholder_mask(
#         self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
#     ):
#         """
#         Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
#         equal to the length of multimodal features. If the lengths are different, an error is raised.
#         """
#         if input_ids is None:
#             special_image_mask = inputs_embeds == self.get_input_embeddings()(
#                 torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
#             )
#             special_image_mask = special_image_mask.all(-1)
#         else:
#             special_image_mask = input_ids == self.config.image_token_id

#         n_image_tokens = special_image_mask.sum()
#         special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
#         n_image_features = image_features.shape[0] * image_features.shape[1]
#         if inputs_embeds[special_image_mask].numel() != image_features.numel():
#             raise ValueError(
#                 f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#             )
#         return special_image_mask


#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Cache] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **lm_kwargs,
#     ) -> tuple:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#             config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

#         Example:

#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

#         >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma32-3b-mix-224")
#         >>> processor = AutoProcessor.from_pretrained("google/gemma32-3b-mix-224")

#         >>> prompt = "Where is the cat standing?"
#         >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(**inputs,)
#         >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Where is the cat standing?\nsnow"
#         ```"""
#         if (input_ids is None) ^ (inputs_embeds is not None):
#             raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # Replace image id with PAD if the image token if OOV, to avoid index-errors
#         if input_ids is not None and self.config.image_token_id >= self.vocab_size:
#             special_image_mask = input_ids == self.config.image_token_id
#             llm_input_ids = input_ids.clone()
#             llm_input_ids[special_image_mask] = 0
#         else:
#             llm_input_ids = input_ids

#         if inputs_embeds is None:
#             inputs_embeds = self.get_input_embeddings()(llm_input_ids)

#         if cache_position is None:
#             past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#             cache_position = torch.arange(
#                 past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#             )

#         # Merge text and images
#         if pixel_values is not None:
#             image_features = self.get_image_features(pixel_values)
#             image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
#             special_image_mask = self.get_placeholder_mask(
#                 input_ids, inputs_embeds=inputs_embeds, image_features=image_features
#             )
#             inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

#         # It may already have been prepared by e.g. `generate`
#         if not isinstance(causal_mask_mapping := attention_mask, dict):
#             # Prepare mask arguments
#             mask_kwargs = {
#                 "config": self.config.get_text_config(),
#                 "input_embeds": inputs_embeds,
#                 "attention_mask": attention_mask,
#                 "cache_position": cache_position,
#                 "past_key_values": past_key_values,
#                 "position_ids": position_ids,
#             }
#             # NOTE: this `is_prefill` logic is not flawless, it fails when we're using a cache eagerly initialized
#             # (e.g. compiled prefill) AND `pixel_values` are not provided. Determining prefill in that case requires
#             # checking data values, which is not compile-compatible.
#             is_prefill = (
#                 not use_cache
#                 or past_key_values is None
#                 or not past_key_values.is_initialized
#                 or pixel_values is not None
#             )
#             if token_type_ids is not None and is_prefill:
#                 # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`

#                 # First find where a new image block starts: 1 if image and previous not image
#                 # The images cannot attend to future images, but can attend to all prev images and to itself
#                 # bidirectionally
#                 is_image = (token_type_ids == 1).to(cache_position.device)
#                 new_image_start = is_image & ~nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
#                 image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1
#                 image_group_ids = torch.where(
#                     is_image, image_group_ids, torch.full_like(token_type_ids, -1, device=is_image.device)
#                 )
#                 mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
#                     token_type_ids.to(cache_position.device), image_group_ids, self.config.mm_tokens_per_image
#                 )

#             # Create the masks
#             causal_mask_mapping = {
#                 "full_attention": create_causal_mask(**mask_kwargs),
#                 "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
#             }

#         outputs = self.language_model(
#             attention_mask=causal_mask_mapping,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#             cache_position=cache_position,
#             **lm_kwargs,
#         )

#         return (
#             outputs,
#             image_features if pixel_values is not None else None,
#         )

# class Gemma3MultiModalProjector(nn.Module):
#     def __init__(self, config: Gemma3Config):
#         super().__init__()

#         self.mm_input_projection_weight = nn.Parameter(
#             torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
#         )

#         self.mm_soft_emb_norm = Gemma3RMSNorm(
#             config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
#         )

#         self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
#         self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
#         self.kernel_size = self.patches_per_image // self.tokens_per_side
#         self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

#     def forward(self, vision_outputs: torch.Tensor):
#         batch_size, _, seq_length = vision_outputs.shape

#         reshaped_vision_outputs = vision_outputs.transpose(1, 2)
#         reshaped_vision_outputs = reshaped_vision_outputs.reshape(
#             batch_size, seq_length, self.patches_per_image, self.patches_per_image
#         )
#         reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

#         pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
#         pooled_vision_outputs = pooled_vision_outputs.flatten(2)
#         pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

#         normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

#         projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
#         return projected_vision_outputs.type_as(vision_outputs)

# def token_type_ids_mask_function(
#     token_type_ids: Optional[torch.Tensor],
#     image_group_ids: Optional[torch.Tensor],
#     tokens_per_image: int,
# ) -> Optional[Callable]:
#     """
#     This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
#     not start and end indices.
#     """
#     # Do not return an additional mask in this case
#     if token_type_ids is None:
#         return None

#     def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
#         # If it's 1 for both query and key/value, we are in an image block
#         # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
#         # Since vmap doesn't support `if statement` we workaround it with `torch.where`
#         safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
#         token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
#         token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)

#         image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_idx]
#         image_group_ids_at_kv_idx = torch.where(kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1)

#         is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)
#         same_image_block = image_group_ids[batch_idx, q_idx] == image_group_ids_at_kv_idx

#         # This is bidirectional attention whenever we are dealing with image tokens
#         return is_image_block & same_image_block

#     return inner_mask




__all__ = [
    "Gemma3PreTrainedModel",
    "Gemma3TextModel",
    "Gemma3ForCausalLM",
    # "Gemma3Model",
]
