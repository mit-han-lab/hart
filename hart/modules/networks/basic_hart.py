"""This file contains code for basic blocks in HART Transformer.

This file is adopted and modified from https://github.com/FoundationVision/VAR/blob/main/models/basic_var.py
"""

import functools
import math
from typing import Tuple, Union

import hart_backend.fused_kernels
import torch
import torch.nn as nn
import torch.nn.functional as F

from hart.modules.networks.utils import DropPath, drop_path

# from visualizer import get_local

# this file only provides the 3 blocks used in VAR transformer
__all__ = [
    "FFN",
    "TimestepEmbedder",
    "LlamaRMSNorm",
    "LlamaRMSNormFused",
    "LlamaMLP",
    "AdaLNSelfAttn",
    "AdaLNBeforeHead",
]


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    pass
# automatically import faster attention implementations
try:
    import xformers
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func  # qkv: BLHc, ret: BLHcq
except ImportError:
    pass
try:
    from torch.nn.functional import (
        scaled_dot_product_attention as slow_attn,  # q, k, v: BHLc
    )
except ImportError:

    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None:
            attn.add_(attn_mask)
        return (
            F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True)
            if dropout_p > 0
            else attn.softmax(dim=-1)
        ) @ value


@functools.cache
def get_position_ids_1d(batch_size, L, device):
    # [batch_size, L]
    return torch.arange(L, device=device).unsqueeze(0).repeat(batch_size, 1)


@functools.cache
def get_position_ids(batch_size, patch_nums, device, si=-1, m_maskgit=None):
    # [batch_size, L]
    all_position_ids = []
    largest_patch_num = patch_nums[-1]
    if si == -1:
        pns = patch_nums
    else:
        pns = patch_nums[si : si + 1]
    for level_idx in range(len(pns)):
        patch_num = pns[level_idx]
        _x = torch.arange(patch_num, device=device)
        _y = torch.arange(patch_num, device=device)
        # [pn, pn, 2]
        cartesian = torch.stack(torch.meshgrid(_x, _y, indexing="ij"), dim=-1)
        # normalize to the size in the largest feature map
        coords = cartesian / patch_num * largest_patch_num
        # [pn * pn, 2]
        coords = coords.reshape(-1, 2)
        all_position_ids.append(coords)
    # [batch_size, L, 2]
    pos_ids = torch.cat(all_position_ids, 0).unsqueeze(0).repeat(batch_size, 1, 1)
    if m_maskgit is None:
        return pos_ids
    pos_ids = pos_ids[m_maskgit]
    return pos_ids.reshape(batch_size, -1, pos_ids.shape[-1])


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# unsqueeze_dim=2 because by default our qk has shape [batch_size, seq_len, heads, head_dim]
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
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


def context_pooling(context_tokens, context_mask=None, mode="avg"):
    # context_tokens: [batch, context_tokens, embed_dim]
    # context_mask: [batch, context_tokens]
    if len(context_tokens.shape) == 2:
        # C2I
        return context_tokens
    assert len(context_tokens.shape) == 3 and context_tokens.shape[1] > 1
    if mode == "avg":
        c_mask = context_mask.unsqueeze(-1)
        # [batch, context_tokens, embed_dim]
        condition = context_tokens * c_mask.to(context_tokens.dtype)
        # [batch, 1, embed_dim] => averaging
        condition = condition / c_mask.sum(1).clamp_(1).unsqueeze(1)
        # [batch, 1, embed_dim]
        condition = condition.sum(1)
    elif mode == "max":
        # [batch, 1, embed_dim]
        condition = context_tokens.max(1, keepdims=False).values
    else:
        raise NotImplementedError
    return condition


@functools.cache
def get_xattn_mask(context_mask):
    return context_mask.sum(1).tolist()


class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(t, freqs, False)
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_forward_func(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(
                grad_output, freqs, False
            )
        elif ctx.tensor_format == "bshd":
            grad_input = hart_backend.fused_kernels.fused_rope_backward_func(
                grad_output.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        return grad_input, None, None


class FusedRoPEFuncWithPos(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,  # [B, S, D]
        tensor_format: str = "sbhd",
        # cu_seqlens: Union[torch.Tensor, None] = None,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(
                t, freqs, False
            )
        elif tensor_format == "bshd":
            output = hart_backend.fused_kernels.fused_rope_with_pos_forward_func(
                t.transpose(0, 1), freqs, True
            ).transpose(0, 1)
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs)
        ctx.tensor_format = tensor_format

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        raise NotImplementedError("Not implemented yet")
        # freqs, = ctx.saved_tensors
        # if ctx.tensor_format == "sbhd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output, freqs, False)
        # elif ctx.tensor_format == "bshd":
        #     grad_input = hart_backend.fused_kernels.fused_rope_backward_func(grad_output.transpose(0, 1), freqs, True).transpose(0, 1)
        # else:
        #     raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")

        # return grad_input, None, None


class FusedLlamaRotaryEmbedding1D(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        seq = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)
        self.embs = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None, tensor_format="bshd"):
        self.embs = self.embs.to(x.device)
        # print(self.embs)
        # print(self.embs.shape)
        # exit()
        return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


class FusedLlamaRotaryEmbedding1DWithPos(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        seq = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)

        self.embs = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            # print(self.embs.shape)
            # print(context_position_ids.shape)
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            embs = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)
        else:  # Original impl
            self.embs = self.embs.to(x.device)
            return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


class FusedLlamaRotaryEmbedding2DWithPos(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Haotian: now we have two directions x and y so inv_freq has a stride 4

        # NOTE: Shang: freq stride is 4 rather than 2. While freq is normalized by dim.
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, seq_len=None, position_ids=None, tensor_format="bshd"):
        if position_ids is not None:
            inv_freq_expanded = (
                self.inv_freq[None, :, None, None]
                .float()
                .expand(position_ids.shape[0], -1, 1, 1)
                .repeat(1, 1, 1, 2)
            )
            # position_ids_expanded: [bs, 1, seq_len, 2]
            position_ids_expanded = position_ids[:, None, :].float()
            inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
            position_ids_expanded = position_ids_expanded.permute(
                0, 3, 1, 2
            ).contiguous()

            device_type = x.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                # freqs: [bs, 2, seq_len, head_size // 4]
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(2, 3)
                embs = torch.cat((freqs, freqs), dim=-1)

                embs = embs.transpose(2, 1).contiguous()
                embs = embs.reshape(embs.size(0), embs.size(1), -1)

            return FusedRoPEFuncWithPos.apply(x, embs, tensor_format)

        else:  # Original impl
            raise NotImplementedError("Not implemented yet")
            self.embs = self.embs.to(x.device)
            return FusedRoPEFunc.apply(x, self.embs[:seq_len], tensor_format)


# From Junsong and Enze's EfficientDiT codebase.
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        flag = False
        if len(t.shape) == 2:
            flag = True
            t = t[0]
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.dtype
        )
        t_emb = self.mlp(t_freq)
        if not flag:
            return t_emb
        else:
            return t_emb.unsqueeze(0)

    @property
    def dtype(self):
        # return the data type of this model
        return next(self.parameters()).dtype


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        scale = kwargs.get("scale", 1.0)
        self.weight = nn.Parameter(torch.ones(hidden_size) * scale)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRMSNormFused(nn.Module):
    # Shang: kwargs for elementwise_affine
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, use_quant: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_quant = use_quant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (
            torch.empty_like(x, dtype=torch.int8)
            if self.use_quant
            else torch.empty_like(x)
        )
        self.weight.data = self.weight.data.to(x)
        hart_backend.fused_kernels.rms_norm(
            out, x, self.weight.data, self.variance_epsilon, self.use_quant
        )
        return out


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding1D(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 2]
        # inv_freq_expanded: [bs, head_size // 2, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        # position_ids_expanded: [bs, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, seq_len, head_size // 2]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, seq_len, head_size]
            cos = emb.cos()
            sin = emb.sin()
        # [bs, seq_len, head_size]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# from hf transformers:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 4, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # position_ids: [bs, seq_len]
        # inv_freq: [head_size // 4]
        # inv_freq_expanded: [bs, head_size // 4, 1, 1]
        inv_freq_expanded = (
            self.inv_freq[None, :, None, None]
            .float()
            .expand(position_ids.shape[0], -1, 1, 1)
            .repeat(1, 1, 1, 2)
        )
        # position_ids_expanded: [bs, 1, seq_len, 2]
        position_ids_expanded = position_ids[:, None, :].float()
        inv_freq_expanded = inv_freq_expanded.permute(0, 3, 1, 2).contiguous()
        position_ids_expanded = position_ids_expanded.permute(0, 3, 1, 2).contiguous()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [bs, 2, seq_len, head_size // 4]
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [bs, 2, seq_len, head_size // 2]
            cos = emb.cos()
            sin = emb.sin()
            # [bs, seq_len, 2, head_size // 2]
            cos = cos.transpose(2, 1).contiguous()
            sin = sin.transpose(2, 1).contiguous()
            cos = cos.reshape(cos.size(0), cos.size(1), -1)
            sin = sin.reshape(sin.size(0), sin.size(1), -1)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_norm=False,
        **block_kwargs,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.init_weights()

    def init_weights(self):
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            raise NotImplementedError
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
        act_func="gelu",
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if act_func == "gelu":
            self.act = nn.GELU(approximate="tanh")
        elif act_func == "silu":
            self.act = nn.SiLU()
        else:
            raise NotImplementedError
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class LlamaMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.gate_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.up_proj = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.down_proj = nn.Linear(self.hidden_features, self.out_features, bias=False)
        self.act_fn = nn.SiLU()
        self.fused_mlp_func = None

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SelfAttention(nn.Module):
    def __init__(
        self,
        block_idx,
        embed_dim=768,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_l2_norm=False,
        flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(
            torch.zeros(embed_dim)
        )
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            False  # flash_if_available and memory_efficient_attention is not None
        )

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self, x, attn_bias, si=-1, context_position_ids=None, context_mask=None
    ):
        B, L, C = x.shape

        qkv = F.linear(
            input=x,
            weight=self.mat_qkv.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


class LlamaAttention(nn.Module):
    def __init__(
        self,
        block_idx,
        patch_nums,
        embed_dim=768,
        num_heads=12,
        attn_drop=0.0,
        proj_drop=0.0,
        max_position_embeddings=4096,
        rope_theta=10000,
        flash_if_available=True,
        attn_l2_norm=False,
        context_token=0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert patch_nums is not None
        self.context_token = context_token
        self.patch_nums = patch_nums
        self.block_idx, self.num_heads, self.head_dim = (
            block_idx,
            num_heads,
            embed_dim // num_heads,
        )  # =64

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attn_l2_norm = False

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.rotary_emb_fused_with_pos = FusedLlamaRotaryEmbedding2DWithPos(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        if context_token != 0:
            self.context_rotary_emb = LlamaRotaryEmbedding1D(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
            self.context_rotary_emb_fused_with_pos = FusedLlamaRotaryEmbedding1DWithPos(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )

        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                requires_grad=True,
            )
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(
            torch.zeros(embed_dim)
        )
        self.register_buffer("zero_k_bias", torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = (
            nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        )
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = (
            False  # flash_if_available and memory_efficient_attention is not None
        )

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    # @get_local('attn')
    def forward(
        self,
        x,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
    ):
        B, L, C = x.shape
        # [B, L, 2]
        if self.context_token == 0:
            position_ids = get_position_ids(
                B, self.patch_nums, x.device, si=si, m_maskgit=m_maskgit
            )
        else:
            # text to image
            # level 0 does not appear in the position_ids
            # since it is included in context tokens
            # should be 679 tokens for 16x16 latent w/ default 10-stage VAR
            if si == -1:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
            elif si > 0:
                _position_ids = get_position_ids(
                    B, self.patch_nums[1:], x.device, si=si - 1, m_maskgit=m_maskgit
                )
                # largest position_id
                position_ids = _position_ids + context_position_ids[:, -1].unsqueeze(
                    -1
                ).unsqueeze(-1)
        # [B, context, 2]
        # if self.context_token > 0 and si <= 0:
        #     context_position_ids = get_position_ids_1d(B, self.context_token, x.device)

        qkv = F.linear(
            input=x,
            weight=self.qkv_proj.weight,
            bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
        ).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = (
            self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        )
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1  # q or k or v: BLHc
            dim_unsqueeze = 2
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2  # q or k or v: BHLc
            dim_unsqueeze = 1

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform:
                scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        ################## Use naive rotary embedding ##################
        # apply position embedding to visual tokens
        if self.context_token == 0:
            # position_ids exist for c2i
            # or t2i when stage id != 0
            # or t2i training phase (stage id = -1)
            cos, sin = self.rotary_emb(v, position_ids)
        elif self.context_token > 0:
            if si == -1:
                # training, all tokens
                cos, sin = self.rotary_emb(v, position_ids)
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
            elif si == 0:
                # inference step 1, only context tokens
                cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
                cos, sin = cos_c, sin_c
            else:
                # si > 0, no need to add rotary emb for context
                # inference step > 1, only new tokens
                cos, sin = self.rotary_emb(v, position_ids)
        else:
            print("Context token cannot be negative", self.context_token)
            raise NotImplementedError
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=dim_unsqueeze)
        ################## Use naive rotary embedding ##################

        ################## Use optimized rotary embedding ##################
        # if self.context_token == 0:
        #     # position_ids exist for c2i
        #     # or t2i when stage id != 0
        #     # or t2i training phase (stage id = -1)
        #     cos, sin = self.rotary_emb(v, position_ids)
        #     q, k = apply_rotary_pos_emb(
        #         q,
        #         k,
        #         cos,
        #         sin,
        #         unsqueeze_dim=dim_unsqueeze
        #     )
        # elif self.context_token > 0:
        #     if si == -1:
        #         # training, all tokens
        #         cos, sin = self.rotary_emb(v, position_ids)
        #         cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
        #         cos, sin = torch.cat([cos_c, cos], 1), torch.cat([sin_c, sin], 1)
        #         q, k = apply_rotary_pos_emb(
        #             q,
        #             k,
        #             cos,
        #             sin,
        #             unsqueeze_dim=dim_unsqueeze
        #         )
        #     elif si == 0:
        #         # inference step 1, only context tokens
        #         # NOTE: This branch for prompt
        #         # cos_c, sin_c = self.context_rotary_emb(v, context_position_ids)
        #         # cos, sin = cos_c, sin_c
        #         q = self.context_rotary_emb_fused_with_pos(q, position_ids=context_position_ids, tensor_format="bshd")
        #         k = self.context_rotary_emb_fused_with_pos(k, position_ids=context_position_ids, tensor_format="bshd")
        #     else:
        #         # NOTE: This branch for multi-scale generation
        #         # si > 0, no need to add rotary emb for context
        #         # inference step > 1, only new tokens
        #         # cos, sin = self.rotary_emb(v, position_ids)
        #         q = self.rotary_emb_fused_with_pos(q, position_ids=position_ids, tensor_format="bshd")
        #         k = self.rotary_emb_fused_with_pos(k, position_ids=position_ids, tensor_format="bshd")
        # else:
        #     print("Context token cannot be negative", self.context_token)
        #     raise NotImplementedError
        ################## Use optimized rotary embedding ##################

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                dropout_p=dropout_p,
                softmax_scale=self.scale,
            ).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(
                q.to(dtype=main_type),
                k.to(dtype=main_type),
                v.to(dtype=main_type),
                attn_bias=(
                    None
                    if attn_bias is None
                    else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1)
                ),
                p=dropout_p,
                scale=self.scale,
            ).view(B, L, C)
        else:
            oup = (
                slow_attn(
                    query=q,
                    key=k,
                    value=v,
                    scale=self.scale,
                    attn_mask=attn_bias,
                    dropout_p=dropout_p,
                )
                .transpose(1, 2)
                .reshape(B, L, C)
            )

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f"using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}"


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self,
        block_idx,
        last_drop_p,
        embed_dim,
        cond_dim,
        shared_aln: bool,
        norm_layer,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        attn_l2_norm=False,
        flash_if_available=False,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        gpt2_mlp_act_func="gelu",
        max_position_embeddings=4096,
        patch_nums=None,
        context_token=0,
        disable_aln=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
    ):
        super().__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.disable_aln = disable_aln
        self.sep_aln_pooling_mode = sep_aln_pooling_mode

        if attn_type == "gpt2":
            self.attn = SelfAttention(
                block_idx=block_idx,
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,
            )
        else:
            self.attn = LlamaAttention(
                block_idx=block_idx,
                patch_nums=patch_nums,
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
                max_position_embeddings=max_position_embeddings,
                rope_theta=10000,
                proj_drop=drop,
                flash_if_available=flash_if_available,
                context_token=context_token,
                attn_l2_norm=attn_l2_norm,
            )
        if mlp_type == "gpt2":
            self.ffn = FFN(
                in_features=embed_dim,
                hidden_features=round(embed_dim * mlp_ratio),
                drop=drop,
                fused_if_available=fused_if_available,
                act_func=gpt2_mlp_act_func,
            )
        elif mlp_type == "llama":
            # MLP ratio = 4: mul 8 / 3
            self.ffn = LlamaMLP(
                in_features=embed_dim,
                hidden_features=int((embed_dim * mlp_ratio * 2) / 3 + 255) // 256 * 256,
                out_features=embed_dim,
                drop=drop,
                fused_if_available=fused_if_available,
            )

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if not self.disable_aln:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        else:
            if self.shared_aln:
                self.scale_shift_table = nn.Parameter(
                    torch.randn(6, embed_dim) / embed_dim**0.5
                )
        self.fused_add_norm_fn = None
        self.use_cross_attn = use_cross_attn

        if self.use_cross_attn:
            self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)
        else:
            self.cross_attn = None

    def forward_function(
        self,
        x_BLC,
        cond_BD_or_gss,
        attn_bias,
        mask,
        context_position_ids=None,
        context_mask=None,
    ):
        return self(
            x=x_BLC,
            cond_BD=cond_BD_or_gss,
            attn_bias=attn_bias,
            m_maskgit=mask,
            context_position_ids=context_position_ids,
            context_mask=context_mask,
        )

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(
        self,
        x,
        cond_BD,
        attn_bias,
        si=-1,
        context_position_ids=None,
        context_mask=None,
        m_maskgit=None,
    ):  # C: embed_dim, D: cond_dim
        # We achieve multi-token conditioning through LLM attention mask.
        if not self.disable_aln:
            # if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            #     cond_BD = cond_BD.max(1, keepdims=True).values
            condition = context_pooling(
                cond_BD, context_mask=context_mask, mode=self.sep_aln_pooling_mode
            ).unsqueeze(1)

            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_lin(condition).view(-1, 1, 6, self.C).unbind(2)
            )
            x = x + self.drop_path(
                self.attn(
                    self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                    attn_bias=attn_bias,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                    si=si,
                    m_maskgit=m_maskgit,
                ).mul_(gamma1)
            )
            if self.use_cross_attn:
                # xattn_mask = get_xattn_mask(context_mask)
                x[:, cond_BD.size(1) :] += self.cross_attn(
                    x[:, cond_BD.size(1) :], cond_BD, None
                )
            x = x + self.drop_path(
                self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)
            )  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        else:
            if not self.shared_aln:
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                    )
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(self.ffn(self.ln_wo_grad(x)))
            else:
                # cond_BD: [batch, 1, embed_dim]
                condition = context_pooling(cond_BD, context_mask, mode="avg")
                # [batch, 6, embed_dim]
                adaln_modulator = self.scale_shift_table[None] + condition.unsqueeze(1)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = adaln_modulator.chunk(
                    6, dim=1
                )
                x = x + self.drop_path(
                    self.attn(
                        self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1),
                        attn_bias=attn_bias,
                        context_position_ids=context_position_ids,
                        context_mask=context_mask,
                        si=si,
                        m_maskgit=m_maskgit,
                    ).mul_(gamma1)
                )
                if self.use_cross_attn:
                    # xattn_mask = get_xattn_mask(context_mask)
                    x[:, cond_BD.size(1) :] += self.cross_attn(
                        x[:, cond_BD.size(1) :], cond_BD, None
                    )
                x = x + self.drop_path(
                    self.ffn(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(
                        gamma2
                    )
                )
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}"


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        # We achieve multi-token conditioning through LLM attention mask.
        if len(cond_BD.shape) == 3 and cond_BD.shape[1] > 1:
            cond_BD = cond_BD.max(1, keepdims=True).values

        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
