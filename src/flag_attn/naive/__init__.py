"""Naive Triton FlashAttention package."""

from flag_attn.naive.core import Attention, attention, get_fwd_config

__all__ = ["Attention", "attention", "get_fwd_config"]
