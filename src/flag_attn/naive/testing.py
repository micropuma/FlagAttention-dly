"""Precision tests for the naive Triton FlashAttention implementation."""

import pytest
import torch

from flag_attn.naive.core import DEVICE, attention


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [2, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_op(batch, seq_len, head_dim, num_heads, dtype, device=DEVICE):
    """Compare Triton FlashAttention output against a naive PyTorch reference."""

    torch.manual_seed(20)
    q = torch.empty(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(
        mean=0.0,
        std=0.5,
    )
    k = torch.empty(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(
        mean=0.0,
        std=0.5,
    )
    v = torch.empty(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(
        mean=0.0,
        std=0.5,
    )
    sm_scale = 0.5

    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v).to(dtype)

    tri_out = attention(q, k, v, sm_scale).to(dtype)
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=0)
