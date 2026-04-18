"""Core implementation of the naive Triton FlashAttention forward kernel."""

import torch
import triton
import triton.language as tl

__all__ = ["Attention", "attention", "DEVICE", "get_fwd_config"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NOTE: this function can be overwritten at runtime to use your custom config.
def get_fwd_config(B, H, M, N, D, causal):
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            if D <= 64:
                block_m, block_n, num_stages, num_warps = 128, 64, 3, 4
            else:
                if M <= 1024:
                    block_m, block_n, num_stages, num_warps = 128, 32, 3, 4
                else:
                    block_m, block_n, num_stages, num_warps = 128, 128, 3, 8
        else:
            if D <= 64:
                block_m, block_n, num_stages, num_warps = 128, 64, 4, 4
            else:
                if M <= 1024:
                    block_m, block_n, num_stages, num_warps = 128, 32, 2, 4
                else:
                    block_m, block_n, num_stages, num_warps = 128, 128, 3, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                block_m, block_n, num_stages, num_warps = 128, 64, 3, 4
            else:
                block_m, block_n, num_stages, num_warps = 128, 32, 2, 4
        else:
            if D <= 64:
                block_m, block_n, num_stages, num_warps = 64, 64, 3, 4
            else:
                block_m, block_n, num_stages, num_warps = 128, 32, 2, 4
    else:
        block_m, block_n, num_stages, num_warps = 32, 32, 1, 4

    return block_m, block_n, num_stages, num_warps

@triton.jit
def _flash_attn_inner(
    q, m_i, l_i, acc, q_scale,
    seq_id,
    qkv_offset, 
    k, 
    v,  
    stride_batch, stride_head, stride_seq, stride_dim,
    stage: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
):
    """
    FlashAttention inner loop.

    1. Iterate kv blocks along the sequence dimension.
    2. Update online softmax state m_i / l_i and acc.
    3. In stage 2, apply causal masking on the diagonal block.
    """

    if stage == 1:
        lo, hi = 0, seq_id * B_r
    elif stage == 2:
        lo, hi = seq_id * B_r, (seq_id+1) * B_r
        lo = tl.multiple_of(lo, B_r)
    else:
        lo, hi = 0, SEQ_LEN

    input_dtype = v.dtype.element_ty

    offs_qblock = seq_id * B_r + tl.arange(0, B_r)
    offs_seq = tl.arange(0, B_c)
    offs_dim = tl.arange(0, HEAD_DIM)
    k_offset = offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim + qkv_offset
    v_offset = offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim + qkv_offset


    for start_n in tl.range(lo, hi, B_c):
        start_n = tl.multiple_of(start_n, B_c)
        k_ptr = k + k_offset + start_n * stride_seq
        v_ptr = v + v_offset + start_n * stride_seq  

        k_mask = (start_n + offs_seq[:, None]) < SEQ_LEN
        k_block = tl.load(k_ptr, mask=k_mask, other=0.0)
        qk = tl.dot(q, tl.trans(k_block))

        if stage == 2:
            mask_causal = offs_qblock[:, None] >= (start_n + tl.arange(0, B_c))[None, :]
            qk = qk * q_scale + tl.where(mask_causal, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * q_scale)
            qk = qk * q_scale - m_ij[:, None]

        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_mask = (start_n + offs_seq[:, None]) < SEQ_LEN
        v_block = tl.load(v_ptr, mask=v_mask, other=0.0)
        p = p.to(input_dtype)
        acc = tl.dot(p, v_block, acc)

        m_i = m_ij

    return acc, m_i, l_i

@triton.jit
def _flash_attn_impl(
    q, k, v, o, lse,      # lse用于backward pass
    stride_batch, stride_head, stride_seq, stride_dim,
    sm_scale,
    HEAD_NUM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    stage: tl.constexpr,
) :
    """Naive FlashAttention forward implementation."""

    input_dtype = q.dtype.element_ty
    tl.static_assert(B_r <= SEQ_LEN, "B_r subblock should be smaller than seqlen")
    tl.static_assert(B_c <= SEQ_LEN, "B_c subblock should be smaller than seqlen")

    seq_id = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)

    batch_id = batch_head_id // HEAD_NUM
    head_id = batch_head_id % HEAD_NUM  

    qkv_offset = (batch_id.to(tl.int64) * stride_batch) + (head_id.to(tl.int64) * stride_head)

    offs_seq = seq_id * B_r + tl.arange(0, B_r)
    offs_dim = tl.arange(0, HEAD_DIM)
    q_ptr = q + qkv_offset + offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    q_mask = offs_seq[:, None] < SEQ_LEN
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    m_i = tl.zeros([B_r], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([B_r], dtype=tl.float32) + 1.0  
    acc = tl.zeros([B_r, HEAD_DIM], dtype=tl.float32)
    q_scale = sm_scale * 1.44269504

    if stage & 1:  
        acc, m_i, l_i = _flash_attn_inner(
            q, m_i, l_i, acc, q_scale,
            seq_id,
            qkv_offset, 
            k, 
            v,  
            stride_batch, stride_head, stride_seq, stride_dim,
            4 - stage, 
            HEAD_NUM,
            SEQ_LEN,
            HEAD_DIM,
            B_r,
            B_c,
        )

    if stage & 2:
        acc, m_i, l_i = _flash_attn_inner(
            q, m_i, l_i, acc, q_scale,
            seq_id,
            qkv_offset, 
            k, 
            v,
            stride_batch, stride_head, stride_seq, stride_dim,
            2,
            HEAD_NUM,
            SEQ_LEN,
            HEAD_DIM,
            B_r,
            B_c,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    lse_ptr = lse + batch_head_id * SEQ_LEN + offs_seq
    lse_mask = offs_seq < SEQ_LEN
    tl.store(lse_ptr, m_i, mask=lse_mask)

    o_ptr = o + qkv_offset + offs_seq[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    o_mask = offs_seq[:, None] < SEQ_LEN
    tl.store(o_ptr, acc.to(input_dtype), mask=o_mask)


class Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                sm_scale: float,
                causal = True,
                ):

        assert q.is_contiguous()
        assert q.dim() == 4

        batch, head_num, seq_len, head_dim = q.shape
        o = torch.empty_like(q)
        lse = torch.empty(
            (batch, head_num, seq_len),
            device=DEVICE,
            dtype=torch.float32,
        )

        block_m, block_n, num_stages, num_warps = get_fwd_config(
            batch,
            head_num,
            seq_len,
            seq_len,
            head_dim,
            causal,
        )

        grid = (triton.cdiv(seq_len, block_m), batch * head_num)

        _flash_attn_impl[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            sm_scale,
            HEAD_NUM=head_num,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            B_r=block_m,
            B_c=block_n,
            stage=3 if causal else 1,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = head_dim
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx,
                 grad_o: torch.Tensor):
        raise NotImplementedError("Naive Triton FlashAttention backward is not implemented.")
        
attention = Attention.apply
