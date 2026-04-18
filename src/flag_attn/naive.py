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

def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

@triton.jit
def _flash_attn_inner(
    q, m_i, l_i, acc, q_scale,
    seq_id,
    kv_offset_k,
    kv_offset_v,
    k, 
    v,  
    stride_k_seq, stride_k_dim,
    stride_v_seq, stride_v_dim,
    stage: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    divisibility_n: tl.constexpr,
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

    # k should be loaded in a transposed layout for better memory access pattern in tl.dot
    k_offset = kv_offset_k + offs_seq[None, :] * stride_k_seq + offs_dim[:, None] * stride_k_dim
    v_offset = kv_offset_v + offs_seq[:, None] * stride_v_seq + offs_dim[None, :] * stride_v_dim    

    for start_n in tl.range(lo, hi, B_c):            # TODO(leon): tl.range() may casue more aggressive software pipelining, cause more compliated smem bank conflicts here
        tl.multiple_of(start_n, B_c)

        k_ptr = k + k_offset + start_n * stride_k_seq
        v_ptr = v + v_offset + start_n * stride_v_seq

        if not divisibility_n:
            k_mask = (start_n + offs_seq[None, :]) < SEQ_LEN
            k_block = tl.load(k_ptr, mask=k_mask, other=0.0, cache_modifier=".cg")
        else:
            k_block = tl.load(k_ptr, cache_modifier=".cg")

        # k is transposed in gmem -> smem load for better memory access pattern in tl.dot
        qk = tl.dot(q, k_block)

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

        if not divisibility_n:
            v_mask = (start_n + offs_seq[:, None]) < SEQ_LEN
            v_block = tl.load(v_ptr, mask=v_mask, other=0.0, cache_modifier=".cg")
        else:
            v_block = tl.load(v_ptr, cache_modifier=".cg")

        p = p.to(input_dtype)
        acc = tl.dot(p, v_block, acc)

        m_i = m_ij

    return acc, m_i, l_i

@triton.jit
def _flash_attn_impl(
    q, k, v, o, lse,      # lse用于backward pass
    stride_q_batch, stride_q_head, stride_q_seq, stride_q_dim,
    stride_k_batch, stride_k_head, stride_k_seq, stride_k_dim,
    stride_v_batch, stride_v_head, stride_v_seq, stride_v_dim,
    sm_scale,
    HEAD_NUM_Q: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    B_r: tl.constexpr,
    B_c: tl.constexpr,
    stage: tl.constexpr,
    divisibility_m: tl.constexpr,
    divisibility_n: tl.constexpr,
) :
    """Naive FlashAttention forward implementation."""

    input_dtype = q.dtype.element_ty
    tl.static_assert(B_r <= SEQ_LEN, "B_r subblock should be smaller than seqlen")
    tl.static_assert(B_c <= SEQ_LEN, "B_c subblock should be smaller than seqlen")

    seq_id = tl.program_id(axis=0)
    batch_head_id = tl.program_id(axis=1)

    batch_id = batch_head_id // HEAD_NUM_Q
    q_head_id = batch_head_id % HEAD_NUM_Q
    # In GQA/MQA, multiple query heads share one KV head.
    kv_head_id = q_head_id // NUM_GROUPS

    q_offset = (batch_id.to(tl.int64) * stride_q_batch) + (q_head_id.to(tl.int64) * stride_q_head)
    kv_offset_k = (batch_id.to(tl.int64) * stride_k_batch) + (kv_head_id.to(tl.int64) * stride_k_head)
    kv_offset_v = (batch_id.to(tl.int64) * stride_v_batch) + (kv_head_id.to(tl.int64) * stride_v_head)

    offs_seq = seq_id * B_r + tl.arange(0, B_r)
    offs_dim = tl.arange(0, HEAD_DIM)
    q_ptr = q + q_offset + offs_seq[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim

    if not divisibility_m:
        q_mask = offs_seq[:, None] < SEQ_LEN
        q = tl.load(q_ptr, mask=q_mask, other=0.0, cache_modifier=".cg")
    else:
        q = tl.load(q_ptr, cache_modifier=".cg")

    # dotI trick here, to help q be loaded into register/fragment that can boost following qk dot product performance.  
    if (HEAD_DIM < 128):
        identity = tl.where(
            offs_dim[:, None] == offs_dim[None, :],  
            tl.full((HEAD_DIM, HEAD_DIM), 1.0, dtype=input_dtype),
            tl.full((HEAD_DIM, HEAD_DIM), 0.0, dtype=input_dtype))
        q = tl.dot(q, identity).to(input_dtype)


    m_i = tl.zeros([B_r], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([B_r], dtype=tl.float32) + 1.0  
    acc = tl.zeros([B_r, HEAD_DIM], dtype=tl.float32)
    q_scale = sm_scale * 1.44269504

    if stage & 1:  
        acc, m_i, l_i = _flash_attn_inner(
            q, m_i, l_i, acc, q_scale,
            seq_id,
            kv_offset_k,
            kv_offset_v,
            k, 
            v,  
            stride_k_seq, stride_k_dim,
            stride_v_seq, stride_v_dim,
            4 - stage, 
            SEQ_LEN,
            HEAD_DIM,
            B_r,
            B_c,
            divisibility_n,
        )

    if stage & 2:
        acc, m_i, l_i = _flash_attn_inner(
            q, m_i, l_i, acc, q_scale,
            seq_id,
            kv_offset_k,
            kv_offset_v,
            k, 
            v,
            stride_k_seq, stride_k_dim,
            stride_v_seq, stride_v_dim,
            2,
            SEQ_LEN,
            HEAD_DIM,
            B_r,
            B_c,
            divisibility_n,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    lse_ptr = lse + batch_head_id * SEQ_LEN + offs_seq
    lse_mask = offs_seq < SEQ_LEN
    tl.store(lse_ptr, m_i, mask=lse_mask, cache_modifier=".cg")

    o_ptr = o + q_offset + offs_seq[:, None] * stride_q_seq + offs_dim[None, :] * stride_q_dim
    o_mask = offs_seq[:, None] < SEQ_LEN
    tl.store(o_ptr, acc.to(input_dtype), mask=o_mask, cache_modifier=".cg")


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
        assert k.dim() == 4
        assert v.dim() == 4

        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        batch, q_head_num, seq_len, head_dim = q.shape
        k_batch, kv_head_num, kv_seq_len, k_head_dim = k.shape
        v_batch, v_head_num, v_seq_len, v_head_dim = v.shape
        assert batch == k_batch == v_batch
        assert kv_head_num == v_head_num
        assert seq_len == kv_seq_len == v_seq_len
        assert head_dim == k_head_dim == v_head_dim
        assert q_head_num % kv_head_num == 0

        num_groups = q_head_num // kv_head_num
        o = torch.empty_like(q)
        lse = torch.empty(
            (batch, q_head_num, seq_len),
            device=DEVICE,
            dtype=torch.float32,
        )

        block_m, block_n, num_stages, num_warps = get_fwd_config(
            batch,
            q_head_num,
            seq_len,
            seq_len,
            head_dim,
            causal,
        )

        divisibility_m = seq_len % block_m == 0
        divisibility_n = seq_len % block_n == 0

        grid = (triton.cdiv(seq_len, block_m), batch * q_head_num)

        _flash_attn_impl[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            sm_scale,
            HEAD_NUM_Q=q_head_num,
            NUM_GROUPS=num_groups,
            SEQ_LEN=seq_len,
            HEAD_DIM=head_dim,
            B_r=block_m,
            B_c=block_n,
            stage=3 if causal else 1,
            num_stages=num_stages,
            num_warps=num_warps,
            divisibility_m=divisibility_m,
            divisibility_n=divisibility_n,
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
