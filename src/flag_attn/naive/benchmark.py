"""Benchmarks and profiling entrypoints for the naive Triton FlashAttention."""

import argparse
import os

import torch
import triton

from flag_attn.naive.core import DEVICE, attention


os.environ.setdefault("TRITON_CACHE_DIR", "./triton_cache_debug")

_BENCH_BATCH = 4
_BENCH_HEADS = 32
_BENCH_SEQ_LEN_XVALS = [2**i for i in range(7, 12)]
_BENCH_BATCH_XVALS = [1, 2, 4, 8, 16]
_BENCH_BATCH_SWEEP_SEQ = 128
_BENCH_DTYPES = [torch.float16, torch.bfloat16]


def _naive_attention(q, k, v, sm_scale, causal=True):
    """Naive PyTorch attention for benchmarking (O(N^2) memory)."""

    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        n = q.shape[2]
        mask = torch.tril(torch.ones(n, n, device=q.device))
        p[:, :, mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v)


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


def _build_seq_bench_configs():
    configs = []
    for head_dim in [64, 128]:
        for dtype in _BENCH_DTYPES:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["seq_len"],
                    x_vals=_BENCH_SEQ_LEN_XVALS,
                    line_arg="provider",
                    line_vals=["triton", "pytorch"],
                    line_names=["Triton FlashAttn", "PyTorch Naive"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="TFLOPS",
                    plot_name=f"flash-attn-B{_BENCH_BATCH}-H{_BENCH_HEADS}-D{head_dim}-{_dtype_name(dtype)}",
                    args={
                        "num_heads": _BENCH_HEADS,
                        "batch": _BENCH_BATCH,
                        "head_dim": head_dim,
                        "dtype": dtype,
                    },
                )
            )
    return configs


def _build_batch_bench_configs():
    configs = []
    for head_dim in [64, 128]:
        for dtype in _BENCH_DTYPES:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["batch"],
                    x_vals=_BENCH_BATCH_XVALS,
                    line_arg="provider",
                    line_vals=["triton", "pytorch"],
                    line_names=["Triton FlashAttn", "PyTorch Naive"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="TFLOPS",
                    plot_name=(
                        f"flash-attn-N{_BENCH_BATCH_SWEEP_SEQ}-H{_BENCH_HEADS}-"
                        f"D{head_dim}-{_dtype_name(dtype)}-batch-sweep"
                    ),
                    args={
                        "num_heads": _BENCH_HEADS,
                        "seq_len": _BENCH_BATCH_SWEEP_SEQ,
                        "head_dim": head_dim,
                        "dtype": dtype,
                    },
                )
            )
    return configs


_SEQ_BENCH_CONFIGS = _build_seq_bench_configs()


@triton.testing.perf_report(_SEQ_BENCH_CONFIGS)
def bench_flash_attn(batch, seq_len, head_dim, num_heads, provider, dtype=torch.float16, device=DEVICE):
    """Benchmark Triton FlashAttention vs naive PyTorch."""

    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    sm_scale = 1.3

    if provider == "triton":
        fn = lambda: attention(q, k, v, sm_scale)
    elif provider == "pytorch":
        mem_bytes = batch * num_heads * seq_len * seq_len * dtype.itemsize
        free_mem = torch.cuda.mem_get_info(device)[0]
        if mem_bytes * 3 > free_mem:
            return float("nan")
        fn = lambda: _naive_attention(q, k, v, sm_scale)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * batch * num_heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul * 0.5
    return total_flops * 1e-12 / (ms * 1e-3)


@triton.testing.perf_report(_build_batch_bench_configs())
def bench_flash_attn_batch(batch, seq_len, head_dim, num_heads, provider, dtype=torch.float16, device=DEVICE):
    return bench_flash_attn.fn(
        batch=batch,
        seq_len=seq_len,
        head_dim=head_dim,
        num_heads=num_heads,
        provider=provider,
        dtype=dtype,
        device=device,
    )


def _run_single_case(batch, seq_len, head_dim, num_heads, causal=True, warmup=10, dtype=torch.float16):
    """Run one concrete shape repeatedly so Nsight Compute can capture a stable case."""

    torch.manual_seed(0)
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE)
    sm_scale = 1.3

    for _ in range(max(warmup, 0)):
        attention(q, k, v, sm_scale, causal)
    torch.cuda.synchronize()

    attention(q, k, v, sm_scale, causal)
    torch.cuda.synchronize()


def _parse_args():
    parser = argparse.ArgumentParser(description="Naive Triton FlashAttention benchmark/profiling driver")
    parser.add_argument(
        "--mode",
        choices=["bench-seqlen", "bench-batch", "profile-once"],
        default="bench-seqlen",
        help=(
            "bench-seqlen: sweep sequence length; "
            "bench-batch: sweep batch size; "
            "profile-once: run one fixed case for Nsight Compute"
        ),
    )
    parser.add_argument("--save-dir", default="./flashattnv2", help="Directory to save benchmark artifacts")
    parser.add_argument("--batch", type=int, default=4, help="Batch size for profile-once mode")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for profile-once mode")
    parser.add_argument("--head-dim", type=int, default=128, choices=[64, 128], help="Head dimension")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16", help="Input dtype")
    parser.add_argument("--causal", action="store_true", help="Use causal attention in profile-once mode")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before the profiled run")
    return parser.parse_args()


def main():
    args = _parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    if args.mode == "profile-once":
        _run_single_case(
            batch=args.batch,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            causal=args.causal,
            warmup=args.warmup,
            dtype=dtype,
        )
        return

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.mode == "bench-seqlen":
        bench_flash_attn.run(print_data=True, save_path=args.save_dir)
    else:
        bench_flash_attn_batch.run(print_data=True, save_path=args.save_dir)


if __name__ == "__main__":
    main()
