import argparse
import math

import torch

import flag_attn


CLIFF_PRESETS = {
    "d64-noncausal-precliff": {
        "n_ctx": 4096,
        "d_head": 64,
        "causal": False,
    },
    "d64-noncausal-cliff": {
        "n_ctx": 8192,
        "d_head": 64,
        "causal": False,
    },
    "d64-causal-pretail": {
        "n_ctx": 16384,
        "d_head": 64,
        "causal": True,
    },
    "d64-causal-tail-cliff": {
        "n_ctx": 32768,
        "d_head": 64,
        "causal": True,
    },
    "d128-causal-pretail": {
        "n_ctx": 16384,
        "d_head": 128,
        "causal": True,
    },
    "d128-causal-tail-cliff": {
        "n_ctx": 32768,
        "d_head": 128,
        "causal": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Microkernel profiling harness for flash attention forward kernels.",
    )
    parser.add_argument(
        "--provider",
        choices=["flag_attn", "naive"],
        default=None,
        help="Implementation to profile.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CLIFF_PRESETS),
        default=None,
        help="Named cliff point from the existing benchmark results.",
    )
    parser.add_argument("--n-ctx", type=int, default=None, help="Sequence length.")
    parser.add_argument("--d-head", type=int, default=None, help="Head dimension.")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking. Ignored when the preset already fixes it.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Defaults to the same rule as flash_benchmark.py.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="Number of heads. Defaults to the same rule as flash_benchmark.py.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations before profiling.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=3,
        help="Iterations inside the profiling region.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--print-tensor-shapes",
        action="store_true",
        help="Print resolved tensor shapes before running.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available preset names and exit.",
    )
    return parser.parse_args()


def resolve_case(args):
    if args.preset is not None:
        preset = CLIFF_PRESETS[args.preset]
        n_ctx = preset["n_ctx"]
        d_head = preset["d_head"]
        causal = preset["causal"]
    else:
        if args.n_ctx is None or args.d_head is None:
            raise ValueError("Either --preset or both --n-ctx and --d-head are required.")
        n_ctx = args.n_ctx
        d_head = args.d_head
        causal = args.causal

    batch = args.batch if args.batch is not None else max(1, 32768 // n_ctx)
    heads = args.heads if args.heads is not None else max(1, 2048 // d_head)
    return n_ctx, d_head, causal, batch, heads


def get_dtype(dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_inputs(batch, heads, n_ctx, d_head, dtype):
    q = torch.randn((batch, heads, n_ctx, d_head), device="cuda", dtype=dtype)
    k = torch.randn((batch, heads, n_ctx, d_head), device="cuda", dtype=dtype)
    v = torch.randn((batch, heads, n_ctx, d_head), device="cuda", dtype=dtype)
    return q, k, v


def build_forward(provider, q, k, v, causal, sm_scale):
    if provider == "flag_attn":
        return lambda: flag_attn.flash_attention(q, k, v, causal=causal, sm_scale=sm_scale)
    if provider == "naive":
        return lambda: flag_attn.naive_attention(q, k, v, sm_scale, causal)
    raise ValueError(f"Unsupported provider: {provider}")


def nvtx_push(message):
    torch.cuda.nvtx.range_push(message)


def nvtx_pop():
    torch.cuda.nvtx.range_pop()


def cuda_profiler_start():
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop():
    torch.cuda.cudart().cudaProfilerStop()


def main():
    args = parse_args()
    if args.list_presets:
        for preset_name, preset in sorted(CLIFF_PRESETS.items()):
            print(
                f"{preset_name}: n_ctx={preset['n_ctx']} "
                f"d_head={preset['d_head']} causal={preset['causal']}"
            )
        return

    if args.provider is None:
        raise ValueError("--provider is required unless --list-presets is used.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    dtype = get_dtype(args.dtype)
    n_ctx, d_head, causal, batch, heads = resolve_case(args)
    sm_scale = 1.0 / math.sqrt(d_head)

    if args.print_tensor_shapes:
        print(
            f"provider={args.provider} preset={args.preset} "
            f"batch={batch} heads={heads} n_ctx={n_ctx} d_head={d_head} "
            f"causal={causal} dtype={dtype}"
        )

    q, k, v = build_inputs(batch, heads, n_ctx, d_head, dtype)
    fn = build_forward(args.provider, q, k, v, causal, sm_scale)
    case_name = args.preset or f"d{d_head}-n{n_ctx}-causal-{int(causal)}"
    range_prefix = f"flash_microkernel/{args.provider}/{case_name}"

    with torch.inference_mode():
        nvtx_push(f"{range_prefix}/compile")
        fn()
        torch.cuda.synchronize()
        nvtx_pop()

        nvtx_push(f"{range_prefix}/warmup")
        for _ in range(args.warmup_iters):
            fn()
        torch.cuda.synchronize()
        nvtx_pop()

        cuda_profiler_start()
        nvtx_push(f"{range_prefix}/profile")
        for _ in range(args.profile_iters):
            fn()
        torch.cuda.synchronize()
        nvtx_pop()
        cuda_profiler_stop()

    print(
        f"profiled provider={args.provider} case={case_name} batch={batch} heads={heads} "
        f"n_ctx={n_ctx} d_head={d_head} causal={causal} dtype={args.dtype} "
        f"warmup_iters={args.warmup_iters} profile_iters={args.profile_iters}"
    )


if __name__ == "__main__":
    main()
