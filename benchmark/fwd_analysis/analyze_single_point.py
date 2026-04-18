import argparse
import math
import pathlib
import shlex
import subprocess
import sys

import torch
import triton

from flash_microkernel_profile import (
    CLIFF_PRESETS,
    build_forward,
    build_inputs,
    get_dtype,
)
from run_flash_cliff_ncu import HARNESS_SCRIPT, TARGETED_SECTIONS, encode_nvtx_push_pop_range


def parse_bool_token(value):
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "causal"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "noncausal", "non-causal"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Unsupported causal flag {value!r}. Use true/false."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare steady-state benchmark timing and lightweight Nsight Compute "
            "profiling for a single flash attention shape."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CLIFF_PRESETS),
        default=None,
        help="Named cliff point from flash_microkernel_profile.py.",
    )
    parser.add_argument("--n-ctx", type=int, default=None, help="Sequence length.")
    parser.add_argument("--d-head", type=int, default=None, help="Head dimension.")
    parser.add_argument(
        "--causal",
        type=parse_bool_token,
        default=None,
        help="Whether to use causal masking when not using --preset.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Input dtype.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["flag_attn", "naive"],
        default=["flag_attn", "naive"],
        help="Implementations to benchmark/profile.",
    )
    parser.add_argument(
        "--bench-warmup",
        type=int,
        default=10,
        help="Warmup iterations for steady-state benchmarking.",
    )
    parser.add_argument(
        "--bench-rep",
        type=int,
        default=10,
        help="Repetition count for steady-state benchmarking.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Override batch size. Defaults to flash_benchmark.py heuristic.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=None,
        help="Override head count. Defaults to flash_benchmark.py heuristic.",
    )
    parser.add_argument(
        "--run-ncu",
        action="store_true",
        help="Run lightweight Nsight Compute after printing benchmark results.",
    )
    parser.add_argument(
        "--ncu-dry-run",
        action="store_true",
        help="Print the lightweight NCU commands without executing them.",
    )
    parser.add_argument(
        "--ncu-bin",
        default="ncu",
        help="Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run the profiling harness.",
    )
    parser.add_argument(
        "--ncu-output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for generated NCU outputs. Defaults under benchmark/results_ncu_single_point/.",
    )
    parser.add_argument(
        "--ncu-warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations inside the Nsight Compute harness.",
    )
    parser.add_argument(
        "--ncu-profile-iters",
        type=int,
        default=1,
        help="Profile iterations inside the Nsight Compute harness.",
    )
    parser.add_argument(
        "--full-sections",
        action="store_true",
        help="Use `--set full` for NCU instead of the lightweight targeted sections.",
    )
    return parser.parse_args()


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def resolve_case(args):
    if args.preset is not None:
        preset = CLIFF_PRESETS[args.preset]
        n_ctx = preset["n_ctx"]
        d_head = preset["d_head"]
        causal = preset["causal"]
        case_name = args.preset
        preset_name = args.preset
    else:
        if args.n_ctx is None or args.d_head is None or args.causal is None:
            raise ValueError(
                "Either --preset or all of --n-ctx, --d-head and --causal are required."
            )
        n_ctx = args.n_ctx
        d_head = args.d_head
        causal = args.causal
        case_name = f"d{d_head}-n{n_ctx}-causal-{int(causal)}"
        preset_name = None
    batch = args.batch if args.batch is not None else max(1, 32768 // n_ctx)
    heads = args.heads if args.heads is not None else max(1, 2048 // d_head)
    return case_name, preset_name, n_ctx, d_head, causal, batch, heads


def build_ncu_profile_command(
    args,
    provider,
    case_name,
    preset_name,
    n_ctx,
    d_head,
    causal,
    report_stem,
):
    nvtx_range = f"flash_microkernel/{provider}/{case_name}/profile"
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--nvtx",
        "--nvtx-include",
        encode_nvtx_push_pop_range(nvtx_range),
        "-f",
        "-o",
        str(report_stem),
    ]
    if args.full_sections:
        cmd.extend(["--set", "full"])
    else:
        for section_name in TARGETED_SECTIONS:
            cmd.extend(["--section", section_name])
    cmd.extend(
        [
            args.python_bin,
            str(HARNESS_SCRIPT),
            "--provider",
            provider,
            "--dtype",
            args.dtype,
            "--warmup-iters",
            str(args.ncu_warmup_iters),
            "--profile-iters",
            str(args.ncu_profile_iters),
            "--print-tensor-shapes",
        ]
    )
    if preset_name is not None:
        cmd.extend(["--preset", preset_name])
    else:
        cmd.extend(["--n-ctx", str(n_ctx), "--d-head", str(d_head)])
        if causal:
            cmd.append("--causal")
    return cmd


def export_text_report(ncu_bin, report_path, output_path):
    commands = [
        [
            ncu_bin,
            "--import",
            str(report_path),
            "--page",
            "details",
            "--print-summary",
            "per-kernel",
            "--print-details",
            "header",
            "--print-nvtx-rename",
            "kernel",
        ],
        [
            ncu_bin,
            "--import",
            str(report_path),
            "--page",
            "details",
            "--print-summary",
            "per-kernel",
            "--print-details",
            "body",
            "--print-nvtx-rename",
            "kernel",
        ],
    ]
    sections = []
    for label, cmd in zip(["header", "body"], commands):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"NCU import failed for {label}: {shell_join(cmd)}\n{result.stderr.strip()}"
            )
        content = result.stdout.strip()
        if content:
            sections.append(f"## {label}\n\n{content}")
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def export_raw_csv(ncu_bin, report_path, output_path):
    cmd = [
        ncu_bin,
        "--import",
        str(report_path),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--print-fp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"NCU raw export failed: {shell_join(cmd)}\n{result.stderr.strip()}"
        )
    output_path.write_text(result.stdout, encoding="utf-8")


def benchmark_provider(fn, batch, heads, n_ctx, d_head, causal, warmup, rep):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    total_flops = 4.0 * batch * heads * n_ctx * n_ctx * d_head
    if causal:
        total_flops *= 0.5
    tflops = total_flops / ms * 1e-9
    return ms, tflops


def ensure_ncu_output_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    for child in ["reports", "text", "csv", "logs"]:
        (base_dir / child).mkdir(exist_ok=True)


def run_ncu(args, case_name, preset_name, n_ctx, d_head, causal):
    output_dir = args.ncu_output_dir
    if output_dir is None:
        output_dir = (
            HARNESS_SCRIPT.parents[1]
            / "results_ncu_single_point"
            / f"{case_name}_{args.dtype}"
        )
    output_dir = output_dir.resolve()
    ensure_ncu_output_dir(output_dir)
    for provider in args.providers:
        stem = f"{provider}_{case_name}_{args.dtype}"
        report_stem = output_dir / "reports" / stem
        report_path = report_stem.with_suffix(".ncu-rep")
        stdout_path = output_dir / "logs" / f"{stem}.stdout.log"
        stderr_path = output_dir / "logs" / f"{stem}.stderr.log"
        text_path = output_dir / "text" / f"{stem}.details.txt"
        csv_path = output_dir / "csv" / f"{stem}.raw.csv"
        cmd = build_ncu_profile_command(
            args, provider, case_name, preset_name, n_ctx, d_head, causal, report_stem
        )
        print(f"[ncu] {provider}")
        print(shell_join(cmd))
        if args.ncu_dry_run:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"NCU profile failed: {shell_join(cmd)}\n{result.stderr.strip()}"
            )
        if not report_path.exists():
            hints = []
            if result.stdout.strip():
                hints.append(result.stdout.strip())
            if result.stderr.strip():
                hints.append(result.stderr.strip())
            raise RuntimeError(
                f"NCU did not produce report: {report_path}\n" + "\n".join(hints)
            )
        export_text_report(args.ncu_bin, report_path, text_path)
        export_raw_csv(args.ncu_bin, report_path, csv_path)
    print(f"ncu_output_dir={output_dir}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    case_name, preset_name, n_ctx, d_head, causal, batch, heads = resolve_case(args)
    dtype = get_dtype(args.dtype)
    sm_scale = 1.0 / math.sqrt(d_head)

    print(
        f"case={case_name} batch={batch} heads={heads} "
        f"n_ctx={n_ctx} d_head={d_head} causal={causal} dtype={args.dtype}"
    )

    rows = []
    for provider in args.providers:
        q, k, v = build_inputs(batch, heads, n_ctx, d_head, dtype)
        fn = build_forward(provider, q, k, v, causal, sm_scale)
        with torch.inference_mode():
            ms, tflops = benchmark_provider(
                fn,
                batch,
                heads,
                n_ctx,
                d_head,
                causal,
                args.bench_warmup,
                args.bench_rep,
            )
        rows.append((provider, ms, tflops))

    print("provider, ms, tflops")
    for provider, ms, tflops in rows:
        print(f"{provider}, {ms:.4f}, {tflops:.4f}")

    if len(rows) == 2:
        base = rows[0][1]
        diff = rows[1][1] / base - 1.0
        print(f"relative_ms_delta({rows[1][0]} vs {rows[0][0]})={diff:+.2%}")

    if args.run_ncu or args.ncu_dry_run:
        run_ncu(args, case_name, preset_name, n_ctx, d_head, causal)


if __name__ == "__main__":
    main()
