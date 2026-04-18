import argparse
import datetime
import pathlib
import shlex
import subprocess
import sys

from flash_microkernel_profile import CLIFF_PRESETS

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HARNESS_SCRIPT = SCRIPT_DIR / "flash_microkernel_profile.py"


DEFAULT_CASES = [
    "d64-noncausal-precliff",
    "d64-noncausal-cliff",
    "d64-causal-pretail",
    "d64-causal-tail-cliff",
    "d128-causal-pretail",
    "d128-causal-tail-cliff",
]

TARGETED_SECTIONS = [
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "SchedulerStats",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Nsight Compute on flash attention cliff cases and export readable reports.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["flag_attn", "naive"],
        default=["flag_attn", "naive"],
        help="Implementations to profile.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CLIFF_PRESETS),
        default=DEFAULT_CASES,
        help="Preset cliff cases to profile.",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Input dtype used by the harness.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations executed before profiling.",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=3,
        help="Iterations inside the NCU capture window.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Directory for generated .ncu-rep, .txt and .csv files.",
    )
    parser.add_argument(
        "--ncu-bin",
        default="ncu",
        help="Nsight Compute CLI executable.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run the harness.",
    )
    parser.add_argument(
        "--set",
        dest="section_set",
        default="full",
        help=(
            "NCU section set to collect. Defaults to `full` so the generated "
            "report keeps all Nsight Compute sections."
        ),
    )
    parser.add_argument(
        "--targeted-sections",
        action="store_true",
        help="Use the smaller targeted section list instead of the default full set.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    parser.add_argument(
        "--open-in-ui",
        action="store_true",
        help="Open the last generated report in Nsight Compute UI after exports finish.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def get_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "benchmark" / "results_ncu_flash_cliff" / timestamp


def ensure_output_dir(output_dir, force):
    if output_dir.exists():
        if any(output_dir.iterdir()) and not force:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass --force or choose a new --output-dir."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    (output_dir / "csv").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def encode_nvtx_push_pop_range(range_name):
    escaped = range_name.replace("\\", "\\\\").replace("/", "\\/")
    return f"{escaped}/"


def run_and_capture(cmd, stdout_path, stderr_path, dry_run):
    if dry_run:
        return
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        details = result.stderr.strip()
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {shell_join(cmd)}"
            + (f"\n{details}" if details else "")
        )


def export_text_report(ncu_bin, report_path, output_path, dry_run):
    header_cmd = [
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
    ]
    body_cmd = [
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
    ]
    if dry_run:
        return [header_cmd, body_cmd]

    sections = []
    for label, cmd in [("header", header_cmd), ("body", body_cmd)]:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            details = result.stderr.strip()
            raise RuntimeError(
                f"NCU import failed with exit code {result.returncode}: {shell_join(cmd)}"
                + (f"\n{details}" if details else "")
            )
        content = result.stdout.strip()
        if content:
            sections.append(f"## {label}\n\n{content}")
    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return [header_cmd, body_cmd]


def export_raw_csv(ncu_bin, report_path, output_path, dry_run):
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
    if dry_run:
        return cmd
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        details = result.stderr.strip()
        raise RuntimeError(
            f"NCU raw export failed with exit code {result.returncode}: {shell_join(cmd)}"
            + (f"\n{details}" if details else "")
        )
    return cmd


def extend_command_lines(command_lines, cmd_or_cmds):
    if not cmd_or_cmds:
        return
    if isinstance(cmd_or_cmds[0], (list, tuple)):
        for cmd in cmd_or_cmds:
            command_lines.append(shell_join(cmd))
    else:
        command_lines.append(shell_join(cmd_or_cmds))


def build_ncu_command(args, provider, case_name, report_stem):
    nvtx_range = f"flash_microkernel/{provider}/{case_name}/profile"
    nvtx_include = encode_nvtx_push_pop_range(nvtx_range)
    harness_cmd = [
        args.python_bin,
        str(HARNESS_SCRIPT),
        "--provider",
        provider,
        "--preset",
        case_name,
        "--dtype",
        args.dtype,
        "--warmup-iters",
        str(args.warmup_iters),
        "--profile-iters",
        str(args.profile_iters),
        "--print-tensor-shapes",
    ]
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--nvtx",
        "--nvtx-include",
        nvtx_include,
        "-f",
        "-o",
        str(report_stem),
    ]
    if args.targeted_sections:
        for section_name in TARGETED_SECTIONS:
            cmd.extend(["--section", section_name])
    elif args.section_set:
        cmd.extend(["--set", args.section_set])
    else:
        for section_name in TARGETED_SECTIONS:
            cmd.extend(["--section", section_name])
    cmd.extend(harness_cmd)
    return cmd


def write_index(output_dir, records):
    lines = [
        "# Flash Cliff NCU Reports",
        "",
        f"Generated at `{datetime.datetime.now().isoformat(timespec='seconds')}`.",
        "",
        "## Files",
        "",
    ]
    for record in records:
        lines.append(
            f"- `{record['provider']}` / `{record['case']}`: "
            f"`{record['report']}`, `{record['text']}`, `{record['csv']}`"
        )
    lines.extend(
        [
            "",
            "## Open In UI",
            "",
            "Use either command below with the generated `.ncu-rep` file:",
            "",
        ]
    )
    if records:
        example_report = records[0]["report"]
        lines.append(f"`ncu --open-in-ui --import {example_report}`")
        lines.append(f"`ncu-ui {example_report}`")
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = get_output_dir(args)
    ensure_output_dir(output_dir, args.force)

    records = []
    command_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    last_report_path = None

    for case_name in args.cases:
        for provider in args.providers:
            file_stem = f"{provider}_{case_name}_{args.dtype}"
            report_stem = output_dir / "reports" / file_stem
            report_path = report_stem.with_suffix(".ncu-rep")
            text_path = output_dir / "text" / f"{file_stem}.details.txt"
            csv_path = output_dir / "csv" / f"{file_stem}.raw.csv"
            stdout_log = output_dir / "logs" / f"{file_stem}.stdout.log"
            stderr_log = output_dir / "logs" / f"{file_stem}.stderr.log"

            profile_cmd = build_ncu_command(args, provider, case_name, report_stem)
            command_lines.append(shell_join(profile_cmd))
            print(f"[run] {provider} {case_name}")
            print(shell_join(profile_cmd))
            run_and_capture(profile_cmd, stdout_log, stderr_log, args.dry_run)
            if not args.dry_run and not report_path.exists():
                raise RuntimeError(
                    "NCU profile completed without producing a report file. "
                    f"Check {stdout_log} and {stderr_log} for profiler warnings."
                )

            text_cmd = export_text_report(args.ncu_bin, report_path, text_path, args.dry_run)
            csv_cmd = export_raw_csv(args.ncu_bin, report_path, csv_path, args.dry_run)
            extend_command_lines(command_lines, text_cmd)
            extend_command_lines(command_lines, csv_cmd)
            command_lines.append("")

            records.append(
                {
                    "provider": provider,
                    "case": case_name,
                    "report": report_path,
                    "text": text_path,
                    "csv": csv_path,
                }
            )
            last_report_path = report_path

    (output_dir / "rerun_commands.sh").write_text("\n".join(command_lines), encoding="utf-8")
    write_index(output_dir, records)

    print(f"results_dir={output_dir}")
    if last_report_path is not None:
        print(f"last_report={last_report_path}")
        print(f"open_ui_cmd=ncu --open-in-ui --import {last_report_path}")
        print(f"open_ui_alt=ncu-ui {last_report_path}")

        if args.open_in_ui:
            cmd = [args.ncu_bin, "--open-in-ui", "--import", str(last_report_path)]
            print(shell_join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
