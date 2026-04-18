---
name: profile-attention
description: Use when the user wants to profile flash attention or compare flag_attn and naive kernels, especially to regenerate results_flash_attention_<date>, find cliff points from benchmark CSVs, and then run single-point or batch Nsight Compute analysis with scripts under benchmark/fwd_analysis.
---

# Profile Attention

This skill is for the FlagAttention repository's flash attention profiling workflow.

## Use This Skill For

- Re-running `benchmark/flash_benchmark.py` to generate fresh performance curves
- Comparing `flag_attn` and `naive` on forward attention
- Detecting likely cliff points from `results_flash_attention_<date>/*.csv`
- Running single-point benchmark + NCU analysis
- Running batch NCU collection for a curated cliff set

## Workflow

1. From the repo root or `benchmark/`, run forward benchmark first:

```bash
python flash_benchmark.py --bench-mode fwd
```

This writes outputs under `benchmark/results_flash_attention_<YYYYMMDD>/`. Treat this as the source of truth for real performance ranking.

2. Read the relevant CSVs, usually:

- `attention_d-64_mode-fwd_causal-False_dtype-torch.float16.csv`
- `attention_d-64_mode-fwd_causal-True_dtype-torch.float16.csv`
- `attention_d-128_mode-fwd_causal-True_dtype-torch.float16.csv`

3. Judge whether `naive` has a real cliff:

- First compare `naive` against itself across neighboring `N_CTX`
- Then compare `naive` against `flag_attn` at the same `N_CTX`
- Treat persistent drops of roughly `>5%` as higher-priority cliff candidates
- Treat isolated `1%-2%` dips as low priority unless they are stable across reruns

4. For a single shape, use:

```bash
python benchmark/fwd_analysis/analyze_single_point.py --preset d64-noncausal-cliff
```

This prints steady-state benchmark `ms / TFLOP/s` first. Add `--run-ncu` for lightweight NCU, or `--run-ncu --full-sections` for full NCU. Single-point NCU outputs go under `benchmark/results_ncu_single_point/<case>_<dtype>/`.

5. For a batch of known cliff points, use:

```bash
python benchmark/fwd_analysis/run_flash_cliff_ncu.py
```

Use `--targeted-sections` when you want faster profiling. By default it uses `--set full`. Batch NCU outputs go under `benchmark/results_ncu_flash_cliff/<timestamp>/`.

## Important Distinction

- `results_flash_attention_*` answers: "who is actually faster?"
- `results_ncu_*` answers: "why is this point faster or slower?"

Do not treat NCU `Duration` as the final performance ranking when `--set full` is enabled. Use benchmark timing for the real ranking, and use NCU counters for attribution.

## Files

- `benchmark/flash_benchmark.py`
- `benchmark/fwd_analysis/analyze_single_point.py`
- `benchmark/fwd_analysis/flash_microkernel_profile.py`
- `benchmark/fwd_analysis/run_flash_cliff_ncu.py`
- `benchmark/fwd_analysis/README_ncu_flash_cliff.md`

## Output Expectations

After single-point or cliff runs, analyze:

- `text/*.details.txt` first
- `csv/*.raw.csv` only when the text report is insufficient
- `logs/*.stdout.log` to confirm shape, iteration count, and NVTX capture

When explaining a cliff, prioritize:

- shape and launch configuration
- occupancy and scheduler metrics
- tensor/ALU/LSU mix
- warp stall reasons
- whether the issue is benchmark-only or also visible in lightweight NCU

## Default Deliverable

When a user asks to "profile attention", the expected sequence is:

1. Run or inspect `flash_benchmark.py` results.
2. Identify suspicious cliff candidates from CSVs.
3. Confirm a chosen cliff point with `analyze_single_point.py`.
4. If needed, collect `details.txt` via NCU.
5. Explain whether the gap is real, where it comes from, and whether it warrants a kernel change.
