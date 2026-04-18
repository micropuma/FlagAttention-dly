# Flash Cliff NCU

This workflow profiles forward-kernel cliff points for `flag_attn` and `naive`, then exports:

- `.ncu-rep` for Nsight Compute UI
- `.details.txt` with both section headers and section bodies
- `.raw.csv` for metric slicing or diffing

By default the runner now uses `ncu --set full`, so the generated report keeps the
full Nsight Compute section set instead of a hand-picked subset.

## Quick Start

Run the default cliff set for both implementations. The default cases are chosen
from the latest `results_flash_attention_*` curves and currently focus on:

- `D=64, non-causal`: `4096 -> 8192`
- `D=64, causal`: `16384 -> 32768`
- `D=128, causal`: `16384 -> 32768`

```bash
python run_flash_cliff_ncu.py
```

Run only the current `D=64, non-causal` pre-cliff and cliff pair:

```bash
python run_flash_cliff_ncu.py \
  --cases d64-noncausal-precliff d64-noncausal-cliff
```

Run only the current `D=64, causal` tail pair:

```bash
python run_flash_cliff_ncu.py \
  --cases d64-causal-pretail d64-causal-tail-cliff \
  --warmup-iters 1 \
  --profile-iters 1
```

Run only your implementation in bf16:

```bash
python run_flash_cliff_ncu.py \
  --providers naive \
  --dtype bf16
```

Use the old lightweight section list when you want faster profiling:

```bash
python run_flash_cliff_ncu.py \
  --targeted-sections
```

## Open In UI

After a run finishes, open any generated report with:

```bash
ncu --open-in-ui --import benchmark/results_ncu_flash_cliff/.../reports/<name>.ncu-rep
```

or:

```bash
ncu-ui benchmark/results_ncu_flash_cliff/.../reports/<name>.ncu-rep
```

## Output Layout

Each run creates a timestamped directory under `benchmark/results_ncu_flash_cliff/` with:

- `reports/`: `.ncu-rep` files
- `text/`: `ncu --page details` exports
- `csv/`: `ncu --page raw --csv` exports
- `logs/`: stdout/stderr from the profiling launch
- `index.md`: generated file index
- `rerun_commands.sh`: exact commands used
