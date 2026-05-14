# Aegra XBOW Benchmark Runner

Standalone runner for executing Aegra against the XBOW validation benchmark suite.

This runner reuses the PentestGPT standalone runner structure and keeps the same Docker lifecycle, output parsing, reporting, and resume state components. The executor is replaced with `AegraExecutor`, which invokes:

```bash
python <Aegra>/benchmarks/xbow/run_aegra_xbow.py --target-url <target_url> --benchmark-id <id>
```

## Usage

```bash
python run_benchmarks.py --range 1-5 --dry-run
python run_benchmarks.py --ids 1,5,10 --timeout 900
python run_benchmarks.py --all --resume
python run_benchmarks.py --range 1-10 --any-flag
python run_benchmarks.py --range 1-10 --pattern-flag
```

Useful options:

- `--range`: benchmark range such as `1-10`
- `--ids`: comma-separated benchmark IDs such as `1,5,10`
- `--all`: run all 104 benchmarks
- `--timeout`: per-benchmark timeout in seconds
- `--resume`: skip benchmarks already marked successful in `state.json`
- `--dry-run`: print the selected benchmarks without launching Docker or Aegra
- `--output-dir`: base output directory, default `./logs`
- `--any-flag`: count any detected flag as success
- `--pattern-flag`: count any strict `FLAG{...}` style flag as success
- `--benchmarks-dir`: optional path to the XBOW benchmark directories

The default benchmark directory is auto-detected from this repository layout:

```text
benchmarks/PentestGPT/benchmark/xbow-validation-benchmarks/benchmarks
```

## Outputs

Each run writes:

```text
logs/benchmark_run_*/summary.json
logs/benchmark_run_*/summary.txt
logs/benchmark_run_*/benchmarks/XBEN-xxx-24.log
```

For test harnesses, set `AEGRA_XBOW_ADAPTER` to a compatible fake adapter script. Production runs use `benchmarks/xbow/run_aegra_xbow.py`.
