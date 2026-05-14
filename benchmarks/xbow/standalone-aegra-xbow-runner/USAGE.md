# Quick Start

Run from `benchmarks/xbow/standalone-aegra-xbow-runner`:

```bash
python run_benchmarks.py --range 1-1 --dry-run
python run_benchmarks.py --range 1-1
```

The runner builds and starts the selected XBOW benchmark with Docker, passes the target URL to Aegra, parses `[FLAG]` and `[DONE]` lines, writes per-benchmark logs, and generates `summary.json` plus `summary.txt`.

## Selection

```bash
python run_benchmarks.py --range 1-10
python run_benchmarks.py --ids 1,5,10
python run_benchmarks.py --all
```

## Execution Options

```bash
python run_benchmarks.py --range 1-10 --timeout 1800
python run_benchmarks.py --range 1-50 --resume
python run_benchmarks.py --range 1-5 --output-dir ./logs
```

## Flag Modes

```bash
python run_benchmarks.py --range 1-10 --any-flag
python run_benchmarks.py --range 1-10 --pattern-flag
```

Default mode checks for the exact flag from each benchmark `.env` file. `--any-flag` accepts any detected flag. `--pattern-flag` accepts a strict `FLAG{...}` style flag with at least 32 characters inside braces.
