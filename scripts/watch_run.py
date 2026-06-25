"""Compact progress snapshot for a running/finished operation.

Usage:
    python scripts/watch_run.py <op_id> [store_dir]

Reads the file-store runtime.json + operation-trace.txt (no server needed).
Defaults to the canonical runtime store (var/runtime, or $AEGRA_RUNTIME_STORE_DIR).
Wrap in a loop for live watching, e.g. PowerShell:
    while ($true) { python scripts/watch_run.py <op>; sleep 5; cls }
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def _find(obj, key, out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            _find(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            _find(v, key, out)
    return out


def main() -> None:
    op = sys.argv[1] if len(sys.argv) > 1 else ""
    default_store = os.getenv("AEGRA_RUNTIME_STORE_DIR") or "var/runtime"
    store = Path(sys.argv[2] if len(sys.argv) > 2 else default_store)
    run_dir = store / op
    if not run_dir.exists():
        print(f"no run dir: {run_dir}")
        return

    trace = run_dir / "operation-trace.txt"
    text = trace.read_text(encoding="utf-8", errors="replace") if trace.exists() else ""
    cycles = re.findall(r"cycle_index: (\d+)", text)
    caps = re.findall(r"capability: ([a-z]+)", text)
    last_action = re.findall(r"action: ([a-z_]+)", text)
    last_reason = re.findall(r"reason: (.+)", text)

    print(f"op={op}")
    print(f"  cycle={cycles[-1] if cycles else '-'}  last_capability={caps[-1] if caps else '-'}")
    print(f"  last_planner_action={last_action[-1] if last_action else '-'}")
    if last_reason:
        print(f"  last_reason={last_reason[-1][:160]}")

    rt = run_dir / "runtime.json"
    if rt.exists():
        conds = _find(json.loads(rt.read_text(encoding="utf-8")), "conditions")
        if conds:
            c = conds[0]
            sat = [n for n, v in c.items() if v.get("satisfied")]
            missing = [n for n, v in c.items() if not v.get("satisfied")]
            print(f"  conditions {len(sat)}/{len(c)} satisfied")
            print(f"    OK : {sat}")
            print(f"    -- : {missing}")


if __name__ == "__main__":
    main()
