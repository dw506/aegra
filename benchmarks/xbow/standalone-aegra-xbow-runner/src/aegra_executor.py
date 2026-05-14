"""Aegra executor for the standalone XBOW benchmark runner."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


class AegraExecutor:
    """Executes the Aegra XBOW adapter with timeout handling."""

    def __init__(
        self,
        adapter_path: Path | None = None,
        python_executable: str | None = None,
    ):
        """
        Initialize executor.

        Args:
            adapter_path: Optional adapter path. Defaults to benchmarks/xbow/run_aegra_xbow.py.
            python_executable: Python executable to use. Defaults to the current interpreter.
        """
        runner_root = Path(__file__).resolve().parents[1]
        repo_root = runner_root.parents[2]
        env_adapter = os.environ.get("AEGRA_XBOW_ADAPTER")
        self.adapter_path = Path(env_adapter).resolve() if env_adapter else (
            adapter_path or repo_root / "benchmarks" / "xbow" / "run_aegra_xbow.py"
        ).resolve()
        self.python_executable = python_executable or os.environ.get("AEGRA_PYTHON") or sys.executable

    async def execute(
        self,
        target_url: str,
        benchmark_id: str,
        timeout_seconds: int,
        output_file: Path,
        benchmark_json: Path | None = None,
        tags: list[str] | None = None,
        level: int | str | None = None,
    ) -> dict:
        """
        Execute Aegra with timeout.

        Args:
            target_url: Target URL to test
            benchmark_id: Benchmark identifier for logging
            timeout_seconds: Timeout in seconds
            output_file: Path to write output

        Returns:
            dict with 'output_lines', 'returncode', 'timed_out'
        """
        benchmark_output_dir = output_file.parent / benchmark_id
        command = self._build_command(
            target_url,
            benchmark_id,
            benchmark_json=benchmark_json,
            tags=tags,
            level=level,
            output_dir=benchmark_output_dir,
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Running Aegra (timeout: {timeout_seconds}s)...")
        print(f"  Target: {target_url}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            try:
                output_lines = await asyncio.wait_for(
                    self._stream_output(process, output_file),
                    timeout=timeout_seconds
                )
                returncode = await process.wait()

                return {
                    "output_lines": output_lines,
                    "returncode": returncode,
                    "timed_out": False
                }

            except asyncio.TimeoutError:
                print(f"  Timeout after {timeout_seconds}s")
                await self._kill_gracefully(process)

                output_lines = []
                if output_file.exists():
                    with open(output_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if " " in line:
                                output_lines.append(line.split(" ", 1)[1].rstrip())

                return {
                    "output_lines": output_lines,
                    "returncode": -1,
                    "timed_out": True
                }

        except Exception as e:
            print(f"  Execution error: {e}")
            return {
                "output_lines": [f"[ERROR] {e}"],
                "returncode": -1,
                "timed_out": False,
                "error": str(e)
            }

    def _build_command(
        self,
        target_url: str,
        benchmark_id: str,
        *,
        benchmark_json: Path | None = None,
        tags: list[str] | None = None,
        level: int | str | None = None,
        output_dir: Path | None = None,
    ) -> list[str]:
        """
        Build the Aegra adapter command.

        The default command is:
            python <Aegra>/benchmarks/xbow/run_aegra_xbow.py --target-url <target_url> --benchmark-id <id>
        """
        command = [
            self.python_executable,
            str(self.adapter_path),
            "--target-url",
            target_url,
            "--benchmark-id",
            benchmark_id,
        ]
        if benchmark_json:
            command.extend(["--benchmark-json", str(benchmark_json)])
        for tag in tags or []:
            command.extend(["--tags", str(tag)])
        if level is not None:
            command.extend(["--level", str(level)])
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])
        return command

    async def _stream_output(
        self,
        process: asyncio.subprocess.Process,
        output_file: Path
    ) -> list[str]:
        """
        Stream stdout line-by-line to file and collect lines.
        Terminates early when [DONE] is detected.
        """
        lines = []

        with open(output_file, "w", encoding="utf-8") as f:
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break

                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} {line}\n")
                f.flush()

                if any(marker in line for marker in ["[FLAG]", "[DONE]", "[ERROR]", "[WARN]"]):
                    print(f"  {line}")

                lines.append(line)

                if "[DONE]" in line:
                    print("  Aegra completed, terminating early...")
                    await self._kill_gracefully(process)
                    break

        return lines

    async def _kill_gracefully(self, process: asyncio.subprocess.Process):
        """Kill process gracefully: SIGTERM, then SIGKILL if needed."""
        try:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"  Warning: Error killing process: {e}")
