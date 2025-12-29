"""End-of-experiments orchestration: figures, RESULTS.md, and paper PDF.

Runs in order: ``plot_results.py`` (figures + summary.json),
``write_results_md.py`` (markdown table), ``patch_paper.py`` (replace
placeholders), then ``tectonic`` to rebuild the paper PDF.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(short: str) -> None:
    print(f"\n--- {short} ---", flush=True)
    path = PROJECT_ROOT / "experiments" / f"{short}.py"
    spec = importlib.util.spec_from_file_location(f"_fin_{short}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    _run("plot_results")
    _run("write_results_md")
    _run("patch_paper")
    print("\n--- tectonic ---", flush=True)
    res = subprocess.run(
        ["tectonic", "main.tex"],
        cwd=PROJECT_ROOT / "paper",
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print("tectonic stderr:\n", res.stderr, flush=True)
        sys.exit(res.returncode)
    print(res.stdout.strip().splitlines()[-1] if res.stdout else "(no output)")
    print(f"\nFinal PDF: {PROJECT_ROOT / 'paper' / 'main.pdf'}")


if __name__ == "__main__":
    main()
