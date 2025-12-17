"""Patch paper sections with the actual numbers from runs/summary.json.

The paper sections contain a few placeholders of the form ``\\textsc{see-log}``
which this script replaces with concrete grok-step numbers and a few
results-dependent prose snippets. Idempotent: running it twice gives
the same output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "runs" / "summary.json"
SECTIONS = PROJECT_ROOT / "paper" / "sections"


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        print(f"warning: {SUMMARY_PATH} does not exist; nothing to patch.", file=sys.stderr)
        sys.exit(0)
    return json.loads(SUMMARY_PATH.read_text())


def _patch(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    if out != text:
        path.write_text(out)
        print(f"patched {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    summary = _load_summary()

    # Replication section: replace the see-log placeholder with the actual grok step.
    add = summary.get("01_baseline_addition")
    if add and add.get("grok_step_overall") is not None:
        gs = add["grok_step_overall"]
        gs_str = f"{gs:,}".replace(",", "{,}")  # LaTeX-friendly thin space
        _patch(
            SECTIONS / "replication.tex",
            [
                (
                    r"$\sim$\textsc{see-log}",
                    f"$\\sim {gs_str}$",
                ),
            ],
        )


if __name__ == "__main__":
    main()
