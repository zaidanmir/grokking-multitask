"""Write a human-readable summary of every run to runs/RESULTS.md.

Reads runs/summary.json (produced by plot_results.py) and renders a
GitHub-Markdown table of grok steps and final accuracies. Runs that
haven't been executed yet are silently skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fmt_step(s) -> str:
    if s is None:
        return "—"
    return f"{int(s):,}"


def _fmt_acc(a) -> str:
    if a is None:
        return "—"
    return f"{float(a):.4f}"


def main() -> None:
    summary_path = PROJECT_ROOT / "runs" / "summary.json"
    if not summary_path.exists():
        print(f"warning: {summary_path} does not exist; nothing to write.", file=sys.stderr)
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# Experimental results\n")
    lines.append("Auto-generated from `runs/summary.json` by `experiments/write_results_md.py`.\n")
    lines.append("All experiments use $p = 113$, train\\_frac = 0.30, AdamW with weight decay = 1.0.\n\n")
    lines.append("## Per-run summary\n\n")
    lines.append("| run | ops | steps | final test acc | grok step (test_acc ≥ 0.95) |\n")
    lines.append("|-----|-----|------:|---------------:|------------------:|\n")
    for name in sorted(summary.keys()):
        e = summary[name]
        ops = "/".join(e.get("per_task", {}).keys()) or "all"
        lines.append(
            f"| `{name}` | {ops} | {_fmt_step(e.get('n_steps_logged'))} | "
            f"{_fmt_acc(e.get('final_test_acc'))} | "
            f"{_fmt_step(e.get('grok_step_overall'))} |\n"
        )

    # Per-task slice for multi-task runs.
    multitask = {n: e for n, e in summary.items() if "per_task" in e and len(e["per_task"]) > 1}
    if multitask:
        lines.append("\n## Per-task grok steps (multi-task runs)\n\n")
        ops_list = sorted({op for e in multitask.values() for op in e["per_task"]})
        header = "| run | " + " | ".join(ops_list) + " |\n"
        sep = "|-----|" + "------:|" * len(ops_list) + "\n"
        lines.append(header)
        lines.append(sep)
        for name in sorted(multitask.keys()):
            cells = []
            for op in ops_list:
                cells.append(_fmt_step(multitask[name]["per_task"].get(op, {}).get("grok_step")))
            lines.append(f"| `{name}` | " + " | ".join(cells) + " |\n")

    out_path = PROJECT_ROOT / "runs" / "RESULTS.md"
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
