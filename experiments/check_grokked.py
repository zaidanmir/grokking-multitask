"""Quick sanity check: did each experiment reach test_acc >= 0.95 by end of run?

Walks ``runs/<name>/history.npz`` for every run, and for each task in each run
reports whether the network grokked, when, and the final test accuracy.
Returns nonzero exit code if any run failed to grok — useful in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import find_grok_step
from src.train import load_history


def main() -> int:
    runs_dir = PROJECT_ROOT / "runs"
    failures: list[str] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "history.npz").exists():
            continue
        h = load_history(d)
        ops = [k.split("_acc_")[1] for k in h.keys() if k.startswith("test_acc_")]
        gs = find_grok_step(h["step"].tolist(), h["test_acc"].tolist())
        final_acc = h["test_acc"][-1]
        marker = "OK " if (gs is not None or final_acc >= 0.95) else "FAIL"
        print(f"{marker}  {d.name:42s}  final acc {final_acc:.4f}  grok step {gs}")
        for op in ops:
            gs_op = find_grok_step(h["step"].tolist(), h[f"test_acc_{op}"].tolist())
            fa_op = h[f"test_acc_{op}"][-1]
            sub_marker = "    OK " if (gs_op is not None or fa_op >= 0.95) else "    FAIL"
            print(f"{sub_marker}  task {op}: final acc {fa_op:.4f}  grok step {gs_op}")
            if gs_op is None and fa_op < 0.95:
                failures.append(f"{d.name}/{op}")
        if gs is None and final_acc < 0.95:
            failures.append(d.name)
    if failures:
        print(f"\n{len(failures)} run(s) did not reach 0.95 test acc:", failures)
        return 1
    print("\nall runs grokked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
