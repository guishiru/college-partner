"""Regenerate the golden baselines.

    python tests/golden/regenerate.py            # 所有可用方法
    python tests/golden/regenerate.py reliability regression

This is deliberately a separate command rather than something the test suite
does on failure. A baseline that updates itself proves nothing: it would turn
every regression into a green run plus a quiet diff. Run this only when you
meant to change a statistical result, then read the diff line by line before
committing it, and say why in docs/decisions/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from golden.cases import CASES, baseline_path, fingerprint, load_dataset  # noqa: E402

from modules.analysis.engine import AnalysisEngine, AnalysisEngineError  # noqa: E402


def main(argv: list[str]) -> int:
    requested = argv or sorted(CASES)
    unknown = [method for method in requested if method not in CASES]
    if unknown:
        print(f"未知方法：{', '.join(unknown)}", file=sys.stderr)
        return 1

    data = load_dataset()
    engine = AnalysisEngine()
    written, skipped = [], []

    for method in requested:
        try:
            result = engine.run(method, data, CASES[method])
        except AnalysisEngineError as exc:
            skipped.append((method, str(exc)))
            continue
        path = baseline_path(method)
        path.write_text(
            json.dumps(fingerprint(result), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        written.append(method)

    for method in written:
        print(f"已写入 {baseline_path(method).relative_to(REPO_ROOT)}")
    for method, reason in skipped:
        print(f"跳过 {method}：{reason}", file=sys.stderr)

    print(f"\n共写入 {len(written)} 个基线，跳过 {len(skipped)} 个。")
    if skipped:
        print("跳过的方法必须登记在 tests/golden/cases.py 的 PENDING 中。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
