#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from strict_answer import format_answer, normalize_gt_letter


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    parse_error = None
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                parse_error = f"line {line_no}: {type(exc).__name__}: {exc}"
                break
            rows.append(row)
    return rows, parse_error


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def convert(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, parse_error = load_jsonl(path)
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()
    for row in rows:
        letter = normalize_gt_letter(str(row.get("solution") or row.get("answer") or row.get("gt_answer") or ""))
        if letter is None:
            stats["skip_unmappable_solution"] += 1
            continue
        new_row = dict(row)
        new_row["solution"] = format_answer(letter)
        new_row["gt_letter"] = letter
        out.append(new_row)
        stats["kept_rows"] += 1
    summary = {
        "input_path": str(path.resolve()),
        "input_rows_parsed": len(rows),
        "output_rows": len(out),
        "removed_rows": len(rows) - len(out),
        "parse_error": parse_error,
        "stats": dict(stats),
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize GRPO/eval JSONL solutions to strict <ANSWER>X</ANSWER> format.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    out, summary = convert(Path(args.input))
    dump_jsonl(Path(args.output), out)
    summary["output_path"] = str(Path(args.output).resolve())
    dump_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
