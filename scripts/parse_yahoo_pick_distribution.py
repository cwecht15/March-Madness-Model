#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROUND_MAP = {
    "RD64": "round_of_64",
    "RD32": "round_of_32",
    "RD16": "sweet_sixteen",
    "RD8": "elite_eight",
    "RD4": "final_four",
    "Final": "championship",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Yahoo public pick distribution text into a structured CSV.")
    parser.add_argument(
        "--input",
        default="Public_data.txt",
        help="Path to the raw copied Yahoo pick distribution text file.",
    )
    parser.add_argument(
        "--output",
        default="data/public_pick_distribution/yahoo_pick_distribution_latest.csv",
        help="Path for the structured output CSV.",
    )
    return parser.parse_args()


def clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_rank(token: str) -> int | None:
    match = re.match(r"^(\d+)", token)
    return int(match.group(1)) if match else None


def parse_seed(token: str) -> int | None:
    match = re.match(r"^\((\d+)\)$", token)
    return int(match.group(1)) if match else None


def parse_pct(token: str) -> float | None:
    token = token.replace("%", "").strip()
    try:
        return float(token) / 100.0
    except ValueError:
        return None


def parse_distribution(lines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_round_raw: str | None = None
    current_round: str | None = None
    i = 0

    while i < len(lines):
        token = lines[i]
        if token in ROUND_MAP:
            current_round_raw = token
            current_round = ROUND_MAP[token]
            i += 1
            continue

        if token == "Rank" and i + 2 < len(lines) and lines[i + 1].startswith("Team"):
            i += 3
            continue

        rank = parse_rank(token)
        if rank is None or current_round is None:
            i += 1
            continue

        if i + 3 >= len(lines):
            break

        team = lines[i + 1]
        seed = parse_seed(lines[i + 2])
        pct = parse_pct(lines[i + 3])
        if seed is None or pct is None:
            i += 1
            continue

        rows.append(
            {
                "round_source": current_round_raw,
                "round_key": current_round,
                "rank": rank,
                "team": team,
                "seed": seed,
                "picked_pct": pct,
            }
        )
        i += 4

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    parsed = parse_distribution(clean_lines(text))
    if parsed.empty:
        raise ValueError(f"No pick distribution rows were parsed from {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(output_path, index=False)
    print(f"Saved {len(parsed)} rows to {output_path}")
    print(parsed.groupby("round_key").size().to_string())


if __name__ == "__main__":
    main()
