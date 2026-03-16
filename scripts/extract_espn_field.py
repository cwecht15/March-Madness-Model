#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REGION_BY_BLOCK = {
    0: "East",
    1: "South",
    2: "West",
    3: "Midwest",
}
REGION_BY_ID = {
    1: "East",
    2: "South",
    3: "West",
    4: "Midwest",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a 68-team tournament field CSV from an ESPN bracket HTML file.")
    parser.add_argument("--html", required=True, help="Path to the saved ESPN bracket HTML page.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def load_espn_payload(html_path: Path) -> dict:
    text = html_path.read_text(encoding="utf-8")
    marker = "window['__espnfitt__']="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find ESPN data payload in {html_path}")
    start += len(marker)

    brace_depth = 0
    in_string = False
    escaped = False
    end = None

    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    end = index + 1
                    break

    if end is None:
        raise ValueError(f"Could not determine the ESPN payload boundary in {html_path}")

    return json.loads(text[start:end])


def block_region(bracket_location: int) -> str:
    return REGION_BY_BLOCK[(int(bracket_location) - 1) // 8]


def build_field(payload: dict) -> pd.DataFrame:
    matchups = payload["page"]["content"]["bracket"]["matchups"]
    rows: list[dict[str, object]] = []

    for matchup in matchups:
        round_id = int(matchup.get("roundId", -1))
        if round_id not in {0, 1}:
            continue

        left = matchup.get("competitorOne") or {}
        right = matchup.get("competitorTwo") or {}

        if round_id == 0:
            region = REGION_BY_ID[int(matchup["regionId"])]
            seed = int(left["seed"])
            play_in_group = f"{region}-{seed}"
            for competitor in (left, right):
                if competitor.get("name") == "TBD":
                    continue
                rows.append(
                    {
                        "team": competitor["name"],
                        "seed": int(competitor["seed"]),
                        "region": region,
                        "play_in_group": play_in_group,
                        "notes": "first_four",
                    }
                )
            continue

        region = block_region(int(matchup["bracketLocation"]))
        for competitor in (left, right):
            if competitor.get("name") == "TBD":
                continue
            rows.append(
                {
                    "team": competitor["name"],
                    "seed": int(competitor["seed"]),
                    "region": region,
                    "play_in_group": "",
                    "notes": "",
                }
            )

    field = pd.DataFrame(rows).drop_duplicates(subset=["team"], keep="first")
    field = field.sort_values(["region", "seed", "team"], kind="stable").reset_index(drop=True)
    return field


def main() -> int:
    args = parse_args()
    payload = load_espn_payload(Path(args.html))
    field = build_field(payload)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    field.to_csv(args.output, index=False)
    print(f"Wrote field CSV to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
