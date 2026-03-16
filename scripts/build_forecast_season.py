#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.build_tournament_dataset import (
    OUTPUT_COLUMNS,
    TeamNameResolver,
    clean_numeric,
    ensure_parent,
    find_column,
    load_aliases,
    load_historical_names,
    load_kenpom,
    load_teamrankings,
    pick_team_column,
    read_table,
)


FEATURE_COLUMNS = [
    "seed",
    "AdjEM",
    "AdjO",
    "AdjD",
    "AdjT",
    "Luck",
    "Seas_PPG",
    "Seas_Succ_3PT",
    "Seas_3PT_Per",
    "Seas_FT_Succ",
    "Seas_3PT_Rate",
    "Seas_Off_Rebound_Per",
    "Seas_Def_Rebound_Per",
    "Seas_Turnovers",
    "Seas_Fouls",
    "Seas_Opp_PPG",
    "Seas_Opp_3PT_Succ",
    "Seas_Opp_3PT_Rate",
    "Seas_Opp_Off_Rebound",
    "Seas_Opp_Def_Rebound",
    "Seas_Opp_Turnover",
    "Seas_Opp_Fouls",
    "Seas_Poss",
]
TARGET_COLUMNS = [
    "Finish",
    "First_Rd",
    "Second_Rd",
    "Sweet_Sixteen",
    "Elite_Eight",
    "Final_Four",
    "Championship",
    "Pts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pre-tournament season dataset for forecasting from field seeds, KenPom, and TeamRankings."
    )
    parser.add_argument("--season", type=int, required=True, help="Tournament year, e.g. 2026.")
    parser.add_argument(
        "--field",
        required=True,
        help="CSV with at least team and seed columns. Can include 68 teams with First Four play-ins.",
    )
    parser.add_argument("--kenpom", required=True, help="Path to the KenPom export for the season.")
    parser.add_argument(
        "--teamrankings-manifest",
        required=True,
        help="CSV manifest that maps each TeamRankings source file to an output column.",
    )
    parser.add_argument(
        "--historical-csv",
        default="data/processed/March_Madness_Train_Model_rebuilt.csv",
        help="Optional historical training CSV used to anchor canonical team names.",
    )
    parser.add_argument(
        "--aliases",
        default="data/team_aliases.csv",
        help="CSV with source-specific team aliases.",
    )
    parser.add_argument("--output", required=True, help="Where to write the built forecast season CSV.")
    parser.add_argument(
        "--match-report",
        default=None,
        help="Optional CSV path for team-name resolution details.",
    )
    return parser.parse_args()


def derive_field(field_path: str, resolver: TeamNameResolver) -> pd.DataFrame:
    df = read_table(field_path)
    team_col = pick_team_column(df)
    seed_col = find_column(df.columns, "seed")
    if seed_col is None:
        raise ValueError("Field file must include a seed column.")

    out = pd.DataFrame({"team": df[team_col].map(lambda value: resolver.resolve("field", value))})
    out["seed"] = clean_numeric(df[seed_col]).astype("Int64")
    out = out.dropna(subset=["team", "seed"]).copy()
    out["seed"] = out["seed"].astype(int)
    out = out.drop_duplicates(subset=["team"], keep="first")
    return out


def build_forecast_frame(
    season: int,
    field_path: str,
    kenpom_path: str,
    teamrankings_manifest: str,
    resolver: TeamNameResolver,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    field_df = derive_field(field_path, resolver)
    kenpom_df = load_kenpom(kenpom_path, resolver)
    teamrankings_df = load_teamrankings(teamrankings_manifest, season, resolver)

    season_df = field_df.merge(kenpom_df, on="team", how="left")
    season_df = season_df.merge(teamrankings_df, on="team", how="left")
    season_df.insert(0, "year", season)
    season_df.insert(2, "year_team", season_df["year"].astype(str) + season_df["team"])

    for column in TARGET_COLUMNS:
        season_df[column] = pd.NA

    for column in OUTPUT_COLUMNS:
        if column not in season_df.columns:
            season_df[column] = pd.NA

    season_df = season_df[OUTPUT_COLUMNS].sort_values(["seed", "team"], kind="stable").reset_index(drop=True)
    return season_df, resolver.report()


def validate_forecast_output(df: pd.DataFrame, season: int) -> list[str]:
    messages: list[str] = []
    missing = df[FEATURE_COLUMNS].isna().sum()
    missing = missing[missing > 0]
    for column, count in missing.items():
        messages.append(f"{season} forecast file has {int(count)} missing values in {column}")

    duplicated = int(df["team"].duplicated().sum())
    if duplicated:
        messages.append(f"{season} forecast file has {duplicated} duplicate team rows after matching.")

    adjem_missing = df[["AdjEM", "AdjO", "AdjD"]].dropna()
    if not adjem_missing.empty:
        delta = (adjem_missing["AdjO"] - adjem_missing["AdjD"] - adjem_missing["AdjEM"]).abs().max()
        if float(delta) > 0.25:
            messages.append(f"{season} forecast file has at least one team where AdjEM does not match AdjO - AdjD.")

    return messages


def main() -> int:
    args = parse_args()

    alias_map = load_aliases(args.aliases)
    historical_lookup = load_historical_names(args.historical_csv)
    resolver = TeamNameResolver(alias_map=alias_map, historical_lookup=historical_lookup)

    season_df, match_report = build_forecast_frame(
        season=args.season,
        field_path=args.field,
        kenpom_path=args.kenpom,
        teamrankings_manifest=args.teamrankings_manifest,
        resolver=resolver,
    )

    ensure_parent(args.output)
    season_df.to_csv(args.output, index=False)

    report_path = args.match_report or str(Path(args.output).with_name(f"{Path(args.output).stem}_match_report.csv"))
    ensure_parent(report_path)
    match_report.to_csv(report_path, index=False)

    warnings = validate_forecast_output(season_df, args.season)
    print(f"Wrote forecast season dataset to {args.output}")
    print(f"Wrote match report to {report_path}")

    if warnings:
        print("Validation warnings:", file=sys.stderr)
        for message in warnings:
            print(f"  - {message}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
