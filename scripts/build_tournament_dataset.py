#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


OUTPUT_COLUMNS = [
    "year",
    "team",
    "year_team",
    "seed",
    "Finish",
    "First_Rd",
    "Second_Rd",
    "Sweet_Sixteen",
    "Elite_Eight",
    "Final_Four",
    "Championship",
    "Pts",
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

ROUND_COLUMNS = [
    "First_Rd",
    "Second_Rd",
    "Sweet_Sixteen",
    "Elite_Eight",
    "Final_Four",
    "Championship",
]

ROUND_TOTALS = {
    "First_Rd": 32,
    "Second_Rd": 16,
    "Sweet_Sixteen": 8,
    "Elite_Eight": 4,
    "Final_Four": 2,
    "Championship": 1,
}

FINISH_LABEL_TO_WINS = {
    "first four": 0,
    "round of 64": 0,
    "round 64": 0,
    "first round": 0,
    "round of 32": 1,
    "round 32": 1,
    "second round": 1,
    "sweet 16": 2,
    "sweet sixteen": 2,
    "elite 8": 3,
    "elite eight": 3,
    "final 4": 4,
    "final four": 4,
    "runner up": 5,
    "runner-up": 5,
    "finalist": 5,
    "championship game": 5,
    "champion": 6,
    "champions": 6,
    "title": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a March Madness season dataset from KenPom, TeamRankings, and tournament results."
    )
    parser.add_argument("--season", type=int, required=True, help="Tournament year, e.g. 2025.")
    parser.add_argument("--kenpom", required=True, help="Path or URL to a KenPom table export for the season.")
    parser.add_argument(
        "--teamrankings-manifest",
        required=True,
        help="CSV manifest that maps each TeamRankings source file to an output column.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a tournament results CSV with at least team, seed, and wins/finish information.",
    )
    parser.add_argument(
        "--historical-csv",
        default=None,
        help="Optional existing training CSV used to anchor canonical team names.",
    )
    parser.add_argument(
        "--aliases",
        default="data/team_aliases.csv",
        help="CSV with source-specific team aliases.",
    )
    parser.add_argument("--output", required=True, help="Where to write the built season CSV.")
    parser.add_argument(
        "--match-report",
        default=None,
        help="Optional CSV path for team-name resolution details.",
    )
    parser.add_argument(
        "--base-dataset",
        default=None,
        help="Optional historical training CSV to append the rebuilt season onto.",
    )
    parser.add_argument(
        "--drop-years",
        nargs="*",
        type=int,
        default=None,
        help="Years to remove from --base-dataset before appending the rebuilt season.",
    )
    parser.add_argument(
        "--merged-output",
        default=None,
        help="Optional path for a merged dataset when --base-dataset is supplied.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if unresolved names create missing values after the season merge.",
    )
    return parser.parse_args()


def normalize_column_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).strip().lower()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ").replace("\xa0", " ")
    text = text.replace("saint", "st")
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\bthe\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def display_team_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_kenpom_team_name(name: str) -> str:
    text = display_team_name(name)
    # KenPom tournament exports often append the NCAA seed to the team name.
    text = re.sub(r"\s+\d+$", "", text).strip()
    return text


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def read_bytes(source: str) -> bytes:
    if is_url(source):
        if requests is None:
            raise RuntimeError("requests is required to read URL sources.")
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return response.content
    return Path(source).read_bytes()


def read_text(source: str) -> str:
    return read_bytes(source).decode("utf-8", errors="ignore")


def read_table(source: str) -> pd.DataFrame:
    suffix = Path(source).suffix.lower() if not is_url(source) else ""

    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(read_bytes(source)))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(read_bytes(source)))

    text = read_text(source)
    if suffix in {".html", ".htm"} or "<table" in text.lower():
        tables = pd.read_html(io.StringIO(text))
        return choose_generic_team_table(tables, source)

    return pd.read_csv(io.StringIO(text))


def read_all_html_tables(source: str) -> list[pd.DataFrame]:
    text = read_text(source)
    return pd.read_html(io.StringIO(text))


def clean_numeric(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def choose_generic_team_table(tables: list[pd.DataFrame], source: str) -> pd.DataFrame:
    for table in tables:
        columns = {normalize_column_name(column) for column in table.columns}
        if not columns.intersection({"team", "school", "name"}):
            continue

        numeric_columns = 0
        for column in table.columns:
            if clean_numeric(table[column]).notna().mean() >= 0.5:
                numeric_columns += 1

        if numeric_columns:
            return table

    raise ValueError(
        f"Unable to find a team/value table in {source!r}. "
        "Use a CSV export or trim the HTML down to the target table."
    )


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_aliases(path: str | None) -> dict[tuple[str, str], str]:
    alias_map: dict[tuple[str, str], str] = {}
    if not path:
        return alias_map

    alias_path = Path(path)
    if not alias_path.exists():
        return alias_map

    with alias_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = (row.get("source") or "all").strip().lower()
            raw_name = normalize_team_name(row.get("source_name", ""))
            canonical = display_team_name(row.get("canonical_name", ""))
            if raw_name and canonical:
                alias_map[(source, raw_name)] = canonical
    return alias_map


def load_historical_names(path: str | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}

    df = pd.read_csv(path)
    if "team" not in df.columns:
        return {}

    buckets: dict[str, set[str]] = {}
    for team_name in sorted(df["team"].dropna().astype(str).unique()):
        buckets.setdefault(normalize_team_name(team_name), set()).add(display_team_name(team_name))

    return {key: next(iter(values)) for key, values in buckets.items() if len(values) == 1}


class TeamNameResolver:
    def __init__(self, alias_map: dict[tuple[str, str], str], historical_lookup: dict[str, str]) -> None:
        self.alias_map = alias_map
        self.historical_lookup = historical_lookup
        self.rows: list[dict[str, str]] = []

    def resolve(self, source: str, raw_name: str) -> str:
        source_key = source.strip().lower()
        display_name = display_team_name(raw_name)
        normalized = normalize_team_name(raw_name)

        method = "cleaned"
        canonical = display_name

        if (source_key, normalized) in self.alias_map:
            canonical = self.alias_map[(source_key, normalized)]
            method = "source_alias"
        elif ("all", normalized) in self.alias_map:
            canonical = self.alias_map[("all", normalized)]
            method = "global_alias"
        elif normalized in self.historical_lookup:
            canonical = self.historical_lookup[normalized]
            method = "historical_exact"

        self.rows.append(
            {
                "source": source_key,
                "raw_team": display_name,
                "normalized_team": normalized,
                "canonical_team": canonical,
                "match_method": method,
            }
        )
        return canonical

    def report(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["source", "raw_team", "normalized_team", "canonical_team", "match_method"])
        return pd.DataFrame(self.rows).drop_duplicates()


def pick_team_column(df: pd.DataFrame) -> str:
    candidates = {
        normalize_column_name(column): column
        for column in df.columns
    }
    for key in ("team", "school", "name"):
        if key in candidates:
            return candidates[key]
    raise ValueError(f"Unable to find a team column. Columns were: {list(df.columns)}")


def pick_numeric_value_column(df: pd.DataFrame, season: int) -> str:
    preferred = [
        f"{season - 1}-{season}",
        f"{season - 1}-{str(season)[-2:]}",
        str(season),
        "value",
        "stat",
        "season",
    ]

    normalized = {normalize_column_name(column): column for column in df.columns}
    for key in preferred:
        normalized_key = normalize_column_name(key)
        if normalized_key in normalized:
            return normalized[normalized_key]

    excluded = {"rank", "rk", "team", "school", "name", "last_3", "last_1", "home", "away"}
    numeric_candidates: list[tuple[int, str]] = []

    for idx, column in enumerate(df.columns):
        column_key = normalize_column_name(column)
        if column_key in excluded:
            continue
        numeric_ratio = clean_numeric(df[column]).notna().mean()
        if numeric_ratio >= 0.8:
            numeric_candidates.append((idx, column))

    if not numeric_candidates:
        raise ValueError(f"Unable to locate a numeric value column in {list(df.columns)}")

    return numeric_candidates[0][1]


def standardize_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [display_team_name(str(column)) for column in cleaned.columns]
    return cleaned


def choose_kenpom_table(source: str) -> pd.DataFrame:
    suffix = Path(source).suffix.lower() if not is_url(source) else ""
    raw_bytes = read_bytes(source)
    if suffix == ".csv":
        if raw_bytes.startswith(b"PK\x03\x04") or raw_bytes.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return standardize_table(pd.read_excel(io.BytesIO(raw_bytes)))
        try:
            return standardize_table(pd.read_csv(io.BytesIO(raw_bytes)))
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
            return standardize_table(pd.read_excel(io.BytesIO(raw_bytes)))
    if suffix in {".xlsx", ".xls"}:
        return standardize_table(pd.read_excel(io.BytesIO(raw_bytes)))

    tables = [standardize_table(df) for df in read_all_html_tables(source)]
    required_sets = [
        {"team", "adjem", "adjo", "adjd", "adjt", "luck"},
        {"team", "netrtg", "ortg", "drtg", "adjt", "luck"},
    ]

    for table in tables:
        keys = {normalize_column_name(column) for column in table.columns}
        if any(required.issubset(keys) for required in required_sets):
            return table

    raise ValueError(
        "Unable to find a KenPom ratings table with Team, AdjEM/NetRtg, AdjO/ORtg, AdjD/DRtg, AdjT, and Luck."
    )


def load_kenpom(source: str, resolver: TeamNameResolver) -> pd.DataFrame:
    df = choose_kenpom_table(source)
    columns = {normalize_column_name(column): column for column in df.columns}

    def require(*options: str) -> str:
        for option in options:
            if option in columns:
                return columns[option]
        raise ValueError(f"Missing one of {options} in KenPom table columns {list(df.columns)}")

    team_col = require("team")
    adjem_col = require("adjem", "netrtg")
    adjo_col = require("adjo", "ortg")
    adjd_col = require("adjd", "drtg")
    adjt_col = require("adjt")
    luck_col = require("luck")

    out = pd.DataFrame(
        {
            "team": df[team_col].map(lambda value: resolver.resolve("kenpom", clean_kenpom_team_name(value))),
            "AdjEM": clean_numeric(df[adjem_col]),
            "AdjO": clean_numeric(df[adjo_col]),
            "AdjD": clean_numeric(df[adjd_col]),
            "AdjT": clean_numeric(df[adjt_col]),
            "Luck": clean_numeric(df[luck_col]),
        }
    )
    out = out.loc[~out["team"].isin({"", "nan", "Team"})].copy()
    return out.dropna(subset=["team"]).drop_duplicates(subset=["team"], keep="first")


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required_columns = {"output_column", "source_file", "scale"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"TeamRankings manifest is missing required columns: {sorted(missing)}")
    return manifest


def maybe_divide_percent(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series
    return series / 100.0 if series.dropna().abs().median() > 1 else series


def resolve_source_path(manifest_path: str | Path, source_file: str) -> str:
    source_path = Path(source_file)
    if source_path.is_absolute() or is_url(source_file):
        return source_file
    if source_path.exists():
        return str(source_path.resolve())
    return str((Path(manifest_path).parent / source_path).resolve())


def load_teamrankings(manifest_path: str, season: int, resolver: TeamNameResolver) -> pd.DataFrame:
    manifest = load_manifest(manifest_path)
    merged: pd.DataFrame | None = None

    for row in manifest.itertuples(index=False):
        output_column = "" if pd.isna(row.output_column) else str(row.output_column).strip()
        source_file = "" if pd.isna(row.source_file) else str(row.source_file).strip()
        scale = "raw" if pd.isna(row.scale) else str(row.scale).strip().lower()

        if not output_column or not source_file:
            continue

        table = standardize_table(read_table(resolve_source_path(manifest_path, source_file)))
        team_col = pick_team_column(table)
        value_col = pick_numeric_value_column(table, season)

        series = clean_numeric(table[value_col])
        if scale == "pct":
            series = maybe_divide_percent(series)
        elif scale != "raw":
            raise ValueError(f"Unsupported scale {scale!r} for {output_column}")

        stat_df = pd.DataFrame(
            {
                "team": table[team_col].map(lambda value: resolver.resolve("teamrankings", value)),
                output_column: series,
            }
        ).dropna(subset=["team"])

        stat_df = stat_df.drop_duplicates(subset=["team"], keep="first")
        merged = stat_df if merged is None else merged.merge(stat_df, on="team", how="outer")

    if merged is None:
        raise ValueError("No TeamRankings source rows were loaded from the manifest.")

    return merged


def find_column(columns: Iterable[str], *options: str) -> str | None:
    mapping = {normalize_column_name(column): column for column in columns}
    for option in options:
        key = normalize_column_name(option)
        if key in mapping:
            return mapping[key]
    return None


def wins_from_finish_label(value: str) -> int | None:
    normalized = normalize_team_name(value)
    return FINISH_LABEL_TO_WINS.get(normalized)


def derive_results(df: pd.DataFrame, resolver: TeamNameResolver) -> pd.DataFrame:
    team_col = pick_team_column(df)
    seed_col = find_column(df.columns, "seed")
    if seed_col is None:
        raise ValueError("Tournament results file must include a seed column.")

    wins_col = find_column(df.columns, "wins", "ncaa_wins", "tournament_wins")
    finish_col = find_column(df.columns, "finish")
    finish_label_col = find_column(df.columns, "finish_label", "finish_round", "round_reached", "finish_text")

    out = pd.DataFrame({"team": df[team_col].map(lambda value: resolver.resolve("results", value))})
    out["seed"] = clean_numeric(df[seed_col]).astype("Int64")

    wins = pd.Series([pd.NA] * len(df), dtype="Int64")
    if wins_col is not None:
        wins = clean_numeric(df[wins_col]).astype("Int64")
    elif finish_col is not None:
        finish_values = clean_numeric(df[finish_col]).astype("Int64")
        wins = finish_values - 1
    elif finish_label_col is not None:
        wins = df[finish_label_col].map(wins_from_finish_label).astype("Int64")
    else:
        raise ValueError(
            "Tournament results file must include wins, finish, or finish_label style information."
        )

    if wins.isna().any():
        missing = out.loc[wins.isna(), "team"].tolist()
        raise ValueError(f"Unable to derive tournament wins for: {missing}")

    if ((wins < 0) | (wins > 6)).any():
        bad = out.loc[(wins < 0) | (wins > 6), "team"].tolist()
        raise ValueError(f"Tournament wins must be between 0 and 6. Bad rows: {bad}")

    out["wins"] = wins.astype(int)

    grouped = out.groupby("team", as_index=False).agg({"seed": "min", "wins": "sum"})
    grouped["wins"] = grouped["wins"].clip(upper=6)
    grouped["Finish"] = grouped["wins"] + 1
    grouped["First_Rd"] = (grouped["wins"] >= 1).astype(int)
    grouped["Second_Rd"] = (grouped["wins"] >= 2).astype(int)
    grouped["Sweet_Sixteen"] = (grouped["wins"] >= 3).astype(int)
    grouped["Elite_Eight"] = (grouped["wins"] >= 4).astype(int)
    grouped["Final_Four"] = (grouped["wins"] >= 5).astype(int)
    grouped["Championship"] = (grouped["wins"] >= 6).astype(int)
    grouped["Pts"] = grouped["wins"].map(lambda value: 0 if int(value) == 0 else (2 ** int(value)) - 1)
    grouped = grouped.drop(columns=["wins"])
    return grouped


def validate_round_totals(df: pd.DataFrame, season: int) -> list[str]:
    messages: list[str] = []
    for column, expected in ROUND_TOTALS.items():
        actual = int(df[column].sum())
        if actual != expected:
            messages.append(f"{season} {column} sum was {actual}, expected {expected}")
    return messages


def validate_output(df: pd.DataFrame, season: int) -> list[str]:
    messages = validate_round_totals(df, season)
    adjem_delta = (df["AdjO"] - df["AdjD"] - df["AdjEM"]).abs()
    if adjem_delta.dropna().max() > 0.25:
        messages.append("AdjEM does not reconcile with AdjO - AdjD for at least one team.")

    missing = df[OUTPUT_COLUMNS].isna().sum()
    missing = missing[missing > 0]
    for column, count in missing.items():
        messages.append(f"{column} has {int(count)} missing values")
    return messages


def build_season_frame(
    season: int,
    results_path: str,
    kenpom_path: str,
    teamrankings_manifest: str,
    resolver: TeamNameResolver,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_df = derive_results(read_table(results_path), resolver)
    kenpom_df = load_kenpom(kenpom_path, resolver)
    teamrankings_df = load_teamrankings(teamrankings_manifest, season, resolver)

    season_df = results_df.merge(kenpom_df, on="team", how="left")
    season_df = season_df.merge(teamrankings_df, on="team", how="left")
    season_df.insert(0, "year", season)
    season_df.insert(2, "year_team", season_df["year"].astype(str) + season_df["team"])

    for column in OUTPUT_COLUMNS:
        if column not in season_df.columns:
            season_df[column] = pd.NA

    season_df = season_df[OUTPUT_COLUMNS].sort_values(["seed", "team"], kind="stable").reset_index(drop=True)
    return season_df, resolver.report()


def merge_with_base(base_path: str, drop_years: list[int], season_df: pd.DataFrame) -> pd.DataFrame:
    base_df = pd.read_csv(base_path)
    drop_set = set(drop_years)
    drop_set.add(int(season_df["year"].iloc[0]))
    merged = base_df.loc[~base_df["year"].isin(drop_set)].copy()
    merged = pd.concat([merged, season_df], ignore_index=True)
    return merged.sort_values(["year", "seed", "team"], kind="stable").reset_index(drop=True)


def main() -> int:
    args = parse_args()

    alias_map = load_aliases(args.aliases)
    historical_lookup = load_historical_names(args.historical_csv or args.base_dataset)
    resolver = TeamNameResolver(alias_map=alias_map, historical_lookup=historical_lookup)

    season_df, match_report = build_season_frame(
        season=args.season,
        results_path=args.results,
        kenpom_path=args.kenpom,
        teamrankings_manifest=args.teamrankings_manifest,
        resolver=resolver,
    )

    validation_errors = validate_output(season_df, args.season)
    if validation_errors and args.strict:
        for message in validation_errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1

    ensure_parent(args.output)
    season_df.to_csv(args.output, index=False)

    report_path = args.match_report or str(Path(args.output).with_name(f"{Path(args.output).stem}_match_report.csv"))
    ensure_parent(report_path)
    match_report.to_csv(report_path, index=False)

    if args.base_dataset and args.merged_output:
        merged = merge_with_base(args.base_dataset, args.drop_years or [], season_df)
        ensure_parent(args.merged_output)
        merged.to_csv(args.merged_output, index=False)

    print(f"Wrote season dataset to {args.output}")
    print(f"Wrote match report to {report_path}")

    if args.base_dataset and args.merged_output:
        print(f"Wrote merged dataset to {args.merged_output}")

    unresolved = match_report.loc[match_report["match_method"] == "cleaned"]
    if not unresolved.empty:
        print(
            f"Name review: {len(unresolved)} rows used cleaned source names instead of aliases or historical matches.",
            file=sys.stderr,
        )

    if validation_errors:
        print("Validation warnings:", file=sys.stderr)
        for message in validation_errors:
            print(f"  - {message}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
