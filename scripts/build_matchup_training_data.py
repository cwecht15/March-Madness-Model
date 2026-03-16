#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.build_tournament_dataset import display_team_name, load_aliases, normalize_team_name


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


ROUND_ORDER = {
    "First Four": 0,
    "Round of 64": 1,
    "First_Rd": 1,
    "South": 1,
    "East": 1,
    "West": 1,
    "Midwest": 1,
    "Round of 32": 2,
    "Second_Rd": 2,
    "Sweet Sixteen": 3,
    "Sweet_Sixteen": 3,
    "Elite Eight": 4,
    "Elite_Eight": 4,
    "Final Four": 5,
    "Championship": 6,
}

REGIONAL_FIRST_ROUND_PODS = [
    {1, 16},
    {8, 9},
    {5, 12},
    {4, 13},
    {6, 11},
    {3, 14},
    {7, 10},
    {2, 15},
]
REGIONAL_SECOND_ROUND_GROUPS = [
    REGIONAL_FIRST_ROUND_PODS[0] | REGIONAL_FIRST_ROUND_PODS[1],
    REGIONAL_FIRST_ROUND_PODS[2] | REGIONAL_FIRST_ROUND_PODS[3],
    REGIONAL_FIRST_ROUND_PODS[4] | REGIONAL_FIRST_ROUND_PODS[5],
    REGIONAL_FIRST_ROUND_PODS[6] | REGIONAL_FIRST_ROUND_PODS[7],
]
REGIONAL_SWEET_SIXTEEN_GROUPS = [
    REGIONAL_SECOND_ROUND_GROUPS[0] | REGIONAL_SECOND_ROUND_GROUPS[1],
    REGIONAL_SECOND_ROUND_GROUPS[2] | REGIONAL_SECOND_ROUND_GROUPS[3],
]

MAIN_BRACKET_ROUND_NAME = {
    1: "round_of_64",
    2: "round_of_32",
    3: "sweet_sixteen",
    4: "elite_eight",
    5: "final_four",
    6: "championship",
}

SPECIAL_ABBREVIATIONS = {
    "uva": "Virginia",
    "ttu": "Texas Tech",
    "conn": "Connecticut",
    "pur": "Purdue",
    "mich": "Michigan",
    "vill": "Villanova",
    "bay": "Baylor",
    "gonz": "Gonzaga",
    "ku": "Kansas",
    "uk": "Kentucky",
    "lou": "Louisville",
    "wis": "Wisconsin",
    "sdsu": "San Diego St",
    "fau": "Fla Atlantic",
    "unc": "N Carolina",
    "uconn": "Connecticut",
    "ucf": "Central Florida",
}


@dataclass
class MatchRecord:
    year: int
    round_index: int
    round_name: str
    team_a: str
    team_b: str
    seed_a: int
    seed_b: int
    score_a: int
    score_b: int
    winner: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical matchup training data from tournament games.")
    parser.add_argument(
        "--team-data",
        default="data/processed/March_Madness_Train_Model_rebuilt.csv",
        help="Path to the rebuilt season-level dataset.",
    )
    parser.add_argument("--raw-root", default="data/raw", help="Root folder containing per-year tournament game files.")
    parser.add_argument(
        "--aliases",
        default="data/team_aliases.csv",
        help="Team alias CSV used for canonical naming.",
    )
    parser.add_argument(
        "--manual-corrections",
        default="data/manual_tournament_game_corrections.csv",
        help="Optional CSV of manual game corrections.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/matchup_training_data.csv",
        help="Output CSV for the game-level training dataset.",
    )
    parser.add_argument(
        "--match-report",
        default="data/processed/matchup_training_match_report.csv",
        help="Output CSV for team-name matching details.",
    )
    return parser.parse_args()


def normalize_for_matching(name: str) -> str:
    text = normalize_team_name(name)
    text = text.replace("christn", "christian")
    text = text.replace("mount st mary s", "mount st marys")
    text = text.replace("st mary s", "st marys")
    text = text.replace("st francis", "st francis")
    text = text.replace("st peter s", "st peters")
    text = text.replace("n dakota", "north dakota")
    text = text.replace("s dakota", "south dakota")
    text = text.replace("e washington", "east washington")
    text = text.replace("g washington", "george washington")
    text = text.replace("e kentucky", "eastern kentucky")
    text = text.replace("w michigan", "western michigan")
    text = text.replace("jax state", "jacksonville st")
    text = text.replace("se missouri", "southeast missouri")
    text = text.replace("sf austin", "stephen f austin")
    text = text.replace("app state", "appalachian st")
    text = text.replace("saint", "st")
    text = text.replace("n western", "northwestern")
    text = text.replace("miami fl", "miami")
    text = text.replace("ole miss", "mississippi")
    text = text.replace("college of charleston", "charleston")
    text = text.replace("col charlestn", "charleston")
    text = text.replace("louisiana lafayette", "louisiana")
    text = text.replace("east tennessee st", "etsu")
    text = text.replace("texas a m corpus chris", "texas a m cc")
    text = text.replace("texas a m corpus christi", "texas a m cc")
    text = text.replace("texas a and m corpus chris", "texas a and m cc")
    text = text.replace("texas a and m corpus christi", "texas a and m cc")
    return " ".join(text.split())


class SeasonTeamMatcher:
    def __init__(self, season_teams: list[str], alias_map: dict[tuple[str, str], str]) -> None:
        self.season_teams = sorted(set(display_team_name(team) for team in season_teams))
        self.alias_map = alias_map
        self.normalized_to_team = {normalize_for_matching(team): team for team in self.season_teams}
        self.rows: list[dict[str, str | int]] = []

    def resolve(self, source: str, raw_name: str) -> str:
        display_name = display_team_name(raw_name)
        normalized = normalize_for_matching(display_name)
        source_key = source.lower()

        method = "season_cleaned"
        canonical = display_name

        if (source_key, normalized) in self.alias_map:
            canonical = self.resolve_alias_target(self.alias_map[(source_key, normalized)])
            method = "source_alias"
        elif ("all", normalized) in self.alias_map:
            canonical = self.resolve_alias_target(self.alias_map[("all", normalized)])
            method = "global_alias"
        elif normalized in SPECIAL_ABBREVIATIONS:
            canonical = SPECIAL_ABBREVIATIONS[normalized]
            method = "abbreviation_alias"
        elif normalized in self.normalized_to_team:
            canonical = self.normalized_to_team[normalized]
            method = "season_exact"
        else:
            candidates = list(self.normalized_to_team.keys())
            matches = difflib.get_close_matches(normalized, candidates, n=2, cutoff=0.74)
            if len(matches) == 1:
                canonical = self.normalized_to_team[matches[0]]
                method = "season_fuzzy"

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

    def resolve_alias_target(self, alias_target: str) -> str:
        display_name = display_team_name(alias_target)
        if display_name in self.season_teams:
            return display_name

        normalized = normalize_for_matching(display_name)
        if normalized in self.normalized_to_team:
            return self.normalized_to_team[normalized]

        candidates = list(self.normalized_to_team.keys())
        matches = difflib.get_close_matches(normalized, candidates, n=1, cutoff=0.8)
        if matches:
            return self.normalized_to_team[matches[0]]

        return display_name

    def report(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["source", "raw_team", "normalized_team", "canonical_team", "match_method"])
        return pd.DataFrame(self.rows).drop_duplicates()


def load_manual_corrections(path: str | Path) -> pd.DataFrame:
    correction_path = Path(path)
    if not correction_path.exists():
        return pd.DataFrame(columns=["year", "round_name", "seed_a", "team_a", "score_a", "seed_b", "team_b", "score_b"])
    return pd.read_csv(correction_path)


def determine_round_index(raw_round_name: str) -> int | None:
    if raw_round_name == "Championship":
        return 6
    if raw_round_name == "Final Four":
        return 5
    if raw_round_name == "First Four":
        return 0
    if raw_round_name in {"East", "West", "South", "Midwest"}:
        return 1
    return ROUND_ORDER.get(raw_round_name)


def infer_regional_round_index(seed_a: int, seed_b: int) -> int:
    seed_pair = {int(seed_a), int(seed_b)}
    if any(seed_pair == pod for pod in REGIONAL_FIRST_ROUND_PODS):
        return 1
    if any(seed_pair.issubset(group) for group in REGIONAL_SECOND_ROUND_GROUPS):
        return 2
    if any(seed_pair.issubset(group) for group in REGIONAL_SWEET_SIXTEEN_GROUPS):
        return 3
    return 4


def load_games_for_year(
    year: int,
    raw_root: Path,
    season_teams: list[str],
    alias_map: dict[tuple[str, str], str],
    manual_corrections: pd.DataFrame,
) -> tuple[list[MatchRecord], pd.DataFrame]:
    games_path = raw_root / str(year) / "tournament_games.csv"
    games_df = pd.read_csv(games_path)

    corrections = manual_corrections.loc[manual_corrections["year"] == year].copy()
    if not corrections.empty:
        games_df = pd.concat([games_df, corrections], ignore_index=True, sort=False)

    matcher = SeasonTeamMatcher(season_teams=season_teams, alias_map=alias_map)
    records: list[MatchRecord] = []

    for row in games_df.itertuples(index=False):
        raw_round = str(row.round_name)
        if raw_round in {"East", "West", "South", "Midwest"}:
            round_index = infer_regional_round_index(int(row.seed_a), int(row.seed_b))
        else:
            round_index = determine_round_index(raw_round)
        if round_index in {None, 0}:
            continue

        score_a = int(row.score_a)
        score_b = int(row.score_b)
        if score_a == score_b:
            continue

        team_a = matcher.resolve("games", row.team_a)
        team_b = matcher.resolve("games", row.team_b)

        winner = team_a if score_a > score_b else team_b
        records.append(
            MatchRecord(
                year=year,
                round_index=round_index,
                round_name=MAIN_BRACKET_ROUND_NAME[round_index],
                team_a=team_a,
                team_b=team_b,
                seed_a=int(row.seed_a),
                seed_b=int(row.seed_b),
                score_a=score_a,
                score_b=score_b,
                winner=winner,
            )
        )

    deduped: list[MatchRecord] = []
    seen: set[tuple[int, str, str, int, int]] = set()
    for record in records:
        key = (
            record.round_index,
            tuple(sorted([record.team_a, record.team_b]))[0],
            tuple(sorted([record.team_a, record.team_b]))[1],
            max(record.score_a, record.score_b),
            min(record.score_a, record.score_b),
        )
        if key not in seen:
            deduped.append(record)
            seen.add(key)

    return deduped, matcher.report()


def build_matchup_rows(games: list[MatchRecord], season_df: pd.DataFrame) -> pd.DataFrame:
    team_lookup = season_df.set_index("team")
    rows: list[dict[str, int | float | str]] = []

    for game in games:
        if game.team_a not in team_lookup.index or game.team_b not in team_lookup.index:
            continue

        a = team_lookup.loc[game.team_a]
        b = team_lookup.loc[game.team_b]

        def add_row(left_name: str, right_name: str, left: pd.Series, right: pd.Series, left_seed: int, right_seed: int, left_score: int, right_score: int, label: int) -> None:
            row: dict[str, int | float | str] = {
                "year": game.year,
                "round_index": game.round_index,
                "round_name": game.round_name,
                "team_left": left_name,
                "team_right": right_name,
                "left_seed": left_seed,
                "right_seed": right_seed,
                "seed_diff": left_seed - right_seed,
                "seed_abs_diff": abs(left_seed - right_seed),
                "left_score": left_score,
                "right_score": right_score,
                "margin": left_score - right_score,
                "left_win": label,
            }

            for feature in FEATURE_COLUMNS:
                if feature == "seed":
                    continue
                left_value = float(left[feature])
                right_value = float(right[feature])
                row[f"{feature}_left"] = left_value
                row[f"{feature}_right"] = right_value
                row[f"{feature}_diff"] = left_value - right_value
            rows.append(row)

        add_row(game.team_a, game.team_b, a, b, game.seed_a, game.seed_b, game.score_a, game.score_b, int(game.winner == game.team_a))
        add_row(game.team_b, game.team_a, b, a, game.seed_b, game.seed_a, game.score_b, game.score_a, int(game.winner == game.team_b))

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    team_df = pd.read_csv(args.team_data)
    alias_map = load_aliases(args.aliases)
    manual_corrections = load_manual_corrections(args.manual_corrections)

    seasons = sorted(team_df["year"].unique().tolist())
    raw_root = Path(args.raw_root)

    all_rows: list[pd.DataFrame] = []
    reports: list[pd.DataFrame] = []
    summary_rows: list[dict[str, int]] = []

    for year in seasons:
        games_path = raw_root / str(year) / "tournament_games.csv"
        if not games_path.exists():
            continue

        season_df = team_df.loc[team_df["year"] == year].copy()
        games, report = load_games_for_year(
            year=year,
            raw_root=raw_root,
            season_teams=season_df["team"].tolist(),
            alias_map=alias_map,
            manual_corrections=manual_corrections,
        )
        matchup_df = build_matchup_rows(games, season_df)
        if not matchup_df.empty:
            all_rows.append(matchup_df)

        report["year"] = year
        reports.append(report)
        summary_rows.append({"year": year, "games": len(games), "rows": len(matchup_df)})

    output = pd.concat(all_rows, ignore_index=True).sort_values(["year", "round_index", "team_left", "team_right"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    report_df = pd.concat(reports, ignore_index=True).drop_duplicates()
    report_path = Path(args.match_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    pd.DataFrame(summary_rows).to_csv(output_path.with_name(f"{output_path.stem}_summary.csv"), index=False)

    print(f"Wrote matchup training data to {output_path}")
    print(f"Wrote match report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
