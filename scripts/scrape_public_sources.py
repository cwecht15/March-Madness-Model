#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode


KENPOM_URL = "https://kenpom.com/index.php?y={season}"
ESPN_BRACKET_URL = "https://www.espn.com/mens-college-basketball/bracket/_/season/{season}/{season}-ncaa-tournament"

TEAMRANKINGS_STATS: list[tuple[str, str, str]] = [
    ("Seas_PPG", "points-per-game", "raw"),
    ("Seas_Succ_3PT", "percent-of-points-from-3-pointers", "pct"),
    ("Seas_3PT_Per", "three-point-pct", "pct"),
    ("Seas_FT_Succ", "free-throw-pct", "pct"),
    ("Seas_3PT_Rate", "three-point-rate", "pct"),
    ("Seas_Off_Rebound_Per", "offensive-rebounding-pct", "pct"),
    ("Seas_Def_Rebound_Per", "defensive-rebounding-pct", "pct"),
    ("Seas_Turnovers", "turnovers-per-game", "raw"),
    ("Seas_Fouls", "personal-fouls-per-game", "raw"),
    ("Seas_Opp_PPG", "opponent-points-per-game", "raw"),
    ("Seas_Opp_3PT_Succ", "opponent-three-point-pct", "pct"),
    ("Seas_Opp_3PT_Rate", "opponent-three-point-rate", "pct"),
    ("Seas_Opp_Off_Rebound", "opponent-offensive-rebounding-pct", "pct"),
    ("Seas_Opp_Def_Rebound", "opponent-defensive-rebounding-pct", "pct"),
    ("Seas_Opp_Turnover", "opponent-turnovers-per-game", "raw"),
    ("Seas_Opp_Fouls", "opponent-personal-fouls-per-game", "raw"),
    ("Seas_Poss", "possessions-per-game", "raw"),
]

TEAMRANKINGS_BASE = "https://www.teamrankings.com/ncaa-basketball/stat/{slug}"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

CHAMPIONSHIP_ALIASES = {
    "DUKE": "Duke",
    "FLA": "Florida",
    "HOU": "Houston",
    "WIS": "Wisconsin",
}

RESULT_COLUMNS = ["team", "seed", "wins", "finish_label", "notes"]


@dataclass
class GameResult:
    round_name: str
    seed_a: int
    team_a: str
    score_a: int
    seed_b: int
    team_b: str
    score_b: int

    @property
    def winner(self) -> str:
        return self.team_a if self.score_a > self.score_b else self.team_b

    @property
    def loser(self) -> str:
        return self.team_b if self.score_a > self.score_b else self.team_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape public March Madness source pages for one season.")
    parser.add_argument("--season", type=int, default=2025, help="Tournament year / end year, e.g. 2025.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Raw output directory. Defaults to data/raw/<season>.",
    )
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory for generated TeamRankings manifests.",
    )
    parser.add_argument(
        "--teamrankings-date",
        default=None,
        help="Optional explicit TeamRankings date parameter, e.g. 2025-04-08 for the 2024-2025 season.",
    )
    parser.add_argument("--skip-kenpom", action="store_true", help="Skip KenPom scraping.")
    parser.add_argument("--skip-teamrankings", action="store_true", help="Skip TeamRankings scraping.")
    parser.add_argument("--skip-results", action="store_true", help="Skip ESPN bracket/results scraping.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


def build_url(base: str, **params: str | None) -> str:
    query = {key: value for key, value in params.items() if value}
    if not query:
        return base
    return f"{base}?{urlencode(query)}"


def fetch_tables(url: str) -> list[pd.DataFrame]:
    return pd.read_html(fetch_text(url))


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def clean_numeric(series: pd.Series) -> pd.Series:
    text = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(text, errors="coerce")


def scrape_kenpom(season: int, out_dir: Path) -> Path:
    url = KENPOM_URL.format(season=season)
    html = fetch_text(url)
    html_path = out_dir / f"kenpom_{season}.html"
    save_text(html_path, html)

    tables = pd.read_html(io.StringIO(html))
    target = None
    for table in tables:
        columns = {normalize_column(col) for col in table.columns}
        if {"team", "netrtg", "ortg", "drtg", "adjt", "luck"}.issubset(columns):
            target = table.copy()
            break

    if target is None:
        raise ValueError("Could not find the KenPom ratings table.")

    columns = {normalize_column(col): col for col in target.columns}
    team_col = columns["team"]
    out = pd.DataFrame(
        {
            "team": target[team_col].astype(str).str.strip(),
            "AdjEM": clean_numeric(target[columns["netrtg"]]),
            "AdjO": clean_numeric(target[columns["ortg"]]),
            "AdjD": clean_numeric(target[columns["drtg"]]),
            "AdjT": clean_numeric(target[columns["adjt"]]),
            "Luck": clean_numeric(target[columns["luck"]]),
        }
    )
    out = out.dropna(subset=["team"]).drop_duplicates(subset=["team"], keep="first")

    csv_path = out_dir / "kenpom_ratings.csv"
    out.to_csv(csv_path, index=False)
    return csv_path


def pick_teamrankings_value_column(df: pd.DataFrame, season: int) -> str:
    columns = {normalize_column(col): col for col in df.columns}
    preferred = [str(season), str(season - 1), "value"]
    for option in preferred:
        key = normalize_column(option)
        if key in columns:
            return columns[key]
    raise ValueError(f"Could not find a {season} value column in {list(df.columns)}")


def find_teamrankings_season_date(season: int) -> str:
    url = TEAMRANKINGS_BASE.format(slug="points-per-game")
    html = fetch_text(url)
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", {"id": "date"})
    if select is None:
        raise ValueError("Could not find the TeamRankings season selector.")

    target_label = f"{season - 1}-{season}"
    for option in select.find_all("option"):
        label = " ".join(option.get_text(" ", strip=True).split())
        if label == target_label:
            value = option.get("value")
            if value:
                return value

    raise ValueError(f"Could not find TeamRankings season option {target_label}.")


def scrape_teamrankings(season: int, out_dir: Path, config_dir: Path, season_date: str | None) -> Path:
    teamrankings_dir = out_dir / "teamrankings"
    ensure_dir(teamrankings_dir)
    manifest_rows: list[dict[str, str]] = []
    resolved_date = season_date or find_teamrankings_season_date(season)

    for output_column, slug, scale in TEAMRANKINGS_STATS:
        url = build_url(TEAMRANKINGS_BASE.format(slug=slug), date=resolved_date)
        html = fetch_text(url)
        html_path = teamrankings_dir / f"{slug}.html"
        save_text(html_path, html)

        tables = pd.read_html(io.StringIO(html))
        if not tables:
            raise ValueError(f"No tables found for TeamRankings stat {slug}")
        table = tables[0]
        columns = {normalize_column(col): col for col in table.columns}
        if "team" not in columns:
            raise ValueError(f"TeamRankings table for {slug} did not include a team column.")

        team_col = columns["team"]
        value_col = pick_teamrankings_value_column(table, season)

        out = pd.DataFrame(
            {
                "team": table[team_col].astype(str).str.strip(),
                output_column: clean_numeric(table[value_col]),
            }
        )
        if scale == "pct":
            out[output_column] = out[output_column] / 100.0

        csv_path = teamrankings_dir / f"{slug}.csv"
        out.to_csv(csv_path, index=False)

        manifest_rows.append(
            {
                "output_column": output_column,
                "source_file": str(csv_path).replace("\\", "/"),
                "scale": scale,
                "notes": f"Scraped from {url}",
            }
        )

    manifest_path = config_dir / f"teamrankings_{season}.csv"
    ensure_dir(manifest_path.parent)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    return manifest_path


def parse_espn_bracket_text(text: str) -> list[GameResult]:
    compact = re.sub(r"\s+", " ", text)
    generic_text = re.sub(
        r"Championship\s+[A-Za-z ,]+\s+\d+\s+\d+\s+[A-Z.]+\s+Final\s+\d+\s+\d+\s+[A-Z.]+",
        " ",
        compact,
    )
    generic_text = re.sub(
        r"Championship Banner\s+\d{4}-\d{2}\s+NCAA Champions",
        " ",
        generic_text,
    )
    round_markers = [
        ("First Four", "First Four"),
        ("SOUTH", "South"),
        ("EAST", "East"),
        ("WEST", "West"),
        ("MIDWEST", "Midwest"),
        ("Final Four", "Final Four"),
        ("Championship", "Championship"),
    ]

    anchors = []
    for label, round_name in round_markers:
        for match in re.finditer(rf"\b{re.escape(label)}\b", generic_text):
            anchors.append((match.start(), round_name))
    anchors.sort()

    def current_round(idx: int) -> str:
        current = "Unknown"
        for start, round_name in anchors:
            if start > idx:
                break
            current = round_name
        return current

    games: list[GameResult] = []
    pattern = re.compile(
        r"Final(?:/\d*OT|/OT)?\s+(\d+)\s+([A-Za-z0-9.'&\- ]+?)\s+(\d+)\s+(\d+)\s+([A-Za-z0-9.'&\- ]+?)\s+(\d+)"
    )
    for match in pattern.finditer(generic_text):
        round_name = current_round(match.start())
        seed_a = int(match.group(1))
        seed_b = int(match.group(4))
        if not (1 <= seed_a <= 16 and 1 <= seed_b <= 16):
            continue
        team_a = " ".join(match.group(2).split())
        team_b = " ".join(match.group(5).split())
        if "championship" in team_a.lower() or "championship" in team_b.lower():
            continue
        if seed_a == seed_b and round_name not in {"Final Four", "Championship"}:
            round_name = "First Four"
        games.append(
            GameResult(
                round_name=round_name,
                seed_a=seed_a,
                team_a=team_a,
                score_a=int(match.group(3)),
                seed_b=seed_b,
                team_b=team_b,
                score_b=int(match.group(6)),
            )
        )

    champ_match = re.search(
        r"Championship\s+[A-Za-z ,]+\s+(\d+)\s+(\d+)\s+([A-Z.]+)\s+Final\s+(\d+)\s+(\d+)\s+([A-Z.]+)",
        compact,
    )
    if champ_match:
        games.append(
            GameResult(
                round_name="Championship",
                seed_a=int(champ_match.group(2)),
                team_a=CHAMPIONSHIP_ALIASES.get(champ_match.group(3), champ_match.group(3)),
                score_a=int(champ_match.group(1)),
                seed_b=int(champ_match.group(5)),
                team_b=CHAMPIONSHIP_ALIASES.get(champ_match.group(6), champ_match.group(6)),
                score_b=int(champ_match.group(4)),
            )
        )

    deduped: list[GameResult] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for game in games:
        key = (game.round_name, game.team_a, game.team_b, game.score_a, game.score_b)
        if key not in seen:
            deduped.append(game)
            seen.add(key)
    return deduped


def finish_label_from_wins(wins: int) -> str:
    labels = {
        0: "First Round",
        1: "Round of 32",
        2: "Sweet 16",
        3: "Elite Eight",
        4: "Final Four",
        5: "Runner-up",
        6: "Champion",
    }
    return labels[wins]


def scrape_tournament_results(season: int, out_dir: Path) -> Path:
    url = ESPN_BRACKET_URL.format(season=season)
    html = fetch_text(url)
    html_path = out_dir / f"espn_bracket_{season}.html"
    save_text(html_path, html)

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    games = parse_espn_bracket_text(page_text)

    seeds: dict[str, int] = {}
    wins: dict[str, int] = {}
    teams: set[str] = set()

    for game in games:
        teams.add(game.team_a)
        teams.add(game.team_b)
        seeds.setdefault(game.team_a, game.seed_a)
        seeds.setdefault(game.team_b, game.seed_b)
        wins.setdefault(game.team_a, 0)
        wins.setdefault(game.team_b, 0)
        if game.round_name != "First Four":
            wins[game.winner] += 1

    rows = []
    for team in sorted(teams, key=lambda name: (seeds.get(name, 99), name)):
        team_wins = wins.get(team, 0)
        rows.append(
            {
                "team": team,
                "seed": seeds.get(team),
                "wins": team_wins,
                "finish_label": finish_label_from_wins(team_wins),
                "notes": "",
            }
        )

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    csv_path = out_dir / "tournament_results.csv"
    results.to_csv(csv_path, index=False)

    games_path = out_dir / "tournament_games.csv"
    pd.DataFrame([game.__dict__ for game in games]).to_csv(games_path, index=False)
    return csv_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir or f"data/raw/{args.season}")
    config_dir = Path(args.config_dir)

    ensure_dir(out_dir)
    ensure_dir(config_dir)

    if not args.skip_kenpom:
        kenpom_csv = scrape_kenpom(args.season, out_dir)
        print(f"Wrote KenPom ratings to {kenpom_csv}")

    if not args.skip_teamrankings:
        manifest_path = scrape_teamrankings(args.season, out_dir, config_dir, args.teamrankings_date)
        print(f"Wrote TeamRankings manifest to {manifest_path}")

    if not args.skip_results:
        results_csv = scrape_tournament_results(args.season, out_dir)
        print(f"Wrote tournament results to {results_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
