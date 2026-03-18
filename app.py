from __future__ import annotations

import __main__
import base64
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scripts.bracket_pdf import generate_bracket_pdf
from scripts.build_matchup_training_data import SeasonTeamMatcher, load_aliases
from scripts.run_tournament_forecast import (
    build_contender_scorecard,
    build_team_rankings,
    load_matchup_payload,
)
from scripts.simulate_bracket import (
    PAIRING_ORDER,
    ROUND_LABELS,
    load_team_model_predictions,
    parse_semifinal_pairs,
    predict_matchup_probability,
    prepare_region_seed_map,
    resolve_field,
    validate_field,
)
from scripts.train_clean_models import EncodedXGBClassifier as TeamEncodedXGBClassifier
from scripts.train_matchup_model import IdentityCalibrator, IsotonicCalibrator, PlattCalibrator


# Compatibility shim for joblib payloads that were trained when scripts were run
# as entry points and pickled custom classes under `__main__`.
EncodedXGBClassifier = TeamEncodedXGBClassifier
if not hasattr(__main__, "EncodedXGBClassifier"):
    __main__.EncodedXGBClassifier = TeamEncodedXGBClassifier
if not hasattr(__main__, "IdentityCalibrator"):
    __main__.IdentityCalibrator = IdentityCalibrator
if not hasattr(__main__, "PlattCalibrator"):
    __main__.PlattCalibrator = PlattCalibrator
if not hasattr(__main__, "IsotonicCalibrator"):
    __main__.IsotonicCalibrator = IsotonicCalibrator


DEFAULT_SEASON_DATA = "data/processed/march_madness_2026.csv"
DEFAULT_FIELD = "data/raw/2026/tournament_field.csv"
DEFAULT_TEAM_MANIFEST = "artifacts/model_benchmarks/saved_model_manifest.csv"
DEFAULT_MATCHUP_MANIFEST = "artifacts/matchup_model/saved_model_manifest.csv"
DEFAULT_ALIASES = "data/team_aliases.csv"
DEFAULT_SEMIFINALS = "East-South,West-Midwest"
DEFAULT_PUBLIC_PICKS = "data/public_pick_distribution/yahoo_pick_distribution_2026-03-18.csv"
ROUND_TITLES = {
    0: "First Four",
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}
ROUND_COLUMN_MAP = {
    "first_four": "win_first_four",
    "round_of_64": "win_round_of_64",
    "round_of_32": "win_round_of_32",
    "sweet_sixteen": "win_sweet_sixteen",
    "elite_eight": "win_elite_eight",
    "final_four": "win_final_four",
    "championship": "win_championship",
}
TEAM_ODDS_PROBABILITY_COLUMNS = [
    "win_first_four",
    "make_round_of_64",
    "make_round_of_32",
    "make_sweet_sixteen",
    "make_elite_eight",
    "make_final_four",
    "make_championship",
    "win_championship",
]
TEAM_ODDS_DELTA_COLUMNS = ["final_four_delta", "title_game_delta", "championship_delta"]
PUBLIC_PICK_ROUND_TITLE_MAP = {
    "Round of 64": "round_of_64",
    "Round of 32": "round_of_32",
    "Sweet 16": "sweet_sixteen",
    "Elite 8": "elite_eight",
    "Final Four": "final_four",
    "Championship": "championship",
}
PUBLIC_PICK_ROUND_WEIGHTS = {
    "round_of_64": 1,
    "round_of_32": 2,
    "sweet_sixteen": 4,
    "elite_eight": 8,
    "final_four": 16,
    "championship": 32,
}
PUBLIC_PICK_TEAM_ODDS_MAP = {
    "round_of_64": "win_round_of_64",
    "round_of_32": "win_round_of_32",
    "sweet_sixteen": "win_sweet_sixteen",
    "elite_eight": "win_elite_eight",
    "final_four": "win_final_four",
    "championship": "win_championship",
}


st.set_page_config(page_title="March Madness Live Bracket", layout="wide")
st.markdown(
    """
    <style>
    .st-key-bracket_sticky_header {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0.5rem !important;
        z-index: 40 !important;
        overflow: visible !important;
    }
    .st-key-bracket_sticky_header > div {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0.5rem !important;
        z-index: 40 !important;
        overflow: visible !important;
        background: var(--secondary-background-color, #1f2430);
        color: var(--text-color, #f5f7fb);
        border: 1px solid rgba(127, 127, 127, 0.28);
        border-radius: 16px;
        padding: 0.9rem 1rem 0.5rem 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(8px);
    }
    .st-key-bracket_sticky_header .stRadio > label {
        font-weight: 700;
    }
    .st-key-bracket_sticky_header [data-testid="stMetricLabel"],
    .st-key-bracket_sticky_header [data-testid="stMetricValue"],
    .st-key-bracket_sticky_header p,
    .st-key-bracket_sticky_header label,
    .st-key-bracket_sticky_header h1,
    .st-key-bracket_sticky_header h2,
    .st-key-bracket_sticky_header h3,
    .st-key-bracket_sticky_header h4 {
        color: var(--text-color, #f5f7fb);
    }
    .game-status-chip {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 0.45rem;
    }
    .game-status-unpicked {
        background: #fff1cc;
        color: #8a5a00;
        border: 1px solid #f0c36d;
    }
    .game-status-picked {
        background: #dff5e8;
        color: #0f5c2f;
        border: 1px solid #8dd3a8;
    }
    .team-line {
        font-size: 0.97rem;
        margin: 0.15rem 0;
    }
    .impact-line {
        font-size: 0.84rem;
        color: #4b5563;
        margin: 0.15rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_resources(
    season_data_path: str,
    field_path: str,
    semifinal_pairs_text: str,
    team_model_manifest: str,
    matchup_model_manifest: str,
    aliases_path: str,
    public_pick_distribution_path: str,
):
    season_df = pd.read_csv(season_data_path).copy()
    season_df = load_team_model_predictions(season_df, Path(team_model_manifest))

    raw_field = pd.read_csv(field_path)
    semifinal_pairs = parse_semifinal_pairs(semifinal_pairs_text)
    resolved_field, match_report = resolve_field(season_df, raw_field, Path(aliases_path))
    validate_field(resolved_field)

    _, matchup_payload = load_matchup_payload(Path(matchup_model_manifest))
    probability_temperature = float(matchup_payload.get("simulation_temperature", 1.0))
    team_lookup = {row.team: row for row in season_df.itertuples(index=False)}
    public_pick_distribution, public_pick_match_report = load_public_pick_distribution(
        season_df=season_df,
        public_pick_distribution_path=public_pick_distribution_path,
        aliases_path=aliases_path,
    )

    return {
        "season_df": season_df,
        "resolved_field": resolved_field,
        "semifinal_pairs": semifinal_pairs,
        "match_report": match_report,
        "matchup_payload": matchup_payload,
        "probability_temperature": probability_temperature,
        "team_lookup": team_lookup,
        "team_rankings": build_team_rankings(season_df),
        "contender_scorecard": build_contender_scorecard(season_df),
        "public_pick_distribution": public_pick_distribution,
        "public_pick_match_report": public_pick_match_report,
    }


@st.cache_data(show_spinner=False)
def load_public_pick_distribution(
    season_df: pd.DataFrame,
    public_pick_distribution_path: str,
    aliases_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(public_pick_distribution_path)
    empty_distribution = pd.DataFrame(columns=["round_key", "rank", "team", "public_team", "seed", "picked_pct"])
    empty_report = pd.DataFrame(columns=["source", "raw_team", "normalized_team", "canonical_team", "match_method"])
    if not path.exists():
        return empty_distribution, empty_report

    distribution = pd.read_csv(path).copy()
    if distribution.empty or "team" not in distribution.columns:
        return empty_distribution, empty_report

    alias_map = load_aliases(Path(aliases_path))
    matcher = SeasonTeamMatcher(season_df["team"].astype(str).tolist(), alias_map)
    distribution["public_team"] = distribution["team"].astype(str)
    distribution["team"] = distribution["team"].astype(str).apply(lambda value: matcher.resolve("public", value))
    distribution["picked_pct"] = distribution["picked_pct"].astype(float)
    distribution["round_key"] = distribution["round_key"].astype(str)
    distribution["seed"] = distribution["seed"].astype(int)
    return distribution, matcher.report()


def build_games(field_df: pd.DataFrame, semifinal_pairs: list[tuple[str, str]]):
    region_seed_map = prepare_region_seed_map(field_df)
    games: dict[str, dict[str, object]] = {}
    order: list[str] = []
    round_groups: dict[int, list[str]] = {index: [] for index in ROUND_TITLES}
    play_in_game_ids: dict[tuple[str, int], str] = {}

    for region in ["East", "South", "West", "Midwest"]:
        seed_map = region_seed_map[region]
        for seed in range(1, 17):
            teams = seed_map[seed]
            if len(teams) == 2:
                game_id = f"FF-{region}-{seed}"
                games[game_id] = {
                    "game_id": game_id,
                    "round_index": 0,
                    "round_title": ROUND_TITLES[0],
                    "label": f"{region} play-in ({seed}-seed)",
                    "left_source": ("team", teams[0]),
                    "right_source": ("team", teams[1]),
                    "region": region,
                }
                order.append(game_id)
                round_groups[0].append(game_id)
                play_in_game_ids[(region, seed)] = game_id

    for region in ["East", "South", "West", "Midwest"]:
        seed_map = region_seed_map[region]
        round_one_ids: list[str] = []
        for left_seed, right_seed in PAIRING_ORDER:
            left_source = (
                ("game", play_in_game_ids[(region, left_seed)])
                if (region, left_seed) in play_in_game_ids
                else ("team", seed_map[left_seed][0])
            )
            right_source = (
                ("game", play_in_game_ids[(region, right_seed)])
                if (region, right_seed) in play_in_game_ids
                else ("team", seed_map[right_seed][0])
            )
            game_id = f"R1-{region}-{left_seed}v{right_seed}"
            games[game_id] = {
                "game_id": game_id,
                "round_index": 1,
                "round_title": ROUND_TITLES[1],
                "label": f"{region} {left_seed}/{right_seed}",
                "left_source": left_source,
                "right_source": right_source,
                "region": region,
            }
            order.append(game_id)
            round_groups[1].append(game_id)
            round_one_ids.append(game_id)

        round_two_ids: list[str] = []
        for index in range(0, len(round_one_ids), 2):
            game_id = f"R2-{region}-{(index // 2) + 1}"
            games[game_id] = {
                "game_id": game_id,
                "round_index": 2,
                "round_title": ROUND_TITLES[2],
                "label": f"{region} Round of 32 {(index // 2) + 1}",
                "left_source": ("game", round_one_ids[index]),
                "right_source": ("game", round_one_ids[index + 1]),
                "region": region,
            }
            order.append(game_id)
            round_groups[2].append(game_id)
            round_two_ids.append(game_id)

        round_three_ids: list[str] = []
        for index in range(0, len(round_two_ids), 2):
            game_id = f"R3-{region}-{(index // 2) + 1}"
            games[game_id] = {
                "game_id": game_id,
                "round_index": 3,
                "round_title": ROUND_TITLES[3],
                "label": f"{region} Sweet 16 {(index // 2) + 1}",
                "left_source": ("game", round_two_ids[index]),
                "right_source": ("game", round_two_ids[index + 1]),
                "region": region,
            }
            order.append(game_id)
            round_groups[3].append(game_id)
            round_three_ids.append(game_id)

        game_id = f"R4-{region}"
        games[game_id] = {
            "game_id": game_id,
            "round_index": 4,
            "round_title": ROUND_TITLES[4],
            "label": f"{region} Elite 8",
            "left_source": ("game", round_three_ids[0]),
            "right_source": ("game", round_three_ids[1]),
            "region": region,
        }
        order.append(game_id)
        round_groups[4].append(game_id)

    semifinal_ids: list[str] = []
    for left_region, right_region in semifinal_pairs:
        game_id = f"R5-{left_region}-{right_region}"
        games[game_id] = {
            "game_id": game_id,
            "round_index": 5,
            "round_title": ROUND_TITLES[5],
            "label": f"{left_region} vs {right_region}",
            "left_source": ("game", f"R4-{left_region}"),
            "right_source": ("game", f"R4-{right_region}"),
            "region": "",
        }
        order.append(game_id)
        round_groups[5].append(game_id)
        semifinal_ids.append(game_id)

    final_id = "R6-Championship"
    games[final_id] = {
        "game_id": final_id,
        "round_index": 6,
        "round_title": ROUND_TITLES[6],
        "label": "National Championship",
        "left_source": ("game", semifinal_ids[0]),
        "right_source": ("game", semifinal_ids[1]),
        "region": "",
    }
    order.append(final_id)
    round_groups[6].append(final_id)
    return games, order, round_groups


def build_parent_lookup(games: dict[str, dict[str, object]]) -> dict[str, str]:
    parent_lookup: dict[str, str] = {}
    for game_id, game in games.items():
        for source_name in ("left_source", "right_source"):
            source_type, source_value = game[source_name]
            if source_type == "game":
                parent_lookup[str(source_value)] = game_id
    return parent_lookup


def resolve_source(source: tuple[str, str], winners: dict[str, str]) -> str | None:
    source_type, source_value = source
    if source_type == "team":
        return source_value
    return winners.get(source_value)


def game_probability(
    game: dict[str, object],
    left_team: str,
    right_team: str,
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_cache: dict[tuple[str, str, int], float],
    probability_temperature: float,
) -> float:
    return predict_matchup_probability(
        matchup_payload,
        team_lookup[left_team],
        team_lookup[right_team],
        int(game["round_index"]),
        probability_cache,
        True,
        probability_temperature,
    )


def sanitize_picks(games: dict[str, dict[str, object]], order: list[str], picks: dict[str, str]) -> dict[str, str]:
    winners: dict[str, str] = {}
    sanitized: dict[str, str] = {}
    for game_id in order:
        game = games[game_id]
        left_team = resolve_source(game["left_source"], winners)
        right_team = resolve_source(game["right_source"], winners)
        if not left_team or not right_team:
            continue
        picked = picks.get(game_id)
        if picked in {left_team, right_team}:
            sanitized[game_id] = picked
            winners[game_id] = picked
    return sanitized


def build_game_rows(
    games: dict[str, dict[str, object]],
    order: list[str],
    picks: dict[str, str],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
):
    rows: list[dict[str, object]] = []
    winners: dict[str, str] = {}
    probability_cache: dict[tuple[str, str, int], float] = {}

    for game_id in order:
        game = games[game_id]
        left_team = resolve_source(game["left_source"], winners)
        right_team = resolve_source(game["right_source"], winners)
        if not left_team or not right_team:
            continue

        left_probability = game_probability(
            game,
            left_team,
            right_team,
            team_lookup,
            matchup_payload,
            probability_cache,
            probability_temperature,
        )
        picked = picks.get(game_id)
        if picked:
            winners[game_id] = picked

        rows.append(
            {
                "game_id": game_id,
                "round_index": int(game["round_index"]),
                "round_title": str(game["round_title"]),
                "label": str(game["label"]),
                "left_team": left_team,
                "right_team": right_team,
                "left_probability": left_probability,
                "right_probability": 1.0 - left_probability,
                "picked_winner": picked or "",
            }
        )

    return rows


def candidate_teams_for_source(
    source: tuple[str, str],
    game_row_map: dict[str, dict[str, object]],
) -> list[str]:
    source_type, source_value = source
    if source_type == "team":
        return [str(source_value)]

    sibling_row = game_row_map.get(str(source_value))
    if sibling_row is None:
        return []
    if sibling_row["picked_winner"]:
        return [str(sibling_row["picked_winner"])]
    return [str(sibling_row["left_team"]), str(sibling_row["right_team"])]


def build_next_round_preview(
    game_id: str,
    row: dict[str, object],
    games: dict[str, dict[str, object]],
    parent_lookup: dict[str, str],
    game_row_map: dict[str, dict[str, object]],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
) -> list[dict[str, object]]:
    parent_id = parent_lookup.get(game_id)
    if not parent_id:
        return []

    parent_game = games[parent_id]
    if parent_game["left_source"] == ("game", game_id):
        sibling_source = parent_game["right_source"]
        candidate_on_left = True
    else:
        sibling_source = parent_game["left_source"]
        candidate_on_left = False

    sibling_candidates = candidate_teams_for_source(sibling_source, game_row_map)
    if not sibling_candidates:
        return []

    current_candidates = [str(row["left_team"]), str(row["right_team"])]
    probability_cache: dict[tuple[str, str, int], float] = {}
    preview_rows: list[dict[str, object]] = []

    for candidate in current_candidates:
        for opponent in sibling_candidates:
            if candidate_on_left:
                left_team, right_team = candidate, opponent
            else:
                left_team, right_team = opponent, candidate

            left_probability = game_probability(
                parent_game,
                left_team,
                right_team,
                team_lookup,
                matchup_payload,
                probability_cache,
                probability_temperature,
            )
            candidate_probability = left_probability if candidate_on_left else 1.0 - left_probability
            opponent_probability = 1.0 - candidate_probability
            preview_rows.append(
                {
                    "candidate": candidate,
                    "opponent": opponent,
                    "candidate_probability": candidate_probability,
                    "opponent_probability": opponent_probability,
                    "next_round_title": parent_game["round_title"],
                }
            )

    return preview_rows


def render_game_card(
    row: dict[str, object],
    games: dict[str, dict[str, object]],
    parent_lookup: dict[str, str],
    game_row_map: dict[str, dict[str, object]],
    seed_lookup: dict[str, int],
    current_odds_lookup: dict[str, dict[str, object]],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    saved_picks: dict[str, str],
    desired_picks: dict[str, str],
    show_championship_odds: bool = False,
) -> None:
    with st.container(border=True):
        picked = saved_picks.get(str(row["game_id"]), "")
        status_class = "game-status-picked" if picked else "game-status-unpicked"
        status_text = "Picked" if picked else "Unpicked"
        st.markdown(
            f"<div class='game-status-chip {status_class}'>{status_text}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**{row['label']}**")
        st.caption(row["round_title"])

        current_pick = saved_picks.get(row["game_id"], "Unpicked")
        options = ["Unpicked", row["left_team"], row["right_team"]]
        default_index = options.index(current_pick) if current_pick in options else 0

        choice = st.radio(
            f"Pick {row['label']}",
            options=options,
            index=default_index,
            key=f"pick::{row['game_id']}",
            horizontal=True,
            label_visibility="collapsed",
        )

        st.write(
            f"{format_seeded_team(str(row['left_team']), seed_lookup)}: {format_probability(row['left_probability'])}\n\n"
            f"{format_seeded_team(str(row['right_team']), seed_lookup)}: {format_probability(row['right_probability'])}"
        )
        left_odds = current_odds_lookup.get(str(row["left_team"]), {})
        right_odds = current_odds_lookup.get(str(row["right_team"]), {})
        odds_text = (
            "Final Four odds"
            f"\n\n{format_seeded_team(str(row['left_team']), seed_lookup)}: {format_probability(left_odds.get('make_final_four', 0.0))}"
            f"\n\n{format_seeded_team(str(row['right_team']), seed_lookup)}: {format_probability(right_odds.get('make_final_four', 0.0))}"
        )
        if show_championship_odds:
            odds_text += (
                "\n\nChampionship odds"
                f"\n\n{format_seeded_team(str(row['left_team']), seed_lookup)}: {format_probability(left_odds.get('win_championship', 0.0))}"
                f"\n\n{format_seeded_team(str(row['right_team']), seed_lookup)}: {format_probability(right_odds.get('win_championship', 0.0))}"
            )
        st.caption(odds_text)

        preview_rows = build_next_round_preview(
            game_id=str(row["game_id"]),
            row=row,
            games=games,
            parent_lookup=parent_lookup,
            game_row_map=game_row_map,
            team_lookup=team_lookup,
            matchup_payload=matchup_payload,
            probability_temperature=probability_temperature,
        )
        if preview_rows:
            st.caption("Next game impact")
            for preview in preview_rows:
                st.write(
                    f"If `{format_seeded_team(str(preview['candidate']), seed_lookup)}` advances: "
                    f"`{format_seeded_team(str(preview['candidate']), seed_lookup)}` "
                    f"{format_probability(preview['candidate_probability'])} "
                    f"vs `{format_seeded_team(str(preview['opponent']), seed_lookup)}` in {preview['next_round_title']}"
                )

        if choice == "Unpicked":
            desired_picks.pop(str(row["game_id"]), None)
        else:
            desired_picks[str(row["game_id"])] = str(choice)


def render_region_bracket(
    region: str,
    round_groups: dict[int, list[str]],
    game_row_map: dict[str, dict[str, object]],
    games: dict[str, dict[str, object]],
    parent_lookup: dict[str, str],
    seed_lookup: dict[str, int],
    current_odds_lookup: dict[str, dict[str, object]],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    saved_picks: dict[str, str],
    desired_picks: dict[str, str],
) -> None:
    play_in_rows = [
        game_row_map[game_id]
        for game_id in round_groups[0]
        if game_id in game_row_map and games[game_id]["region"] == region
    ]
    if play_in_rows:
        st.markdown("#### First Four")
        play_cols = st.columns(max(len(play_in_rows), 1))
        for index, row in enumerate(play_in_rows):
            with play_cols[index]:
                render_game_card(
                    row=row,
                    games=games,
                    parent_lookup=parent_lookup,
                    game_row_map=game_row_map,
                    seed_lookup=seed_lookup,
                    current_odds_lookup=current_odds_lookup,
                    team_lookup=team_lookup,
                    matchup_payload=matchup_payload,
                    probability_temperature=probability_temperature,
                    saved_picks=saved_picks,
                    desired_picks=desired_picks,
                    show_championship_odds=False,
                )

    st.markdown(f"#### {region} Bracket")
    round_layout = [(1, "Round of 64"), (2, "Round of 32"), (3, "Sweet 16"), (4, "Elite 8")]
    cols = st.columns(len(round_layout))
    for col, (round_index, title) in zip(cols, round_layout):
        with col:
            st.markdown(f"**{title}**")
            for game_id in round_groups[round_index]:
                if games[game_id]["region"] != region or game_id not in game_row_map:
                    continue
                render_game_card(
                    row=game_row_map[game_id],
                    games=games,
                    parent_lookup=parent_lookup,
                    game_row_map=game_row_map,
                    seed_lookup=seed_lookup,
                    current_odds_lookup=current_odds_lookup,
                    team_lookup=team_lookup,
                    matchup_payload=matchup_payload,
                    probability_temperature=probability_temperature,
                    saved_picks=saved_picks,
                    desired_picks=desired_picks,
                    show_championship_odds=False,
                )


def render_national_bracket(
    round_groups: dict[int, list[str]],
    game_row_map: dict[str, dict[str, object]],
    games: dict[str, dict[str, object]],
    parent_lookup: dict[str, str],
    seed_lookup: dict[str, int],
    current_odds_lookup: dict[str, dict[str, object]],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    saved_picks: dict[str, str],
    desired_picks: dict[str, str],
) -> None:
    st.markdown("#### Final Four and Championship")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Final Four**")
        for game_id in round_groups[5]:
            if game_id not in game_row_map:
                continue
            render_game_card(
                row=game_row_map[game_id],
                games=games,
                parent_lookup=parent_lookup,
                game_row_map=game_row_map,
                seed_lookup=seed_lookup,
                current_odds_lookup=current_odds_lookup,
                team_lookup=team_lookup,
                matchup_payload=matchup_payload,
                probability_temperature=probability_temperature,
                saved_picks=saved_picks,
                desired_picks=desired_picks,
                show_championship_odds=True,
            )
    with cols[1]:
        st.markdown("**Championship**")
        for game_id in round_groups[6]:
            if game_id not in game_row_map:
                continue
            render_game_card(
                row=game_row_map[game_id],
                games=games,
                parent_lookup=parent_lookup,
                game_row_map=game_row_map,
                seed_lookup=seed_lookup,
                current_odds_lookup=current_odds_lookup,
                team_lookup=team_lookup,
                matchup_payload=matchup_payload,
                probability_temperature=probability_temperature,
                saved_picks=saved_picks,
                desired_picks=desired_picks,
                show_championship_odds=True,
            )


def simulate_with_picks(
    games: dict[str, dict[str, object]],
    order: list[str],
    field_df: pd.DataFrame,
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    picks: dict[str, str],
    n_sims: int,
) -> pd.DataFrame:
    teams = sorted(field_df["team"].astype(str).tolist())
    counts = {team: {label: 0 for label in ROUND_LABELS.values()} for team in teams}
    rng = np.random.default_rng(42)
    probability_cache: dict[tuple[str, str, int], float] = {}

    for _ in range(n_sims):
        winners: dict[str, str] = {}
        for game_id in order:
            game = games[game_id]
            left_team = resolve_source(game["left_source"], winners)
            right_team = resolve_source(game["right_source"], winners)
            if not left_team or not right_team:
                continue

            if game_id in picks:
                winner = picks[game_id]
            else:
                left_probability = game_probability(
                    game,
                    left_team,
                    right_team,
                    team_lookup,
                    matchup_payload,
                    probability_cache,
                    probability_temperature,
                )
                winner = left_team if rng.random() < left_probability else right_team

            winners[game_id] = winner
            counts[winner][ROUND_LABELS[int(game["round_index"])]] += 1

    rows: list[dict[str, object]] = []
    for row in field_df.itertuples(index=False):
        team_counts = counts[row.team]
        rows.append(
            {
                "team": row.team,
                "seed": int(row.seed),
                "region": row.region,
                "play_in_group": getattr(row, "play_in_group", ""),
                "expected_wins": sum(team_counts.values()) / n_sims,
                "win_first_four": team_counts["first_four"] / n_sims,
                "win_round_of_64": team_counts["round_of_64"] / n_sims,
                "win_round_of_32": team_counts["round_of_32"] / n_sims,
                "win_sweet_sixteen": team_counts["sweet_sixteen"] / n_sims,
                "win_elite_eight": team_counts["elite_eight"] / n_sims,
                "win_final_four": team_counts["final_four"] / n_sims,
                "win_championship": team_counts["championship"] / n_sims,
            }
        )

    odds = pd.DataFrame(rows)
    play_in_mask = odds["play_in_group"].fillna("").astype(str).str.strip().ne("")
    odds["make_round_of_64"] = np.where(play_in_mask, odds["win_first_four"], 1.0)
    odds["make_round_of_32"] = odds["win_round_of_64"]
    odds["make_sweet_sixteen"] = odds["win_round_of_32"]
    odds["make_elite_eight"] = odds["win_sweet_sixteen"]
    odds["make_final_four"] = odds["win_elite_eight"]
    odds["make_championship"] = odds["win_final_four"]
    return odds.sort_values(["win_championship", "make_final_four", "expected_wins"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def cached_simulation(
    season_data_path: str,
    field_path: str,
    semifinal_pairs_text: str,
    team_model_manifest: str,
    matchup_model_manifest: str,
    aliases_path: str,
    public_pick_distribution_path: str,
    picks_key: tuple[tuple[str, str], ...],
    n_sims: int,
):
    resources = load_resources(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        team_model_manifest,
        matchup_model_manifest,
        aliases_path,
        public_pick_distribution_path,
    )
    games, order, _ = build_games(resources["resolved_field"], resources["semifinal_pairs"])
    picks = dict(picks_key)
    picks = sanitize_picks(games, order, picks)
    return simulate_with_picks(
        games=games,
        order=order,
        field_df=resources["resolved_field"],
        team_lookup=resources["team_lookup"],
        matchup_payload=resources["matchup_payload"],
        probability_temperature=resources["probability_temperature"],
        picks=picks,
        n_sims=n_sims,
    )


def format_probability(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def build_delta_table(current_odds: pd.DataFrame, baseline_odds: pd.DataFrame) -> pd.DataFrame:
    merged = current_odds.merge(
        baseline_odds[["team", "win_championship", "make_final_four", "make_championship"]],
        on="team",
        suffixes=("", "_baseline"),
        how="left",
    )
    merged["championship_delta"] = merged["win_championship"] - merged["win_championship_baseline"]
    merged["final_four_delta"] = merged["make_final_four"] - merged["make_final_four_baseline"]
    merged["title_game_delta"] = merged["make_championship"] - merged["make_championship_baseline"]
    return merged


def build_bracket_public_pick_table(
    export_df: pd.DataFrame,
    public_pick_distribution: pd.DataFrame,
    current_odds_lookup: dict[str, dict[str, object]],
) -> pd.DataFrame:
    picked = export_df.loc[export_df["picked_winner"].astype(str).str.strip().ne("")].copy()
    if picked.empty or public_pick_distribution.empty:
        return pd.DataFrame()

    picked["round_key"] = picked["round_title"].map(PUBLIC_PICK_ROUND_TITLE_MAP)
    picked = picked.loc[picked["round_key"].notna()].copy()
    if picked.empty:
        return pd.DataFrame()

    picked["team"] = picked["picked_winner"].astype(str)
    merged = picked.merge(
        public_pick_distribution[["team", "round_key", "picked_pct", "rank", "public_team"]],
        on=["team", "round_key"],
        how="left",
    )
    merged["round_weight"] = merged["round_key"].map(PUBLIC_PICK_ROUND_WEIGHTS).astype(float)
    merged["picked_pct"] = merged["picked_pct"].fillna(0.0).astype(float)
    merged["public_rank"] = merged["rank"].fillna(999).astype(int)
    merged["model_round_prob"] = merged.apply(
        lambda row: float(
            current_odds_lookup.get(str(row["team"]), {}).get(PUBLIC_PICK_TEAM_ODDS_MAP[str(row["round_key"])], 0.0)
        ),
        axis=1,
    )
    merged["leverage"] = merged["model_round_prob"] - merged["picked_pct"]
    merged["expected_same_picks_in_pool"] = 0.0
    return merged


def score_bracket_pool_fit(
    bracket_pick_table: pd.DataFrame,
    pool_size: int,
) -> dict[str, object]:
    if bracket_pick_table.empty:
        return {
            "profile": "Unavailable",
            "pool_fit": "No picked rounds to score",
            "recommended_pool_min": None,
            "recommended_pool_max": None,
            "weighted_public_pct": np.nan,
            "weighted_late_public_pct": np.nan,
            "weighted_leverage_pct": np.nan,
            "champion_public_pct": np.nan,
            "expected_same_champion_entries": np.nan,
            "final_four_avg_public_pct": np.nan,
            "title_game_avg_public_pct": np.nan,
            "final_four_unique_count": 0,
            "title_game_unique_count": 0,
            "late_path_popularity_pct": np.nan,
        }

    weight_sum = float(bracket_pick_table["round_weight"].sum())
    weighted_public_pct = float((bracket_pick_table["picked_pct"] * bracket_pick_table["round_weight"]).sum() / weight_sum)
    weighted_leverage_pct = float((bracket_pick_table["leverage"] * bracket_pick_table["round_weight"]).sum() / weight_sum)
    late_rows = bracket_pick_table.loc[
        bracket_pick_table["round_key"].isin(["elite_eight", "final_four", "championship"])
    ].copy()
    late_weight_sum = float(late_rows["round_weight"].sum()) if not late_rows.empty else 0.0
    weighted_late_public_pct = (
        float((late_rows["picked_pct"] * late_rows["round_weight"]).sum() / late_weight_sum) if late_weight_sum else 0.0
    )

    final_four_rows = bracket_pick_table.loc[bracket_pick_table["round_key"] == "elite_eight"].copy()
    title_game_rows = bracket_pick_table.loc[bracket_pick_table["round_key"] == "final_four"].copy()
    champion_rows = bracket_pick_table.loc[bracket_pick_table["round_key"] == "championship"]
    champion_public_pct = float(champion_rows["picked_pct"].iloc[0]) if not champion_rows.empty else np.nan
    expected_same_champion_entries = champion_public_pct * pool_size if not math.isnan(champion_public_pct) else np.nan
    final_four_avg_public_pct = (
        float(final_four_rows["picked_pct"].mean()) if not final_four_rows.empty else np.nan
    )
    title_game_avg_public_pct = (
        float(title_game_rows["picked_pct"].mean()) if not title_game_rows.empty else np.nan
    )
    final_four_unique_count = int((final_four_rows["picked_pct"] <= 0.12).sum()) if not final_four_rows.empty else 0
    title_game_unique_count = int((title_game_rows["picked_pct"] <= 0.12).sum()) if not title_game_rows.empty else 0

    champion_component = 0.0 if math.isnan(champion_public_pct) else champion_public_pct
    title_component = 0.0 if math.isnan(title_game_avg_public_pct) else title_game_avg_public_pct
    final_four_component = 0.0 if math.isnan(final_four_avg_public_pct) else final_four_avg_public_pct
    diversity_bonus = (0.025 * final_four_unique_count) + (0.015 * title_game_unique_count)
    late_path_popularity_pct = max(
        0.0,
        (0.50 * champion_component) + (0.30 * title_component) + (0.20 * final_four_component) - diversity_bonus,
    )

    if (
        late_path_popularity_pct >= 0.30 and final_four_unique_count == 0
    ) or (
        champion_component >= 0.25
        and final_four_component >= 0.18
        and final_four_unique_count == 0
        and title_game_unique_count == 0
    ):
        profile = "Chalky"
        recommended_pool_min, recommended_pool_max = 10, 40
    elif late_path_popularity_pct >= 0.12:
        profile = "Balanced"
        recommended_pool_min, recommended_pool_max = 25, 150
    elif late_path_popularity_pct >= 0.06:
        profile = "Contrarian"
        recommended_pool_min, recommended_pool_max = 75, 500
    else:
        profile = "Very Contrarian"
        recommended_pool_min, recommended_pool_max = 250, None

    if pool_size < recommended_pool_min:
        pool_fit = "Too crazy for this pool size"
    elif recommended_pool_max is not None and pool_size > recommended_pool_max:
        pool_fit = "Not crazy enough for this pool size"
    else:
        pool_fit = "Reasonable fit for this pool size"

    return {
        "profile": profile,
        "pool_fit": pool_fit,
        "recommended_pool_min": recommended_pool_min,
        "recommended_pool_max": recommended_pool_max,
        "weighted_public_pct": weighted_public_pct,
        "weighted_late_public_pct": weighted_late_public_pct,
        "weighted_leverage_pct": weighted_leverage_pct,
        "champion_public_pct": champion_public_pct,
        "expected_same_champion_entries": expected_same_champion_entries,
        "final_four_avg_public_pct": final_four_avg_public_pct,
        "title_game_avg_public_pct": title_game_avg_public_pct,
        "final_four_unique_count": final_four_unique_count,
        "title_game_unique_count": title_game_unique_count,
        "late_path_popularity_pct": late_path_popularity_pct,
    }


def build_simulated_bracket_public_summary(
    bracket_results: list[dict[str, object]],
    games: dict[str, dict[str, object]],
    order: list[str],
    seed_lookup: dict[str, int],
    public_pick_distribution: pd.DataFrame,
    current_odds_lookup: dict[str, dict[str, object]],
    pool_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in bracket_results:
        export_df = export_picks_dataframe(games, order, dict(result["picks"]), seed_lookup)
        public_table = build_bracket_public_pick_table(export_df, public_pick_distribution, current_odds_lookup)
        score = score_bracket_pool_fit(public_table, pool_size)
        rows.append(
            {
                "bracket_id": int(result["bracket_id"]),
                "champion": str(result["champion"]),
                "runner_up": str(result.get("runner_up", "")),
                "underdog_wins": int(result.get("underdog_wins", 0)),
                "profile": score["profile"],
                "pool_fit": score["pool_fit"],
                "weighted_public_pct": score["weighted_public_pct"],
                "weighted_late_public_pct": score["weighted_late_public_pct"],
                "weighted_leverage_pct": score["weighted_leverage_pct"],
                "champion_public_pct": score["champion_public_pct"],
                "expected_same_champion_entries": score["expected_same_champion_entries"],
                "final_four_avg_public_pct": score["final_four_avg_public_pct"],
                "title_game_avg_public_pct": score["title_game_avg_public_pct"],
                "final_four_unique_count": score["final_four_unique_count"],
                "title_game_unique_count": score["title_game_unique_count"],
                "late_path_popularity_pct": score["late_path_popularity_pct"],
            }
        )
    return pd.DataFrame(rows)


def build_team_odds_view(current_odds: pd.DataFrame, baseline_odds: pd.DataFrame) -> pd.DataFrame:
    odds_view = build_delta_table(current_odds, baseline_odds).copy()
    ordered_columns = [
        "team",
        "seed",
        "region",
        "play_in_group",
        "expected_wins",
        "win_first_four",
        "make_round_of_64",
        "make_round_of_32",
        "make_sweet_sixteen",
        "make_elite_eight",
        "make_final_four",
        "make_championship",
        "win_championship",
        "final_four_delta",
        "title_game_delta",
        "championship_delta",
    ]
    odds_view = odds_view[ordered_columns].sort_values(
        ["win_championship", "make_final_four", "expected_wins"],
        ascending=False,
    )
    odds_view["play_in_group"] = odds_view["play_in_group"].fillna("")
    return odds_view.reset_index(drop=True)


def percent_column_config(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f%%")


def points_delta_column_config(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%+.1f pts")


def format_seeded_team(team: str, seed_lookup: dict[str, int]) -> str:
    seed = seed_lookup.get(str(team))
    if seed is None or (isinstance(seed, float) and math.isnan(seed)):
        return str(team)
    return f"({int(seed)}) {team}"


def auto_pick_winner(
    left_team: str,
    right_team: str,
    strategy: str,
    seed_lookup: dict[str, int],
    strength_lookup: dict[str, float],
) -> str:
    left_seed = int(seed_lookup.get(left_team, 99))
    right_seed = int(seed_lookup.get(right_team, 99))
    left_strength = float(strength_lookup.get(left_team, float("-inf")))
    right_strength = float(strength_lookup.get(right_team, float("-inf")))

    if strategy == "seed":
        if left_seed != right_seed:
            return left_team if left_seed < right_seed else right_team
        if left_strength != right_strength:
            return left_team if left_strength > right_strength else right_team
        return min(left_team, right_team)

    if left_strength != right_strength:
        return left_team if left_strength > right_strength else right_team
    if left_seed != right_seed:
        return left_team if left_seed < right_seed else right_team
    return min(left_team, right_team)


def autofill_picks(
    games: dict[str, dict[str, object]],
    order: list[str],
    seed_lookup: dict[str, int],
    strength_lookup: dict[str, float],
    strategy: str,
) -> dict[str, str]:
    picks: dict[str, str] = {}
    winners: dict[str, str] = {}
    for game_id in order:
        game = games[game_id]
        left_team = resolve_source(game["left_source"], winners)
        right_team = resolve_source(game["right_source"], winners)
        if not left_team or not right_team:
            continue
        winner = auto_pick_winner(
            left_team=left_team,
            right_team=right_team,
            strategy=strategy,
            seed_lookup=seed_lookup,
            strength_lookup=strength_lookup,
        )
        picks[game_id] = winner
        winners[game_id] = winner
    return picks


def encode_picks_for_query(picks: dict[str, str]) -> str:
    payload = json.dumps(picks, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decode_picks_from_query(encoded: str) -> dict[str, str]:
    if not encoded:
        return {}
    try:
        payload = base64.urlsafe_b64decode(encoded.encode("ascii"))
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            return {}
        return {str(key): str(value) for key, value in parsed.items()}
    except Exception:
        return {}


def sync_picks_query_params(picks: dict[str, str]) -> None:
    if picks:
        st.query_params["picks"] = encode_picks_for_query(picks)
    elif "picks" in st.query_params:
        del st.query_params["picks"]


def load_picks_from_query_params() -> dict[str, str]:
    encoded = st.query_params.get("picks", "")
    if isinstance(encoded, list):
        encoded = encoded[0] if encoded else ""
    return decode_picks_from_query(str(encoded))


def soften_probability(probability: float, randomness: float) -> float:
    randomness = max(0.0, min(1.0, float(randomness)))
    probability = float(probability)
    softened = 0.5 + ((probability - 0.5) * (1.0 - randomness))
    return max(1e-6, min(1.0 - 1e-6, softened))


def is_underdog_win(winner: str, left_team: str, right_team: str, seed_lookup: dict[str, int]) -> bool:
    left_seed = seed_lookup.get(left_team)
    right_seed = seed_lookup.get(right_team)
    if left_seed is None or right_seed is None or int(left_seed) == int(right_seed):
        return False
    favorite = left_team if int(left_seed) < int(right_seed) else right_team
    return winner != favorite


def simulate_single_bracket(
    games: dict[str, dict[str, object]],
    order: list[str],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    seed_lookup: dict[str, int],
    base_picks: dict[str, str],
    locked_champion: str | None,
    randomness: float,
    rng: np.random.Generator,
    probability_cache: dict[tuple[str, str, int], float],
) -> tuple[dict[str, str], int]:
    winners: dict[str, str] = {}
    picks: dict[str, str] = {}
    underdog_wins = 0

    for game_id in order:
        game = games[game_id]
        left_team = resolve_source(game["left_source"], winners)
        right_team = resolve_source(game["right_source"], winners)
        if not left_team or not right_team:
            continue

        if game_id in base_picks:
            winner = str(base_picks[game_id])
        elif locked_champion and locked_champion in {left_team, right_team}:
            winner = locked_champion
        else:
            left_probability = game_probability(
                game,
                left_team,
                right_team,
                team_lookup,
                matchup_payload,
                probability_cache,
                probability_temperature,
            )
            left_probability = soften_probability(left_probability, randomness)
            winner = left_team if rng.random() < left_probability else right_team

        picks[game_id] = winner
        winners[game_id] = winner
        if is_underdog_win(winner, left_team, right_team, seed_lookup):
            underdog_wins += 1

    return picks, underdog_wins


def generate_simulated_brackets(
    games: dict[str, dict[str, object]],
    order: list[str],
    round_groups: dict[int, list[str]],
    team_lookup: dict[str, object],
    matchup_payload: dict,
    probability_temperature: float,
    seed_lookup: dict[str, int],
    base_picks: dict[str, str],
    n_brackets: int,
    randomness: float,
    locked_champion: str | None = None,
    attempts_per_bracket: int = 60,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(42)
    results: list[dict[str, object]] = []
    probability_cache: dict[tuple[str, str, int], float] = {}

    for bracket_index in range(1, n_brackets + 1):
        best_candidate: dict[str, object] | None = None

        for _ in range(max(1, attempts_per_bracket)):
            picks, underdog_wins = simulate_single_bracket(
                games=games,
                order=order,
                team_lookup=team_lookup,
                matchup_payload=matchup_payload,
                probability_temperature=probability_temperature,
                seed_lookup=seed_lookup,
                base_picks=base_picks,
                locked_champion=locked_champion,
                randomness=randomness,
                rng=rng,
                probability_cache=probability_cache,
            )
            champion = picks.get(order[-1], "")
            if locked_champion and champion != locked_champion:
                continue

            best_candidate = {
                "bracket_id": bracket_index,
                "picks": picks,
                "champion": champion,
                "underdog_wins": underdog_wins,
            }
            break

        if best_candidate is None:
            continue

        final_game = games[order[-1]]
        runner_up = ""
        left_final_source = resolve_source(final_game["left_source"], best_candidate["picks"])
        right_final_source = resolve_source(final_game["right_source"], best_candidate["picks"])
        for team in [left_final_source, right_final_source]:
            if team and team != best_candidate["champion"]:
                runner_up = str(team)
                break

        final_four_teams = [best_candidate["picks"].get(game_id, "") for game_id in round_groups[4] if game_id in best_candidate["picks"]]
        best_candidate["runner_up"] = runner_up
        best_candidate["final_four"] = ", ".join([team for team in final_four_teams if team])
        results.append(best_candidate)

    return results


def build_simulation_pdf_zip(
    bracket_results: list[dict[str, object]],
    games: dict[str, dict[str, object]],
    order: list[str],
    seed_lookup: dict[str, int],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result in bracket_results:
            champion_slug = str(result["champion"]).replace(" ", "_").replace("/", "-")
            pdf_bytes = generate_bracket_pdf(games, order, dict(result["picks"]), seed_lookup)
            file_name = f"sim_bracket_{int(result['bracket_id']):02d}_{champion_slug}.pdf"
            zf.writestr(file_name, pdf_bytes)
    buffer.seek(0)
    return buffer.getvalue()


def summarize_simulated_brackets(
    bracket_results: list[dict[str, object]],
    games: dict[str, dict[str, object]],
    order: list[str],
    field_df: pd.DataFrame,
) -> pd.DataFrame:
    teams = field_df["team"].astype(str).tolist()
    play_in_lookup = (
        field_df[["team", "play_in_group"]]
        .assign(play_in_group=lambda df: df["play_in_group"].fillna("").astype(str))
        .set_index("team")["play_in_group"]
        .to_dict()
    )
    counters: dict[str, dict[str, float]] = {
        team: {
            "lose_first_game": 0.0,
            "make_round_of_64": 0.0,
            "make_round_of_32": 0.0,
            "make_sweet_sixteen": 0.0,
            "make_elite_eight": 0.0,
            "make_final_four": 0.0,
            "make_championship": 0.0,
            "win_championship": 0.0,
        }
        for team in teams
    }

    if not bracket_results:
        return pd.DataFrame()

    for result in bracket_results:
        picks = dict(result["picks"])
        winners: dict[str, str] = {}
        first_game_seen: set[str] = set()
        reached: dict[str, dict[str, bool]] = {
            team: {
                "lose_first_game": False,
                "make_round_of_64": play_in_lookup.get(team, "").strip() == "",
                "make_round_of_32": False,
                "make_sweet_sixteen": False,
                "make_elite_eight": False,
                "make_final_four": False,
                "make_championship": False,
                "win_championship": False,
            }
            for team in teams
        }

        for game_id in order:
            game = games[game_id]
            left_team = resolve_source(game["left_source"], winners)
            right_team = resolve_source(game["right_source"], winners)
            if not left_team or not right_team:
                continue

            winner = str(picks.get(game_id, ""))
            if not winner:
                continue

            for team in (left_team, right_team):
                if team not in first_game_seen:
                    first_game_seen.add(team)
                    if team != winner:
                        reached[team]["lose_first_game"] = True

            round_index = int(game["round_index"])
            if round_index == 0:
                reached[winner]["make_round_of_64"] = True
            elif round_index == 1:
                reached[winner]["make_round_of_32"] = True
            elif round_index == 2:
                reached[winner]["make_sweet_sixteen"] = True
            elif round_index == 3:
                reached[winner]["make_elite_eight"] = True
            elif round_index == 4:
                reached[winner]["make_final_four"] = True
            elif round_index == 5:
                reached[winner]["make_championship"] = True
            elif round_index == 6:
                reached[winner]["win_championship"] = True

            winners[game_id] = winner

        for team in teams:
            for column, value in reached[team].items():
                counters[team][column] += float(value)

    n_brackets = float(len(bracket_results))
    summary_rows: list[dict[str, object]] = []
    for row in field_df.itertuples(index=False):
        team = str(row.team)
        summary_rows.append(
            {
                "team": team,
                "seed": int(row.seed),
                "region": str(row.region),
                "play_in_group": str(getattr(row, "play_in_group", "") or ""),
                "lose_first_game": counters[team]["lose_first_game"] / n_brackets,
                "make_round_of_64": counters[team]["make_round_of_64"] / n_brackets,
                "make_round_of_32": counters[team]["make_round_of_32"] / n_brackets,
                "make_sweet_sixteen": counters[team]["make_sweet_sixteen"] / n_brackets,
                "make_elite_eight": counters[team]["make_elite_eight"] / n_brackets,
                "make_final_four": counters[team]["make_final_four"] / n_brackets,
                "make_championship": counters[team]["make_championship"] / n_brackets,
                "win_championship": counters[team]["win_championship"] / n_brackets,
            }
        )

    return pd.DataFrame(summary_rows).sort_values(
        ["win_championship", "make_final_four", "make_sweet_sixteen"],
        ascending=False,
    ).reset_index(drop=True)


def fingerprint_simulated_brackets(bracket_results: list[dict[str, object]]) -> tuple:
    return tuple(
        (
            int(result["bracket_id"]),
            str(result["champion"]),
            int(result["underdog_wins"]),
            tuple(sorted(dict(result["picks"]).items())),
        )
        for result in bracket_results
    )


def export_picks_dataframe(
    games: dict[str, dict[str, object]],
    order: list[str],
    picks: dict[str, str],
    seed_lookup: dict[str, int],
) -> pd.DataFrame:
    winners: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for game_id in order:
        game = games[game_id]
        left_team = resolve_source(game["left_source"], winners)
        right_team = resolve_source(game["right_source"], winners)
        if not left_team or not right_team:
            continue
        picked = picks.get(game_id, "")
        if picked:
            winners[game_id] = picked
        rows.append(
            {
                "game_id": game_id,
                "round_title": game["round_title"],
                "label": game["label"],
                "left_team": left_team,
                "left_seed": seed_lookup.get(left_team, ""),
                "right_team": right_team,
                "right_seed": seed_lookup.get(right_team, ""),
                "picked_winner": picked,
                "picked_seed": seed_lookup.get(picked, "") if picked else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    st.title("March Madness Live Bracket")
    st.caption("Make picks and watch the conditional odds for the remaining bracket update after every choice.")

    with st.sidebar:
        st.header("Data")
        season_data_path = st.text_input("Season data", DEFAULT_SEASON_DATA)
        field_path = st.text_input("Field file", DEFAULT_FIELD)
        semifinal_pairs_text = st.text_input("Semifinal pairings", DEFAULT_SEMIFINALS)
        public_pick_distribution_path = st.text_input("Public pick distribution", DEFAULT_PUBLIC_PICKS)
        n_sims = st.slider("Conditional sims", min_value=1000, max_value=10000, value=3000, step=1000)
        if st.button("Reset picks", use_container_width=True):
            st.session_state["bracket_picks"] = {}
            sync_picks_query_params({})
            st.rerun()
        st.caption("Higher simulation counts are smoother but slower.")

    resources = load_resources(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
        public_pick_distribution_path,
    )
    games, order, round_groups = build_games(resources["resolved_field"], resources["semifinal_pairs"])
    parent_lookup = build_parent_lookup(games)
    seed_lookup = (
        resources["resolved_field"][["team", "seed"]]
        .drop_duplicates(subset=["team"])
        .set_index("team")["seed"]
        .astype(int)
        .to_dict()
    )
    strength_lookup = (
        resources["team_rankings"][["team", "team_strength_score"]]
        .drop_duplicates(subset=["team"])
        .set_index("team")["team_strength_score"]
        .astype(float)
        .to_dict()
    )

    if "bracket_picks" not in st.session_state:
        st.session_state["bracket_picks"] = load_picks_from_query_params()
    saved_picks = st.session_state.get("bracket_picks", {})
    saved_picks = sanitize_picks(games, order, saved_picks)
    st.session_state["bracket_picks"] = saved_picks

    with st.sidebar:
        st.header("Picks")
        if st.button("Autofill by seed", use_container_width=True):
            auto_picks = autofill_picks(
                games=games,
                order=order,
                seed_lookup=seed_lookup,
                strength_lookup=strength_lookup,
                strategy="seed",
            )
            st.session_state["bracket_picks"] = auto_picks
            sync_picks_query_params(auto_picks)
            st.rerun()
        if st.button("Autofill by team strength", use_container_width=True):
            auto_picks = autofill_picks(
                games=games,
                order=order,
                seed_lookup=seed_lookup,
                strength_lookup=strength_lookup,
                strategy="team_strength",
            )
            st.session_state["bracket_picks"] = auto_picks
            sync_picks_query_params(auto_picks)
            st.rerun()

    baseline_odds = cached_simulation(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
        public_pick_distribution_path,
        tuple(),
        n_sims,
    )
    current_odds = cached_simulation(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
        public_pick_distribution_path,
        tuple(sorted(saved_picks.items())),
        n_sims,
    )
    current_odds_lookup = current_odds.set_index("team").to_dict("index")
    export_df = export_picks_dataframe(games, order, saved_picks, seed_lookup)
    export_csv = export_df.to_csv(index=False).encode("utf-8")
    export_pdf = generate_bracket_pdf(games, order, saved_picks, seed_lookup)

    with st.sidebar:
        st.download_button(
            "Export picks CSV",
            data=export_csv,
            file_name="march_madness_bracket_picks.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Export picks PDF",
            data=export_pdf,
            file_name="march_madness_bracket_picks.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    tabs = st.tabs(["Bracket Builder", "Live Odds", "Team Odds", "Bracket Sims", "Pool Fit", "Team Lens"])

    with tabs[0]:
        rendered_rows = build_game_rows(
            games,
            order,
            saved_picks,
            resources["team_lookup"],
            resources["matchup_payload"],
            resources["probability_temperature"],
        )
        game_row_map = {str(row["game_id"]): row for row in rendered_rows}
        desired_picks: dict[str, str] = dict(saved_picks)

        sticky_header = st.container(key="bracket_sticky_header")
        with sticky_header:
            st.subheader("Bracket Builder")
            st.caption(
                "Pick through one region at a time. Each card shows the current game odds and the immediate next-game impact of each option."
            )
            header_cols = st.columns([1.15, 1, 1])
            header_cols[0].metric("Teams in field", len(resources["resolved_field"]))
            header_cols[1].metric("Picks locked", len(saved_picks))
            header_cols[2].metric(
                "Play-in teams",
                int(resources["resolved_field"]["play_in_group"].fillna("").astype(str).str.strip().ne("").sum()),
            )

            view_options = ["East", "South", "West", "Midwest", "Final Four"]
            selected_view = st.radio(
                "Bracket view",
                options=view_options,
                horizontal=True,
                key="region_view",
            )

        if selected_view == "Final Four":
            render_national_bracket(
                round_groups=round_groups,
                game_row_map=game_row_map,
                games=games,
                parent_lookup=parent_lookup,
                seed_lookup=seed_lookup,
                current_odds_lookup=current_odds_lookup,
                team_lookup=resources["team_lookup"],
                matchup_payload=resources["matchup_payload"],
                probability_temperature=resources["probability_temperature"],
                saved_picks=saved_picks,
                desired_picks=desired_picks,
            )
        else:
            render_region_bracket(
                region=selected_view,
                round_groups=round_groups,
                game_row_map=game_row_map,
                games=games,
                parent_lookup=parent_lookup,
                seed_lookup=seed_lookup,
                current_odds_lookup=current_odds_lookup,
                team_lookup=resources["team_lookup"],
                matchup_payload=resources["matchup_payload"],
                probability_temperature=resources["probability_temperature"],
                saved_picks=saved_picks,
                desired_picks=desired_picks,
            )

        sanitized_desired = sanitize_picks(games, order, desired_picks)
        if sanitized_desired != saved_picks:
            st.session_state["bracket_picks"] = sanitized_desired
            sync_picks_query_params(sanitized_desired)
            st.rerun()

        if not game_row_map:
            st.info("No games are currently available. Check the field or semifinal pairing inputs.")

    current_picks = sanitize_picks(games, order, st.session_state.get("bracket_picks", {}))
    current_odds = cached_simulation(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
        public_pick_distribution_path,
        tuple(sorted(current_picks.items())),
        n_sims,
    )
    delta_table = build_delta_table(current_odds, baseline_odds)
    team_odds_view = build_team_odds_view(current_odds, baseline_odds)

    with tabs[1]:
        st.subheader("Conditional Odds")
        if current_picks:
            st.caption("These odds are conditioned on the picks you have already locked in.")
        else:
            st.caption("These are the baseline bracket odds with no user picks locked in.")

        favorite = current_odds.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Current title favorite", favorite["team"])
        col2.metric("Title odds", format_probability(favorite["win_championship"]))
        col3.metric("Final Four odds", format_probability(favorite["make_final_four"]))

        st.markdown("#### Biggest risers from your picks")
        risers = delta_table.sort_values("championship_delta", ascending=False).head(10).copy()
        risers["title_odds"] = risers["win_championship"] * 100.0
        risers["title_delta"] = risers["championship_delta"] * 100.0
        risers["final_four_prob"] = risers["make_final_four"] * 100.0
        risers["title_game_prob"] = risers["make_championship"] * 100.0
        st.dataframe(
            risers[["team", "seed", "region", "title_odds", "title_delta", "final_four_prob", "title_game_prob"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "title_odds": percent_column_config("title_odds"),
                "title_delta": points_delta_column_config("title_delta"),
                "final_four_prob": percent_column_config("final_four_prob"),
                "title_game_prob": percent_column_config("title_game_prob"),
            },
        )

        st.markdown("#### Championship odds")
        title_view = delta_table.copy()
        title_view["title_odds"] = title_view["win_championship"] * 100.0
        title_view["final_four_odds"] = title_view["make_final_four"] * 100.0
        title_view["title_game_odds"] = title_view["make_championship"] * 100.0
        title_view["title_delta"] = title_view["championship_delta"] * 100.0
        st.dataframe(
            title_view[["team", "seed", "region", "title_odds", "title_delta", "final_four_odds", "title_game_odds"]]
            .head(24),
            use_container_width=True,
            hide_index=True,
            column_config={
                "title_odds": percent_column_config("title_odds"),
                "title_delta": points_delta_column_config("title_delta"),
                "final_four_odds": percent_column_config("final_four_odds"),
                "title_game_odds": percent_column_config("title_game_odds"),
            },
        )

        st.markdown("#### Available game probabilities")
        game_rows = build_game_rows(
            games,
            order,
            current_picks,
            resources["team_lookup"],
            resources["matchup_payload"],
            resources["probability_temperature"],
        )
        open_games = [row for row in game_rows if not row["picked_winner"]]
        if open_games:
            matchup_df = pd.DataFrame(open_games)
            matchup_df["favorite"] = np.where(
                matchup_df["left_probability"] >= matchup_df["right_probability"],
                matchup_df["left_team"],
                matchup_df["right_team"],
            )
            matchup_df["favorite_probability"] = np.maximum(matchup_df["left_probability"], matchup_df["right_probability"])
            matchup_df["underdog"] = np.where(
                matchup_df["left_probability"] < matchup_df["right_probability"],
                matchup_df["left_team"],
                matchup_df["right_team"],
            )
            matchup_df["favorite_probability"] = matchup_df["favorite_probability"] * 100.0
            st.dataframe(
                matchup_df[["round_title", "label", "left_team", "right_team", "favorite", "favorite_probability"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "favorite_probability": percent_column_config("favorite_probability"),
                },
            )
        else:
            st.success("Every currently reachable game has been picked.")

    with tabs[2]:
        st.subheader("Team Odds")
        if current_picks:
            st.caption("This table matches the `team_odds.csv` style output, updated for your current locked picks.")
        else:
            st.caption("This table matches the baseline `team_odds.csv` output with no picks locked in.")

        team_odds_column_config: dict[str, st.column_config.Column] = {
            "expected_wins": st.column_config.NumberColumn("expected_wins", format="%.2f"),
            "play_in_group": st.column_config.TextColumn("play_in_group"),
        }
        for column in TEAM_ODDS_PROBABILITY_COLUMNS:
            team_odds_column_config[column] = percent_column_config(column)
        for column in TEAM_ODDS_DELTA_COLUMNS:
            team_odds_column_config[column] = points_delta_column_config(column)

        display_team_odds = team_odds_view.copy()
        for column in TEAM_ODDS_PROBABILITY_COLUMNS + TEAM_ODDS_DELTA_COLUMNS:
            display_team_odds[column] = display_team_odds[column] * 100.0

        st.dataframe(
            display_team_odds,
            use_container_width=True,
            hide_index=True,
            column_config=team_odds_column_config,
        )

        export_team_odds_csv = current_odds.sort_values(
            ["win_championship", "make_final_four", "expected_wins"],
            ascending=False,
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current team odds CSV",
            data=export_team_odds_csv,
            file_name="current_team_odds.csv",
            mime="text/csv",
            use_container_width=False,
        )

    with tabs[3]:
        st.subheader("Bracket Simulations")
        st.caption("Generate complete brackets from the model. Current locked picks are treated as fixed constraints.")

        sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
        n_generated_brackets = sim_col1.slider("Number of brackets", min_value=1, max_value=25, value=5, step=1)
        randomness = sim_col2.slider("Randomness", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        champion_options = ["No lock"] + sorted(resources["resolved_field"]["team"].astype(str).tolist())
        locked_champion = sim_col3.selectbox("Lock champion", options=champion_options, index=0)
        sim_col4.caption("Current locked picks stay fixed in every simulated bracket.")

        if st.button("Simulate brackets", use_container_width=True):
            with st.spinner("Generating simulated brackets..."):
                simulated_results = generate_simulated_brackets(
                    games=games,
                    order=order,
                    round_groups=round_groups,
                    team_lookup=resources["team_lookup"],
                    matchup_payload=resources["matchup_payload"],
                    probability_temperature=resources["probability_temperature"],
                    seed_lookup=seed_lookup,
                    base_picks=current_picks,
                    n_brackets=n_generated_brackets,
                    randomness=randomness,
                    locked_champion=None if locked_champion == "No lock" else locked_champion,
                )
            st.session_state["simulated_brackets"] = simulated_results
            st.session_state["simulated_brackets_fingerprint"] = fingerprint_simulated_brackets(simulated_results)
            st.session_state.pop("simulated_brackets_zip", None)
            st.session_state.pop("simulated_selected_pdf", None)
            st.session_state.pop("simulated_selected_pdf_id", None)
            st.rerun()

        simulated_brackets = st.session_state.get("simulated_brackets", [])
        if simulated_brackets:
            summary_rows = pd.DataFrame(
                [
                    {
                        "bracket_id": row["bracket_id"],
                        "champion": row["champion"],
                        "runner_up": row["runner_up"],
                        "underdog_wins": row["underdog_wins"],
                        "final_four": row["final_four"],
                    }
                    for row in simulated_brackets
                ]
            )
            st.markdown("#### Simulated bracket summary")
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            aggregate_sim_summary = summarize_simulated_brackets(
                bracket_results=simulated_brackets,
                games=games,
                order=order,
                field_df=resources["resolved_field"],
            )
            if not aggregate_sim_summary.empty:
                st.markdown("#### Aggregate team results across simulated brackets")
                display_aggregate_sim_summary = aggregate_sim_summary.copy()
                aggregate_probability_columns = [
                    "lose_first_game",
                    "make_round_of_64",
                    "make_round_of_32",
                    "make_sweet_sixteen",
                    "make_elite_eight",
                    "make_final_four",
                    "make_championship",
                    "win_championship",
                ]
                for column in aggregate_probability_columns:
                    display_aggregate_sim_summary[column] = display_aggregate_sim_summary[column] * 100.0

                aggregate_column_config: dict[str, st.column_config.Column] = {
                    "play_in_group": st.column_config.TextColumn("play_in_group"),
                }
                for column in aggregate_probability_columns:
                    aggregate_column_config[column] = percent_column_config(column)

                st.dataframe(
                    display_aggregate_sim_summary,
                    use_container_width=True,
                    hide_index=True,
                    column_config=aggregate_column_config,
                )

                st.download_button(
                    "Download aggregate simulation summary CSV",
                    data=aggregate_sim_summary.to_csv(index=False).encode("utf-8"),
                    file_name="simulated_bracket_team_summary.csv",
                    mime="text/csv",
                    use_container_width=False,
                )

            current_fingerprint = fingerprint_simulated_brackets(simulated_brackets)
            if st.button("Prepare PDF bundle", use_container_width=False):
                with st.spinner("Preparing PDF bundle..."):
                    st.session_state["simulated_brackets_zip"] = build_simulation_pdf_zip(
                        simulated_brackets,
                        games,
                        order,
                        seed_lookup,
                    )
                    st.session_state["simulated_brackets_fingerprint"] = current_fingerprint
            zip_bytes = st.session_state.get("simulated_brackets_zip")
            zip_fingerprint = st.session_state.get("simulated_brackets_fingerprint")
            if zip_bytes is not None and zip_fingerprint == current_fingerprint:
                st.download_button(
                    "Download all simulated brackets as PDFs",
                    data=zip_bytes,
                    file_name="simulated_brackets_pdfs.zip",
                    mime="application/zip",
                    use_container_width=False,
                )

            selected_bracket_id = st.selectbox(
                "Preview simulated bracket",
                options=[row["bracket_id"] for row in simulated_brackets],
                format_func=lambda value: f"Bracket {int(value):02d}",
            )
            selected_result = next(row for row in simulated_brackets if int(row["bracket_id"]) == int(selected_bracket_id))
            selected_export_df = export_picks_dataframe(games, order, dict(selected_result["picks"]), seed_lookup)
            st.dataframe(selected_export_df, use_container_width=True, hide_index=True)

            if st.button("Prepare selected bracket PDF", use_container_width=False):
                with st.spinner("Preparing selected bracket PDF..."):
                    st.session_state["simulated_selected_pdf"] = generate_bracket_pdf(
                        games,
                        order,
                        dict(selected_result["picks"]),
                        seed_lookup,
                    )
                    st.session_state["simulated_selected_pdf_id"] = int(selected_result["bracket_id"])
            selected_pdf = st.session_state.get("simulated_selected_pdf")
            selected_pdf_id = st.session_state.get("simulated_selected_pdf_id")
            if selected_pdf is not None and selected_pdf_id == int(selected_result["bracket_id"]):
                st.download_button(
                    "Download selected bracket PDF",
                    data=selected_pdf,
                    file_name=f"simulated_bracket_{int(selected_result['bracket_id']):02d}.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
        else:
            st.info("No simulated brackets yet. Use the controls above to generate some.")

    with tabs[4]:
        st.subheader("Pool Fit")
        st.caption(
            "Compare your bracket to public Yahoo pick rates to see whether it is chalky, balanced, or too contrarian for the size of your pool."
        )

        if resources["public_pick_distribution"].empty:
            st.info(
                "No public pick distribution file is loaded yet. Add a Yahoo pick distribution CSV or HTML parse output to score bracket chalk and leverage."
            )
        else:
            pool_size = st.slider("Pool size", min_value=10, max_value=2000, value=50, step=10)
            current_public_table = build_bracket_public_pick_table(
                export_df,
                resources["public_pick_distribution"],
                current_odds_lookup,
            )
            current_pool_score = score_bracket_pool_fit(current_public_table, pool_size)

            st.markdown("#### Current bracket profile")
            score_cols = st.columns(4)
            score_cols[0].metric("Profile", str(current_pool_score["profile"]))
            score_cols[1].metric("Pool fit", str(current_pool_score["pool_fit"]))
            score_cols[2].metric(
                "Champion public pick rate",
                format_probability(current_pool_score["champion_public_pct"])
                if not math.isnan(float(current_pool_score["champion_public_pct"]))
                else "N/A",
            )
            score_cols[3].metric(
                "Expected same champion entries",
                (
                    f"{float(current_pool_score['expected_same_champion_entries']):.1f}"
                    if not math.isnan(float(current_pool_score["expected_same_champion_entries"]))
                    else "N/A"
                ),
            )

            detail_cols = st.columns(3)
            detail_cols[0].metric(
                "Weighted public pick rate",
                format_probability(current_pool_score["weighted_public_pct"])
                if not math.isnan(float(current_pool_score["weighted_public_pct"]))
                else "N/A",
            )
            detail_cols[1].metric(
                "Late-round public pick rate",
                format_probability(current_pool_score["weighted_late_public_pct"])
                if not math.isnan(float(current_pool_score["weighted_late_public_pct"]))
                else "N/A",
            )
            detail_cols[2].metric(
                "Model leverage",
                f"{100.0 * float(current_pool_score['weighted_leverage_pct']):+.1f} pts"
                if not math.isnan(float(current_pool_score["weighted_leverage_pct"]))
                else "N/A",
            )
            late_cols = st.columns(4)
            late_cols[0].metric(
                "Final Four avg public rate",
                format_probability(current_pool_score["final_four_avg_public_pct"])
                if not math.isnan(float(current_pool_score["final_four_avg_public_pct"]))
                else "N/A",
            )
            late_cols[1].metric(
                "Title game avg public rate",
                format_probability(current_pool_score["title_game_avg_public_pct"])
                if not math.isnan(float(current_pool_score["title_game_avg_public_pct"]))
                else "N/A",
            )
            late_cols[2].metric("Unique Final Four teams", int(current_pool_score["final_four_unique_count"]))
            late_cols[3].metric(
                "Adjusted late-path popularity",
                format_probability(current_pool_score["late_path_popularity_pct"])
                if not math.isnan(float(current_pool_score["late_path_popularity_pct"]))
                else "N/A",
            )

            min_pool = current_pool_score["recommended_pool_min"]
            max_pool = current_pool_score["recommended_pool_max"]
            if min_pool is not None:
                if max_pool is None:
                    st.caption(f"Recommended pool size range for this profile: {int(min_pool)}+ entries.")
                else:
                    st.caption(f"Recommended pool size range for this profile: {int(min_pool)}-{int(max_pool)} entries.")

            if current_public_table.empty:
                st.info("Lock in at least one bracket pick to score your current bracket against the public field.")
            else:
                display_current_public = current_public_table[
                    [
                        "round_title",
                        "label",
                        "team",
                        "picked_pct",
                        "model_round_prob",
                        "leverage",
                        "public_rank",
                    ]
                ].copy()
                display_current_public["picked_pct"] = display_current_public["picked_pct"] * 100.0
                display_current_public["model_round_prob"] = display_current_public["model_round_prob"] * 100.0
                display_current_public["leverage"] = display_current_public["leverage"] * 100.0
                st.markdown("#### Current bracket pick-by-pick leverage")
                st.dataframe(
                    display_current_public,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "picked_pct": percent_column_config("picked_pct"),
                        "model_round_prob": percent_column_config("model_round_prob"),
                        "leverage": points_delta_column_config("leverage"),
                    },
                )

            simulated_brackets = st.session_state.get("simulated_brackets", [])
            if simulated_brackets:
                simulated_public_summary = build_simulated_bracket_public_summary(
                    bracket_results=simulated_brackets,
                    games=games,
                    order=order,
                    seed_lookup=seed_lookup,
                    public_pick_distribution=resources["public_pick_distribution"],
                    current_odds_lookup=current_odds_lookup,
                    pool_size=pool_size,
                )
                if not simulated_public_summary.empty:
                    st.markdown("#### Simulated bracket pool-fit summary")
                    display_simulated_public_summary = simulated_public_summary.copy()
                    for column in [
                        "weighted_public_pct",
                        "weighted_late_public_pct",
                        "weighted_leverage_pct",
                        "champion_public_pct",
                        "final_four_avg_public_pct",
                        "title_game_avg_public_pct",
                        "late_path_popularity_pct",
                    ]:
                        display_simulated_public_summary[column] = display_simulated_public_summary[column] * 100.0

                    st.dataframe(
                        display_simulated_public_summary.sort_values(
                            ["profile", "late_path_popularity_pct", "weighted_leverage_pct"],
                            ascending=[True, True, False],
                        ),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "weighted_public_pct": percent_column_config("weighted_public_pct"),
                            "weighted_late_public_pct": percent_column_config("weighted_late_public_pct"),
                            "weighted_leverage_pct": points_delta_column_config("weighted_leverage_pct"),
                            "champion_public_pct": percent_column_config("champion_public_pct"),
                            "final_four_avg_public_pct": percent_column_config("final_four_avg_public_pct"),
                            "title_game_avg_public_pct": percent_column_config("title_game_avg_public_pct"),
                            "late_path_popularity_pct": percent_column_config("late_path_popularity_pct"),
                            "expected_same_champion_entries": st.column_config.NumberColumn(
                                "expected_same_champion_entries",
                                format="%.1f",
                            ),
                        },
                    )

                    st.download_button(
                        "Download simulated bracket pool-fit CSV",
                        data=simulated_public_summary.to_csv(index=False).encode("utf-8"),
                        file_name="simulated_bracket_pool_fit.csv",
                        mime="text/csv",
                        use_container_width=False,
                    )

            with st.expander("Public pick match report", expanded=False):
                st.dataframe(resources["public_pick_match_report"], use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Team Lens")
        st.caption("Use this view to compare the bracket state with the underlying team profile and contender screen.")

        lens_cols = st.columns(2)
        with lens_cols[0]:
            st.markdown("#### Team rankings")
            st.dataframe(resources["team_rankings"].head(24), use_container_width=True, hide_index=True)
        with lens_cols[1]:
            st.markdown("#### Contender scorecard")
            st.dataframe(resources["contender_scorecard"].head(24), use_container_width=True, hide_index=True)

        st.markdown("#### Match report")
        st.dataframe(resources["match_report"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
