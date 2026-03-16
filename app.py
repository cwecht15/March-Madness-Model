from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from scripts.bracket_pdf import generate_bracket_pdf
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


DEFAULT_SEASON_DATA = "data/processed/march_madness_2026.csv"
DEFAULT_FIELD = "data/raw/2026/tournament_field.csv"
DEFAULT_TEAM_MANIFEST = "artifacts/model_benchmarks/saved_model_manifest.csv"
DEFAULT_MATCHUP_MANIFEST = "artifacts/matchup_model/saved_model_manifest.csv"
DEFAULT_ALIASES = "data/team_aliases.csv"
DEFAULT_SEMIFINALS = "East-South,West-Midwest"
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
    }


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
        n_sims = st.slider("Conditional sims", min_value=1000, max_value=10000, value=3000, step=1000)
        if st.button("Reset picks", use_container_width=True):
            st.session_state["bracket_picks"] = {}
            st.rerun()
        st.caption("Higher simulation counts are smoother but slower.")

    resources = load_resources(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
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

    saved_picks = st.session_state.get("bracket_picks", {})
    saved_picks = sanitize_picks(games, order, saved_picks)

    with st.sidebar:
        st.header("Picks")
        if st.button("Autofill by seed", use_container_width=True):
            st.session_state["bracket_picks"] = autofill_picks(
                games=games,
                order=order,
                seed_lookup=seed_lookup,
                strength_lookup=strength_lookup,
                strategy="seed",
            )
            st.rerun()
        if st.button("Autofill by team strength", use_container_width=True):
            st.session_state["bracket_picks"] = autofill_picks(
                games=games,
                order=order,
                seed_lookup=seed_lookup,
                strength_lookup=strength_lookup,
                strategy="team_strength",
            )
            st.rerun()

    baseline_odds = cached_simulation(
        season_data_path,
        field_path,
        semifinal_pairs_text,
        DEFAULT_TEAM_MANIFEST,
        DEFAULT_MATCHUP_MANIFEST,
        DEFAULT_ALIASES,
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

    tabs = st.tabs(["Bracket Builder", "Live Odds", "Team Lens"])

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
        tuple(sorted(current_picks.items())),
        n_sims,
    )
    delta_table = build_delta_table(current_odds, baseline_odds)

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
        risers["title_odds"] = risers["win_championship"].map(format_probability)
        risers["title_delta"] = risers["championship_delta"].map(lambda value: f"{value * 100:+.1f} pts")
        st.dataframe(
            risers[["team", "seed", "region", "title_odds", "title_delta", "make_final_four", "make_championship"]]
            .rename(columns={"make_final_four": "final_four_prob", "make_championship": "title_game_prob"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Championship odds")
        title_view = delta_table.copy()
        title_view["title_odds"] = title_view["win_championship"].map(format_probability)
        title_view["final_four_odds"] = title_view["make_final_four"].map(format_probability)
        title_view["title_game_odds"] = title_view["make_championship"].map(format_probability)
        title_view["title_delta"] = title_view["championship_delta"].map(lambda value: f"{value * 100:+.1f} pts")
        st.dataframe(
            title_view[["team", "seed", "region", "title_odds", "title_delta", "final_four_odds", "title_game_odds"]]
            .head(24),
            use_container_width=True,
            hide_index=True,
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
            matchup_df["favorite_probability"] = matchup_df["favorite_probability"].map(format_probability)
            st.dataframe(
                matchup_df[["round_title", "label", "left_team", "right_team", "favorite", "favorite_probability"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("Every currently reachable game has been picked.")

    with tabs[2]:
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
