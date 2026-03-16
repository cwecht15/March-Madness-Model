#!/usr/bin/env python
from __future__ import annotations

import __main__
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.simulate_bracket import (
    infer_field_from_historical_games,
    load_team_model_predictions,
    parse_semifinal_pairs,
    resolve_field,
    simulate_bracket,
    validate_field,
)
from scripts.train_clean_models import EncodedXGBClassifier as TeamEncodedXGBClassifier
from scripts.train_matchup_model import (
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)


# Compatibility shim for joblib payloads trained from script entry points.
if not hasattr(__main__, "EncodedXGBClassifier"):
    __main__.EncodedXGBClassifier = TeamEncodedXGBClassifier
if not hasattr(__main__, "IdentityCalibrator"):
    __main__.IdentityCalibrator = IdentityCalibrator
if not hasattr(__main__, "PlattCalibrator"):
    __main__.PlattCalibrator = PlattCalibrator
if not hasattr(__main__, "IsotonicCalibrator"):
    __main__.IsotonicCalibrator = IsotonicCalibrator


PRIMARY_SCREENING_COLUMNS = [
    "AdjEM",
    "AdjO",
    "AdjD",
]
SECONDARY_SCREENING_COLUMNS = [
    "Seas_PPG",
    "Seas_Off_Rebound_Per",
    "Seas_Turnovers",
    "Seas_3PT_Per",
]
TEAM_RANKING_WEIGHTS = {
    "adjem_pct": 0.26,
    "adjo_pct": 0.13,
    "adjd_pct": 0.11,
    "meta_points_pct": 0.18,
    "meta_finish_pct": 0.12,
    "meta_round_of_32": 0.08,
    "meta_sweet_sixteen": 0.06,
    "meta_elite_eight": 0.03,
    "meta_semifinal": 0.02,
    "meta_championship": 0.01,
}
TIER_ORDER = {
    "title": 0,
    "final_four": 1,
    "second_weekend": 2,
    "darkhorse": 3,
    "fade": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one-stop team rankings, contender screens, upset flags, and bracket odds."
    )
    parser.add_argument(
        "--season-data",
        required=True,
        help="Path to the season-level team dataset for the tournament field.",
    )
    parser.add_argument(
        "--field",
        help="Optional CSV with columns team, seed, region for a 64-team main bracket field.",
    )
    parser.add_argument(
        "--historical-games",
        help="Optional tournament_games.csv path used to infer a historical field and semifinal pairings.",
    )
    parser.add_argument(
        "--team-model-manifest",
        default="artifacts/model_benchmarks/saved_model_manifest.csv",
        help="Saved team-level model manifest for generating bracket-independent team predictions.",
    )
    parser.add_argument(
        "--matchup-model-manifest",
        default="artifacts/matchup_model/saved_model_manifest.csv",
        help="Saved matchup model manifest.",
    )
    parser.add_argument(
        "--aliases",
        default="data/team_aliases.csv",
        help="Alias CSV used for team-name matching.",
    )
    parser.add_argument(
        "--semifinal-pairs",
        default="",
        help="Comma-separated semifinal region pairings like East-West,South-Midwest. Required when --field is used without --historical-games.",
    )
    parser.add_argument("--n-sims", type=int, default=20000, help="Number of Monte Carlo bracket simulations to run.")
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Use raw matchup model probabilities without the saved post-hoc calibrator.",
    )
    parser.add_argument(
        "--probability-temperature",
        type=float,
        default=None,
        help="Optional temperature applied to matchup logits before simulation. Values above 1.0 soften probabilities.",
    )
    parser.add_argument(
        "--upset-threshold",
        type=float,
        default=0.35,
        help="Minimum underdog win probability to flag a game as a live upset.",
    )
    parser.add_argument(
        "--strong-upset-threshold",
        type=float,
        default=0.45,
        help="Minimum underdog win probability to flag a game as a strong upset.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/tournament_forecast",
        help="Directory for forecast outputs.",
    )
    return parser.parse_args()


def pct_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    rank = series.rank(method="average", ascending=not higher_is_better)
    denominator = max(len(series) - 1, 1)
    return 1.0 - ((rank - 1.0) / denominator)


def add_stat_context(season_df: pd.DataFrame) -> pd.DataFrame:
    enriched = season_df.copy()
    enriched["adjem_rank"] = enriched["AdjEM"].rank(method="min", ascending=False).astype(int)
    enriched["adjo_rank"] = enriched["AdjO"].rank(method="min", ascending=False).astype(int)
    enriched["adjd_rank"] = enriched["AdjD"].rank(method="min", ascending=True).astype(int)
    enriched["seas_ppg_rank"] = enriched["Seas_PPG"].rank(method="min", ascending=False).astype(int)
    enriched["off_rebound_rank"] = enriched["Seas_Off_Rebound_Per"].rank(method="min", ascending=False).astype(int)
    enriched["turnover_rank"] = enriched["Seas_Turnovers"].rank(method="min", ascending=True).astype(int)
    enriched["three_point_rank"] = enriched["Seas_3PT_Per"].rank(method="min", ascending=False).astype(int)

    enriched["adjem_pct"] = pct_rank(enriched["AdjEM"], higher_is_better=True)
    enriched["adjo_pct"] = pct_rank(enriched["AdjO"], higher_is_better=True)
    enriched["adjd_pct"] = pct_rank(-enriched["AdjD"], higher_is_better=True)
    enriched["meta_points_pct"] = pct_rank(enriched["meta_points"], higher_is_better=True)
    enriched["meta_finish_pct"] = pct_rank(enriched["meta_finish"], higher_is_better=True)
    return enriched


def build_team_rankings(season_df: pd.DataFrame) -> pd.DataFrame:
    ranked = add_stat_context(season_df)
    score = np.zeros(len(ranked), dtype=float)
    for column, weight in TEAM_RANKING_WEIGHTS.items():
        score = score + (ranked[column].astype(float) * weight)

    ranked["team_strength_score"] = 100.0 * score
    ranked["team_rank"] = ranked["team_strength_score"].rank(method="first", ascending=False).astype(int)
    ranking_columns = [
        "team_rank",
        "team",
        "seed",
        "team_strength_score",
        "AdjEM",
        "AdjO",
        "AdjD",
        "adjem_rank",
        "adjo_rank",
        "adjd_rank",
        "meta_points",
        "meta_finish",
        "meta_round_of_64",
        "meta_round_of_32",
        "meta_sweet_sixteen",
        "meta_elite_eight",
        "meta_semifinal",
        "meta_championship",
        "Seas_PPG",
        "Seas_Off_Rebound_Per",
        "Seas_Turnovers",
        "Seas_3PT_Per",
    ]
    return ranked.sort_values(["team_rank", "team"]).loc[:, ranking_columns].reset_index(drop=True)


def secondary_supports(row: pd.Series) -> list[str]:
    supports: list[str] = []
    if row["seas_ppg_rank"] <= 20:
        supports.append("top_20_scoring")
    if row["off_rebound_rank"] <= 20:
        supports.append("top_20_off_rebounding")
    if row["turnover_rank"] <= 20:
        supports.append("top_20_turnover_control")
    if row["three_point_rank"] <= 20:
        supports.append("top_20_three_point_pct")
    return supports


def risk_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    if row["adjem_rank"] > 20:
        flags.append("outside_top_20_adjem")
    if row["adjo_rank"] > 20:
        flags.append("outside_top_20_offense")
    if row["adjd_rank"] > 25:
        flags.append("outside_top_25_defense")
    if row["turnover_rank"] > 40:
        flags.append("loose_turnover_profile")
    return flags


def contender_tier(row: pd.Series) -> str:
    if row["adjem_rank"] <= 10 and row["adjo_rank"] <= 10 and row["adjd_rank"] <= 15:
        return "title"
    if row["adjem_rank"] <= 15 and row["adjo_rank"] <= 15 and row["adjd_rank"] <= 20:
        return "final_four"
    if row["adjem_rank"] <= 20 and (row["adjo_rank"] <= 15 or row["adjd_rank"] <= 15):
        return "second_weekend"
    if row["adjem_rank"] <= 25:
        return "darkhorse"
    return "fade"


def build_contender_scorecard(season_df: pd.DataFrame) -> pd.DataFrame:
    scored = add_stat_context(season_df)
    rows: list[dict[str, object]] = []

    for row in scored.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        supports = secondary_supports(row_series)
        flags = risk_flags(row_series)
        rows.append(
            {
                "team": row.team,
                "seed": int(row.seed),
                "contender_tier": contender_tier(row_series),
                "AdjEM": row.AdjEM,
                "AdjO": row.AdjO,
                "AdjD": row.AdjD,
                "adjem_rank": int(row.adjem_rank),
                "adjo_rank": int(row.adjo_rank),
                "adjd_rank": int(row.adjd_rank),
                "Seas_PPG": row.Seas_PPG,
                "Seas_Off_Rebound_Per": row.Seas_Off_Rebound_Per,
                "Seas_Turnovers": row.Seas_Turnovers,
                "Seas_3PT_Per": row.Seas_3PT_Per,
                "seas_ppg_rank": int(row.seas_ppg_rank),
                "off_rebound_rank": int(row.off_rebound_rank),
                "turnover_rank": int(row.turnover_rank),
                "three_point_rank": int(row.three_point_rank),
                "secondary_support_count": len(supports),
                "secondary_supports": ",".join(supports),
                "risk_flags": ",".join(flags),
                "meta_points": row.meta_points,
                "meta_finish": row.meta_finish,
                "meta_semifinal": row.meta_semifinal,
                "meta_championship": row.meta_championship,
            }
        )

    scorecard = pd.DataFrame(rows)
    scorecard["tier_order"] = scorecard["contender_tier"].map(TIER_ORDER)
    scorecard = scorecard.sort_values(
        ["tier_order", "secondary_support_count", "meta_championship", "meta_semifinal", "team"],
        ascending=[True, False, False, False, True],
    ).drop(columns="tier_order")
    return scorecard.reset_index(drop=True)


def load_matchup_payload(manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"No saved matchup model found in {manifest_path}")
    matchup_model_path = Path(manifest.iloc[0]["model_path"])
    return manifest, joblib.load(matchup_model_path)


def make_reach_columns(team_odds: pd.DataFrame) -> pd.DataFrame:
    enriched = team_odds.copy()
    play_in_group = enriched.get("play_in_group")
    if play_in_group is None:
        play_in_mask = pd.Series(False, index=enriched.index)
    else:
        play_in_mask = play_in_group.fillna("").astype(str).str.strip().ne("")

    if "win_first_four" not in enriched.columns:
        enriched["win_first_four"] = 0.0

    enriched["make_round_of_64"] = np.where(play_in_mask, enriched["win_first_four"], 1.0)
    enriched["make_round_of_32"] = enriched["win_round_of_64"]
    enriched["make_sweet_sixteen"] = enriched["win_round_of_32"]
    enriched["make_elite_eight"] = enriched["win_sweet_sixteen"]
    enriched["make_final_four"] = enriched["win_elite_eight"]
    enriched["make_championship"] = enriched["win_final_four"]
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
        "win_round_of_64",
        "win_round_of_32",
        "win_sweet_sixteen",
        "win_elite_eight",
        "win_final_four",
    ]
    return enriched.loc[:, ordered_columns].sort_values(
        ["win_championship", "make_final_four", "expected_wins"],
        ascending=False,
    )


def annotate_bracket_matchups(
    most_likely_bracket: pd.DataFrame,
    field_df: pd.DataFrame,
    upset_threshold: float,
    strong_upset_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_lookup = field_df.set_index("team")["seed"].astype(int).to_dict()
    annotated = most_likely_bracket.copy()
    annotated["left_seed"] = annotated["team_left"].map(seed_lookup).astype(int)
    annotated["right_seed"] = annotated["team_right"].map(seed_lookup).astype(int)
    annotated["seed_gap"] = (annotated["left_seed"] - annotated["right_seed"]).abs()
    annotated["right_win_probability"] = 1.0 - annotated["left_win_probability"]

    rows: list[dict[str, object]] = []
    for row in annotated.itertuples(index=False):
        if row.left_seed == row.right_seed:
            underdog_team = ""
            underdog_seed = row.left_seed
            favorite_team = ""
            favorite_seed = row.left_seed
            underdog_probability = 0.0
            favorite_probability = max(float(row.left_win_probability), float(row.right_win_probability))
            upset_level = "same_seed"
        elif row.left_seed < row.right_seed:
            favorite_team = row.team_left
            favorite_seed = int(row.left_seed)
            favorite_probability = float(row.left_win_probability)
            underdog_team = row.team_right
            underdog_seed = int(row.right_seed)
            underdog_probability = float(row.right_win_probability)
            upset_level = (
                "strong"
                if underdog_probability >= strong_upset_threshold
                else "live"
                if underdog_probability >= upset_threshold
                else ""
            )
        else:
            favorite_team = row.team_right
            favorite_seed = int(row.right_seed)
            favorite_probability = float(row.right_win_probability)
            underdog_team = row.team_left
            underdog_seed = int(row.left_seed)
            underdog_probability = float(row.left_win_probability)
            upset_level = (
                "strong"
                if underdog_probability >= strong_upset_threshold
                else "live"
                if underdog_probability >= upset_threshold
                else ""
            )

        rows.append(
            {
                "round_index": int(row.round_index),
                "round_name": row.round_name,
                "game_number": int(row.game_number),
                "team_left": row.team_left,
                "left_seed": int(row.left_seed),
                "team_right": row.team_right,
                "right_seed": int(row.right_seed),
                "left_win_probability": float(row.left_win_probability),
                "right_win_probability": float(row.right_win_probability),
                "picked_winner": row.picked_winner,
                "favorite_team": favorite_team,
                "favorite_seed": favorite_seed,
                "favorite_win_probability": favorite_probability,
                "underdog_team": underdog_team,
                "underdog_seed": underdog_seed,
                "underdog_win_probability": underdog_probability,
                "seed_gap": int(row.seed_gap),
                "upset_level": upset_level,
                "upset_flag": bool(upset_level in {"live", "strong"}),
            }
        )

    matchup_probabilities = pd.DataFrame(rows)
    upset_flags = matchup_probabilities.loc[matchup_probabilities["upset_flag"]].copy()
    upset_flags = upset_flags.sort_values(
        ["underdog_win_probability", "seed_gap", "round_index"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return matchup_probabilities, upset_flags


def run_simulation_outputs(
    season_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    if args.historical_games:
        field_df, semifinal_pairs = infer_field_from_historical_games(Path(args.historical_games))
        field_source = Path(args.historical_games)
    elif args.field:
        field_df = pd.read_csv(args.field)
        semifinal_pairs = parse_semifinal_pairs(args.semifinal_pairs)
        field_source = Path(args.field)
    else:
        raise ValueError("Provide either --field or --historical-games to run bracket simulation outputs.")

    if len(semifinal_pairs) != 2:
        raise ValueError("Exactly two semifinal region pairings are required.")

    resolved_field, match_report = resolve_field(season_df, field_df, Path(args.aliases))
    validate_field(resolved_field)

    _, matchup_payload = load_matchup_payload(Path(args.matchup_model_manifest))
    probability_temperature = (
        float(args.probability_temperature)
        if args.probability_temperature is not None
        else float(matchup_payload.get("simulation_temperature", 1.0))
    )

    team_odds, most_likely_bracket = simulate_bracket(
        season_df=season_df,
        field_df=resolved_field,
        semifinal_pairs=semifinal_pairs,
        model_payload=matchup_payload,
        n_sims=args.n_sims,
        use_calibration=not args.disable_calibration,
        probability_temperature=probability_temperature,
    )
    round_odds = make_reach_columns(team_odds)
    matchup_probabilities, upset_flags = annotate_bracket_matchups(
        most_likely_bracket,
        resolved_field,
        upset_threshold=args.upset_threshold,
        strong_upset_threshold=args.strong_upset_threshold,
    )
    round_odds["field_source"] = str(field_source).replace("\\", "/")
    return round_odds, most_likely_bracket, matchup_probabilities, upset_flags, match_report, probability_temperature


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    season_df = pd.read_csv(args.season_data).copy()
    season_df = load_team_model_predictions(season_df, Path(args.team_model_manifest))

    team_rankings = build_team_rankings(season_df)
    contender_scorecard = build_contender_scorecard(season_df)
    team_rankings.to_csv(output_dir / "team_rankings.csv", index=False)
    contender_scorecard.to_csv(output_dir / "contender_scorecard.csv", index=False)

    summary: dict[str, object] = {
        "season_data": str(Path(args.season_data)).replace("\\", "/"),
        "team_ranking_weights": TEAM_RANKING_WEIGHTS,
        "primary_screening_columns": PRIMARY_SCREENING_COLUMNS,
        "secondary_screening_columns": SECONDARY_SCREENING_COLUMNS,
        "simulation_run": False,
    }

    if args.field or args.historical_games:
        team_odds, most_likely_bracket, matchup_probabilities, upset_flags, match_report, probability_temperature = (
            run_simulation_outputs(season_df, args)
        )
        team_odds.to_csv(output_dir / "team_odds.csv", index=False)
        most_likely_bracket.to_csv(output_dir / "most_likely_bracket.csv", index=False)
        matchup_probabilities.to_csv(output_dir / "matchup_probabilities.csv", index=False)
        upset_flags.to_csv(output_dir / "upset_flags.csv", index=False)
        match_report.to_csv(output_dir / "field_match_report.csv", index=False)

        summary.update(
            {
                "simulation_run": True,
                "n_sims": args.n_sims,
                "use_calibration": not args.disable_calibration,
                "probability_temperature": probability_temperature,
                "favorite": team_odds.iloc[0][["team", "win_championship", "make_final_four"]].to_dict(),
                "upset_threshold": args.upset_threshold,
                "strong_upset_threshold": args.strong_upset_threshold,
            }
        )

    (output_dir / "forecast_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote team rankings to {output_dir / 'team_rankings.csv'}")
    print(f"Wrote contender scorecard to {output_dir / 'contender_scorecard.csv'}")
    if summary["simulation_run"]:
        print(f"Wrote team odds to {output_dir / 'team_odds.csv'}")
        print(f"Wrote matchup probabilities to {output_dir / 'matchup_probabilities.csv'}")
        print(f"Wrote upset flags to {output_dir / 'upset_flags.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
