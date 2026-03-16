#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.build_matchup_training_data import SeasonTeamMatcher, load_aliases
from scripts.simulate_bracket import (
    EncodedXGBClassifier,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    REGIONAL_FIRST_ROUND_PODS,
    load_team_model_predictions,
    resolve_field,
    simulate_bracket,
)


def clean_team_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["team"] = cleaned["team"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest and tune bracket-level probability temperature.")
    parser.add_argument(
        "--team-data",
        default="data/processed/March_Madness_Train_Model_rebuilt.csv",
        help="Season-level dataset containing all historical tournament teams.",
    )
    parser.add_argument("--raw-root", default="data/raw", help="Root directory containing per-year tournament_games.csv files.")
    parser.add_argument(
        "--team-model-manifest",
        default="artifacts/model_benchmarks/saved_model_manifest.csv",
        help="Saved team-level model manifest used to build matchup meta-features.",
    )
    parser.add_argument(
        "--matchup-model-manifest",
        default="artifacts/matchup_model/saved_model_manifest.csv",
        help="Saved matchup model manifest to audit and update.",
    )
    parser.add_argument("--aliases", default="data/team_aliases.csv", help="Alias CSV used for team-name reconciliation.")
    parser.add_argument(
        "--manual-corrections",
        default="data/manual_tournament_game_corrections.csv",
        help="Optional CSV of manually corrected tournament game rows.",
    )
    parser.add_argument(
        "--temperatures",
        default="1.0,1.1,1.25,1.4,1.6,1.8,2.0,2.25,2.5,3.0",
        help="Comma-separated list of candidate bracket temperature values.",
    )
    parser.add_argument("--n-sims", type=int, default=3000, help="Number of simulations per season and temperature.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/bracket_backtest",
        help="Directory for backtest outputs.",
    )
    parser.add_argument(
        "--disable-calibration",
        action="store_true",
        help="Ignore the saved game-level calibrator while tuning bracket temperature.",
    )
    parser.add_argument(
        "--update-model-payload",
        action="store_true",
        help="Write the selected temperature back into the saved matchup model payload.",
    )
    return parser.parse_args()


def parse_temperature_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def available_years(team_df: pd.DataFrame, raw_root: Path) -> list[int]:
    years = []
    for year in sorted(team_df["year"].unique().tolist()):
        if (raw_root / str(year) / "tournament_games.csv").exists():
            years.append(int(year))
    return years


def resolve_actual_champion(season_df: pd.DataFrame, games_path: Path, aliases_path: Path) -> str:
    games_df = pd.read_csv(games_path)
    title_game = games_df.loc[games_df["round_name"] == "Championship"].iloc[0]
    winner_raw = str(title_game["team_a"] if int(title_game["score_a"]) > int(title_game["score_b"]) else title_game["team_b"])

    matcher = SeasonTeamMatcher(season_df["team"].tolist(), load_aliases(aliases_path))
    return matcher.resolve("games", winner_raw)


def complete_field_from_season(season_df: pd.DataFrame, resolved_field: pd.DataFrame) -> pd.DataFrame:
    season_names = set(season_df["team"])
    field_names = set(resolved_field["team"])
    extra_field_teams = sorted(field_names - season_names)
    if extra_field_teams:
        raise ValueError(f"field contains teams not present in season data: {', '.join(extra_field_teams)}")

    if len(resolved_field) == 64:
        return resolved_field.copy()

    missing_teams = sorted(season_names - field_names)
    missing_slots: list[dict[str, object]] = []
    for region in sorted(resolved_field["region"].unique()):
        present_seeds = set(resolved_field.loc[resolved_field["region"] == region, "seed"].astype(int).tolist())
        for seed in range(1, 17):
            if seed not in present_seeds:
                missing_slots.append({"region": region, "seed": seed})

    if len(missing_slots) != len(missing_teams):
        raise ValueError(
            f"could not uniquely infer missing field teams: {len(missing_teams)} teams for {len(missing_slots)} slots"
        )

    remaining = season_df.loc[season_df["team"].isin(missing_teams), ["team", "seed"]].copy()
    additions: list[dict[str, object]] = []
    for slot in missing_slots:
        candidates = remaining.loc[remaining["seed"].astype(int) == int(slot["seed"])].copy()
        if len(candidates) != 1:
            raise ValueError(
                f"could not uniquely assign missing seed {int(slot['seed'])} in {slot['region']}: {len(candidates)} candidates"
            )
        team_name = str(candidates.iloc[0]["team"])
        additions.append({"team": team_name, "seed": int(slot["seed"]), "region": str(slot["region"])})
        remaining = remaining.loc[remaining["team"] != team_name].copy()

    return (
        pd.concat([resolved_field.copy(), pd.DataFrame(additions)], ignore_index=True)
        .sort_values(["region", "seed", "team"])
        .reset_index(drop=True)
    )


def load_corrected_games(year: int, raw_root: Path, manual_corrections_path: Path) -> pd.DataFrame:
    games_df = pd.read_csv(raw_root / str(year) / "tournament_games.csv")
    if manual_corrections_path.exists():
        corrections = pd.read_csv(manual_corrections_path)
        corrections = corrections.loc[corrections["year"] == year].copy()
        if not corrections.empty:
            games_df = pd.concat([games_df, corrections], ignore_index=True, sort=False)
    return games_df


def infer_field_from_games_df(games_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    field_rows: list[dict[str, str | int]] = []

    for region in ["East", "West", "South", "Midwest"]:
        region_games = games_df.loc[games_df["round_name"] == region].copy()
        if region_games.empty:
            continue
        first_round_games = region_games.loc[
            region_games.apply(lambda row: {int(row["seed_a"]), int(row["seed_b"])} in REGIONAL_FIRST_ROUND_PODS, axis=1)
        ]
        for row in first_round_games.itertuples(index=False):
            field_rows.append({"team": row.team_a, "seed": int(row.seed_a), "region": region})
            field_rows.append({"team": row.team_b, "seed": int(row.seed_b), "region": region})

    field_df = pd.DataFrame(field_rows).drop_duplicates(subset=["team"]).sort_values(["region", "seed", "team"])
    region_lookup = dict(zip(field_df["team"], field_df["region"]))

    semifinal_pairs: list[tuple[str, str]] = []
    final_four = games_df.loc[games_df["round_name"] == "Final Four"]
    for row in final_four.itertuples(index=False):
        left_region = region_lookup.get(row.team_a)
        right_region = region_lookup.get(row.team_b)
        if not left_region or not right_region:
            continue
        semifinal_pairs.append((left_region, right_region))

    return field_df.reset_index(drop=True), semifinal_pairs


def multiclass_brier(probabilities: np.ndarray, champion_index: int) -> float:
    target = np.zeros(len(probabilities), dtype=float)
    target[champion_index] = 1.0
    return float(np.square(probabilities - target).sum())


def build_year_contexts(
    all_team_df: pd.DataFrame,
    years: list[int],
    raw_root: Path,
    aliases_path: Path,
    manual_corrections_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    contexts: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for year in years:
        season_df = clean_team_names(all_team_df.loc[all_team_df["year"] == year].copy())
        games_path = raw_root / str(year) / "tournament_games.csv"

        try:
            games_df = load_corrected_games(year, raw_root, manual_corrections_path)
            field_df, semifinal_pairs = infer_field_from_games_df(games_df)
            resolved_field, _ = resolve_field(season_df, field_df, aliases_path)
            resolved_field = complete_field_from_season(season_df, resolved_field)
            if len(resolved_field) != 64:
                raise ValueError(f"resolved field has {len(resolved_field)} teams")

            if len(semifinal_pairs) != 2:
                raise ValueError(f"expected 2 semifinal pairs, found {len(semifinal_pairs)}")

            title_game = games_df.loc[games_df["round_name"] == "Championship"].iloc[0]
            winner_raw = str(title_game["team_a"] if int(title_game["score_a"]) > int(title_game["score_b"]) else title_game["team_b"])
            matcher = SeasonTeamMatcher(season_df["team"].tolist(), load_aliases(aliases_path))
            actual_champion = matcher.resolve("games", winner_raw)
            contexts.append(
                {
                    "year": year,
                    "season_df": season_df,
                    "games_path": games_path,
                    "resolved_field": resolved_field,
                    "semifinal_pairs": semifinal_pairs,
                    "actual_champion": actual_champion,
                }
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append({"year": year, "reason": str(exc)})

    return contexts, skipped


def audit_temperature(
    contexts: list[dict[str, object]],
    matchup_payload: dict,
    n_sims: int,
    use_calibration: bool,
    temperature: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_rows: list[dict[str, object]] = []
    title_rows: list[dict[str, object]] = []

    for context in contexts:
        year = int(context["year"])
        season_df = context["season_df"]
        resolved_field = context["resolved_field"]
        semifinal_pairs = context["semifinal_pairs"]
        team_odds, _ = simulate_bracket(
            season_df=season_df,
            field_df=resolved_field,
            semifinal_pairs=semifinal_pairs,
            model_payload=matchup_payload,
            n_sims=n_sims,
            use_calibration=use_calibration,
            probability_temperature=temperature,
        )

        actual_champion = str(context["actual_champion"])
        actual_row = team_odds.loc[team_odds["team"] == actual_champion].iloc[0]
        champion_index = int(team_odds.index[team_odds["team"] == actual_champion][0])
        probabilities = team_odds["win_championship"].to_numpy()
        top_row = team_odds.iloc[0]

        season_rows.append(
            {
                "year": year,
                "temperature": temperature,
                "actual_champion": actual_champion,
                "actual_champion_prob": float(actual_row["win_championship"]),
                "predicted_champion": top_row["team"],
                "predicted_champion_prob": float(top_row["win_championship"]),
                "top4_mass": float(team_odds["win_championship"].head(4).sum()),
                "top8_mass": float(team_odds["win_championship"].head(8).sum()),
                "top_pick_hit": int(top_row["team"] == actual_champion),
                "multiclass_brier": multiclass_brier(probabilities, champion_index),
                "champion_log_loss": float(-math.log(max(float(actual_row["win_championship"]), 1e-9))),
            }
        )

        for row in team_odds.itertuples(index=False):
            title_rows.append(
                {
                    "year": year,
                    "temperature": temperature,
                    "team": row.team,
                    "win_championship": float(row.win_championship),
                    "is_champion": int(row.team == actual_champion),
                }
            )

    return pd.DataFrame(season_rows), pd.DataFrame(title_rows)


def summarize_grid(season_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for temperature, group in season_summary.groupby("temperature"):
        rows.append(
            {
                "temperature": float(temperature),
                "n_years": int(len(group)),
                "mean_actual_champion_prob": float(group["actual_champion_prob"].mean()),
                "median_actual_champion_prob": float(group["actual_champion_prob"].median()),
                "mean_top_pick_prob": float(group["predicted_champion_prob"].mean()),
                "mean_top4_mass": float(group["top4_mass"].mean()),
                "mean_top8_mass": float(group["top8_mass"].mean()),
                "top_pick_hit_rate": float(group["top_pick_hit"].mean()),
                "mean_multiclass_brier": float(group["multiclass_brier"].mean()),
                "mean_champion_log_loss": float(group["champion_log_loss"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_champion_log_loss", "mean_multiclass_brier", "temperature"])


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    team_df = pd.read_csv(args.team_data)
    team_df = load_team_model_predictions(team_df, Path(args.team_model_manifest))
    team_df = clean_team_names(team_df)

    raw_root = Path(args.raw_root)
    aliases_path = Path(args.aliases)
    manual_corrections_path = Path(args.manual_corrections)
    years = available_years(team_df, raw_root)
    contexts, skipped_years = build_year_contexts(team_df, years, raw_root, aliases_path, manual_corrections_path)

    matchup_manifest = pd.read_csv(args.matchup_model_manifest)
    matchup_model_path = Path(matchup_manifest.iloc[0]["model_path"])
    matchup_payload = joblib.load(matchup_model_path)

    season_frames: list[pd.DataFrame] = []
    title_frames: list[pd.DataFrame] = []
    for temperature in parse_temperature_list(args.temperatures):
        season_summary, title_table = audit_temperature(
            contexts=contexts,
            matchup_payload=matchup_payload,
            n_sims=args.n_sims,
            use_calibration=not args.disable_calibration,
            temperature=temperature,
        )
        season_frames.append(season_summary)
        title_frames.append(title_table)

    all_seasons = pd.concat(season_frames, ignore_index=True)
    all_titles = pd.concat(title_frames, ignore_index=True)
    grid_summary = summarize_grid(all_seasons)
    best_temperature = float(grid_summary.iloc[0]["temperature"])

    best_season_summary = all_seasons.loc[all_seasons["temperature"] == best_temperature].copy()
    best_title_table = all_titles.loc[all_titles["temperature"] == best_temperature].copy()

    grid_summary.to_csv(output_dir / "temperature_grid_summary.csv", index=False)
    all_seasons.to_csv(output_dir / "season_summary_all_temperatures.csv", index=False)
    best_season_summary.to_csv(output_dir / "season_summary_best_temperature.csv", index=False)
    best_title_table.to_csv(output_dir / "title_probabilities_best_temperature.csv", index=False)

    if args.update_model_payload:
        matchup_payload["simulation_temperature"] = best_temperature
        joblib.dump(matchup_payload, matchup_model_path)

    summary_payload = {
        "years": years,
        "included_years": [int(context["year"]) for context in contexts],
        "skipped_years": skipped_years,
        "n_sims": args.n_sims,
        "use_calibration": not args.disable_calibration,
        "best_temperature": best_temperature,
        "updated_model_payload": bool(args.update_model_payload),
        "best_grid_row": grid_summary.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote temperature grid summary to {output_dir / 'temperature_grid_summary.csv'}")
    print(f"Wrote season backtest summary to {output_dir / 'season_summary_best_temperature.csv'}")
    print(f"Best probability temperature: {best_temperature:.3f}")
    if args.update_model_payload:
        print(f"Updated matchup model payload at {matchup_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
