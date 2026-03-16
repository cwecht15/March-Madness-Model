#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.build_matchup_training_data import FEATURE_COLUMNS as TEAM_FEATURE_COLUMNS
from scripts.build_matchup_training_data import SeasonTeamMatcher, load_aliases


PAIRING_ORDER = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
ROUND_LABELS = {
    0: "first_four",
    1: "round_of_64",
    2: "round_of_32",
    3: "sweet_sixteen",
    4: "elite_eight",
    5: "final_four",
    6: "championship",
}
REGIONAL_FIRST_ROUND_PODS = [set(pairing) for pairing in PAIRING_ORDER]


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)

        params = dict(self.kwargs)
        n_classes = len(self.encoder_.classes_)
        if n_classes <= 2:
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")
        else:
            params.setdefault("objective", "multi:softprob")
            params["num_class"] = n_classes
            params.setdefault("eval_metric", "mlogloss")

        self.model_ = XGBClassifier(**params)
        self.model_.fit(X, y_encoded)
        self.classes_ = self.encoder_.classes_
        return self

    def predict_proba(self, X):
        probabilities = self.model_.predict_proba(X)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        return probabilities

    def predict(self, X):
        encoded_predictions = self.model_.predict(X)
        return self.encoder_.inverse_transform(encoded_predictions.astype(int))


class IdentityCalibrator:
    def fit(self, probability: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=5000, random_state=42)

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        feature = self._transform(probability)
        self.model.fit(feature, y)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        feature = self._transform(probability)
        calibrated = self.model.predict_proba(feature)[:, 1]
        return np.clip(calibrated, 1e-6, 1 - 1e-6)

    @staticmethod
    def _transform(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        return logits.reshape(-1, 1)


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.model = None

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        calibrated = self.model.predict(np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6))
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate an NCAA tournament bracket using the trained matchup model.")
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
        help="Saved team-level model manifest for generating matchup meta-features.",
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
        "--output-dir",
        default="artifacts/bracket_simulation",
        help="Directory for simulation outputs.",
    )
    return parser.parse_args()


def infer_field_from_historical_games(games_path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    games_df = pd.read_csv(games_path)
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


def parse_semifinal_pairs(text: str) -> list[tuple[str, str]]:
    if not text.strip():
        return []

    pairs: list[tuple[str, str]] = []
    for item in text.split(","):
        left, right = [piece.strip() for piece in item.split("-", maxsplit=1)]
        pairs.append((left, right))
    return pairs


def load_team_model_predictions(season_df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    predicted = season_df.copy()
    predicted["team"] = predicted["team"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    for row in manifest.itertuples(index=False):
        payload = joblib.load(Path(row.model_path))
        model = payload["model"]
        features = payload["features"]
        X = predicted[features]

        if row.task == "points":
            predicted["meta_points"] = model.predict(X)
        elif row.task == "finish":
            probabilities = model.predict_proba(X)
            class_values = np.array(model.named_steps["model"].classes_)
            predicted["meta_finish"] = probabilities @ class_values.astype(float)
        else:
            probabilities = model.predict_proba(X)
            class_values = np.array(model.named_steps["model"].classes_)
            positive_index = int(np.where(class_values == 1)[0][0])
            predicted[f"meta_{row.task}"] = probabilities[:, positive_index]

    return predicted


def resolve_field(season_df: pd.DataFrame, field_df: pd.DataFrame, aliases_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_map = load_aliases(aliases_path)
    matcher = SeasonTeamMatcher(season_df["team"].tolist(), alias_map)

    resolved = field_df.copy()
    resolved["team"] = resolved["team"].apply(lambda value: matcher.resolve("field", value))
    resolved = resolved.drop_duplicates(subset=["team"]).reset_index(drop=True)
    return resolved, matcher.report()


def build_team_lookup(season_df: pd.DataFrame) -> dict[str, pd.Series]:
    return {row.team: row for row in season_df.itertuples(index=False)}


def matchup_feature_row(
    left_team: pd.Series,
    right_team: pd.Series,
    round_index: int,
    required_features: list[str],
) -> pd.DataFrame:
    row: dict[str, float | int] = {
        "round_index": round_index,
        "left_seed": int(left_team.seed),
        "right_seed": int(right_team.seed),
        "seed_diff": int(left_team.seed) - int(right_team.seed),
        "seed_abs_diff": abs(int(left_team.seed) - int(right_team.seed)),
    }

    for feature in TEAM_FEATURE_COLUMNS:
        if feature == "seed":
            continue
        row[f"{feature}_left"] = float(getattr(left_team, feature))
        row[f"{feature}_right"] = float(getattr(right_team, feature))
        row[f"{feature}_diff"] = float(getattr(left_team, feature) - getattr(right_team, feature))

    for column in required_features:
        if not column.startswith("meta_") or not column.endswith("_diff"):
            continue
        meta_name = column[: -len("_diff")]
        row[column] = float(getattr(left_team, meta_name) - getattr(right_team, meta_name))

    return pd.DataFrame([row], columns=required_features).fillna(0.0)


def predict_matchup_probability(
    model_payload: dict,
    left_team: pd.Series,
    right_team: pd.Series,
    round_index: int,
    probability_cache: dict[tuple[str, str, int], float],
    use_calibration: bool,
    probability_temperature: float,
) -> float:
    cache_key = (left_team.team, right_team.team, round_index, round(probability_temperature, 6), int(use_calibration))
    if cache_key in probability_cache:
        return probability_cache[cache_key]

    feature_names = model_payload["features"]
    feature_row = matchup_feature_row(left_team, right_team, round_index, feature_names)
    model = model_payload["model"]
    probability = float(model.predict_proba(feature_row)[:, 1][0])
    if use_calibration and model_payload.get("calibrator") is not None:
        probability = float(model_payload["calibrator"].predict(np.array([probability]))[0])
    if probability_temperature != 1.0:
        clipped = max(1e-6, min(1 - 1e-6, probability))
        logit = np.log(clipped / (1.0 - clipped))
        probability = float(1.0 / (1.0 + np.exp(-(logit / probability_temperature))))
    probability = max(1e-6, min(1 - 1e-6, probability))
    probability_cache[cache_key] = probability
    return probability


def simulate_round(
    games: list[tuple[str, str]],
    round_index: int,
    team_lookup: dict[str, pd.Series],
    model_payload: dict,
    probability_cache: dict[tuple[str, str, int], float],
    use_calibration: bool,
    probability_temperature: float,
    rng: np.random.Generator,
    counts: dict[str, dict[str, int]],
    deterministic_games: list[dict[str, object]] | None = None,
) -> list[str]:
    winners: list[str] = []
    round_label = ROUND_LABELS[round_index]

    for game_number, (left_name, right_name) in enumerate(games, start=1):
        left_team = team_lookup[left_name]
        right_team = team_lookup[right_name]
        left_probability = predict_matchup_probability(
            model_payload,
            left_team,
            right_team,
            round_index,
            probability_cache,
            use_calibration,
            probability_temperature,
        )

        if deterministic_games is not None:
            winner = left_name if left_probability >= 0.5 else right_name
            deterministic_games.append(
                {
                    "round_index": round_index,
                    "round_name": round_label,
                    "game_number": game_number,
                    "team_left": left_name,
                    "team_right": right_name,
                    "left_win_probability": left_probability,
                    "picked_winner": winner,
                }
            )
        else:
            winner = left_name if rng.random() < left_probability else right_name

        winners.append(winner)
        counts[winner][round_label] += 1

    return winners


def region_games_for_round(winners: list[str]) -> list[tuple[str, str]]:
    return [(winners[index], winners[index + 1]) for index in range(0, len(winners), 2)]


def prepare_region_seed_map(field_df: pd.DataFrame) -> dict[str, dict[int, list[str]]]:
    region_team_map: dict[str, dict[int, list[str]]] = {}
    for region, region_df in field_df.groupby("region"):
        seed_map: dict[int, list[str]] = {}
        for seed, seed_df in region_df.groupby(region_df["seed"].astype(int)):
            seed_map[int(seed)] = seed_df["team"].astype(str).tolist()
        region_team_map[str(region)] = seed_map
    return region_team_map


def validate_field(field_df: pd.DataFrame) -> None:
    required_columns = {"team", "seed", "region"}
    missing = required_columns - set(field_df.columns)
    if missing:
        raise ValueError(f"Field is missing required columns: {sorted(missing)}")

    region_team_map = prepare_region_seed_map(field_df)
    if set(region_team_map) != {"East", "West", "South", "Midwest"}:
        raise ValueError(f"Field must include East, West, South, and Midwest regions. Found {sorted(region_team_map)}")

    for region, seed_map in region_team_map.items():
        if set(seed_map) != set(range(1, 17)):
            raise ValueError(f"{region} field must contain seeds 1-16 exactly once or twice for play-ins.")
        for seed, teams in seed_map.items():
            if len(teams) not in {1, 2}:
                raise ValueError(f"{region} seed {seed} must have one team or one play-in pair, found {len(teams)} teams.")

    if len(field_df) not in {64, 68}:
        raise ValueError(f"Expected a 64-team or 68-team field, found {len(field_df)} teams.")


def simulate_region(
    seed_map: dict[int, list[str]],
    team_lookup: dict[str, pd.Series],
    model_payload: dict,
    probability_cache: dict[tuple[str, str, int], float],
    use_calibration: bool,
    probability_temperature: float,
    rng: np.random.Generator,
    counts: dict[str, dict[str, int]],
    deterministic_games: list[dict[str, object]] | None = None,
) -> str:
    slot_winners: dict[int, str] = {}
    play_in_games = [(seed, teams) for seed, teams in sorted(seed_map.items()) if len(teams) == 2]

    if play_in_games:
        play_in_pairs = [(teams[0], teams[1]) for _, teams in play_in_games]
        play_in_winners = simulate_round(
            play_in_pairs,
            0,
            team_lookup,
            model_payload,
            probability_cache,
            use_calibration,
            probability_temperature,
            rng,
            counts,
            deterministic_games,
        )
        for (seed, teams), winner in zip(play_in_games, play_in_winners):
            slot_winners[seed] = winner

    for seed, teams in seed_map.items():
        if len(teams) == 1:
            slot_winners[seed] = teams[0]

    round_one_games = [(slot_winners[left_seed], slot_winners[right_seed]) for left_seed, right_seed in PAIRING_ORDER]
    round_one_winners = simulate_round(
        round_one_games,
        1,
        team_lookup,
        model_payload,
        probability_cache,
        use_calibration,
        probability_temperature,
        rng,
        counts,
        deterministic_games,
    )
    round_two_winners = simulate_round(
        region_games_for_round(round_one_winners),
        2,
        team_lookup,
        model_payload,
        probability_cache,
        use_calibration,
        probability_temperature,
        rng,
        counts,
        deterministic_games,
    )
    sweet_sixteen_winners = simulate_round(
        region_games_for_round(round_two_winners),
        3,
        team_lookup,
        model_payload,
        probability_cache,
        use_calibration,
        probability_temperature,
        rng,
        counts,
        deterministic_games,
    )
    elite_eight_winner = simulate_round(
        region_games_for_round(sweet_sixteen_winners),
        4,
        team_lookup,
        model_payload,
        probability_cache,
        use_calibration,
        probability_temperature,
        rng,
        counts,
        deterministic_games,
    )
    return elite_eight_winner[0]


def simulate_bracket(
    season_df: pd.DataFrame,
    field_df: pd.DataFrame,
    semifinal_pairs: list[tuple[str, str]],
    model_payload: dict,
    n_sims: int,
    use_calibration: bool = True,
    probability_temperature: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_lookup = build_team_lookup(season_df)
    validate_field(field_df)
    teams = sorted(field_df["team"].tolist())
    counts = {team: {label: 0 for label in ROUND_LABELS.values()} for team in teams}
    rng = np.random.default_rng(42)
    probability_cache: dict[tuple[str, str, int], float] = {}
    region_team_map = prepare_region_seed_map(field_df)

    for _ in range(n_sims):
        region_winners: dict[str, str] = {}

        for region, seed_map in region_team_map.items():
            region_winners[region] = simulate_region(
                seed_map=seed_map,
                team_lookup=team_lookup,
                model_payload=model_payload,
                probability_cache=probability_cache,
                use_calibration=use_calibration,
                probability_temperature=probability_temperature,
                rng=rng,
                counts=counts,
            )

        semifinal_winners: list[str] = []
        for left_region, right_region in semifinal_pairs:
            semifinal_game = [(region_winners[left_region], region_winners[right_region])]
            semifinal_winners.extend(
                simulate_round(
                    semifinal_game,
                    5,
                    team_lookup,
                    model_payload,
                    probability_cache,
                    use_calibration,
                    probability_temperature,
                    rng,
                    counts,
                )
            )

        simulate_round(
            [(semifinal_winners[0], semifinal_winners[1])],
            6,
            team_lookup,
            model_payload,
            probability_cache,
            use_calibration,
            probability_temperature,
            rng,
            counts,
        )

    rows: list[dict[str, object]] = []
    for row in field_df.itertuples(index=False):
        team_counts = counts[row.team]
        rows.append(
            {
                "team": row.team,
                "seed": int(row.seed),
                "region": row.region,
                "play_in_group": getattr(row, "play_in_group", ""),
                "win_first_four": team_counts["first_four"] / n_sims,
                "win_round_of_64": team_counts["round_of_64"] / n_sims,
                "win_round_of_32": team_counts["round_of_32"] / n_sims,
                "win_sweet_sixteen": team_counts["sweet_sixteen"] / n_sims,
                "win_elite_eight": team_counts["elite_eight"] / n_sims,
                "win_final_four": team_counts["final_four"] / n_sims,
                "win_championship": team_counts["championship"] / n_sims,
                "expected_wins": sum(team_counts.values()) / n_sims,
            }
        )

    team_odds = pd.DataFrame(rows).sort_values(["win_championship", "win_final_four", "expected_wins"], ascending=False)

    deterministic_games: list[dict[str, object]] = []
    region_winners: dict[str, str] = {}
    dummy_counts = {team: {label: 0 for label in ROUND_LABELS.values()} for team in teams}
    deterministic_rng = np.random.default_rng(42)

    for region, seed_map in region_team_map.items():
        region_winners[region] = simulate_region(
            seed_map=seed_map,
            team_lookup=team_lookup,
            model_payload=model_payload,
            probability_cache=probability_cache,
            use_calibration=use_calibration,
            probability_temperature=probability_temperature,
            rng=deterministic_rng,
            counts=dummy_counts,
            deterministic_games=deterministic_games,
        )

    semifinal_winners: list[str] = []
    for left_region, right_region in semifinal_pairs:
        semifinal_game = [(region_winners[left_region], region_winners[right_region])]
        semifinal_winners.extend(
            simulate_round(
                semifinal_game,
                5,
                team_lookup,
                model_payload,
                probability_cache,
                use_calibration,
                probability_temperature,
                deterministic_rng,
                dummy_counts,
                deterministic_games,
            )
        )
    simulate_round(
        [(semifinal_winners[0], semifinal_winners[1])],
        6,
        team_lookup,
        model_payload,
        probability_cache,
        use_calibration,
        probability_temperature,
        deterministic_rng,
        dummy_counts,
        deterministic_games,
    )

    most_likely_bracket = pd.DataFrame(deterministic_games)
    return team_odds.reset_index(drop=True), most_likely_bracket


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    season_df = pd.read_csv(args.season_data).copy()
    season_df = load_team_model_predictions(season_df, Path(args.team_model_manifest))

    if args.historical_games:
        field_df, semifinal_pairs = infer_field_from_historical_games(Path(args.historical_games))
    elif args.field:
        field_df = pd.read_csv(args.field)
        semifinal_pairs = parse_semifinal_pairs(args.semifinal_pairs)
    else:
        raise ValueError("Provide either --field or --historical-games.")

    if len(semifinal_pairs) != 2:
        raise ValueError("Exactly two semifinal region pairings are required.")

    resolved_field, match_report = resolve_field(season_df, field_df, Path(args.aliases))
    validate_field(resolved_field)

    matchup_manifest = pd.read_csv(args.matchup_model_manifest)
    matchup_model_path = Path(matchup_manifest.iloc[0]["model_path"])
    matchup_payload = joblib.load(matchup_model_path)
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

    team_odds.to_csv(output_dir / "team_odds.csv", index=False)
    most_likely_bracket.to_csv(output_dir / "most_likely_bracket.csv", index=False)
    match_report.to_csv(output_dir / "field_match_report.csv", index=False)

    summary = {
        "season_data": str(Path(args.season_data)).replace("\\", "/"),
        "field_source": str(Path(args.historical_games if args.historical_games else args.field)).replace("\\", "/"),
        "n_sims": args.n_sims,
        "use_calibration": not args.disable_calibration,
        "probability_temperature": probability_temperature,
        "semifinal_pairs": semifinal_pairs,
        "favorite": team_odds.iloc[0][["team", "win_championship", "win_final_four"]].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote team odds to {output_dir / 'team_odds.csv'}")
    print(f"Wrote most-likely bracket to {output_dir / 'most_likely_bracket.csv'}")
    print(f"Wrote field match report to {output_dir / 'field_match_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
