#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


SEED_COLUMNS = ["left_seed", "right_seed", "seed_diff", "seed_abs_diff"]
NON_FEATURE_COLUMNS = {
    "year",
    "round_name",
    "team_left",
    "team_right",
    "left_score",
    "right_score",
    "margin",
    "left_win",
}


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        self.model_ = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **self.kwargs,
        )
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
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        self.model.fit(clipped, y)
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        calibrated = self.model.predict(clipped)
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bracket-aware matchup win-probability models.")
    parser.add_argument(
        "--data",
        default="data/processed/matchup_training_data.csv",
        help="Path to the historical matchup training CSV.",
    )
    parser.add_argument(
        "--team-benchmark-dir",
        default="artifacts/model_benchmarks",
        help="Directory containing team-level benchmark outputs and OOF predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/matchup_model",
        help="Directory for matchup model outputs and saved models.",
    )
    return parser.parse_args()


def make_linear_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def make_tree_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def build_models() -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic": lambda: make_linear_pipeline(
            LogisticRegression(
                max_iter=5000,
                C=1.2,
                class_weight="balanced",
                random_state=42,
            )
        ),
        "hist_gb": lambda: make_tree_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_depth=3,
                max_iter=250,
                min_samples_leaf=10,
                random_state=42,
            )
        ),
        "xgboost": lambda: make_tree_pipeline(
            EncodedXGBClassifier(
                n_estimators=325,
                learning_rate=0.035,
                max_depth=3,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
            )
        ),
    }


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    metrics = {
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, (probability >= 0.5).astype(int))),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probability))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def build_calibrators() -> dict[str, Callable[[], object]]:
    return {
        "identity": IdentityCalibrator,
        "platt": PlattCalibrator,
        "isotonic": IsotonicCalibrator,
    }


def load_team_level_meta_features(benchmark_dir: Path) -> pd.DataFrame:
    best_models = pd.read_csv(benchmark_dir / "best_models.csv")
    oof_dir = benchmark_dir / "oof_predictions"

    meta_frames: list[pd.DataFrame] = []
    for row in best_models.itertuples(index=False):
        oof_path = oof_dir / f"{row.task}__{row.feature_set}__{row.model_name}.csv"
        oof_df = pd.read_csv(oof_path)

        if row.task == "points":
            metric_column = "prediction"
            meta_name = "meta_points"
        elif row.task == "finish":
            metric_column = "expected_value"
            meta_name = "meta_finish"
        else:
            metric_column = "probability"
            meta_name = f"meta_{row.task}"

        meta_frames.append(
            oof_df[["year", "team", metric_column]].rename(columns={"team": "team_name", metric_column: meta_name})
        )

    meta_df = meta_frames[0]
    for frame in meta_frames[1:]:
        meta_df = meta_df.merge(frame, on=["year", "team_name"], how="inner")
    return meta_df


def add_meta_features(matchup_df: pd.DataFrame, team_meta: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    meta_columns = [column for column in team_meta.columns if column not in {"year", "team_name"}]

    left_meta = team_meta.rename(columns={"team_name": "team_left", **{column: f"{column}_left" for column in meta_columns}})
    right_meta = team_meta.rename(
        columns={"team_name": "team_right", **{column: f"{column}_right" for column in meta_columns}}
    )

    enriched = matchup_df.merge(left_meta, on=["year", "team_left"], how="left")
    enriched = enriched.merge(right_meta, on=["year", "team_right"], how="left")

    diff_columns: list[str] = []
    for column in meta_columns:
        left_column = f"{column}_left"
        right_column = f"{column}_right"
        diff_column = f"{column}_diff"
        enriched[diff_column] = enriched[left_column] - enriched[right_column]
        diff_columns.append(diff_column)

    return enriched, diff_columns


def feature_sets(df: pd.DataFrame, meta_diff_columns: list[str]) -> dict[str, list[str]]:
    base_features = sorted(column for column in df.columns if column not in NON_FEATURE_COLUMNS and not column.startswith("meta_"))
    no_seed = [column for column in base_features if column not in SEED_COLUMNS]
    seed_only = ["round_index"] + [column for column in SEED_COLUMNS if column in df.columns]

    return {
        "all_features": base_features,
        "no_seed": no_seed,
        "seed_only": seed_only,
        "meta_all_features": base_features + meta_diff_columns,
        "meta_no_seed": no_seed + meta_diff_columns,
        "meta_seed_only": seed_only + meta_diff_columns,
    }


def evaluate_model(df: pd.DataFrame, feature_names: list[str], model_factory: Callable[[], Pipeline]) -> tuple[dict[str, float], pd.DataFrame]:
    X = df[feature_names]
    y = df["left_win"].to_numpy()
    groups = df["year"].to_numpy()
    splitter = LeaveOneGroupOut()

    rows: list[dict[str, float | int]] = []
    probabilities = np.zeros(len(df), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = model_factory()
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        holdout_year = int(groups[test_idx][0])

        model.fit(X_train, y_train)
        fold_probabilities = model.predict_proba(X_test)[:, 1]
        probabilities[test_idx] = fold_probabilities

        fold_metrics = binary_metrics(y_test, fold_probabilities)
        rows.append({"holdout_year": holdout_year, "fold": fold, "n_test": len(test_idx), **fold_metrics})

    overall_metrics = binary_metrics(y, probabilities)
    fold_df = pd.DataFrame(rows)
    oof = pd.DataFrame({"probability": probabilities, "prediction": (probabilities >= 0.5).astype(int)})
    return overall_metrics | {"fold_metric_mean": float(fold_df["brier"].mean())}, oof


def benchmark_models(df: pd.DataFrame, feature_map: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    models = build_models()
    rows: list[dict[str, float | str]] = []
    oof_predictions: dict[tuple[str, str], pd.DataFrame] = {}

    for feature_set_name, feature_names in feature_map.items():
        for model_name, model_factory in models.items():
            metrics, oof = evaluate_model(df, feature_names, model_factory)
            row = {
                "task": "matchup_win",
                "model_name": model_name,
                "feature_set": feature_set_name,
                "n_features": len(feature_names),
                "feature_names": ",".join(feature_names),
            }
            row.update(metrics)
            rows.append(row)

            oof_predictions[(feature_set_name, model_name)] = pd.concat(
                [df[["year", "round_index", "round_name", "team_left", "team_right", "left_win"]].reset_index(drop=True), oof],
                axis=1,
            )

    return pd.DataFrame(rows), oof_predictions


def evaluate_calibrators(
    df: pd.DataFrame,
    oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrator_factories = build_calibrators()
    y = df["left_win"].to_numpy()
    base_probability = oof["probability"].to_numpy()
    groups = df["year"].to_numpy()
    splitter = LeaveOneGroupOut()

    rows: list[dict[str, float | str]] = []
    calibrated_oof_frames: list[pd.DataFrame] = []

    for calibrator_name, factory in calibrator_factories.items():
        calibrated_probability = np.zeros(len(df), dtype=float)
        fold_rows: list[dict[str, float | int | str]] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(base_probability, y, groups), start=1):
            calibrator = factory()
            calibrator.fit(base_probability[train_idx], y[train_idx])
            fold_probability = calibrator.predict(base_probability[test_idx])
            calibrated_probability[test_idx] = fold_probability

            holdout_year = int(groups[test_idx][0])
            metrics = binary_metrics(y[test_idx], fold_probability)
            fold_rows.append({"calibrator": calibrator_name, "holdout_year": holdout_year, "fold": fold, **metrics})

        metrics = binary_metrics(y, calibrated_probability)
        rows.append(
            {
                "calibrator": calibrator_name,
                **metrics,
                "fold_metric_mean": float(pd.DataFrame(fold_rows)["brier"].mean()),
            }
        )
        calibrated_oof_frames.append(
            pd.DataFrame(
                {
                    "calibrator": calibrator_name,
                    "year": df["year"].to_numpy(),
                    "round_index": df["round_index"].to_numpy(),
                    "round_name": df["round_name"].to_numpy(),
                    "team_left": df["team_left"].to_numpy(),
                    "team_right": df["team_right"].to_numpy(),
                    "left_win": y,
                    "base_probability": base_probability,
                    "calibrated_probability": calibrated_probability,
                }
            )
        )

    return pd.DataFrame(rows).sort_values(["brier", "log_loss", "roc_auc"], ascending=[True, True, False]), pd.concat(
        calibrated_oof_frames, ignore_index=True
    )


def fit_selected_calibrator(base_probability: np.ndarray, y: np.ndarray, calibrator_name: str) -> object:
    calibrator = build_calibrators()[calibrator_name]()
    calibrator.fit(base_probability, y)
    return calibrator


def select_best_model(benchmark: pd.DataFrame) -> pd.Series:
    return benchmark.sort_values(["brier", "log_loss", "roc_auc"], ascending=[True, True, False]).iloc[0]


def build_feature_set_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    summary = benchmark.sort_values(["feature_set", "brier", "log_loss"]).groupby("feature_set", as_index=False).first()
    if {"all_features", "no_seed"}.issubset(set(summary["feature_set"])):
        pass
    return summary


def build_seed_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    by_feature_set = benchmark.sort_values(["feature_set", "brier", "log_loss"]).groupby("feature_set", as_index=False).first()
    row: dict[str, float | str] = {"task": "matchup_win", "primary_metric": "brier"}
    for feature_set in by_feature_set["feature_set"]:
        match = by_feature_set.loc[by_feature_set["feature_set"] == feature_set].iloc[0]
        row[f"{feature_set}_model"] = match["model_name"]
        row[f"{feature_set}_brier"] = float(match["brier"])

    if "all_features_brier" in row and "no_seed_brier" in row:
        row["seed_delta"] = float(row["all_features_brier"] - row["no_seed_brier"])
    if "meta_all_features_brier" in row and "meta_no_seed_brier" in row:
        row["meta_seed_delta"] = float(row["meta_all_features_brier"] - row["meta_no_seed_brier"])
    return pd.DataFrame([row])


def build_round_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for round_index, round_df in oof.groupby("round_index"):
        metrics = binary_metrics(round_df["left_win"].to_numpy(), round_df["probability"].to_numpy())
        rows.append(
            {
                "round_index": int(round_index),
                "round_name": round_df["round_name"].iloc[0],
                "n_games": int(len(round_df) // 2),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("round_index").reset_index(drop=True)


def save_best_model(
    df: pd.DataFrame,
    best_row: pd.Series,
    feature_map: dict[str, list[str]],
    calibrator_name: str,
    calibrator: object,
    output_dir: Path,
) -> pd.DataFrame:
    models_dir = output_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_features = feature_map[str(best_row["feature_set"])]
    model = build_models()[str(best_row["model_name"])]()
    model.fit(df[selected_features], df["left_win"])

    model_path = models_dir / f"matchup_win_{best_row['model_name']}_{best_row['feature_set']}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": selected_features,
            "task": "matchup_win",
            "target": "left_win",
            "feature_set": str(best_row["feature_set"]),
            "model_name": str(best_row["model_name"]),
            "calibrator_name": calibrator_name,
            "calibrator": calibrator,
            "simulation_temperature": 1.0,
        },
        model_path,
    )

    manifest = pd.DataFrame(
        [
            {
                "task": "matchup_win",
                "model_name": str(best_row["model_name"]),
                "feature_set": str(best_row["feature_set"]),
                "calibrator_name": calibrator_name,
                "model_path": str(model_path).replace("\\", "/"),
                "features": ",".join(selected_features),
            }
        ]
    )
    manifest.to_csv(output_dir / "saved_model_manifest.csv", index=False)
    return manifest


def main() -> int:
    args = parse_args()
    data_path = Path(args.data)
    benchmark_dir = Path(args.team_benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matchup_df = pd.read_csv(data_path)
    team_meta = load_team_level_meta_features(benchmark_dir)
    matchup_df, meta_diff_columns = add_meta_features(matchup_df, team_meta)

    feature_map = feature_sets(matchup_df, meta_diff_columns)
    benchmark, oof_predictions = benchmark_models(matchup_df, feature_map)
    best_model = select_best_model(benchmark)
    feature_summary = build_feature_set_summary(benchmark)
    seed_summary = build_seed_summary(benchmark)
    best_oof = oof_predictions[(str(best_model["feature_set"]), str(best_model["model_name"]))]
    calibration_results, calibrated_oof = evaluate_calibrators(matchup_df, best_oof)
    selected_calibrator_name = str(calibration_results.iloc[0]["calibrator"])
    selected_calibrator = fit_selected_calibrator(
        best_oof["probability"].to_numpy(),
        matchup_df["left_win"].to_numpy(),
        selected_calibrator_name,
    )
    saved_manifest = save_best_model(
        matchup_df,
        best_model,
        feature_map,
        selected_calibrator_name,
        selected_calibrator,
        output_dir,
    )

    benchmark.to_csv(output_dir / "benchmark_results.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_set_summary.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)
    pd.DataFrame([best_model]).to_csv(output_dir / "best_models.csv", index=False)
    calibration_results.to_csv(output_dir / "calibration_results.csv", index=False)

    oof_dir = output_dir / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for (feature_set_name, model_name), prediction_df in oof_predictions.items():
        prediction_df.to_csv(oof_dir / f"matchup_win__{feature_set_name}__{model_name}.csv", index=False)
    calibrated_oof.to_csv(oof_dir / "matchup_win__best_model__calibration_oof.csv", index=False)

    round_metrics = build_round_metrics(best_oof)
    round_metrics.to_csv(output_dir / "best_model_round_metrics.csv", index=False)
    calibrated_round_metrics = []
    selected_oof = calibrated_oof.loc[calibrated_oof["calibrator"] == selected_calibrator_name].copy()
    for round_index, round_df in selected_oof.groupby("round_index"):
        metrics = binary_metrics(round_df["left_win"].to_numpy(), round_df["calibrated_probability"].to_numpy())
        calibrated_round_metrics.append(
            {
                "round_index": int(round_index),
                "round_name": round_df["round_name"].iloc[0],
                **metrics,
            }
        )
    pd.DataFrame(calibrated_round_metrics).sort_values("round_index").to_csv(
        output_dir / "best_model_calibrated_round_metrics.csv", index=False
    )

    summary_payload = {
        "data_path": str(data_path).replace("\\", "/"),
        "team_benchmark_dir": str(benchmark_dir).replace("\\", "/"),
        "n_rows": int(len(matchup_df)),
        "years": sorted(matchup_df["year"].unique().tolist()),
        "feature_sets": feature_map,
        "meta_features": meta_diff_columns,
        "best_model": best_model.to_dict(),
        "selected_calibrator": selected_calibrator_name,
        "saved_models": saved_manifest.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote benchmark results to {output_dir / 'benchmark_results.csv'}")
    print(f"Wrote feature-set summary to {output_dir / 'feature_set_summary.csv'}")
    print(f"Wrote best-model summary to {output_dir / 'best_models.csv'}")
    print(f"Wrote calibration results to {output_dir / 'calibration_results.csv'}")
    print(f"Wrote round metrics to {output_dir / 'best_model_round_metrics.csv'}")
    print(f"Wrote saved models to {output_dir / 'saved_models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
