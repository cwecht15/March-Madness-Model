#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


IDENTIFIER_COLUMNS = ["year", "team", "year_team"]
ROUND_TARGETS = [
    ("First_Rd", "round_of_64"),
    ("Second_Rd", "round_of_32"),
    ("Sweet_Sixteen", "sweet_sixteen"),
    ("Elite_Eight", "elite_eight"),
    ("Final_Four", "semifinal"),
    ("Championship", "championship"),
]
TARGET_COLUMNS = ["Finish", "Pts"] + [name for name, _ in ROUND_TARGETS]
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
FEATURE_SETS = {
    "all_features": FEATURE_COLUMNS,
    "no_seed": [column for column in FEATURE_COLUMNS if column != "seed"],
    "seed_only": ["seed"],
}


@dataclass
class TaskConfig:
    name: str
    target: str
    kind: str
    primary_metric: str
    greater_is_better: bool
    positive_label: int | None = None


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        y_encoded = self.encoder_.fit_transform(y)
        n_classes = len(self.encoder_.classes_)

        params = dict(self.kwargs)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark cleaner March Madness models.")
    parser.add_argument(
        "--data",
        default="data/processed/March_Madness_Train_Model_rebuilt.csv",
        help="Path to the rebuilt training CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/model_benchmarks",
        help="Directory for experiment outputs and saved models.",
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


def build_regression_models() -> dict[str, Callable[[], Pipeline]]:
    return {
        "ridge": lambda: make_linear_pipeline(Ridge(alpha=2.0)),
        "poisson": lambda: make_linear_pipeline(PoissonRegressor(alpha=0.5, max_iter=2000)),
        "hist_gb": lambda: make_tree_pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_depth=3,
                max_iter=300,
                min_samples_leaf=12,
                random_state=42,
            )
        ),
        "xgboost": lambda: make_tree_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=350,
                learning_rate=0.035,
                max_depth=3,
                min_child_weight=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
            )
        ),
    }


def build_multiclass_models() -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic": lambda: make_linear_pipeline(
            LogisticRegression(
                max_iter=5000,
                multi_class="multinomial",
                C=1.5,
                class_weight="balanced",
                random_state=42,
            )
        ),
        "hist_gb": lambda: make_tree_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_depth=3,
                max_iter=300,
                min_samples_leaf=12,
                random_state=42,
            )
        ),
        "xgboost": lambda: make_tree_pipeline(
            EncodedXGBClassifier(
                n_estimators=350,
                learning_rate=0.035,
                max_depth=3,
                min_child_weight=6,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=1,
            )
        ),
    }


def build_binary_models() -> dict[str, Callable[[], Pipeline]]:
    return {
        "logistic": lambda: make_linear_pipeline(
            LogisticRegression(
                max_iter=5000,
                C=1.5,
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


def tasks() -> list[TaskConfig]:
    configs = [
        TaskConfig(name="points", target="Pts", kind="regression", primary_metric="mae", greater_is_better=False),
        TaskConfig(
            name="finish",
            target="Finish",
            kind="multiclass",
            primary_metric="expected_finish_mae",
            greater_is_better=False,
        ),
    ]
    configs.extend(
        TaskConfig(
            name=task_name,
            target=target,
            kind="binary",
            primary_metric="brier",
            greater_is_better=False,
            positive_label=1,
        )
        for target, task_name in ROUND_TARGETS
    )
    return configs


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    spearman = spearmanr(y_true, prediction).statistic
    if np.isnan(spearman):
        spearman = 0.0
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": float(spearman),
    }


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    prediction = classes[np.argmax(probabilities, axis=1)]
    expected_value = probabilities @ classes.astype(float)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "expected_finish_mae": float(mean_absolute_error(y_true, expected_value)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, prediction, weights="quadratic")),
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


def evaluate_task(
    df: pd.DataFrame,
    feature_names: list[str],
    task: TaskConfig,
    model_factory: Callable[[], Pipeline],
) -> tuple[dict[str, float], pd.DataFrame]:
    X = df[feature_names]
    y = df[task.target].to_numpy()
    groups = df["year"].to_numpy()
    splitter = LeaveOneGroupOut()

    rows: list[dict[str, float | int | str]] = []
    regression_predictions = np.zeros(len(df), dtype=float)
    binary_probabilities = np.zeros(len(df), dtype=float)
    multiclass_probabilities = None
    multiclass_classes = None

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = model_factory()
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        holdout_year = int(groups[test_idx][0])

        model.fit(X_train, y_train)

        if task.kind == "regression":
            prediction = model.predict(X_test)
            regression_predictions[test_idx] = prediction
            fold_metrics = regression_metrics(y_test, prediction)
        elif task.kind == "multiclass":
            probabilities = model.predict_proba(X_test)
            if multiclass_probabilities is None:
                multiclass_classes = np.array(model.named_steps["model"].classes_ if hasattr(model, "named_steps") else model.classes_)
                multiclass_probabilities = np.zeros((len(df), len(multiclass_classes)), dtype=float)
            multiclass_probabilities[test_idx] = probabilities
            fold_metrics = multiclass_metrics(y_test, probabilities, multiclass_classes)
        else:
            probabilities = model.predict_proba(X_test)
            class_values = np.array(model.named_steps["model"].classes_ if hasattr(model, "named_steps") else model.classes_)
            positive_index = int(np.where(class_values == task.positive_label)[0][0])
            positive_probability = probabilities[:, positive_index]
            binary_probabilities[test_idx] = positive_probability
            fold_metrics = binary_metrics(y_test, positive_probability)

        row = {
            "holdout_year": holdout_year,
            "fold": fold,
            "n_test": len(test_idx),
        }
        row.update(fold_metrics)
        rows.append(row)

    if task.kind == "regression":
        overall_metrics = regression_metrics(y, regression_predictions)
        oof = pd.DataFrame({"prediction": regression_predictions})
    elif task.kind == "multiclass":
        overall_metrics = multiclass_metrics(y, multiclass_probabilities, multiclass_classes)
        prediction = multiclass_classes[np.argmax(multiclass_probabilities, axis=1)]
        expected_value = multiclass_probabilities @ multiclass_classes.astype(float)
        oof = pd.DataFrame(
            {
                "prediction": prediction,
                "expected_value": expected_value,
            }
        )
        for class_index, class_value in enumerate(multiclass_classes):
            oof[f"prob_{class_value}"] = multiclass_probabilities[:, class_index]
    else:
        overall_metrics = binary_metrics(y, binary_probabilities)
        oof = pd.DataFrame({"probability": binary_probabilities, "prediction": (binary_probabilities >= 0.5).astype(int)})

    fold_df = pd.DataFrame(rows)
    return overall_metrics | {"fold_metric_mean": float(fold_df[task.primary_metric].mean())}, oof


def benchmark_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    regressors = build_regression_models()
    multiclass_models = build_multiclass_models()
    binary_models = build_binary_models()

    benchmark_rows: list[dict[str, float | str]] = []
    oof_predictions: dict[tuple[str, str, str], pd.DataFrame] = {}

    for task in tasks():
        model_factories = (
            regressors if task.kind == "regression" else multiclass_models if task.kind == "multiclass" else binary_models
        )

        for feature_set_name, feature_names in FEATURE_SETS.items():
            for model_name, model_factory in model_factories.items():
                metrics, oof = evaluate_task(df, feature_names, task, model_factory)
                row = {
                    "task": task.name,
                    "target": task.target,
                    "kind": task.kind,
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "n_features": len(feature_names),
                    "feature_names": ",".join(feature_names),
                }
                row.update(metrics)
                benchmark_rows.append(row)

                oof = pd.concat([df[IDENTIFIER_COLUMNS + [task.target]].reset_index(drop=True), oof], axis=1)
                oof_predictions[(task.name, feature_set_name, model_name)] = oof

    benchmark = pd.DataFrame(benchmark_rows)
    return benchmark, oof_predictions


def scan_single_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    task_list = tasks()

    for task in task_list:
        if task.kind == "regression":
            factory = lambda: make_linear_pipeline(Ridge(alpha=2.0))
        elif task.kind == "multiclass":
            factory = lambda: make_linear_pipeline(
                LogisticRegression(max_iter=5000, multi_class="multinomial", class_weight="balanced", random_state=42)
            )
        else:
            factory = lambda: make_linear_pipeline(
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
            )

        for feature_name in FEATURE_COLUMNS:
            metrics, _ = evaluate_task(df, [feature_name], task, factory)
            row = {
                "task": task.name,
                "target": task.target,
                "kind": task.kind,
                "feature": feature_name,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def select_best_models(benchmark: pd.DataFrame) -> pd.DataFrame:
    selections: list[pd.Series] = []
    for task in tasks():
        task_rows = benchmark.loc[benchmark["task"] == task.name].copy()
        selections.append(task_rows.sort_values(task.primary_metric, ascending=not task.greater_is_better).iloc[0])
    return pd.DataFrame(selections).reset_index(drop=True)


def save_models(df: pd.DataFrame, best_models: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    models_dir = output_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_rows = []
    reg_models = build_regression_models()
    multi_models = build_multiclass_models()
    bin_models = build_binary_models()

    for row in best_models.itertuples(index=False):
        task = next(task for task in tasks() if task.name == row.task)
        feature_names = FEATURE_SETS[row.feature_set]
        factories = reg_models if task.kind == "regression" else multi_models if task.kind == "multiclass" else bin_models
        model = factories[row.model_name]()
        model.fit(df[feature_names], df[task.target])

        model_path = models_dir / f"{task.name}_{row.model_name}_{row.feature_set}.joblib"
        joblib.dump({"model": model, "features": feature_names, "task": task.name, "target": task.target}, model_path)

        saved_rows.append(
            {
                "task": task.name,
                "target": task.target,
                "model_name": row.model_name,
                "feature_set": row.feature_set,
                "model_path": str(model_path).replace("\\", "/"),
                "features": ",".join(feature_names),
            }
        )

    saved = pd.DataFrame(saved_rows)
    saved.to_csv(output_dir / "saved_model_manifest.csv", index=False)
    return saved


def build_seed_summary(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in tasks():
        task_rows = benchmark.loc[benchmark["task"] == task.name]
        by_feature_set = (
            task_rows.sort_values(task.primary_metric, ascending=not task.greater_is_better)
            .groupby("feature_set", as_index=False)
            .first()
        )

        record = {"task": task.name, "primary_metric": task.primary_metric}
        for feature_set in FEATURE_SETS:
            match = by_feature_set.loc[by_feature_set["feature_set"] == feature_set]
            if match.empty:
                continue
            record[f"{feature_set}_model"] = match.iloc[0]["model_name"]
            record[f"{feature_set}_{task.primary_metric}"] = float(match.iloc[0][task.primary_metric])

        if f"all_features_{task.primary_metric}" in record and f"no_seed_{task.primary_metric}" in record:
            record["seed_delta"] = (
                record[f"all_features_{task.primary_metric}"] - record[f"no_seed_{task.primary_metric}"]
            )
        rows.append(record)

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data).copy()
    benchmark, oof_predictions = benchmark_models(df)
    single_feature = scan_single_features(df)
    best_models = select_best_models(benchmark)
    seed_summary = build_seed_summary(benchmark)
    saved_models = save_models(df, best_models, output_dir)

    benchmark.to_csv(output_dir / "benchmark_results.csv", index=False)
    single_feature.to_csv(output_dir / "single_feature_scan.csv", index=False)
    best_models.to_csv(output_dir / "best_models.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_summary.csv", index=False)

    oof_dir = output_dir / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    for (task_name, feature_set, model_name), prediction_df in oof_predictions.items():
        prediction_df.to_csv(oof_dir / f"{task_name}__{feature_set}__{model_name}.csv", index=False)

    summary_payload = {
        "data_path": args.data,
        "n_rows": int(len(df)),
        "years": sorted(df["year"].unique().tolist()),
        "feature_sets": {name: columns for name, columns in FEATURE_SETS.items()},
        "best_models": best_models.to_dict(orient="records"),
        "saved_models": saved_models.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote benchmark results to {output_dir / 'benchmark_results.csv'}")
    print(f"Wrote single-feature scan to {output_dir / 'single_feature_scan.csv'}")
    print(f"Wrote best-model summary to {output_dir / 'best_models.csv'}")
    print(f"Wrote seed comparison summary to {output_dir / 'seed_summary.csv'}")
    print(f"Wrote saved models to {output_dir / 'saved_models'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
