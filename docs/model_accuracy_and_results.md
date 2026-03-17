# Model Accuracy And Results

Date: 2026-03-16

This file summarizes the saved benchmark and forecast results currently in the repository.

## 1. Data scope

Historical modeling is based on the rebuilt tournament dataset in:

- `data/processed/March_Madness_Train_Model_rebuilt.csv`

Usable tournament years included in the bracket backtest:

- `2011-2019`
- `2021-2025`

`2020` is not included because there was no NCAA tournament.

## 2. Best team-level models

Source:

- `artifacts/model_benchmarks/best_models.csv`

### Points (`Pts`)

- Model: `poisson`
- Feature set: `no_seed`
- Features: `22`
- MAE: `2.9055`
- RMSE: `7.3528`
- R^2: `0.3065`
- Spearman: `0.6466`

### Finish (`Finish`)

- Model: `xgboost`
- Feature set: `no_seed`
- Features: `22`
- Accuracy: `56.77%`
- Expected-finish MAE: `0.6633`
- Log loss: `1.1283`
- Quadratic kappa: `0.5912`

### Round advancement models

- `Round of 64`
  - Model: `logistic`
  - Feature set: `no_seed`
  - Brier: `0.1617`
  - Accuracy: `75.76%`
  - ROC AUC: `0.8446`
- `Round of 32`
  - Model: `xgboost`
  - Feature set: `all_features`
  - Brier: `0.1187`
  - Accuracy: `84.17%`
  - ROC AUC: `0.8651`
- `Sweet 16`
  - Model: `hist_gb`
  - Feature set: `all_features`
  - Brier: `0.0808`
  - Accuracy: `89.96%`
  - ROC AUC: `0.8407`
- `Elite Eight`
  - Model: `xgboost`
  - Feature set: `no_seed`
  - Brier: `0.0510`
  - Accuracy: `93.67%`
  - ROC AUC: `0.7987`
- `Final Four`
  - Model: `xgboost`
  - Feature set: `all_features`
  - Brier: `0.0247`
  - Accuracy: `96.83%`
  - ROC AUC: `0.8688`
- `Championship`
  - Model: `hist_gb`
  - Feature set: `no_seed`
  - Brier: `0.0122`
  - Accuracy: `98.58%`
  - ROC AUC: `0.9142`

## 3. Seed impact in team-level models

Source:

- `artifacts/model_benchmarks/seed_summary.csv`

Takeaway:

- Seed is helpful for some tasks, but not universally.
- Best models for `Pts`, `Finish`, `Round of 64`, `Elite Eight`, and `Championship` exclude seed.
- Best models for `Round of 32`, `Sweet 16`, and `Final Four` include seed.
- The gains from seed in the team-level models are generally small.

Examples:

- `Pts`: no-seed MAE `2.9055` vs all-features MAE `2.9094`
- `Round of 32`: all-features Brier `0.1187` vs no-seed Brier `0.1206`
- `Championship`: no-seed Brier `0.0122` vs all-features Brier `0.0126`

## 4. Best matchup model

Source:

- `artifacts/matchup_model/best_models.csv`

Selected game-level model:

- Task: `matchup_win`
- Model: `logistic`
- Feature set: `meta_all_features`
- Features: `79`

Overall out-of-fold metrics:

- Brier: `0.1480`
- Log loss: `0.4672`
- Average precision: `0.8613`
- Accuracy: `78.02%`
- ROC AUC: `0.8670`

## 5. Matchup model accuracy by round

Source:

- `artifacts/matchup_model/best_model_round_metrics.csv`

Accuracy on games that actually occurred:

- `Round of 64`: `78.70%` on `446` games
- `Round of 32`: `77.68%` on `224` games
- `Sweet 16`: `74.11%` on `112` games
- `Elite Eight`: `78.57%` on `56` games
- `Final Four`: `75.00%` on `28` games
- `Championship`: `100.00%` on `12` games

Important caveat:

- The championship-game number is based on only `12` games, so it is not stable enough to treat as a solved problem.

## 6. Seed impact in the matchup model

Source:

- `artifacts/matchup_model/seed_summary.csv`

Takeaway:

- Seed helps much more clearly in the matchup model than in the team-level models.
- The best matchup model includes seed plus meta-features from the team models.

Examples:

- `meta_all_features` Brier: `0.1480`
- `meta_no_seed` Brier: `0.1606`
- `all_features` Brier: `0.1491`
- `no_seed` Brier: `0.1636`

## 7. Historical bracket backtest

Source:

- `artifacts/bracket_backtest/summary.json`

Settings and result summary:

- Included seasons: `14`
- Simulations per season: `3000`
- Calibration enabled: `true`
- Best temperature: `1.0`

Best-grid summary:

- Mean actual champion probability: `0.7203`
- Mean top-pick probability: `0.7311`
- Mean top-4 title probability mass: `0.9295`
- Top-pick hit rate: `0.9286`
- Mean multiclass Brier: `0.1674`
- Mean champion log loss: `0.4094`

Interpretation:

- The model is historically very sharp.
- That sharpness produced a very high top-pick hit rate in the historical backtest.
- It also means championship odds should be treated carefully because they may be overconfident in practice.

## 8. Qualitative modeling findings

Based on the rebuilt dataset and saved benchmark outputs:

- `AdjEM` is the single strongest overall signal.
- `AdjO` and `AdjD` are the next most important structural inputs.
- Seed is a useful contextual signal, especially at the matchup level.
- Useful secondary factors include:
  - scoring
  - offensive rebounding
  - turnover control
  - `3PT%`
- Weaker signals in this setup include:
  - tempo
  - luck
  - raw `3PT rate`

## 9. 2026 forecast snapshot

Sources:

- `artifacts/tournament_forecast/2026/team_rankings.csv`
- `artifacts/tournament_forecast/2026/team_odds.csv`

### Top bracket-independent teams by team-strength score

1. Arizona
2. Duke
3. Michigan
4. Florida
5. Houston
6. Illinois
7. Iowa State
8. Purdue
9. Virginia
10. Connecticut

### Highest 2026 championship odds

1. Michigan: `51.31%`
2. Duke: `38.55%`
3. Houston: `4.62%`
4. Arizona: `2.92%`
5. Florida: `1.03%`
6. Purdue: `0.79%`
7. Arkansas: `0.30%`
8. Texas Tech: `0.14%`
9. Iowa State: `0.13%`
10. Illinois: `0.09%`

Interpretation:

- The bracket-independent strength ranking and the bracket simulation do not produce the exact same ordering.
- In the current `2026` run:
  - Arizona rates best as a pure team-strength profile
  - Michigan and Duke are the strongest championship favorites once bracket path is incorporated

## 10. Relevant artifact files

Team-level benchmarks:

- `artifacts/model_benchmarks/best_models.csv`
- `artifacts/model_benchmarks/seed_summary.csv`
- `artifacts/model_benchmarks/single_feature_scan.csv`

Matchup-level benchmarks:

- `artifacts/matchup_model/best_models.csv`
- `artifacts/matchup_model/best_model_round_metrics.csv`
- `artifacts/matchup_model/seed_summary.csv`

Backtest:

- `artifacts/bracket_backtest/summary.json`
- `artifacts/bracket_backtest/season_summary_best_temperature.csv`
- `artifacts/bracket_backtest/title_probabilities_best_temperature.csv`

2026 forecast:

- `artifacts/tournament_forecast/2026/team_rankings.csv`
- `artifacts/tournament_forecast/2026/contender_scorecard.csv`
- `artifacts/tournament_forecast/2026/team_odds.csv`
- `artifacts/tournament_forecast/2026/upset_flags.csv`
