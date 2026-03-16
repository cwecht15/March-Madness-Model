# March Madness Pipeline

This folder now has a script-first rebuild path for the tournament dataset instead of relying on notebook-only prep.

## What was added

- `scripts/build_tournament_dataset.py`: builds one season in the same schema as `March_Madness_Train_Model.csv`
- `scripts/build_forecast_season.py`: builds a pre-tournament season file for a live bracket year when final results do not exist yet
- `scripts/scrape_public_sources.py`: scrapes KenPom, TeamRankings, and ESPN bracket pages into local raw files
- `scripts/build_matchup_training_data.py`: converts tournament games into bracket-aware matchup training rows
- `scripts/train_clean_models.py`: benchmarks team-level models for points, finish, and round-by-round advancement
- `scripts/train_matchup_model.py`: benchmarks game-level win-probability models with season-based validation
- `scripts/backtest_bracket_calibration.py`: tunes bracket-level probability temperature against historical tournaments
- `scripts/simulate_bracket.py`: runs Monte Carlo bracket simulations from the saved matchup model
- `scripts/run_tournament_forecast.py`: one-command future-season forecast with team rankings, contender tiers, upset flags, and bracket odds
- `app.py`: interactive Streamlit bracket app that lets a user make picks and see the remaining odds update live
- `config/teamrankings_manifest.template.csv`: starter manifest for the TeamRankings source files
- `data/team_aliases.csv`: starter team-name alias map
- `data/templates/bracket_field_template.csv`: template for a 64-team field file when simulating a future bracket
- `data/templates/tournament_results_template.csv`: template for team-level tournament outcomes

## Expected source split

- KenPom: `AdjEM`, `AdjO`, `AdjD`, `AdjT`, `Luck`
- TeamRankings: season stat columns from `Seas_PPG` through `Seas_Poss`
- Tournament results file: `team`, `seed`, and either `wins`, `Finish`, or `finish_label`

## Results file logic

The script derives the tournament outcome columns from team wins in the main bracket:

- `wins = 0` -> `Finish = 1`, `Pts = 0`
- `wins = 1` -> `Finish = 2`, `Pts = 1`
- `wins = 2` -> `Finish = 3`, `Pts = 3`
- `wins = 3` -> `Finish = 4`, `Pts = 7`
- `wins = 4` -> `Finish = 5`, `Pts = 15`
- `wins = 5` -> `Finish = 6`, `Pts = 31`
- `wins = 6` -> `Finish = 7`, `Pts = 63`

The round flags are then derived automatically.

## TeamRankings manifest

Copy `config/teamrankings_manifest.template.csv` to a season-specific file and update the file paths to the exports you saved locally.

The manifest columns are:

- `output_column`: final dataset column name
- `source_file`: local path or URL for the TeamRankings export
- `scale`: `raw` or `pct`
- `notes`: free-form reminder column for you

`Seas_Succ_3PT` is left intentionally unresolved in the template because the current CSV does not make its source stat obvious. Fill that one with the TeamRankings page you intend to use.

## Basic workflow

1. Run the public scraper or save the source pages manually.
2. Add alias rows to `data/team_aliases.csv` whenever the match report flags a team-name mismatch.
3. Run the builder.

## Public scraper example

```powershell
python scripts/scrape_public_sources.py --season 2025
```

That command writes:

- `data/raw/2025/kenpom_2025.html`
- `data/raw/2025/kenpom_ratings.csv`
- `data/raw/2025/teamrankings/*.html`
- `data/raw/2025/teamrankings/*.csv`
- `data/raw/2025/espn_bracket_2025.html`
- `data/raw/2025/tournament_results.csv`
- `data/raw/2025/tournament_games.csv`
- `config/teamrankings_2025.csv`

If KenPom blocks direct requests on your network, you can still scrape the public sources with:

```powershell
python scripts/scrape_public_sources.py --season 2025 --skip-kenpom
```

## Example command

```powershell
python scripts/build_tournament_dataset.py `
  --season 2025 `
  --kenpom data/raw/2025/kenpom_2025.html `
  --teamrankings-manifest config/teamrankings_2025.csv `
  --results data/raw/2025/tournament_results.csv `
  --historical-csv March_Madness_Train_Model.csv `
  --aliases data/team_aliases.csv `
  --output data/processed/march_madness_2025.csv `
  --base-dataset March_Madness_Train_Model.csv `
  --drop-years 2024 `
  --merged-output data/processed/March_Madness_Train_Model_rebuilt.csv
```

## Forecast-season build

For a live tournament year such as `2026`, build the season file from the field, KenPom, and TeamRankings even before games are played:

```powershell
python scripts/build_forecast_season.py `
  --season 2026 `
  --field data/raw/2026/tournament_field.csv `
  --kenpom data/raw/2026/kenpom/kenpom_data.csv `
  --teamrankings-manifest config/public/teamrankings_2026.csv `
  --output data/processed/march_madness_2026.csv
```

Notes:

- Drop the KenPom file at `data/raw/2026/kenpom/kenpom_data.csv`
- `data/raw/2026/tournament_field.csv` can include all `68` teams, including First Four pairs
- the forecast season file is for rankings, contender screens, and pre-bracket analysis
- the bracket simulator still expects the resolved `64`-team main bracket field once the play-in winners are known

## Validation built into the script

The builder checks:

- round totals sum to `32, 16, 8, 4, 2, 1`
- `AdjEM` roughly matches `AdjO - AdjD`
- missing values after the season merge

Use `--strict` if you want those warnings to fail the run.

## Matchup model workflow

Build the historical game-level dataset:

```powershell
python scripts/build_matchup_training_data.py
```

Train and benchmark the matchup model:

```powershell
python scripts/train_matchup_model.py
```

That writes:

- `data/processed/matchup_training_data.csv`
- `artifacts/matchup_model/benchmark_results.csv`
- `artifacts/matchup_model/best_models.csv`
- `artifacts/matchup_model/best_model_round_metrics.csv`
- `artifacts/matchup_model/saved_models/*.joblib`

The matchup trainer also tests post-hoc probability calibrators and stores the selected calibrator in the saved matchup model payload.

## Bracket simulation

For a historical validation run, you can infer the field and Final Four pairings from the scraped game file:

```powershell
python scripts/simulate_bracket.py `
  --season-data data/processed/march_madness_2025.csv `
  --historical-games data/raw/2025/tournament_games.csv `
  --n-sims 10000 `
  --output-dir artifacts/bracket_simulation/2025
```

For a future bracket, supply a 64-team file with `team`, `seed`, and `region`, plus the semifinal region pairings:

```powershell
python scripts/simulate_bracket.py `
  --season-data data/processed/march_madness_2026.csv `
  --field data/templates/bracket_field_template.csv `
  --semifinal-pairs East-West,South-Midwest `
  --n-sims 10000 `
  --output-dir artifacts/bracket_simulation/2026
```

The simulator writes:

- `team_odds.csv`: per-team round-win and title probabilities
- `most_likely_bracket.csv`: deterministic picks from the saved matchup model
- `field_match_report.csv`: how input team names were reconciled to the season dataset

## One-command forecast workflow

For a future season, use the forecast wrapper to generate bracket-independent team rankings first, then matchup and simulation outputs if you also supply a bracket field:

```powershell
python scripts/run_tournament_forecast.py `
  --season-data data/processed/march_madness_2026.csv `
  --field data/templates/bracket_field_template.csv `
  --semifinal-pairs East-West,South-Midwest `
  --n-sims 10000 `
  --output-dir artifacts/tournament_forecast/2026
```

That writes:

- `team_rankings.csv`: bracket-independent team rankings driven by `AdjEM`, `AdjO`, `AdjD`, and the saved team-level models
- `contender_scorecard.csv`: title / Final Four / second-weekend screening plus secondary stat support from scoring, offensive rebounding, turnover control, and 3PT%
- `team_odds.csv`: simulation odds to reach each round and win the title
- `most_likely_bracket.csv`: deterministic bracket path from the matchup model
- `matchup_probabilities.csv`: projected matchup probabilities for the most-likely bracket path
- `upset_flags.csv`: projected games where the seeded underdog has at least the chosen upset probability threshold
- `field_match_report.csv`: team-name reconciliation details
- `forecast_summary.json`: run settings, ranking weights, and favorite summary

## Interactive bracket app

Launch the local app with:

```powershell
streamlit run app.py
```

The app uses the current season dataset, field file, and saved matchup model to:

- let a user make picks round by round
- unlock later games as earlier picks are made
- recompute remaining team odds conditioned on those picks
- show which teams gained or lost title equity because of the user's bracket
- autofill the bracket by seed or team strength
- export the current picks as both `CSV` and a printable bracket-style `PDF`

## Public deployment

This repo is now set up for a simple public deployment.

Files added for deployment:

- `requirements.txt`: Python dependencies for the app and model stack
- `.streamlit/config.toml`: Streamlit server and theme settings
- `Procfile`: start command for platforms like Render
- `runtime.txt`: Python runtime version

### Recommended option: Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud and create a new app from that repo.
3. Set the main file path to `app.py`.
4. Deploy.

### Alternative option: Render

1. Push this project to GitHub.
2. Create a new `Web Service` on Render from the repo.
3. Use the included `Procfile`, or set the start command to:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

4. Deploy.

Notes:

- The app expects the current local data files in `data/` and `artifacts/`, so those need to be included in the deployed repo.
- If you later want a lighter public app, we can strip out the training scripts and keep only the forecast artifacts plus the Streamlit front end.
- autofill the full bracket by `seed` or by `team strength`
- export the current picks as a CSV

## Historical bracket audit

To test whether the simulator is too sharp or too soft at the title level, run the historical backtest:

```powershell
python scripts/backtest_bracket_calibration.py `
  --n-sims 3000 `
  --update-model-payload
```

That writes:

- `artifacts/bracket_backtest/temperature_grid_summary.csv`
- `artifacts/bracket_backtest/season_summary_best_temperature.csv`
- `artifacts/bracket_backtest/title_probabilities_best_temperature.csv`

If `--update-model-payload` is set, the selected bracket temperature is written back into the saved matchup model and future simulations will use it automatically.
