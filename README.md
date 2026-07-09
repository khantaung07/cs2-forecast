# cs2-predictor

A local CS2 match forecasting pipeline that ingests historical tournament data from Liquipedia, parses completed matches/maps into SQLite, backtests forecasting models chronologically, and predicts match outcomes from the local database.

The project is split into three layers:

1. **Ingestion layer**
   Fetches raw Liquipedia wikitext through the MediaWiki API and stores it locally.

2. **Parsing/storage layer**
   Converts raw wikitext into normalized SQLite tables for `events`, `teams`, `matches`, and `map_results`.

3. **Prediction layer**
   Replays historical results from SQLite to build model state, then produces match probabilities without fetching live data.

## Current model

The current best model is a blended probability model:

```text
final_probability =
    match_weight * enhanced_dynamic_match_probability
  + (1 - match_weight) * map_series_probability
```

The match-level component uses:

```text
Dynamic Elo
+ opponent-adjusted recent form
+ shrinked head-to-head adjustment
```

When a best-of value or map slots are supplied, the predictor also calculates a series probability using overall map-level Elo and blends it with the match-level probability.

The supplied map names currently provide series context only. They do not produce different map-specific probabilities because team-on-map Elo underperformed overall map Elo during backtesting.

If no best-of value or maps are supplied, the predictor uses the enhanced dynamic match model only.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Initialise the local SQLite database:

```bash
cs2forecast init-db
```

## Data ingestion

Tournament pages are listed in:

```text
seeds/tournaments.txt
```

Fetch all seeded tournament pages:

```bash
cs2forecast scrape-events
```

Re-fetch pages even if they are already cached:

```bash
cs2forecast scrape-events --refresh
```

Fetch individual Liquipedia pages manually:

```bash
cs2forecast scrape "Intel Extreme Masters/2025/Katowice"
```

List cached raw pages:

```bash
cs2forecast list-pages
```

Inspect templates on a cached page:

```bash
cs2forecast inspect-templates "Intel Extreme Masters/2025/Katowice"
```

Show examples of a specific template:

```bash
cs2forecast show-template "Intel Extreme Masters/2025/Katowice" Match
```

## Parsing

Parse cached raw pages into normalized SQLite tables:

```bash
cs2forecast parse
```

Inspect parsed matches:

```bash
cs2forecast list-matches
```

Inspect parsed map results:

```bash
cs2forecast list-maps
```

Inspect parsed teams:

```bash
cs2forecast list-teams
```

Inspect teams from a specific source page:

```bash
cs2forecast list-teams --source-page "Intel Extreme Masters/2025/Katowice"
```

## Backtesting

Run baseline Elo backtests:

```bash
cs2forecast backtest-elo
```

Run enhanced match-level backtests:

```bash
cs2forecast backtest-enhanced
```

Run enhanced match-level backtests with a stricter mature-team filter:

```bash
cs2forecast backtest-enhanced --min-team-matches 10
```

Run enhanced map-level experiments:

```bash
cs2forecast backtest-enhanced-map
```

Run series-level map aggregation backtests:

```bash
cs2forecast backtest-series
```

Run blended match and map-series backtests:

```bash
cs2forecast backtest-blended-series
```

Run blended backtests on mature teams:

```bash
cs2forecast backtest-blended-series --min-team-matches 10
```

Run the chronological machine-learning holdout comparison:

```bash
cs2forecast backtest-ml
```

Run the ML comparison using the mature-team filter:

```bash
cs2forecast backtest-ml --min-team-matches 10
```

This compares logistic regression and histogram gradient boosting against the enhanced dynamic and blended forecasting models.

See [`README_BACKTESTING.md`](README_BACKTESTING.md) for detailed results, evaluation methodology, and model-selection decisions.

## Testing

Run static checks:

```bash
ruff check src tests
```

Run the test suite:

```bash
python -m pytest
```

The tests cover core probability calculations, binary evaluation metrics, team alias normalization, chronological dataset splitting, ML feature construction, and predictor helper logic.

## Prediction

Predict a match using the current local database:

```bash
cs2forecast predict-match spirit vitality
```

Predict a known best-of-three:

```bash
cs2forecast predict-match spirit vitality --best-of 3
```

Predict with supplied map slots:

```bash
cs2forecast predict-match spirit vitality --maps dust2,mirage,inferno
```

Alternatively, pass maps as repeatable options:

```bash
cs2forecast predict-match spirit vitality -m dust2 -m mirage -m inferno
```

Example output:

```text
spirit vs vit

Model                                   spirit   vit
Enhanced Dynamic Match Elo                29.8%  70.2%
Series from Overall Map Elo               42.8%  57.2%
Blended Final Probability (match_w=0.5)   36.3%  63.7%
```

The predictor works by replaying all completed historical matches/maps from SQLite to rebuild the latest model state. It then predicts the requested matchup from that state. It does not fetch live data.

## Team alias normalization

Liquipedia pages may refer to the same team using different identifiers across pages or templates. For example, Team Vitality may appear as both `vit` and `vitality`.

Since the forecasting model treats `team_id` as the stable team identity, aliases must be normalized consistently before writing parsed matches to the database.

Team aliases are defined in:

```text
src/cs2forecast/parsing/normalization.py
```

When adding new tournament pages, inspect parsed teams with:

```bash
cs2forecast list-teams --source-page "<Liquipedia page title>"
```

If the same real team appears under multiple IDs, add an entry to `TEAM_ID_ALIASES` and reparse the data.

Example:

```python
TEAM_ID_ALIASES = {
    "vitality": "vit",
    "team_vitality": "vit",
}
```

After updating aliases, rebuild the parsed tables so Elo, recent form, H2H, and map-level features use the corrected canonical team IDs.

## Valve Major pages

For Valve Majors, seed the individual stage pages rather than the root overview page.

Good:

```text
BLAST/Major/2025/Austin/Stage_1
BLAST/Major/2025/Austin/Stage_2
BLAST/Major/2025/Austin/Stage_3
BLAST/Major/2025/Austin/Playoffs
```

Avoid:

```text
BLAST/Major/2025/Austin
```

The root overview page usually contains only summary/showmatch templates, while the real match data is stored on the stage subpages.

## Notes and limitations

* The predictor operates entirely from the local SQLite database.
* It does not scrape HLTV or fetch live match information.
* The database must be refreshed and reparsed to incorporate new tournament results.
* The predictor does not simulate map veto or pick-ban decisions.
* Supplied map names currently provide series context only. The selected series model uses overall map-level Elo because map-specific Elo and map-specific recent form underperformed during backtesting.
* The ML models were evaluated experimentally but did not outperform the hand-built blended model.
* Potential future extensions include additional Liquipedia tournament coverage, automated data-quality checks, total-round prediction, and over/under modelling.