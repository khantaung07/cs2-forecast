# Backtesting Report

This project uses chronological backtesting to evaluate CS2 match and map prediction models. Each model only uses results available before the match being predicted, then updates its internal state after the match result is observed.

The main evaluation metrics are:

* **Log Loss**: primary metric; rewards calibrated probabilities and heavily penalises confident wrong predictions.
* **Brier Score**: secondary calibration metric.
* **Accuracy**: useful sanity check, but less important than log loss/Brier because betting-style or forecasting models need calibrated probabilities.

## Dataset

Current parsed dataset:

* **Matches:** 1,025 completed matches
* **Maps:** 2,261 completed map results

For filtered evaluations, `min_team_matches` only controls when predictions are scored. The models still update on earlier matches. For example, `min_team_matches=10` means a prediction is only included once both teams have at least 10 prior parsed matches.

## 1. Baseline Elo Backtests

Initial chronological Elo tests showed that overall team Elo is a strong baseline.

```
Model                     N      Accuracy   Log Loss   Brier
Constant 50/50 Match      1025   0.546      0.693      0.250
Overall Elo               1025   0.618      0.643      0.226

Constant 50/50 Map        2261   0.537      0.693      0.250
Plain Map Elo             2261   0.567      0.676      0.242
```

Overall match Elo substantially improved over the 50/50 baseline. Plain map-specific Elo improved over the map baseline, but was much weaker than match-level Elo.

## 2. Enhanced Match Elo

The enhanced match model adds:

```
overall Elo
+ dynamic K-factor
+ opponent-adjusted recent form
+ shrinked head-to-head adjustment
```

Chosen configuration:

```
form_decay = 0.95
form_weight = 100
h2h_shrinkage = 3
h2h_weight = 50
```

### Results: `min_team_matches=5`

```
Model                                      N     Accuracy   Log Loss   Brier
Constant 50/50                            783   0.548      0.693      0.250
Overall Elo                               783   0.644      0.628      0.220
Dynamic Elo                               783   0.637      0.629      0.220
Enhanced Elo                              783   0.635      0.625      0.218
Enhanced Dynamic Elo                      783   0.639      0.625      0.218
```

### Results: `min_team_matches=10`

```
Model                                      N     Accuracy   Log Loss   Brier
Constant 50/50                            632   0.554      0.693      0.250
Overall Elo                               632   0.661      0.623      0.217
Dynamic Elo                               632   0.655      0.623      0.217
Enhanced Elo                              632   0.655      0.620      0.216
Enhanced Dynamic Elo                      632   0.657      0.619      0.215
```

The enhanced dynamic match model is the best standalone match-level model. Recent form and H2H provide a small but meaningful improvement in log loss/Brier over overall Elo.

## 3. Enhanced Map Elo Experiments

The map-level experiments tested whether map-specific information improves individual map prediction.

Compared models:

```
Overall Map Elo       = team strength across all maps
Plain Map Elo         = separate rating per team per map
Overall + Map Elo     = overall map strength + map-specific adjustment
Enhanced Map Elo      = overall + map-specific Elo + map-specific recent form
```

### Results: `min_team_maps=5`

```
Model                                      N      Accuracy   Log Loss   Brier
Constant 50/50 Map                         1996   0.535      0.693      0.250
Overall Map Elo                            1996   0.609      0.661      0.234
Plain Map Elo                              1996   0.567      0.675      0.241
Overall + Map Elo                          1996   0.610      0.662      0.234
Enhanced Map Elo                           1996   0.610      0.663      0.235
```

### Results: `min_team_maps=10`

```
Model                                      N      Accuracy   Log Loss   Brier
Constant 50/50 Map                         1776   0.535      0.693      0.250
Overall Map Elo                            1776   0.620      0.656      0.231
Plain Map Elo                              1776   0.569      0.673      0.240
Overall + Map Elo                          1776   0.620      0.657      0.232
Enhanced Map Elo                           1776   0.620      0.658      0.232
```

The best individual map model is **Overall Map Elo**. Naive team-on-map Elo and map recent form did not improve results. This suggests that map-specific win/loss history is too sparse/noisy without richer veto context such as pick/ban order, decider maps, and map-pool tendencies.

## 4. Series-Level Map Backtest

The series-level backtest converts map win probabilities into Bo1/Bo3/Bo5 match probabilities.

This answers:

```
Given the maps in the series, can map-level probabilities predict the match winner?
```

This does **not** simulate veto. It uses observed/known maps from the parsed match data. For unplayed decider maps, the model falls back to an overall map-level probability rather than pretending the decider was known.

### Results: `min_team_matches=5`

```
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Series              783   0.548      0.693      0.250
Series from Overall Map Elo        783   0.637      0.629      0.219
Series from Plain Map Elo          783   0.644      0.630      0.220
```

### Results: `min_team_matches=10`

```
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Series              632   0.554      0.693      0.250
Series from Overall Map Elo        632   0.650      0.627      0.217
Series from Plain Map Elo          632   0.646      0.628      0.219
```

The pure map-series model is competitive, but does not beat the enhanced dynamic match model. Overall map Elo remains slightly better than plain map Elo on the main benchmark.

A diagnostic `--require-full-map-list` run showed stronger plain map Elo performance on series that went the distance, but this subset is outcome-dependent and should not be treated as the main benchmark.

## 5. Blended Match + Map-Series Model

The best model so far blends:

```
Enhanced Dynamic Match Elo
+
Series probability from Overall Map Elo
```

The final probability is:

```
final_prob =
    match_weight * enhanced_dynamic_match_probability
  + (1 - match_weight) * map_series_probability
```

This tests whether map-series probabilities provide complementary signal on top of the stronger match-level model.

### Results: `min_team_matches=10`

```
Model                                  N     Accuracy   Log Loss   Brier
Blend Match+Map match_w=1.00           632   0.657      0.619      0.215
Blend Match+Map match_w=0.90           632   0.658      0.618      0.215
Blend Match+Map match_w=0.75           632   0.649      0.616      0.214
Blend Match+Map match_w=0.65           632   0.655      0.616      0.214
Blend Match+Map match_w=0.60           632   0.653      0.616      0.214
Blend Match+Map match_w=0.55           632   0.652      0.616      0.214
Blend Match+Map match_w=0.50           632   0.657      0.616      0.214
Blend Match+Map match_w=0.45           632   0.653      0.617      0.214
Blend Match+Map match_w=0.40           632   0.653      0.617      0.214
Blend Match+Map match_w=0.25           632   0.657      0.619      0.215
Blend Match+Map match_w=0.00           632   0.650      0.627      0.217
```

The pure enhanced dynamic match model achieved `0.619` log loss. The pure map-series model achieved `0.627` log loss. Blended models around `match_weight=0.50–0.75` achieved the best result of approximately:

```
Log Loss: 0.616
Brier:    0.214
```

This is the best backtested result so far.

## 6. Machine Learning Comparison

Logistic regression and histogram gradient boosting were evaluated using a fixed chronological holdout.

Pre-match features included:

```text
enhanced dynamic match probability
overall-map series probability
dynamic Elo rating difference
recent-form difference
shrinked H2H score
overall map-Elo difference
team match-history counts
best-of format
```

All features were calculated before the corresponding match result was processed. The earliest 70% of eligible matches were used for training, and the final 30% were retained for testing. The ML models remained fixed during the test period, while the underlying Elo, form, H2H, and map-rating states continued to update chronologically as results became available.

### Holdout results: `min_team_matches=5`

```text
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Holdout             235   0.562      0.693      0.250
Enhanced Dynamic Match Holdout     235   0.668      0.592      0.203
Blended Match+Map Holdout          235   0.681      0.590      0.203
Logistic Regression                235   0.647      0.602      0.210
Histogram Gradient Boosting        235   0.621      0.630      0.221
```

Training rows: 548. Test rows: 235. The test period began on February 15, 2026.

### Holdout results: `min_team_matches=10`

```text
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Holdout             190   0.537      0.693      0.250
Enhanced Dynamic Match Holdout     190   0.647      0.607      0.211
Blended Match+Map Holdout          190   0.658      0.606      0.211
Logistic Regression                190   0.632      0.615      0.215
Histogram Gradient Boosting        190   0.626      0.638      0.224
```

Training rows: 442. Test rows: 190. The test period began on February 21, 2026.

Neither ML model outperformed the manually constructed blended model. Logistic regression remained competitive, but did not combine the existing signals more effectively than the validated match-and-map blend. Histogram gradient boosting performed substantially worse, likely because the available training dataset was too small for stable nonlinear modelling.

The holdout scores above should not be compared directly with the earlier `0.616` blended-backtest result because they use different evaluation periods. Within both evaluation regimes, however, the blended match-and-map model achieved the best probability metrics.

## Current Best Model

When no series context is supplied:

```text
Enhanced Dynamic Match Elo
```

When a best-of value or series context is supplied:

```text
Blended Match + Map-Series Model
```

The practical default is:

```text
match_weight = 0.50
```

The final probability is therefore:

```text
final_probability =
    0.50 * enhanced_dynamic_match_probability
  + 0.50 * overall_map_series_probability
```

This combines match-level Elo, recent form, H2H, and dynamic K-factor information with series structure derived from overall map-level strength.

The current series component does not simulate vetoes and does not assign different probabilities to individual named maps. Overall map Elo was selected because map-specific Elo and map-specific recent form underperformed during backtesting.

## Main Conclusions

1. Overall Elo provides a strong baseline for CS2 match prediction.
2. Dynamic K-factor, opponent-adjusted recent form, and shrinked H2H improve match-level probability calibration.
3. Overall map Elo predicts individual maps more effectively than sparse team-on-map Elo.
4. A standalone map-series model is competitive, but weaker than the enhanced dynamic match model.
5. Blending match-level and map-series probabilities produces the strongest results.
6. Logistic regression and histogram gradient boosting did not improve on the hand-built blended model.
7. Richer map modelling would likely require reliable veto, pick/ban, and map-pool data.

## Final Status

The completed forecasting pipeline supports:

```text
Liquipedia MediaWiki API ingestion
local raw-page caching
wikitext parsing
normalized SQLite storage
team alias normalization
chronological model backtesting
match-level and map-level Elo models
series probability calculation
blended match and map-series prediction
logistic regression and gradient-boosting comparisons
local command-line match prediction
```

Example predictions:

```bash
cs2forecast predict-match vitality faze
cs2forecast predict-match vitality faze --best-of 3
cs2forecast predict-match vitality faze --maps nuke,mirage,ancient
```

The predictor replays the completed historical data in SQLite to construct the latest model state, then produces one probability for the requested matchup. It does not fetch live match data.
