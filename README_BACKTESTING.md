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

```text id="3dkglt"
Model                     N      Accuracy   Log Loss   Brier
Constant 50/50 Match      1025   0.546      0.693      0.250
Overall Elo               1025   0.618      0.643      0.226

Constant 50/50 Map        2261   0.537      0.693      0.250
Plain Map Elo             2261   0.567      0.676      0.242
```

Overall match Elo substantially improved over the 50/50 baseline. Plain map-specific Elo improved over the map baseline, but was much weaker than match-level Elo.

## 2. Enhanced Match Elo

The enhanced match model adds:

```text id="02me2j"
overall Elo
+ dynamic K-factor
+ opponent-adjusted recent form
+ shrinked head-to-head adjustment
```

Chosen configuration:

```text id="cwsbku"
form_decay = 0.95
form_weight = 100
h2h_shrinkage = 3
h2h_weight = 50
```

### Results: `min_team_matches=5`

```text id="4kagfo"
Model                                      N     Accuracy   Log Loss   Brier
Constant 50/50                            783   0.548      0.693      0.250
Overall Elo                               783   0.644      0.628      0.220
Dynamic Elo                               783   0.637      0.629      0.220
Enhanced Elo                              783   0.635      0.625      0.218
Enhanced Dynamic Elo                      783   0.639      0.625      0.218
```

### Results: `min_team_matches=10`

```text id="qfhgys"
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

```text id="0qyixa"
Overall Map Elo       = team strength across all maps
Plain Map Elo         = separate rating per team per map
Overall + Map Elo     = overall map strength + map-specific adjustment
Enhanced Map Elo      = overall + map-specific Elo + map-specific recent form
```

### Results: `min_team_maps=5`

```text id="qkv4iq"
Model                                      N      Accuracy   Log Loss   Brier
Constant 50/50 Map                         1996   0.535      0.693      0.250
Overall Map Elo                            1996   0.609      0.661      0.234
Plain Map Elo                              1996   0.567      0.675      0.241
Overall + Map Elo                          1996   0.610      0.662      0.234
Enhanced Map Elo                           1996   0.610      0.663      0.235
```

### Results: `min_team_maps=10`

```text id="ydh9m0"
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

```text id="e14cx2"
Given the maps in the series, can map-level probabilities predict the match winner?
```

This does **not** simulate veto. It uses observed/known maps from the parsed match data. For unplayed decider maps, the model falls back to an overall map-level probability rather than pretending the decider was known.

### Results: `min_team_matches=5`

```text id="vy5u33"
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Series              783   0.548      0.693      0.250
Series from Overall Map Elo        783   0.637      0.629      0.219
Series from Plain Map Elo          783   0.644      0.630      0.220
```

### Results: `min_team_matches=10`

```text id="0ks4p8"
Model                              N     Accuracy   Log Loss   Brier
Constant 50/50 Series              632   0.554      0.693      0.250
Series from Overall Map Elo        632   0.650      0.627      0.217
Series from Plain Map Elo          632   0.646      0.628      0.219
```

The pure map-series model is competitive, but does not beat the enhanced dynamic match model. Overall map Elo remains slightly better than plain map Elo on the main benchmark.

A diagnostic `--require-full-map-list` run showed stronger plain map Elo performance on series that went the distance, but this subset is outcome-dependent and should not be treated as the main benchmark.

## 5. Blended Match + Map-Series Model

The best model so far blends:

```text id="z8dqmm"
Enhanced Dynamic Match Elo
+
Series probability from Overall Map Elo
```

The final probability is:

```text id="2smdb3"
final_prob =
    match_weight * enhanced_dynamic_match_probability
  + (1 - match_weight) * map_series_probability
```

This tests whether map-series probabilities provide complementary signal on top of the stronger match-level model.

### Results: `min_team_matches=10`

```text id="os5nft"
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

```text id="es4xap"
Log Loss: 0.616
Brier:    0.214
```

This is the best backtested result so far.

## Current Best Model

When maps are unknown:

```text id="hltvi0"
Enhanced Dynamic Match Elo
```

When maps are supplied:

```text id="ovbynq"
Blended Match + Map-Series Model
```

Practical default:

```text id="w3qiy7"
match_weight = 0.50
```

This uses both match-level team strength/form/H2H and map-derived series structure.

## Main Conclusions

1. Overall Elo is a strong baseline for CS2 match prediction.
2. Dynamic K, recent form, and shrinked H2H improve match-level calibration.
3. Individual map-specific Elo is weaker than overall map Elo, likely due to sparse team-map history.
4. Map-series probabilities are competitive but weaker than the enhanced match model on their own.
5. Blending match-level and map-series probabilities gives the best result so far.
6. Future map improvements likely require veto/pick-ban context rather than simply increasing map Elo or map form weight.

## Next Steps

```text id="me5k82"
Build the actual predictor command.
```

Expected CLI shape:

```bash id="h0ct3z"
cs2forecast predict-match vitality mouz
cs2forecast predict-match vitality mouz --maps nuke mirage ancient
```

When maps are not provided, the predictor should use the enhanced dynamic match model.

When maps are provided, the predictor should also calculate the map-series probability and return the blended probability.
