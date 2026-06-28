# cs2-predictor

< insert mini description >

1. Ingestion layer
   Fetches raw Liquipedia wikitext through MediaWiki API and stores it.

2. Parsing/storage layer
   Converts raw wikitext into local SQLite tables:
   matches, map_results, teams, events.

3. Prediction layer
   Reads only from SQLite and produces predictions.

# How to use

1. Call data from Liquipedia MediaWiki API

2. Run predictor for a match


## Team alias normalization

Liquipedia pages may refer to the same team using different identifiers across pages or templates. For example, Team Vitality may appear as both `vit` and `vitality`. Since the forecasting model treats `team_id` as the stable team identity, these aliases must be normalized consistently before writing parsed matches to the database.

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


### Major pages

For Valve Majors, seed the individual stage pages rather than the root overview page.

Good:

```text
BLAST/Major/2025/Austin/Stage_1
BLAST/Major/2025/Austin/Stage_2
BLAST/Major/2025/Austin/Stage_3
BLAST/Major/2025/Austin/Playoffs

Avoid:

BLAST/Major/2025/Austin

The root overview page contains only summary/showmatch templates, while the real match data is usually stored on the stage subpages.