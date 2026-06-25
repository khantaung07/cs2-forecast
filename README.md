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