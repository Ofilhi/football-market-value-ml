# Football Players Market Value Predictor

Can a few simple statistics explain how much a football player is worth?

This project trains a **linear regression** model that predicts a player's
market value from their performance and profile, then compares the prediction
with the player's actual market value to spot who looks over- or under-valued.

> **Status:** work in progress. The project skeleton is in place; data
> collection, modelling and results are being added step by step (see the
> [roadmap](#roadmap)).

## Approach

| | |
|---|---|
| **Features** | goals, assists, minutes played, age, position |
| **Target** | market value (EUR) |
| **Model** | linear regression (scikit-learn) |
| **Scope** | top 5 European leagues, 2025-26 season |
| **Stack** | Python, pandas, scikit-learn, matplotlib |

An LLM is used on top of the model to turn its numbers into short, readable
scouting reports. It only comments on the model's output; it never makes the
prediction itself.

## Data sources

- **[football-data.org](https://www.football-data.org/)** (free plan): goals,
  assists, date of birth and position. The API only lists players who scored
  at least one goal in the season, so the dataset is limited to those players.
- **[Transfermarkt public dataset](https://github.com/dcaribou/transfermarkt-datasets)**:
  market value and minutes played, which the free plan of football-data.org
  does not provide. The market value is each player's latest valuation at the
  end of the season.

The two sources share no identifier, so players are matched on their name and
date of birth, among the players who appeared in the same league. About 99% of
the players are matched; the others are listed in a report instead of being
dropped silently.

## Project structure

```
├── src/football_value/   # Python package (config, data, features, model, ...)
├── tests/                # Unit tests
├── data/
│   ├── raw/              # Cached API responses and downloads (not versioned)
│   └── processed/        # Cleaned, model-ready datasets (not versioned)
├── reports/figures/      # Charts produced by the pipeline
├── .env.example          # Template for the required environment variables
├── requirements.txt      # Pinned runtime dependencies
└── pyproject.toml        # Package metadata and tool configuration
```

## Getting started

Requires Python 3.11 or newer.

```bash
git clone https://github.com/Ofilhi/football-market-value-ml.git
cd football-market-value-ml

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
pip install -e .
```

Then create your own `.env` file:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `FOOTBALL_DATA_API_KEY` | football-data.org token ([free registration](https://www.football-data.org/client/register)) |
| `RESPAN_API_KEY` | key for the LLM gateway |
| `LLM_BASE_URL` | OpenAI-compatible endpoint of the LLM gateway |
| `LLM_MODEL` | model used for the scouting reports |

The `.env` file is ignored by git, so your keys never leave your machine.

## Usage

Collect the players' statistics from football-data.org:

```bash
python -m football_value.collect
```

This sends one request per league, caches the raw responses in
`data/raw/football_data/` and writes `data/raw/football_data_players.csv`.
Running it again reads the cache instead of calling the API.

Then add the market values and minutes played from the Transfermarkt dataset:

```bash
python -m football_value.merge
```

This downloads the dataset once into `data/raw/transfermarkt/` (about 55 MB)
and writes `data/processed/players_merged.csv`, plus
`data/processed/unmatched_players.csv` for the players that could not be matched.

## Development

```bash
pytest          # run the tests
ruff check .    # lint
```

## Roadmap

- [x] Project skeleton, configuration and tooling
- [x] Data collection from football-data.org (with local cache and rate limiting)
- [x] Market values and minutes from the Transfermarkt dataset, joined to the API data
- [ ] Cleaning and feature engineering
- [ ] Exploratory data analysis
- [ ] Linear regression: training and evaluation against a baseline
- [ ] Visualisations: predicted vs actual value, residuals, over/under-valued players
- [ ] LLM scouting reports
- [ ] Results, limitations and possible improvements
