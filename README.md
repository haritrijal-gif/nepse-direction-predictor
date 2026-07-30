# NEPSE Next-Day Direction Predictor

## Problem Statement

Retail investors on the Nepal Stock Exchange (NEPSE) often trade based on
rumor and herd behavior rather than data. This project asks a simple,
testable question: **can a stock's next-day price direction (up or down) be
predicted from its own recent price and volume history alone** — no news, no
company fundamentals, no insider information?

- **Who this is for:** retail investors looking for a data-driven signal
  instead of informal tips, and as an honest audit of whether simple
  technical-indicator trading signals actually carry predictive power on
  NEPSE.
- **What the system outputs:** for a chosen NEPSE-listed company, a
  prediction of whether tomorrow's closing price will be up or down versus
  today, with a confidence score.
- **Why it matters:** if a simple, transparent model can't beat a coin flip,
  that's genuinely useful information — it tells an investor not to trust
  informal technical-indicator-based trading tips as a reliable edge.

## Dataset

- **Source:** [Aabishkar2/nepse-data](https://github.com/Aabishkar2/nepse-data),
  a public, auto-updating GitHub repository scraping daily OHLCV price data
  for every NEPSE-listed company.
- **Scope used:** 236 companies with at least ~3 years of trading history
  (750+ trading days), spanning 1995–2026, ~445,000 company-days total.
- **Why this dataset:** real, substantial, currently-updating data — not a
  toy dataset — directly relevant to a real market and real investors.

## Project Structure

nepse-direction-predictor/
├── data/ # raw and processed data (not committed to git)
├── notebooks/
│ └── eda_and_modeling.ipynb # full EDA, modeling, and error analysis
├── models/ # trained model + metrics (metrics committed, model file not)
├── src/
│ ├── list_tickers.py # get the list of available NEPSE tickers
│ ├── download_data.py # download each company's price history
│ ├── retry_failed.py # retry any tickers that failed to download
│ ├── check_data_quality.py # measure history length per company
│ ├── build_panel.py # combine all companies into one clean dataset
│ ├── features.py # feature engineering logic (shared by everything)
│ ├── run_features.py # run feature engineering, save the result
│ ├── train.py # train baseline + real models, evaluate, save best
│ └── error_analysis.py # honest error analysis on the test set
├── app/ # Streamlit app (coming soon)
├── requirements.txt
├── Dockerfile # (coming soon)
└── README.md


## Setup

git clone https://github.com/haritrijal-gif/nepse-direction-predictor.git
cd nepse-direction-predictor

python -m venv venv
venv\Scripts\Activate.ps1 # Windows PowerShell
pip install -r requirements.txt


## How to Run

Run these in order from the project root:

python src\list_tickers.py
python src\download_data.py
python src\retry_failed.py # only if any tickers failed
python src\check_data_quality.py
python src\build_panel.py
python src\run_features.py
python src\train.py
python src\error_analysis.py


Then open `notebooks\eda_and_modeling.ipynb` in Jupyter to see the full
analysis with charts:

jupyter notebook


## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority-class baseline | 56.3% | 0.0 | 0.0 | 0.0 | — |
| Naive momentum (repeat today) | 50.2% | 43.1% | 43.3% | 43.2% | — |
| Logistic Regression | 50.7% | 43.7% | 43.8% | 43.7% | 0.501 |
| **Random Forest (best)** | 52.0% | 44.6% | 40.2% | 42.3% | **0.513** |

**Honest conclusion:** the best model (Random Forest) beats random guessing
only marginally (ROC-AUC 0.513 vs. 0.50). This is a legitimate, expected
finding, not a failure — short-horizon stock direction using only technical
indicators is close to a random walk, mirroring well-documented real-world
market behavior. Full reasoning, error analysis, and charts are in the
notebook.

## Limitations

- Historical prices are not fully bonus/rights-share adjusted; corporate
  action days are flagged and winsorized rather than fully corrected.
- Only price/volume technical indicators are used — no news, fundamentals,
  or macroeconomic data.
- 1-day-ahead prediction is a very difficult horizon; longer horizons may
  show different results.

## Author

Harit Rijal — IT Project Manager, Kathmandu, Nepal