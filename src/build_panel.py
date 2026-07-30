import pandas as pd

summary = pd.read_csv("data/ticker_summary.csv")
qualified = sorted(summary[summary["n_rows"] >= 750]["ticker"].tolist())

frames = []
for t in qualified:
    df = pd.read_csv(f"data/company-wise/{t}.csv")
    df["ticker"] = t
    frames.append(df)

panel = pd.concat(frames, ignore_index=True)
panel["published_date"] = pd.to_datetime(panel["published_date"])

panel = panel.drop_duplicates(subset=["ticker", "published_date"], keep="last")
panel = panel.sort_values(["ticker", "published_date"]).reset_index(drop=True)

print("rows:", len(panel), "| companies:", panel["ticker"].nunique())
print("date range:", panel["published_date"].min(), "to", panel["published_date"].max())
panel.to_parquet("data/panel_raw.parquet")