import urllib.request

BASE_URL = "https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise/{}.csv"

failed_tickers = ["HURJA"]  # add more here if you ever get multiple failures

for ticker in failed_tickers:
    url = BASE_URL.format(ticker)
    dest = f"data/company-wise/{ticker}.csv"
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"{ticker}: success")
    except Exception as e:
        print(f"{ticker}: still failed — {e}")