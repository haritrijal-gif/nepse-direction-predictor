import time
import urllib.request
import urllib.error

BASE_URL = "https://raw.githubusercontent.com/Aabishkar2/nepse-data/main/data/company-wise/{}.csv"

def main():
    with open("data/tickers.txt") as f:
        tickers = [t.strip() for t in f if t.strip()]

    ok, failed = 0, []
    for i, ticker in enumerate(tickers):
        url = BASE_URL.format(ticker)
        dest = f"data/company-wise/{ticker}.csv"
        try:
            urllib.request.urlretrieve(url, dest)
            ok += 1
        except urllib.error.HTTPError as e:
            failed.append((ticker, str(e)))
        if (i + 1) % 50 == 0:
            print(f"...{i+1}/{len(tickers)} done")
        time.sleep(0.03)

    print(f"\nDownloaded: {ok}/{len(tickers)}")
    if failed:
        print("Failed:", failed)

if __name__ == "__main__":
    main()