import json
import requests

url = "https://api.github.com/repos/Aabishkar2/nepse-data/git/trees/main?recursive=1"
response = requests.get(url)
data = response.json()

with open("data/tree.json", "w") as f:
    json.dump(data, f)

files = [
    x["path"].split("/")[-1].replace(".csv", "")
    for x in data["tree"]
    if x["path"].startswith("data/company-wise/") and x["path"].endswith(".csv")
]

print(f"{len(files)} tickers found")
with open("data/tickers.txt", "w") as f:
    f.write("\n".join(files))