#!/usr/bin/env python
"""
Hämtar kandidatrör från SGU API och sparar till ref_cache.json.

Kör detta skript SEPARAT före projekt_nivaer.py:
    py fetch_candidates.py

Strategi: SGU:s API stryper efter ~8-10 anrop på samma TLS-session.
Vi skapar en HELT NY SGUClient per station (= ny TCP-anslutning),
med 3 sekunders paus mellan varje anrop. Dessutom sparas cachen
inkrementellt efter varje lyckat anrop, så data inte förloras vid hängning.
"""

import os, ssl, json, time, sys
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings()

import requests
import pandas as pd
from pathlib import Path
from sgu_client import SGUClient

CACHE_PATH = Path(__file__).parent / "ref_cache.json"
METADATA   = Path(__file__).parent / "stationer_metadata.csv"
BASE_ID    = "22W102"
MAX_ATTEMPTS = 50      # prova max 50 stationer
MAX_DOWNLOAD = 15      # spara max 15 giltiga
SLEEP_SECS   = 3       # paus mellan anrop


def fetch_one_station(sid: str) -> dict:
    """Hämta en stations tidsserier med en ny SGUClient-instans."""
    with SGUClient() as client:
        client._base_client._session.verify = False
        m  = client.levels.observed.get_measurements_by_name(station_id=sid)
        df = m.to_dataframe()
        date_col  = next(c for c in df.columns
                         if "date" in c.lower() or "time" in c.lower())
        level_col = next(c for c in df.columns
                         if "level" in c.lower() or "water" in c.lower())
        df[date_col] = df[date_col].astype(str)
        records = {}
        for d, v in zip(df[date_col], df[level_col]):
            try:
                records[d] = float(v)
            except (ValueError, TypeError):
                records[d] = None
        return records


def main():
    # 1. Läs metadata
    meta = pd.read_csv(METADATA)
    row  = meta[meta["station_id"] == BASE_ID].iloc[0]
    akvifer, jordart = row["akvifer"], row["jordart"]
    print(f"Basröret {BASE_ID}: akvifer={akvifer}, jordart={jordart}")

    # 2. Hämta kandidater från OGC API
    url = ("https://api.sgu.se/oppnadata/grundvattennivaer-observerade"
           "/ogc/features/v1/collections/stationer/items")
    resp = requests.get(url, params={"akvifer": akvifer, "jordart": jordart,
                                     "limit": 500, "f": "json"},
                        timeout=30, verify=False)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    candidates = [
        f["properties"]["platsbeteckning"]
        for f in features
        if f["properties"].get("platsbeteckning") not in (None, BASE_ID)
    ][:MAX_ATTEMPTS]
    print(f"Hämtar tidsserier för {len(candidates)} kandidatstationer …")
    print(f"(ny SGUClient per station, {SLEEP_SECS}s paus mellan anrop)\n")

    # 3. Läs in eventuell befintlig cache (inkrementell)
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            results = json.load(f)
        print(f"Befintlig cache laddad: {len(results)} stationer")
    else:
        results = {}

    n_ok = sum(1 for v in results.values() if "__error__" not in v)

    # 4. Hämta varje station — ny SGUClient per anrop
    for i, sid in enumerate(candidates, 1):
        if n_ok >= MAX_DOWNLOAD:
            print(f"\n✓ {MAX_DOWNLOAD} giltiga rör hämtade — stoppar.")
            break

        # Skippa om redan hämtad
        if sid in results and "__error__" not in results[sid]:
            print(f"  [{i}/{len(candidates)}] ⏩ {sid}: redan i cache")
            continue

        t0 = time.time()
        try:
            records = fetch_one_station(sid)
            results[sid] = records
            n_ok += 1
            print(f"  [{i}/{len(candidates)}] ✓ {sid}: {len(records)} rader "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            results[sid] = {"__error__": str(e)}
            print(f"  [{i}/{len(candidates)}] ✗ {sid}: {e} ({time.time()-t0:.1f}s)")

        # Spara inkrementellt efter varje station
        with open(CACHE_PATH, "w") as f:
            json.dump(results, f)

        # Paus innan nästa anrop
        time.sleep(SLEEP_SECS)

    # 5. Sammanfattning
    ok_count   = sum(1 for v in results.values() if "__error__" not in v)
    fail_count = sum(1 for v in results.values() if "__error__" in v)
    print(f"\n✓ Cache sparad till {CACHE_PATH}")
    print(f"  Giltiga: {ok_count}, Misslyckade: {fail_count}, Totalt: {len(results)}")


if __name__ == "__main__":
    main()
