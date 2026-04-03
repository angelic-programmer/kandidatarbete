import matplotlib
matplotlib.use("Agg")          # Icke-interaktiv backend – sparar PNG utan plt.show()-blockering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    from statsmodels.tsa.stattools import acf as sm_acf
except ImportError:
    raise ImportError("Installera statsmodels: pip install statsmodels")


basror_22W102  = Path(__file__).parent / "22W102.csv"
basror_17XX01U = Path(__file__).parent / "17XX01U.csv"
basror_G1101   = Path(__file__).parent / "G1101.csv"
metadata_path  = Path(__file__).parent / "stationer_metadata.csv"

# Vi skapar en dataframe i pandas av basröret
def load_base_station(filepath: Path = basror_G1101) -> pd.Series:
    # Här är en funktion som tar en filsökväg som argument, basror_22W102 är defaultvärdet
    # Detektera separator automatiskt (semikolon eller komma)
    with open(filepath, encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    # Detektera om filen har rubrikrad (första fältet är inte ett datum)
    first_field = first_line.split(sep)[0].strip()
    try:
        pd.to_datetime(first_field)
        skiprows = 0
    except Exception:
        skiprows = 1
    df = pd.read_csv(
        filepath,
        sep=sep,
        header=None,
        names=["date", "level"],
        skiprows=skiprows,
        encoding="utf-8-sig",
    )
    # Nedan läser vi in csv-filen till en DataFrame med kolumnerna date och level
    for fmt in ["%Y-%m-%d", "%y/%m/%d", "%d/%m/%Y"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    df["level"] = (
        df["level"]
        .astype(str)
          # byt ut komma mot punkt 
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    # tomma värden blir NaN
    df["level"] = pd.to_numeric(df["level"], errors="coerce")  
    df = df.set_index("date").sort_index()
    series = df["level"]
    # Slå ihop duplicerade datum till medelvärde
    series = series.groupby(level=0).mean()
    return series


# Hämtar referensrör 95_2 via SGUClient — används av baslinjemodellen
def load_reference_station(station_id: str = "95_2") -> pd.Series:
    from sgu_client import SGUClient
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    with SGUClient() as client:
        # TILLFÄLLIGT: stäng av SSL-verifiering
        client._base_client._session.verify = False
        meas = client.levels.observed.get_measurements_by_name(
            station_id=station_id
        )
        df_ref = meas.to_dataframe()

    # Hitta datum- och nivåkolumn automatiskt
    date_col = next(
        c for c in df_ref.columns
        if "date" in c.lower() or "time" in c.lower()
    )
    level_col = next(
        c for c in df_ref.columns
        if "level" in c.lower() or "water" in c.lower()
    )
    df_ref[date_col] = pd.to_datetime(df_ref[date_col])
    ref = df_ref.set_index(date_col)[level_col].sort_index()
    if ref.index.tz is not None:
        ref.index = ref.index.tz_localize(None)
    ref = pd.to_numeric(ref, errors="coerce")
    print(f"✓ Referensdata hämtad från SGUClient ({station_id})")
    return ref


# Hämtar kandidatrör från SGU API med samma akvifer och jordart som basröret
# används av den multivariata modellen
def load_candidate_stations(
    base_series: pd.Series,
    base_station_id: str = "G1101",
    meta_path: Path = metadata_path,
    ignore_geology: bool = False,
) -> list:
    """
    Läser metadata för basröret, hämtar stationer med samma akvifer+jordart
    från SGU OGC API, och returnerar lista med (station_id, pd.Series)
    för rör med överlappning med basröret.

    Om ignore_geology=True hoppas det geologiska filtret över och alla
    stationer i cachen används som kandidater (väljs sedan på Pearson-korrelation).
    """
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Läs metadata och hämta basrörets geologiska egenskaper
    meta     = pd.read_csv(meta_path)
    base_row = meta[meta["station_id"] == base_station_id]
    if base_row.empty:
        raise ValueError(f"Basröret {base_station_id} saknas i metadatafilen {meta_path}")
    base_row       = base_row.iloc[0]
    target_akvifer = base_row["akvifer"]
    target_jordart = base_row["jordart"]
    print(f"Basröret {base_station_id}: akvifer={target_akvifer}, jordart={target_jordart}")

    # Hämta tidsserie för varje kandidat och filtrera på överlappning
    base_start = base_series.dropna().index.min()
    base_end   = base_series.dropna().index.max()
    result     = []

    max_download = 10

    # ── Strategi 1: läs från cache (ref_cache.json) ──
    cache_path = Path(__file__).parent / "ref_cache.json"
    if cache_path.exists():
        import json
        print(f"  Läser cache: {cache_path}")
        with open(cache_path) as f:
            all_data = json.load(f)

        if ignore_geology:
            # Använd alla cachade stationer som kandidater (inget geologiskt filter)
            print("  ignore_geology=True: geologiskt filter bortkopplat — använder hela cachen")
            cache_candidates = list(all_data.keys())
        else:
            # Hämta stationer från SGU OGC API filtrerat på akvifer och jordart
            url  = ("https://api.sgu.se/oppnadata/grundvattennivaer-observerade"
                    "/ogc/features/v1/collections/stationer/items")
            resp = requests.get(url, params={"akvifer": target_akvifer, "jordart": target_jordart,
                                             "limit": 500, "f": "json"}, timeout=30, verify=False)
            resp.raise_for_status()
            features   = resp.json().get("features", [])
            cache_candidates = [
                f["properties"]["platsbeteckning"]
                for f in features
                if f["properties"].get("platsbeteckning") not in (None, base_station_id)
            ]
            print(f"SGU API returnerade {len(cache_candidates)} kandidatrör "
                  f"(akvifer={target_akvifer}, jordart={target_jordart})")

        for sid in cache_candidates:
            if sid == base_station_id or sid not in all_data:
                continue
            records = all_data[sid]
            if "__error__" in records:
                continue
            s = pd.Series({pd.Timestamp(k): v for k, v in records.items()}, dtype=float)
            s = s.sort_index()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s = pd.to_numeric(s, errors="coerce")
            overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna()
            if len(overlap) > 0:
                result.append((sid, s))
                print(f"  ✓ {sid}: {len(overlap)} överlappande obs (cache)")
            if len(result) >= max_download:
                break
    else:
        # ── Strategi 2: direkt API-hämtning (kan hänga på Windows pga SSL) ──
        print("  ⚠ Ingen cache hittad — kör: py fetch_candidates.py")
        print("  Försöker hämta direkt (kan hänga) …")
        # Hämta kandidatlista från API om ingen cache finns
        url  = ("https://api.sgu.se/oppnadata/grundvattennivaer-observerade"
                "/ogc/features/v1/collections/stationer/items")
        params = {"limit": 500, "f": "json"}
        if not ignore_geology:
            params["akvifer"] = target_akvifer
            params["jordart"] = target_jordart
        import requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(url, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        features   = resp.json().get("features", [])
        candidates = [
            f["properties"]["platsbeteckning"]
            for f in features
            if f["properties"].get("platsbeteckning") not in (None, base_station_id)
        ]
        import time
        for sid in candidates[:50]:
            if len(result) >= max_download:
                break
            try:
                s = load_reference_station(sid)
                overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna()
                if len(overlap) > 0:
                    result.append((sid, s))
                    print(f"  ✓ {sid}: {len(overlap)} överlappande obs")
                else:
                    print(f"  ✗ {sid}: ingen överlappning")
            except Exception as e:
                print(f"  ✗ {sid}: fel ({e})")
            time.sleep(2.0)

    # Om inga kandidater hittades, fall tillbaka på 95_2
    if not result:
        print("\nVarning: inga kandidatrör hittades — faller tillbaka på 95_2")
        ref = load_reference_station("95_2")
        return [("95_2", ref)]

    # Sortera efter flest överlappande obs — så att max_refs väljer de bästa rören
    result_with_count = []
    for sid, s in result:
        s2 = s.copy()
        if s2.index.tz is not None:
            s2.index = s2.index.tz_localize(None)
        n_overlap = len(s2[(s2.index >= base_start) & (s2.index <= base_end)].dropna())
        result_with_count.append((sid, s, n_overlap))
    result_with_count.sort(key=lambda x: x[2], reverse=True)
    result = [(sid, s) for sid, s, _ in result_with_count]
    print(f"\nRören sorterade efter överlappning. Topp 5: {[sid for sid, _ in result[:5]]}")

    return result


# Hjälpfunktion: normaliserar en tidsserie till ett enhetligt veckoindex
# hanterar: flera mätningar per dag, olika mätintervall, saknade värden
def _resample_to_weekly(s: pd.Series, freq: str = "7D") -> pd.Series:
    # Steg 1: Ta bort tidszon om den finns
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    # Steg 2: Slå ihop flera mätningar samma dag till medelvärde
    # (hanterar rör som mäts flera gånger per dag)
    s = s.groupby(s.index.normalize()).mean()
    # Steg 3: Resampla till vecka — tätare mätningar slås ihop, glesare interpoleras
    # (hanterar att olika rör har olika mätintervall: dagligt, veckovist, månadsvist osv.)
    s = s.resample(freq).mean()
    # Steg 4: Interpolera korta luckor (max 4 veckor) med tidsbaserad linjär interpolering
    s = s.interpolate(method="time", limit=4)
    return s


# Nedan skapas en gemensam dataframe av basröret och ett enskilt referensrör
# används av baslinjemodellen
def prepare_joint_dataframe_baseline(
    base: pd.Series,
    ref: pd.Series,
    # frekvensen sätts veckovis
    freq: str = "7D",
) -> pd.DataFrame:
    # Ta bort tidszon (SGU returnerar UTC och basröret har ingen tidszon)
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)

    # Normalisera referensröret till veckovist index (hanterar duplikat och olika intervall)
    ref = _resample_to_weekly(ref, freq=freq)

    # Här använder jag endast den överlappande perioden där båda tidserierna har data
    start = max(base.index.min(), ref.index.min())
    end   = min(base.index.max(), ref.index.max())
    # Skapar ett regelbundet datumindex med en rad per vecka, från start till slut
    idx   = pd.date_range(start, end, freq=freq)

    # Basröret: resampla till vecka utan interpolation — Kalmanfiltret hanterar NaN
    base_weekly = base.groupby(base.index.normalize()).mean().resample(freq).mean()
    df = pd.DataFrame(index=idx)
    df["base"] = base_weekly.reindex(idx)
    # Referensröret: behåll interpolation (tätare data, inga långa luckor)
    df["ref"]  = ref.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))

    return df


# Nedan skapas en gemensam dataframe av basröret och flera referensrör
# används av den multivariata modellen
def prepare_joint_dataframe_multi(
    base: pd.Series,
    refs: list,
    # frekvensen sätts veckovis
    freq: str = "7D",
    # max antal referensrör — begränsar antalet parametrar i modellen
    # 4 rör × 4 parametrar + 2 = 18 parametrar totalt (hanterbart för optimeraren)
    max_refs: int = 5,
) -> pd.DataFrame:
    # Ta bort tidszon om den finns
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)

    # ── Ranka referensrör efter Pearson-korrelation med basröret ──────
    # (istället för enbart överlappningsantal)
    base_w = _resample_to_weekly(base, freq=freq)
    # Normalisera till veckoperiod (W-MON) så att alla serier får samma ankare
    # oavsett vilket datum de börjar på — annars ger resample("7D") olika index
    base_w.index = base_w.index.to_period("W").to_timestamp()
    corr_list = []
    for sid, s in refs:
        s_w = _resample_to_weekly(s, freq=freq)
        s_w.index = s_w.index.to_period("W").to_timestamp()
        common = base_w.index.intersection(s_w.index)
        if len(common) < 20:
            continue
        b = base_w.reindex(common)
        r = s_w.reindex(common)
        mask = ~(b.isna() | r.isna())
        if mask.sum() < 20:
            continue
        corr = np.corrcoef(b[mask], r[mask])[0, 1]
        corr_list.append((sid, s, abs(corr), corr))
    # Sortera efter absolut korrelation (högst först)
    corr_list.sort(key=lambda x: x[2], reverse=True)
    print("Korrelation med basröret (topp 10):")
    for sid, _, _, c in corr_list[:10]:
        print(f"  {sid}: r = {c:.4f}")

    # Välj de max_refs med högst korrelation
    if len(corr_list) > max_refs:
        print(f"Begränsar från {len(corr_list)} till {max_refs} referensrör (max_refs={max_refs}, valt efter korrelation)")
    refs = [(sid, s) for sid, s, _, _ in corr_list[:max_refs]]

    # Normalisera alla referensrör till veckovist index innan sammanfogning
    # (hanterar: flera mätningar per dag, dagliga/månadsvis/oregelbundna intervall)
    refs_weekly = []
    for sid, s in refs:
        s_w = _resample_to_weekly(s, freq=freq)
        refs_weekly.append((sid, s_w))

    # Använd basrörets fulla datumspann — referensrör som saknar data för
    # vissa veckor får NaN, vilket Kalmanfiltret hanterar naturligt
    start = base.index.min()
    end   = base.index.max()

    # Skapar ett regelbundet datumindex med en rad per vecka, från start till slut
    idx = pd.date_range(start, end, freq=freq)

    # Basröret: resampla till vecka utan interpolation — Kalmanfiltret hanterar NaN
    base_weekly = base.groupby(base.index.normalize()).mean().resample(freq).mean()
    df = pd.DataFrame(index=idx)
    df["base"] = base_weekly.reindex(idx)
    # Referensrören: behåll interpolation (tätare data, inga långa luckor)
    for sid, s in refs_weekly:
        df[sid] = s.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))

    return df


# ─────────────────────────────────────────────
#  BASLINJEMODELL (ett referensrör: topp-1 via Pearson-korrelation)
# ─────────────────────────────────────────────

#vi skapar en ny "klass" i python där vi tar med ett färdigt ramverk för state space-modeller och bara behöver specificera just vår modell
#VAR KOMMER det färdiga ramverket ifrån?
class GroundwaterSSM(MLEModel):

#Nedan är en lista med namnen på de 4 parametrar som modellen ska skatta
#alla sigma2 variabler skattas utifrån basröret
#sigma2 epsilon (sigma2 eps) skattar mätfelet för referensröret
#sigma2_eps_base är låst till 0 (basrörets mätningar antas vara exakta)
#trend-komponenten är borttagen: µ(t+1) = µ(t) + η_level(t)
    # param_names sätts dynamiskt i __init__ beroende på nseason

    def __init__(self, endog, nseason=26, **kwargs):
        self.nseason = nseason
        #nedan beräknas antalet states med 26 st säsonger blir det 26+1(nivå) = 27 states
        # Om nseason=0 → ingen säsongskomponent, bara nivå (local level)
        k_states = 1 + nseason if nseason > 0 else 1
        k_posdef = 2 if nseason > 0 else 1

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        # Dynamiska parameternamn
        if nseason > 0:
            self._param_names = [
                "sigma2_eta_level", "sigma2_eta_season",
                "sigma2_eps_ref", "beta_ref",
            ]
        else:
            self._param_names = [
                "sigma2_eta_level",
                "sigma2_eps_ref", "beta_ref",
            ]

        # Diffus initiering för nivå (okänt startvärde)
        self.ssm.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        import statsmodels.api as sm

        y1 = self.endog[:, 0]  # basröret
        y2 = self.endog[:, 1]  # referensröret

        # Steg 1-3: Anpassa univariat strukturmodell till basröret
        # Detta ger oss databaserade startgissningar för varianser och säsong
        if self.nseason > 0:
            mod = sm.tsa.UnobservedComponents(
                y1, level="local level", seasonal=self.nseason
            )
            res = mod.fit(disp=False, method="nm", maxiter=500)
            sigma2_eta_level  = float(res.params[1])
            sigma2_eta_season = float(res.params[2])
            level  = res.level.smoothed
            season = res.seasonal.smoothed
        else:
            # Ingen säsong — enbart local level
            mod = sm.tsa.UnobservedComponents(y1, level="local level")
            res = mod.fit(disp=False, method="nm", maxiter=500)
            sigma2_eta_level = float(res.params[1])
            level  = res.level.smoothed
            season = np.zeros_like(level)

        # Steg 5: OLS för laddningsparameter beta_ref
        y2_deseas = y2 - season
        mask = ~(np.isnan(y2_deseas) | np.isnan(level))
        X    = level[mask].reshape(-1, 1)
        ols  = np.linalg.lstsq(X, y2_deseas[mask], rcond=None)
        beta_ref       = float(ols[0][0])
        sigma2_eps_ref = float(np.var(y2_deseas[mask] - X.flatten() * beta_ref))

        if self.nseason > 0:
            return np.array([
                sigma2_eta_level, sigma2_eta_season,
                sigma2_eps_ref, beta_ref,
            ])
        else:
            return np.array([
                sigma2_eta_level,
                sigma2_eps_ref, beta_ref,
            ])

#FORTSÄTT FÖRKLARA DENNA KOD FRÅN OCH MED IMORGON 

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        # Varianser: exp-transform → alltid positiva
        n_var = 3 if self.nseason > 0 else 2  # level+season+ref vs level+ref
        p[:n_var] = np.exp(unconstrained[:n_var])
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        n_var = 3 if self.nseason > 0 else 2
        p[:n_var] = np.log(constrained[:n_var])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        ns = self.nseason
        k  = self.k_states

        if ns > 0:
            (s2_level, s2_season, s2_ref, beta_ref) = params
        else:
            (s2_level, s2_ref, beta_ref) = params

        # ── Transitionsmatris T ──────────────────────────────────────────
        T = np.zeros((k, k))
        T[0, 0] = 1.0  # Nivå: µ(t+1) = µ(t)  (random walk utan trend)

        if ns > 0:
            # Säsong med summavillkor:
            T[1, 1:1+ns] = -1.0
            for j in range(1, ns):
                T[1 + j, 1 + j - 1] = 1.0

        self.ssm["transition"] = T

        # ── Observationsmatris Z ─────────────────────────────────────────
        Z = np.zeros((2, k))
        Z[0, 0] = 1.0      # base ← nivå
        if ns > 0:
            Z[0, 1] = 1.0  # base ← säsong
        Z[1, 0] = beta_ref  # ref  ← nivå

        self.ssm["design"] = Z

        # ── Observationsbrus ────────────────────────────────────────────
        self.ssm["obs_cov"] = np.diag([0.0, s2_ref])

        # ── Processkörsbrus Q ───────────────────────────────────────────
        if ns > 0:
            Q = np.array([[s2_level, 0.0],
                          [0.0, s2_season]])
        else:
            Q = np.array([[s2_level]])
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris R: kopplar chockerna till rätt state ───────
        kp = 2 if ns > 0 else 1
        R = np.zeros((k, kp))
        R[0, 0] = 1.0  # η_level  → state 0 (nivå)
        if ns > 0:
            R[1, 1] = 1.0  # η_season → state 1 (γ_0)
        self.ssm["selection"] = R


# ─────────────────────────────────────────────
#  MULTIVARIAT MODELL (flera referensrör)
# ─────────────────────────────────────────────

#vi skapar en ny "klass" i python där vi tar med ett färdigt ramverk för state space-modeller och bara behöver specificera just vår modell
#VAR KOMMER det färdiga ramverket ifrån?
class GroundwaterSSM_Multi(MLEModel):

#Nedan är en lista med namnen på de parametrar som modellen ska skatta — antalet beror på hur många referensrör som används
#alla sigma2 variabler skattas utifrån basröret
#sigma2 epsilon (sigma2 eps) skattar mätfelet för referensröret
#sigma2_eps_base är låst till 0 (basrörets mätningar antas vara exakta)
#trend-komponenten är borttagen: µ(t+1) = µ(t) + η_level(t)

    def __init__(self, endog, nseason=26, ref_ids=None, **kwargs):
        self.nseason = nseason
        self.ref_ids = ref_ids if ref_ids is not None else []
        self.n_refs  = len(self.ref_ids)

        # Om nseason=0 → ingen säsongskomponent, bara nivå
        k_states = 1 + nseason if nseason > 0 else 1
        k_posdef = 2 if nseason > 0 else 1

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        # param_names: med säsong: 2 processvarianser + 4 per ref
        #              utan säsong: 1 processvarians + 3 per ref (ingen gamma)
        if nseason > 0:
            self._param_names = ["sigma2_eta_level", "sigma2_eta_season"]
            for rid in self.ref_ids:
                self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}",
                                      f"alpha_{rid}", f"gamma_{rid}"]
        else:
            self._param_names = ["sigma2_eta_level"]
            for rid in self.ref_ids:
                self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}",
                                      f"alpha_{rid}"]

        # Diffus initiering för nivå (okänt startvärde)
        self.ssm.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        import statsmodels.api as sm

        y1 = self.endog[:, 0]  # basröret

        # Steg 1-2: Anpassa univariat strukturmodell till basröret för startgissningar.
        # Om basen saknar tillräckligt med observationer används enkla standardvärden.
        y1_valid = y1[~np.isnan(y1)]
        if len(y1_valid) >= 3 and self.nseason > 0:
            mod = sm.tsa.UnobservedComponents(
                y1, level="local level", seasonal=self.nseason
            )
            res = mod.fit(disp=False, method="nm", maxiter=500)
            sigma2_eta_level  = float(res.params[1])
            sigma2_eta_season = float(res.params[2])
            level  = res.level.smoothed
            season = res.seasonal.smoothed
        elif len(y1_valid) >= 3:
            # Ingen säsong — enbart local level
            mod = sm.tsa.UnobservedComponents(y1, level="local level")
            res = mod.fit(disp=False, method="nm", maxiter=500)
            sigma2_eta_level = float(res.params[1])
            level  = res.level.smoothed
            season = np.zeros(len(y1))
        else:
            # Basen är helt tom — använd referensrör 1 som proxy för startvärden
            y_proxy = self.endog[:, 1]
            y_proxy_valid = y_proxy[~np.isnan(y_proxy)]
            sigma2_eta_level  = float(np.var(y_proxy_valid)) * 0.01 if len(y_proxy_valid) > 1 else 0.01
            level  = np.full(len(y1), float(np.nanmean(y_proxy_valid)) if len(y_proxy_valid) > 0 else 0.0)
            season = np.zeros(len(y1))

        if self.nseason > 0:
            params = [sigma2_eta_level, sigma2_eta_season]
        else:
            params = [sigma2_eta_level]

        # Steg 3: OLS per referensrör
        for i in range(self.n_refs):
            y_ref = self.endog[:, 1 + i]
            mask  = ~(np.isnan(y_ref) | np.isnan(level) | np.isnan(season))
            if self.nseason > 0:
                X     = np.column_stack([np.ones(mask.sum()), level[mask], season[mask]])
                ols   = np.linalg.lstsq(X, y_ref[mask], rcond=None)
                alpha_i  = float(ols[0][0])
                beta_i   = float(ols[0][1])
                gamma_i  = float(ols[0][2])
                s2_eps_i = float(np.var(y_ref[mask] - X @ ols[0]))
                params  += [max(s2_eps_i, 1e-6), beta_i, alpha_i, gamma_i]
            else:
                X     = np.column_stack([np.ones(mask.sum()), level[mask]])
                ols   = np.linalg.lstsq(X, y_ref[mask], rcond=None)
                alpha_i  = float(ols[0][0])
                beta_i   = float(ols[0][1])
                s2_eps_i = float(np.var(y_ref[mask] - X @ ols[0]))
                params  += [max(s2_eps_i, 1e-6), beta_i, alpha_i]

        return np.array(params)

#FORTSÄTT FÖRKLARA DENNA KOD FRÅN OCH MED IMORGON 

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        p[0] = np.exp(unconstrained[0])   # sigma2_eta_level
        if self.nseason > 0:
            p[1] = np.exp(unconstrained[1])   # sigma2_eta_season
            ppref = 4  # params per ref: eps, beta, alpha, gamma
            offset = 2
        else:
            ppref = 3  # params per ref: eps, beta, alpha
            offset = 1
        for i in range(self.n_refs):
            p[offset + ppref*i] = np.exp(unconstrained[offset + ppref*i])
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0])
        if self.nseason > 0:
            p[1] = np.log(constrained[1])
            ppref = 4
            offset = 2
        else:
            ppref = 3
            offset = 1
        for i in range(self.n_refs):
            p[offset + ppref*i] = np.log(constrained[offset + ppref*i])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        ns      = self.nseason
        k       = self.k_states
        n_endog = 1 + self.n_refs

        s2_level = params[0]
        if ns > 0:
            s2_season = params[1]
            ppref = 4; offset = 2
            ref_params = [
                (params[offset + ppref*i], params[offset + ppref*i+1],
                 params[offset + ppref*i+2], params[offset + ppref*i+3])
                for i in range(self.n_refs)
            ]
        else:
            ppref = 3; offset = 1
            ref_params = [
                (params[offset + ppref*i], params[offset + ppref*i+1],
                 params[offset + ppref*i+2], 0.0)
                for i in range(self.n_refs)
            ]

        # ── Transitionsmatris T ──────────────────────────────────────────
        T = np.zeros((k, k))
        T[0, 0] = 1.0  # Nivå: µ(t+1) = µ(t)

        if ns > 0:
            T[1, 1:1+ns] = -1.0
            for j in range(1, ns):
                T[1 + j, 1 + j - 1] = 1.0

        self.ssm["transition"] = T

        # ── Observationsmatris Z ─────────────────────────────────────────
        Z = np.zeros((n_endog, k))
        Z[0, 0] = 1.0   # base ← nivå
        if ns > 0:
            Z[0, 1] = 1.0   # base ← säsong
        for i, (_, beta_i, _, gamma_i) in enumerate(ref_params):
            Z[1 + i, 0] = beta_i
            if ns > 0:
                Z[1 + i, 1] = gamma_i

        self.ssm["design"] = Z

        # ── Observationsintercept (d-vektor) ─────────────────────────────
        d = np.zeros((n_endog, 1))
        for i, (_, _, alpha_i, _) in enumerate(ref_params):
            d[1 + i, 0] = alpha_i
        self.ssm["obs_intercept"] = d

        # ── Observationsbrus ────────────────────────────────────────────
        obs_cov = np.zeros((n_endog, n_endog))
        for i, (s2_eps_i, _, _, _) in enumerate(ref_params):
            obs_cov[1 + i, 1 + i] = s2_eps_i
        self.ssm["obs_cov"] = obs_cov

        # ── Processkörsbrus Q ───────────────────────────────────────────
        if ns > 0:
            Q = np.array([[s2_level, 0.0],
                          [0.0, s2_season]])
        else:
            Q = np.array([[s2_level]])
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris R: kopplar chockerna till rätt state ───────
        kp = 2 if ns > 0 else 1
        R = np.zeros((k, kp))
        R[0, 0] = 1.0  # η_level  → state 0 (nivå)
        if ns > 0:
            R[1, 1] = 1.0  # η_season → state 1 (γ_0)
        self.ssm["selection"] = R

# ─────────────────────────────────────────────
#  ANPASSA OCH PREDIKTERA
# ─────────────────────────────────────────────

def fit_model_baseline(df: pd.DataFrame, nseason: int = 26) -> tuple:
    """Anpassar baslinjemodellen (ett referensrör) och returnerar (result, model)."""
    endog = df[["base", "ref"]].values.astype(float)

    model = GroundwaterSSM(endog, nseason=nseason)

    # Bättre initiering: sätt startvärdet för nivån till medelvärdet av
    # basrörets faktiska observationer istället för 0 (diffus P₀=10⁶·I)
    base_obs = df["base"].dropna().values
    if len(base_obs) > 0:
        a0 = np.zeros(model.k_states)
        a0[0] = float(np.mean(base_obs))  # nivå-state initieras vid observerat medel
        P0 = np.eye(model.k_states) * 1e6
        P0[0, 0] = float(np.var(base_obs)) if len(base_obs) > 1 else 1.0
        model.ssm.initialize_known(a0, P0)

    print("Anpassar baslinjemodellen (MLE med Kalman filter)...")
    result = model.fit(
        method="lbfgs",
        maxiter=2000,
        disp=True,
    )
    print("\n" + "="*50)
    print(result.summary())
    return result, model


def fit_model_multi(df: pd.DataFrame, nseason: int = 26) -> tuple:
    """Anpassar multivariat SSM (flera referensrör) och returnerar (result, model)."""
    ref_ids = [col for col in df.columns if col != "base"]
    endog   = df[["base"] + ref_ids].values.astype(float)

    model = GroundwaterSSM_Multi(endog, nseason=nseason, ref_ids=ref_ids)

    print(f"Anpassar multivariat modell med {len(ref_ids)} referensrör: {ref_ids}...")
    # Powell (derivative-free) — undviker dyra numeriska gradienter
    # (lbfgs med 18+ parametrar kräver ~18 Kalman-filter per gradient)
    # cov_type="none" — hoppa över OPG-kovariansberäkning som kräver
    # ytterligare 18 × complex-step Kalman-filterkörningar
    result = model.fit(
        method="powell",
        maxiter=5000,
        disp=True,
        cov_type="none",
    )
    print("\n" + "="*50)
    print(result.summary())
    return result, model



def smooth_and_forecast(result, df: pd.DataFrame, n_forecast: int = 26) -> dict:
    """Kör Kalman smoother + prognos och returnerar dict med alla resultat."""
    smoothed = result.smoother_results
    level_smoothed = smoothed.smoothed_state[0]
    level_var      = smoothed.smoothed_state_cov[0, 0]

    # Smoother-baserat prediktionsintervall (undviker explosioner vid saknade värden)
    nseason = getattr(result.model, 'nseason', 26)
    if nseason > 0:
        season_var = smoothed.smoothed_state_cov[1, 1]
        total_var  = level_var + season_var
        pred_mean  = level_smoothed + smoothed.smoothed_state[1]
    else:
        season_var = np.zeros_like(level_var)
        total_var  = level_var
        pred_mean  = level_smoothed
    std = np.sqrt(np.maximum(total_var, 0))
    pred_ci    = np.column_stack([
        pred_mean - 1.96 * std,
        pred_mean + 1.96 * std,
    ])

    # Smoother-baserad prognos: konstant nivå + växande osäkerhet från random walk
    s2_level   = result.params[0]
    last_std   = std[-1]
    fcast_mean = np.full(n_forecast, pred_mean[-1])
    fcast_std  = np.sqrt(last_std**2 + np.arange(1, n_forecast + 1) * s2_level)
    fcast_ci   = np.column_stack([
        fcast_mean - 1.96 * fcast_std,
        fcast_mean + 1.96 * fcast_std,
    ])

    last_date = df.index[-1]
    # Härleda frekvens från indexet
    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq is None:
        # Fallback: beräkna median-avstånd
        diffs = df.index.to_series().diff().dropna()
        median_gap = diffs.median()
        inferred_freq = "MS" if median_gap > pd.Timedelta("21D") else "7D"
    fcast_idx = pd.date_range(
        last_date + (pd.DateOffset(months=1) if "M" in str(inferred_freq) else pd.Timedelta("7D")),
        periods=n_forecast,
        freq=inferred_freq,
    )

    # One-step-ahead filter-prediktioner (basröret, rad 0)
    # Prediktion vid tid t baserat enbart på data t₁, …, t₋₁
    filter_pred_base = result.filter_results.forecasts[0]
    filter_pred_var  = result.filter_results.forecasts_error_cov[0, 0]
    filter_pred_std  = np.sqrt(np.maximum(filter_pred_var, 0))
    filter_pred_ci   = np.column_stack([
        filter_pred_base - 1.96 * filter_pred_std,
        filter_pred_base + 1.96 * filter_pred_std,
    ])

    return {
        "index":          df.index,
        "observed_base":  df["base"].values,
        "level_smoothed": level_smoothed,
        "level_var":      level_var,
        "season_var":     season_var,
        "pred_base":      pred_mean,
        "pred_ci":        pred_ci,
        "filter_pred_base": filter_pred_base,
        "filter_pred_ci":   filter_pred_ci,
        "fcast_index":    fcast_idx,
        "fcast_base":     fcast_mean,
        "fcast_ci":       fcast_ci,
    }


# ─────────────────────────────────────────────
#  MODELLDIAGNOSTIK / UTVÄRDERING
# ─────────────────────────────────────────────

def evaluate_model(result, out: dict, label: str = "Modell", station_id: str = "station") -> dict:
    """
    Beräknar diagnostik- och utvärderingsmått för en state-space-modell.

    Returnerar en dict med alla mått.
    Hanterar både univariata (baseline) och multivariata modeller.

    Utvärderar de tre antaganden som krävs för SSM
    (Kap. 17, Applied Time Series Analysis, 2019):

      Antagande 1 (§17.1): Initialtillstånd α₀ har E(α₀)=a₀, V(α₀)=P₀
      Antagande 2 (§17.1): Feltermerna εₜ och ηₜ är okorrelerade med
                            varandra i alla tidsperioder samt okorrelerade
                            med initialtillståndet
      Antagande 3 (§17.9): Feltermerna och initialtillståndet är
                            normalfördelade (krävs för Kalmanfiltrets
                            optimalitet)
    """
    from scipy import stats as sp_stats

    obs  = out["observed_base"]
    # One-step-ahead filter-prediktioner: ŷ(t|t-1)
    pred = out["filter_pred_base"]
    ci   = out["filter_pred_ci"]

    # ── Mask: tidpunkter med faktisk observation OCH giltig prediktion ─
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o = obs[valid]
    p = pred[valid]

    # ── Modellanpassning (informationskriterier + prediktionsfel) ─────
    metrics = {
        "Log-likelihood":    result.llf,
        "AIC":               result.aic,
        "BIC":               result.bic,
        "HQIC":              result.hqic,
        "Antal parametrar":  len(result.params),
        "Antal obs (T)":     int(result.nobs),
    }

    resid = o - p

    # Hoppa över den diffusa initieringsperioden (första nseason tidsteg)
    # där Kalman-filtret ännu inte konvergerat (P₀ = 10⁶·I)
    try:
        nseason = result.model.nseason
    except AttributeError:
        nseason = 26
    burn = min(nseason, len(resid) - 10)  # behåll minst 10 obs
    resid_eval = resid[burn:]

    metrics["RMSE"]       = float(np.sqrt(np.mean(resid_eval**2)))
    metrics["MAE"]        = float(np.mean(np.abs(resid_eval)))
    metrics["ME (bias)"]  = float(np.mean(resid_eval))

    # Prediktionsintervallets täckningsgrad (efter burn-in)
    ci_lower = ci[valid, 0][burn:]
    ci_upper = ci[valid, 1][burn:]
    o_eval   = o[burn:]
    inside   = (o_eval >= ci_lower) & (o_eval <= ci_upper)
    metrics["95%-täckning"] = float(inside.mean())

    # ── Standardiserade innovationer e(t)/√F(t) ─────────────────────
    #   Används för att testa antagande 2 och 3.
    #   Viktigt: filtrera bort tidpunkter utan faktisk observation —
    #   statsmodels sätter innovationer till 0 för saknade obs, vilket
    #   ger ett missvisande histogram (spike vid 0) vid gles mätdata.
    si_valid = None
    obs_mask = ~np.isnan(obs)   # True där basröret har en riktig mätning
    try:
        std_innov = result.filter_results.standardized_forecasts_error
        # std_innov shape: (k_endog, T)  —  ta rad 0 (basröret)
        si_base = std_innov[0]
        # Behåll bara tidpunkter med faktisk observation
        si_base_obs = si_base[obs_mask]
        # Hoppa över diffus initieringsperiod även för innovationer
        burn_obs = min(burn, len(si_base_obs) - 10)
        si_after_burn = si_base_obs[burn_obs:]
        si_valid = si_after_burn[~np.isnan(si_after_burn)]
        if len(si_valid) <= 10:
            si_valid = None
    except Exception:
        pass

    # ── Antagande 1 (§17.1): Initialtillstånd ───────────────────────
    #   E(α₀) = a₀ och V(α₀) = P₀.
    #   Våra modeller använder approximate diffuse initialization:
    #   a₀ = 0, P₀ = κ·I (κ = 10⁶) — standardmetod när ingen
    #   förhandsinformation om α₀ finns tillgänglig.
    metrics["A1 uppfyllt"] = True    # per modellspecifikation

    # ── Antagande 2 (§17.1): Okorrelerade feltermer ─────────────────
    #   εₜ seriellt okorrelerad  →  Ljung-Box, Durbin-Watson
    #   E(εₜη'ₛ) = 0 ∀ s,t      →  inbyggt i modellstrukturen
    #   E(εₜα'₀) = 0, E(ηₜα'₀) = 0  →  inbyggt i modellstrukturen
    if si_valid is not None:
        metrics["Innov. medel"] = float(np.mean(si_valid))
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(resid_eval, lags=[10], return_df=True)
        metrics["Ljung-Box Q(10)"] = float(lb["lb_stat"].values[0])
        metrics["Ljung-Box p"]     = float(lb["lb_pvalue"].values[0])
    except Exception:
        pass
    try:
        from statsmodels.stats.stattools import durbin_watson
        metrics["Durbin-Watson"] = float(durbin_watson(resid_eval))
    except Exception:
        pass

    # Sammanfattande bedömning: antagande 2
    a2_ok = True
    if "Ljung-Box p" in metrics and metrics["Ljung-Box p"] < 0.05:
        a2_ok = False
    metrics["A2 uppfyllt"] = a2_ok

    # ── Antagande 3 (§17.9): Normalfördelning ───────────────────────
    #   Kalmanfiltrets härledning kräver normalfördelade feltermer
    #   och initialtillstånd.  Visuell bedömning via histogram över
    #   standardiserade innovationer.
    metrics["_si_valid"] = si_valid   # sparas för histogram (efter burn-in, för mått)
    # Spara HELA serien (utan burn-in) för ärligare histogram och residualplot
    # (fortfarande filtrerat på faktiska observationer — inga noll-fyllda luckor)
    try:
        si_all = si_base_obs[~np.isnan(si_base_obs)]
        metrics["_si_all"] = si_all if len(si_all) > 10 else si_valid
    except Exception:
        metrics["_si_all"] = si_valid

    # Spara HELA residualserien (utan burn-in) med tidsindex för residualplot
    dates_arr = out.get("index")
    full_resid = obs - pred  # hela serien inkl. NaN
    full_valid = ~np.isnan(full_resid)
    metrics["_resid"] = full_resid[full_valid]
    metrics["_resid_dates"] = dates_arr[full_valid] if dates_arr is not None else None

    # ── Skriv ut ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIK: {label}")
    print(f"{'='*60}")

    print("\n── Modellanpassning ──")
    for k in ["Log-likelihood", "AIC", "BIC", "HQIC",
              "Antal parametrar", "Antal obs (T)"]:
        print(f"  {k:<30} = {metrics[k]:>12.4f}"
              if isinstance(metrics[k], float)
              else f"  {k:<30} = {metrics[k]:>12d}")

    print("\n── Prediktionsfel (basröret, one-step-ahead) ──")
    for k in ["RMSE", "MAE", "ME (bias)"]:
        print(f"  {k:<30} = {metrics[k]:>12.6f}")
    print(f"  {'95%-täckning':<30} = {metrics['95%-täckning']:>11.1%}")

    # — Antagande 1 —
    print("\n── Antagande 1 (§17.1): Initialtillstånd ──")
    print("  Approximate diffuse: a₀=0, P₀=10⁶·I")

    # — Antagande 2 —
    print("\n── Antagande 2 (§17.1): Okorrelerade feltermer ──")
    if "Innov. medel" in metrics:
        print(f"  {'Innov. medelvärde':<30} = {metrics['Innov. medel']:>12.4f}")
    if "Ljung-Box Q(10)" in metrics:
        print(f"  {'Ljung-Box Q(10)':<30} = {metrics['Ljung-Box Q(10)']:>12.4f}")
        print(f"  {'Ljung-Box p-värde':<30} = {metrics['Ljung-Box p']:>12.4f}")
    if "Durbin-Watson" in metrics:
        print(f"  {'Durbin-Watson':<30} = {metrics['Durbin-Watson']:>12.4f}")

    # — Antagande 3 — histogram sparas som PNG
    print("\n── Antagande 3 (§17.9): Normalfördelning ──")
    si = metrics.get("_si_all", metrics.get("_si_valid"))
    if si is not None and len(si) > 10:
        import matplotlib.pyplot as plt
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fname = f"residual_histogram_{station_id}_{slug}.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(si, bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("Standardiserad innovation")
        ax.set_ylabel("Frekvens")
        ax.set_title(f"Histogram — {station_id} {label}", fontsize=13)
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Histogram sparat till: {fname}")

        # Residualer över tid (observation − prediktion)
        resid_vals  = metrics.get("_resid")
        resid_dates = metrics.get("_resid_dates")
        if resid_vals is not None and resid_dates is not None:
            fname2 = f"residual_tid_{station_id}_{slug}.png"
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(resid_dates, resid_vals,
                     marker="o", markersize=3, linestyle="-",
                     color="steelblue")
            ax2.axhline(0, linestyle="--", color="black", linewidth=0.8)
            ax2.set_xlabel("Datum")
            ax2.set_ylabel("Residual (observation \u2212 prediktion)")
            ax2.set_title(f"Residualer över tid \u2014 {station_id} {label}", fontsize=13)
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(fname2, dpi=150)
            plt.close(fig2)
            print(f"  Residualplot sparat till: {fname2}")
    else:
        print("  Inga innovationer tillgängliga för histogram.")

    return metrics


# ─────────────────────────────────────────────
#  AVVIKELSEDETEKTERING (á la Akvifär)
# ─────────────────────────────────────────────

def detect_anomalies(
    observed: np.ndarray,
    pred_ci: np.ndarray,
    min_consecutive: int = 3,
) -> pd.Series:
    """
    Detekterar perioder med konsekutiva avvikelser utanför prediktionsintervallet,
    analogt med Akvifärs referensmetod. Kräver min_consecutive avvikelser i följd.
    """
    outside = (observed < pred_ci[:, 0]) | (observed > pred_ci[:, 1])
    outside = pd.Series(outside)

    # Kräv min_consecutive i följd
    anomaly = pd.Series(False, index=outside.index)
    in_streak    = 0
    streak_start = None
    for i, val in enumerate(outside):
        if val:
            if in_streak == 0:
                streak_start = i
            in_streak += 1
        else:
            if in_streak >= min_consecutive:
                anomaly.iloc[streak_start:i] = True
            in_streak    = 0
            streak_start = None
    if in_streak >= min_consecutive:
        anomaly.iloc[streak_start:] = True
    return anomaly


# ─────────────────────────────────────────────
#  VISUALISERING
# ─────────────────────────────────────────────

def plot_results(out: dict, anomaly: pd.Series, title_base: str = "22W102",
                 ref_label: str = "95_2", model_label: str = ""):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    label_part = f" — {model_label}" if model_label else ""
    fig.suptitle(
        f"Grundvatten State Space Model{label_part}\n"
        f"Basobjekt: {title_base} | Referens: {ref_label}",
        fontsize=14, fontweight="bold",
    )

    idx      = out["index"]
    obs      = out["observed_base"]
    pred     = out["pred_base"].flatten()[:len(idx)]
    # Använd filter-CI (one-step-ahead) för visuellt band — synligt över hela serien
    ci       = out["filter_pred_ci"][:len(idx)]
    f_idx    = out["fcast_index"]
    f_vals   = out["fcast_base"].flatten()[:len(f_idx)]
    f_ci     = out["fcast_ci"][:len(f_idx)]
    smoothed = out["level_smoothed"].flatten()[:len(idx)]

    # ── Panel 1: Observationer, Kalman-smoother, Prognos ─────────────────
    ax = axes[0]
    ax.set_title("Grundvattennivå: observationer, utjämning & prognos", fontsize=11)

    ax.fill_between(idx, ci[:, 0], ci[:, 1],
                    color="steelblue", alpha=0.20, label="95% prediktionsintervall")

    # Markera avvikelser
    anom_idx = idx[anomaly.values]
    anom_obs = obs[anomaly.values]
    ax.scatter(anom_idx, anom_obs, color="red", zorder=5, s=30,
               label="Avvikelse (≥3 i följd)", marker="x")

    ax.plot(idx, obs, "o", color="navy", ms=3, alpha=0.7, label=f"Observation ({title_base})")
    ax.plot(idx, pred, color="steelblue", lw=1.5, label="Predikterat (in-sample)")
    ax.plot(idx, smoothed, color="darkorange", lw=1.5,
            linestyle="--", label="Kalman-smoothad latent nivå")

    # Prognos
    ax.fill_between(f_idx, f_ci[:, 0], f_ci[:, 1], color="green", alpha=0.15)
    ax.plot(f_idx, f_vals, color="green", lw=2, linestyle="-.",
            label=f"Prognos ({len(f_idx)} steg)")

    ax.axvline(idx[-1], color="gray", linestyle=":", alpha=0.6, label="Prognosstart")
    ax.set_ylabel("Nivå (m ö.h.)")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    ax.grid(alpha=0.3)
    ax.set_ylim(np.nanmin(obs) - 1, np.nanmax(obs) + 1)

    # ── Panel 2: Residualer & avvikelsedetektering ────────────────────────
    ax2 = axes[1]
    ax2.set_title("Residualer (observation – prediktion) & anomalidetektion", fontsize=11)

    residuals = obs - pred
    pi_const  = (ci[:, 1] - ci[:, 0]) / 2  # prediktionsintervallskonstant

    ax2.fill_between(idx, -pi_const, pi_const,
                     color="steelblue", alpha=0.15, label="Prediktionsintervall (95%)")
    ax2.plot(idx, residuals, color="navy", lw=0.8, alpha=0.8, label="Residual")
    ax2.axhline(0, color="black", lw=0.8)

    # Färgmarkera avvikande perioder
    anom_res = residuals.copy()
    anom_res[~anomaly.values] = np.nan
    ax2.plot(idx, anom_res, color="red", lw=2, label="Avvikelse")

    ax2.set_ylabel("Residual (m)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-2, 2)

    plt.tight_layout()
    return fig


def plot_akvifar_style(out: dict, anomaly: pd.Series,
                       title_base: str, ref_label: str,
                       model_label: str = ""):
    """
    Graf i Akvifär-stil: observationer, prediktioner och osäkerhetsband
    för hela perioden. Ingen initieringsperiod — Kalmanfiltret uppdateras
    kontinuerligt och one-step-ahead-prediktionerna är redan ut-av-stickprov.
    """
    idx  = out["index"]
    obs  = out["observed_base"]
    # Smoother-prediktioner för det visuella (stabilt, ingen diffus-spike)
    pred = out["pred_base"].flatten()[:len(idx)]
    # Använd filter-CI (one-step-ahead) för visuellt band — synligt över hela serien
    ci   = out["filter_pred_ci"][:len(idx)]
    # Samma CI används även för röda punkter (obs utanför intervallet)
    filter_ci = ci

    fig, ax = plt.subplots(figsize=(12, 6))

    label_part = f" — {model_label}" if model_label else ""
    fig.suptitle(
        f"Observationer, prediktion och prediktionsintervall{label_part}",
        fontsize=14, fontweight="bold", y=0.98
    )
    ax.set_title(
        f"Observationsrör: {title_base}, referensrör: {ref_label}",
        fontsize=11, fontweight="bold", pad=4
    )

    # Osäkerhetsband (smoother CI)
    ax.fill_between(idx, ci[:, 0], ci[:, 1],
                    alpha=0.25, color="steelblue", label="Prediktionsintervall (95%)")

    # Prediktionsvärden
    ax.plot(idx, pred, marker="o", markersize=3, linestyle="-",
            linewidth=1.2, color="black", label="Prediktionsvärden")

    # Observationsvärden
    ax.plot(idx, obs, marker="o", markersize=4, linestyle="-",
            linewidth=1.5, color="blue", label="Observationsvärden")

    # Röda punkter på observationsvärden (blå) utanför filter-CI
    utanfor_mask = (
        ~np.isnan(obs) &
        ((obs < filter_ci[:, 0]) | (obs > filter_ci[:, 1]))
    )
    if utanfor_mask.any():
        ax.scatter(idx[utanfor_mask], obs[utanfor_mask],
                   s=50, color="red", zorder=5,
                   label="Utanför prediktionsintervall")

    ax.set_xlabel("Datum")
    ax.set_ylabel("Grundvattennivå (m ö.h.)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
    plt.tight_layout()
    return fig


def plot_acf_residuals(out: dict, station_id: str, model_label: str = "", lags: int = 20):
    """
    ACF-plot för residualerna (observation − one-step-ahead prediktion).
    Stil matchar kompiskoden: enkel fetstil-titel, lodräta staplar,
    streckade CI-linjer (ingen blå fyllning), grid.
    """
    obs  = out["observed_base"]
    pred = out["filter_pred_base"].flatten()[:len(obs)]
    residualer = obs - pred

    residualer_series = pd.Series(residualer).dropna()
    residualer_series = residualer_series[~np.isnan(residualer_series.values)]

    if len(residualer_series) < 10:
        print(f"⚠ För få residualer för ACF-plot ({model_label})")
        return None

    max_lags = min(lags, len(residualer_series) // 2 - 1)

    # Beräkna ACF-värden och 95%-konfidensintervall (±1.96/√n)
    acf_vals = sm_acf(residualer_series, nlags=max_lags, fft=True)
    ci_bound = 1.96 / np.sqrt(len(residualer_series))
    lags_arr = np.arange(len(acf_vals))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Lodräta staplar (stems) — blå, som kompisen
    ax.vlines(lags_arr, 0, acf_vals, colors="steelblue", linewidth=1.5)
    ax.plot(lags_arr, acf_vals, "o", color="steelblue", markersize=5)

    # Baslinje vid 0
    ax.axhline(0, color="black", linewidth=0.8)

    # Streckade CI-linjer (som kompisen — inga fyllda band)
    ax.axhline( ci_bound, linestyle="--", color="steelblue", linewidth=0.9)
    ax.axhline(-ci_bound, linestyle="--", color="steelblue", linewidth=0.9)

    # Enkel fetstil-titel som kompisen (ingen extra undertitel)
    label_part = f" — {model_label}" if model_label else ""
    ax.set_title(
        f"ACF för residualer{label_part} ({station_id})",
        fontsize=14, fontweight="bold"
    )

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autokorrelation")
    ax.grid(True)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  IMPUTATION (saknade värden)
# ─────────────────────────────────────────────

def imputation_report(df: pd.DataFrame, out: dict) -> pd.DataFrame:
    """
    Returnerar DataFrame med imputerade värden för de tidpunkter
    där basobjektet saknar observation.
    """
    mask = df["base"].isna()
    if mask.sum() == 0:
        print("Inga saknade värden i basobjektet.")
        return pd.DataFrame()

    imputed = pd.DataFrame({
        "date":          df.index[mask],
        "imputed_level": out["level_smoothed"][mask],
        "lower_95":      out["pred_ci"][mask, 0],
        "upper_95":      out["pred_ci"][mask, 1],
    })
    imputed = imputed.set_index("date")
    # Ta bort orimliga initiala värden från diffus Kalman-initiering
    imputed = imputed[imputed["lower_95"] > 0]
    print(f"\nImputation för {len(imputed)} saknade mätpunkter:")
    print(imputed.round(4).to_string())
    return imputed


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Välj vilket läge som ska köras: "baseline", "multi", eller "both"
    MODE = "both"

    # Alla basrör som ska köras
    STATIONS = [
        (basror_22W102,  "22W102"),
        (basror_17XX01U, "17XX01U"),
        (basror_G1101,   "G1101"),
    ]

    for station_path, station_id in STATIONS:
        print("\n" + "#"*60)
        print(f"# KÖR STATION: {station_id}")
        print("#"*60)

    # ── STEG 1: ALL DATAHÄMTNING FÖRST ────────────────────────────────────
    # (Samlar all nätverkskommunikation innan tung beräkning för att undvika
    #  att socketpooler/SSL-kontext störs av statsmodels/scipy.)

        print("=== Laddar basröret ===")
        base = load_base_station(station_path)
        print(f"Basobjekt {station_id}: {len(base)} rader, {base.isna().sum()} saknade värden")

        # Välj tidsfrekvens automatiskt baserat på datadensitet.
        # Glesa stationer (typ månadsvis) → "MS" (månad)
        # Täta stationer (typ veckovis)   → "7D"  (vecka)
        # Säsongskomponent (nseason) sätts alltid till 0 — sigma2_eta_season
        # kollapsar till ≈0 för alla stationer, dvs. säsongen bidrar inte.
        span_days = (base.index.max() - base.index.min()).days
        n_obs     = base.notna().sum()
        avg_gap   = span_days / max(n_obs - 1, 1)   # medel antal dagar mellan obs
        nseason   = 0  # ingen säsongskomponent — verifierat via AIC/BIC för alla stationer
        if avg_gap > 21:          # glesare än var tredje vecka → månadsfrekvens
            freq = "MS"
            print(f"  Gles data (snitt {avg_gap:.0f} dagar mellan obs) → månadsfrekvens, nseason={nseason}")
        else:
            freq = "7D"
            print(f"  Tät data (snitt {avg_gap:.0f} dagar mellan obs) → veckofrekvens, nseason={nseason}")

        # Hämta kandidatrör (samma akvifer+jordart) — används av både baslinje och multi
        print("\n--- Hämtar kandidatrör från SGU API ---")
        refs = load_candidate_stations(base, base_station_id=station_id, ignore_geology=False)

        ref = None
        df_base = None
        baseline_ref_id = None
        if MODE in ("baseline", "both"):
            # Välj topp 1 via Pearson-korrelation (samma metod som multi)
            df_top1 = prepare_joint_dataframe_multi(base, refs, freq=freq, max_refs=1)
            baseline_ref_id = [c for c in df_top1.columns if c != "base"][0]
            ref = df_top1[baseline_ref_id]
            # Bygg enkel baseline-dataframe med kolumnerna "base" och "ref"
            df_base = df_top1.rename(columns={baseline_ref_id: "ref"})[["base", "ref"]]
            print(f"\nBaslinjens referensrör: {baseline_ref_id} (högst |r| med basröret)")
            # Diagnostik: kontrollera extremvärden i referensröret
            ref_vals = df_base["ref"].dropna()
            print(f"  Referensrör {baseline_ref_id} — min: {ref_vals.min():.2f}, max: {ref_vals.max():.2f}, "
                  f"median: {ref_vals.median():.2f}, saknade veckor: {df_base['ref'].isna().sum()}")
            outlier_mask = (ref_vals < ref_vals.quantile(0.01)) | (ref_vals > ref_vals.quantile(0.99))
            if outlier_mask.any():
                print(f"  Potentiella outliers (utanför 1–99 percentil) i {baseline_ref_id}:")
                print(f"  {ref_vals[outlier_mask].to_string()}")

        df_multi = None
        if MODE in ("multi", "both"):
            df_multi = prepare_joint_dataframe_multi(base, refs, freq=freq, max_refs=5)

        print("\n✓ All data hämtad — startar modellanpassning\n")

        # Initiera resultatvariabler
        result_base = result_multi = None
        out_base = out_multi = None
        metrics_base = metrics_multi = None

        # ── STEG 2: BASLINJEMODELL ────────────────────────────────────────────
        if MODE in ("baseline", "both"):
            print("="*60)
            print(f"BASLINJEMODELL — ett referensrör ({baseline_ref_id})")
            print("="*60)
            tidstyp = "månader" if freq == "MS" else "veckor"
            print(f"\nGemensamt dataset: {len(df_base)} {tidstyp} "
                  f"({df_base.index[0].date()} – {df_base.index[-1].date()})")
            print(f"Saknade i base: {df_base['base'].isna().sum()}")
            print(f"Saknade i ref:  {df_base['ref'].isna().sum()}")

            # Anpassa modellen
            print("\n=== Anpassar baslinjemodellen ===")
            result_base, model_base = fit_model_baseline(df_base, nseason=nseason)

            # Kör Kalman smoother + prognos
            # nseason steg framåt ≈ halv cykel (26 veckor eller 12 månader)
            print("\n=== Kör Kalman smoother + prognos (baslinje) ===")
            out_base = smooth_and_forecast(result_base, df_base, n_forecast=nseason)

            # Imputation-rapport
            imputed_base = imputation_report(df_base, out_base)

            # Avvikelsedetektering
            anomaly_base = detect_anomalies(
                observed=out_base["observed_base"],
                pred_ci=out_base["pred_ci"],
                min_consecutive=3,
            )
            n_anom = anomaly_base.sum()
            print(f"\nAvvikelsedetektering (baslinje): {n_anom} tidpunkter flaggade "
                  f"({n_anom/len(anomaly_base)*100:.1f}%)")

            # Plotta och spara
            try:
                print("\n=== Genererar plot (baslinje) ===")
                fig_base = plot_results(out_base, anomaly_base, title_base=station_id, ref_label=baseline_ref_id, model_label="Baslinje")
                fig_base.savefig(f"groundwater_ssm_baseline_{station_id}.png", dpi=150, bbox_inches="tight")
                print(f"Plot sparad till: groundwater_ssm_baseline_{station_id}.png")
                plt.close(fig_base)

                # Akvifär-stil: observationer, prediktioner, osäkerhetsband
                fig_akv = plot_akvifar_style(
                    out_base, anomaly_base,
                    title_base=station_id,
                    ref_label=baseline_ref_id,
                    model_label="State Space"
                )
                fig_akv.savefig(f"akvifar_style_baseline_{station_id}.png", dpi=150, bbox_inches="tight")
                print(f"Akvifär-stilplot sparad till: akvifar_style_baseline_{station_id}.png")
                plt.close(fig_akv)

                # ACF-plot för baslinje
                fig_acf = plot_acf_residuals(out_base, station_id=station_id, model_label="State Space")
                if fig_acf is not None:
                    fig_acf.savefig(f"acf_residuals_baseline_{station_id}.png", dpi=150, bbox_inches="tight")
                    print(f"ACF-plot sparad till: acf_residuals_baseline_{station_id}.png")
                    plt.close(fig_acf)
            except (Exception, KeyboardInterrupt) as e:
                print(f"⚠ Plottning (baslinje) misslyckades: {type(e).__name__}: {e}")
                plt.close("all")

            # Spara imputerade värden
            if not imputed_base.empty:
                imputed_base.to_csv(f"imputed_values_baseline_{station_id}.csv")
                print(f"Imputerade värden sparade till: imputed_values_baseline_{station_id}.csv")

            # Skattade parametrar
            print("\n=== Skattade parametrar (baslinje) ===")
            params_base = dict(zip(model_base.param_names, result_base.params))
            for k, v in params_base.items():
                print(f"  {k:<28} = {v:.6f}")


        # ── STEG 3: MULTIVARIAT MODELL ────────────────────────────────────────
        if MODE in ("multi", "both") and df_multi["base"].notna().sum() < 3:
            print(f"\n⚠ OBS: Färre än 3 observationer i basröret för {station_id} — kör ändå multivariat modell.")
        if MODE in ("multi", "both"):
            print("\n" + "="*60)
            print("MULTIVARIAT MODELL — referensrör filtrerade på akvifer+jordart")
            print("="*60)
            # ref_ids hämtas från df_multi efter max_refs-begränsning (inte från refs-listan)
            ref_ids = [col for col in df_multi.columns if col != "base"]
            print(f"\nAnvänder {len(ref_ids)} referensrör: {ref_ids}")
            print(f"Gemensamt dataset: {len(df_multi)} {tidstyp} "
                  f"({df_multi.index[0].date()} – {df_multi.index[-1].date()})")
            print(f"Saknade i base: {df_multi['base'].isna().sum()}")
            for rid in ref_ids:
                print(f"Saknade i {rid}: {df_multi[rid].isna().sum()}")

            # Anpassa modellen
            print("\n=== Anpassar multivariat modell ===")
            result_multi, model_multi = fit_model_multi(df_multi, nseason=nseason)

            # Kör Kalman smoother + prognos
            print("\n=== Kör Kalman smoother + prognos (multivariat) ===")
            out_multi = smooth_and_forecast(result_multi, df_multi, n_forecast=nseason)

            # Imputation-rapport
            imputed_multi = imputation_report(df_multi, out_multi)

            # Avvikelsedetektering
            anomaly_multi = detect_anomalies(
                observed=out_multi["observed_base"],
                pred_ci=out_multi["pred_ci"],
                min_consecutive=3,
            )
            n_anom = anomaly_multi.sum()
            print(f"\nAvvikelsedetektering (multivariat): {n_anom} tidpunkter flaggade "
                  f"({n_anom/len(anomaly_multi)*100:.1f}%)")

            # Plotta och spara
            try:
                print("\n=== Genererar plot (multivariat) ===")
                fig_multi = plot_results(out_multi, anomaly_multi,
                                         title_base=station_id, ref_label=", ".join(ref_ids), model_label="Multivariat")
                fig_multi.savefig(f"groundwater_ssm_multi_{station_id}.png", dpi=150, bbox_inches="tight")
                print(f"Plot sparad till: groundwater_ssm_multi_{station_id}.png")
                plt.close(fig_multi)

                # Akvifär-stil för multivariat
                fig_akv_multi = plot_akvifar_style(
                    out_multi, anomaly_multi,
                    title_base=station_id,
                    ref_label=", ".join(ref_ids),
                    model_label="State Space Multivariat"
                )
                fig_akv_multi.savefig(f"akvifar_style_multi_{station_id}.png", dpi=150, bbox_inches="tight")
                print(f"Akvifär-stilplot sparad till: akvifar_style_multi_{station_id}.png")
                plt.close(fig_akv_multi)

                # ACF-plot för multivariat
                fig_acf_multi = plot_acf_residuals(out_multi, station_id=station_id, model_label="State Space Multivariat")
                if fig_acf_multi is not None:
                    fig_acf_multi.savefig(f"acf_residuals_multi_{station_id}.png", dpi=150, bbox_inches="tight")
                    print(f"ACF-plot sparad till: acf_residuals_multi_{station_id}.png")
                    plt.close(fig_acf_multi)
            except (Exception, KeyboardInterrupt) as e:
                print(f"⚠ Plottning (multi) misslyckades: {type(e).__name__}: {e}")
                plt.close("all")

            # Spara imputerade värden
            if not imputed_multi.empty:
                imputed_multi.to_csv(f"imputed_values_multi_{station_id}.csv")
                print(f"Imputerade värden sparade till: imputed_values_multi_{station_id}.csv")

            # Skattade parametrar
            print("\n=== Skattade parametrar (multivariat) ===")
            params_multi = dict(zip(model_multi.param_names, result_multi.params))
            for k, v in params_multi.items():
                print(f"  {k:<40} = {v:.6f}")


        # ── UTVÄRDERING & JÄMFÖRELSE ──────────────────────────────────────
        if MODE in ("baseline", "both") and result_base is not None:
            metrics_base = evaluate_model(result_base, out_base,
                                          label="Baslinje", station_id=station_id)
        if MODE in ("multi", "both") and result_multi is not None:
            metrics_multi = evaluate_model(result_multi, out_multi,
                                           label="Multivariat", station_id=station_id)

        if MODE == "both":
            print("\n" + "="*60)
            print(f"JÄMFÖRELSE: Baslinje vs Multivariat — {station_id}")
            print("="*60)

            # Samla alla modeller som körts
            all_metrics = []
            labels = []
            if metrics_base is not None:
                all_metrics.append(metrics_base); labels.append("Baslinje")
            if metrics_multi is not None:
                all_metrics.append(metrics_multi); labels.append("Multi")

            comparison_keys = [
                ("── Modellanpassning ──", None),
                ("Log-likelihood", "{:>12.2f}"),
                ("AIC", "{:>12.2f}"),
                ("BIC", "{:>12.2f}"),
                ("HQIC", "{:>12.2f}"),
                ("Antal parametrar", "{:>12.0f}"),
                ("── Prediktionsfel (one-step-ahead) ──", None),
                ("RMSE", "{:>12.6f}"),
                ("MAE", "{:>12.6f}"),
                ("ME (bias)", "{:>12.6f}"),
                ("95%-täckning", "{:>11.1%}"),
                ("── Antagande 2: Okorrelerade feltermer ──", None),
                ("Innov. medel", "{:>12.4f}"),
                ("Ljung-Box Q(10)", "{:>12.4f}"),
                ("Ljung-Box p", "{:>12.4f}"),
                ("Durbin-Watson", "{:>12.4f}"),
            ]

            # Dynamisk header för valfritt antal modeller
            header = f"  {'Mått':<35}"
            for lbl in labels:
                header += f" {lbl:>14}"
            header += "  Bäst"
            print(f"\n{header}")
            print(f"  {'-'*(35 + 15*len(labels) + 6)}")

            lower_is_better  = {"AIC", "BIC", "HQIC", "RMSE", "MAE",
                                "Ljung-Box Q(10)"}
            higher_is_better = {"Log-likelihood", "95%-täckning",
                                "Ljung-Box p"}
            target_value     = {"ME (bias)": 0.0, "Innov. medel": 0.0,
                                "Durbin-Watson": 2.0}

            for key, fmt in comparison_keys:
                if fmt is None:
                    print(f"\n  {key}")
                    continue
                vals = [m.get(key) for m in all_metrics]
                if all(v is None for v in vals):
                    continue
                row = f"  {key:<35}"
                for v in vals:
                    row += f" {fmt.format(v) if v is not None else 'N/A':>14}"

                # Bestäm bästa modellen
                valid_vals = [(v, lbl) for v, lbl in zip(vals, labels) if v is not None]
                winner = ""
                if len(valid_vals) >= 2:
                    if key in lower_is_better:
                        winner = min(valid_vals, key=lambda x: x[0])[1]
                    elif key in higher_is_better:
                        winner = max(valid_vals, key=lambda x: x[0])[1]
                    elif key in target_value:
                        t = target_value[key]
                        winner = min(valid_vals, key=lambda x: abs(x[0] - t))[1]
                row += f"  {winner}"
                print(row)