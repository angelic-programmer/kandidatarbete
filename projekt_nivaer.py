import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.statespace.mlemodel import MLEModel
except ImportError:
    raise ImportError("Installera statsmodels: pip install statsmodels")


basror_22W102 = Path(__file__).parent / "22W102.csv"
metadata_path = Path(__file__).parent / "stationer_metadata.csv"

# Vi skapar en dataframe i pandas av basror_22W102
def load_base_station(filepath: Path = basror_22W102) -> pd.Series:
    # Här är en funktion som tar en filsökväg som argument, basror_22W102 är defaultvärdet
    df = pd.read_csv(
        filepath,
        sep=";",
        header=None,
        names=["date", "level"],
        encoding="utf-8-sig",
    )
    # Nedan läser vi in csv-filen till en DataFrame med kolumnerna date och level
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
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
    return df["level"]


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
    base_station_id: str = "22W102",
    meta_path: Path = metadata_path,
    min_overlap_weeks: int = 52,
) -> list:
    """
    Läser metadata för basröret, hämtar stationer med samma akvifer+jordart
    från SGU OGC API, och returnerar lista med (station_id, pd.Series)
    för rör med tillräcklig överlappning med basröret.
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

    # Hämta stationer från SGU OGC API filtrerat på akvifer och jordart
    url  = ("https://api.sgu.se/oppnadata/grundvattennivaer-observerade"
            "/ogc/features/v1/collections/stationer/items")
    resp = requests.get(url, params={"akvifer": target_akvifer, "jordart": target_jordart,
                                     "limit": 500, "f": "json"}, timeout=30, verify=False)
    resp.raise_for_status()
    features   = resp.json().get("features", [])

    # Plocka ut station-ID för alla träffar utom basröret självt
    candidates = [
        f["properties"]["platsbeteckning"]
        for f in features
        if f["properties"].get("platsbeteckning") not in (None, base_station_id)
    ]
    print(f"SGU API returnerade {len(candidates)} kandidatrör "
          f"(akvifer={target_akvifer}, jordart={target_jordart})")

    # Hämta tidsserie för varje kandidat och filtrera på överlappning
    base_start = base_series.dropna().index.min()
    base_end   = base_series.dropna().index.max()
    result     = []

    max_download = 10

    # Läs från cache (ref_cache.json) — kör fetch_candidates.py separat om cachen saknas
    cache_path = Path(__file__).parent / "ref_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache saknas: {cache_path}\n"
            "Kör först: py fetch_candidates.py"
        )
    import json
    print(f"  Läser cache: {cache_path}")
    with open(cache_path) as f:
        all_data = json.load(f)
    for sid, records in all_data.items():
        if "__error__" in records:
            continue
        s = pd.Series({pd.Timestamp(k): v for k, v in records.items()}, dtype=float)
        s = s.sort_index()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s = pd.to_numeric(s, errors="coerce")
        overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna()
        if len(overlap) >= min_overlap_weeks:
            result.append((sid, s))
            print(f"  ✓ {sid}: {len(overlap)} överlappande obs (cache)")
        if len(result) >= max_download:
            break

    # Om inga kandidater hittades, fall tillbaka på 95_2
    if not result:
        print("\nVarning: inga kandidatrör hittades — faller tillbaka på 95_2")
        ref = load_reference_station("95_2")
        return [("95_2", ref)]

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

    # Skapar en tom tabell och fyller i närmaste mätning inom 4 dagars tolerans
    df = pd.DataFrame(index=idx)
    df["base"] = base.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))
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
    max_refs: int = 4,
) -> pd.DataFrame:
    # Ta bort tidszon om den finns
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)

    # ── Ranka referensrör efter Pearson-korrelation med basröret ──────
    # (istället för enbart överlappningsantal)
    base_w = _resample_to_weekly(base, freq=freq)
    corr_list = []
    for sid, s in refs:
        s_w = _resample_to_weekly(s, freq=freq)
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

    # Här använder jag endast den överlappande perioden där alla tidsserier har data
    start = base.index.min()
    end   = base.index.max()
    for _, s in refs_weekly:
        start = max(start, s.index.min())
        end   = min(end,   s.index.max())

    # Skapar ett regelbundet datumindex med en rad per vecka, från start till slut
    idx = pd.date_range(start, end, freq=freq)

    # Skapar en tom tabell och fyller i närmaste mätning inom 4 dagars tolerans
    df = pd.DataFrame(index=idx)
    df["base"] = base.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))
    for sid, s in refs_weekly:
        df[sid] = s.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))

    return df


# ─────────────────────────────────────────────
#  BASLINJEMODELL (ett referensrör: 95_2)
# ─────────────────────────────────────────────

#vi skapar en ny "klass" i python där vi tar med ett färdigt ramverk för state space-modeller och bara behöver specificera just vår modell
#VAR KOMMER det färdiga ramverket ifrån?
class GroundwaterSSM(MLEModel):

#Nedan är en lista med namnen på de 4 parametrar som modellen ska skatta
#alla sigma2 variabler skattas utifrån basröret
#sigma2 epsilon (sigma2 eps) skattar mätfelet för referensröret
#sigma2_eps_base är låst till 0 (basrörets mätningar antas vara exakta)
#trend-komponenten är borttagen: µ(t+1) = µ(t) + η_level(t)
    param_names = [
        "sigma2_eta_level",   # process noise: nivå
        "sigma2_eta_season",  # process noise: säsong
        "sigma2_eps_ref",     # obs noise: referensobjekt
        "beta_ref",           # koppling nivå → referens
    ]

    def __init__(self, endog, nseason=26, **kwargs):
        self.nseason = nseason
        #nedan beräknas antalet states med 26 st säsonger blir det 26+1(nivå) = 27 states
        k_states = 1 + nseason
        # Bara 2 chockkällor (nivå + säsong) — resten av states är deterministiska skift
        k_posdef = 2

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        # Diffus initiering för nivå (okänt startvärde)
        self.ssm.initialize_approximate_diffuse()

    @property
    def start_params(self):
        import statsmodels.api as sm

        y1 = self.endog[:, 0]  # basröret 22W102
        y2 = self.endog[:, 1]  # referensröret 95_2

        # Steg 1-3: Anpassa univariat strukturmodell till basröret
        # Detta ger oss databaserade startgissningar för varianser och säsong
        mod = sm.tsa.UnobservedComponents(
            y1, level="local level", seasonal=self.nseason
        )
        res = mod.fit(disp=False, method="nm", maxiter=500)

        sigma2_eta_level  = float(res.params[1])  # process noise nivå
        sigma2_eta_season = float(res.params[2])  # process noise säsong

        # Steg 5: OLS för laddningsparameter beta_ref
        level     = res.level.smoothed
        season    = res.seasonal.smoothed
        y2_deseas = y2 - season

        # mask använder level (samma längd som y2_deseas)
        mask = ~(np.isnan(y2_deseas) | np.isnan(level))
        X    = level[mask].reshape(-1, 1)
        ols  = np.linalg.lstsq(X, y2_deseas[mask], rcond=None)
        beta_ref       = float(ols[0][0])
        sigma2_eps_ref = float(np.var(y2_deseas[mask] - X.flatten() * beta_ref))

        return np.array([
            sigma2_eta_level,    # process noise: nivå
            sigma2_eta_season,   # process noise: säsong
            sigma2_eps_ref,      # obs noise: referensröret
            beta_ref,            # koppling nivå → referens
        ])

#FORTSÄTT FÖRKLARA DENNA KOD FRÅN OCH MED IMORGON 

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        # Varianser: exp-transform → alltid positiva
        p[:3] = np.exp(unconstrained[:3])
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[:3] = np.log(constrained[:3])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        (s2_level, s2_season,
         s2_ref, beta_ref) = params

        ns = self.nseason
        k  = self.k_states

        # ── Transitionsmatris T ──────────────────────────────────────────
        T = np.zeros((k, k))

        # Nivå: µ(t+1) = µ(t)  (random walk utan trend)
        T[0, 0] = 1.0

        # Säsong med summavillkor:
        #   γ_0(t+1) = -γ_0(t) - γ_1(t) - ... - γ_{s-2}(t)
        #   γ_j(t+1) = γ_{j-1}(t)  för j=1..s-1
        T[1, 1:1+ns] = -1.0
        for j in range(1, ns):
            T[1 + j, 1 + j - 1] = 1.0

        self.ssm["transition"] = T

        # ── Observationsmatris Z ─────────────────────────────────────────
        # y_base = µ + γ_0          (direkt nivå + säsong)
        # y_ref  = beta*µ            (referens kopplad via nivå)
        Z = np.zeros((2, k))
        Z[0, 0] = 1.0      # base ← nivå
        Z[0, 1] = 1.0      # base ← säsong
        Z[1, 0] = beta_ref # ref  ← nivå

        self.ssm["design"] = Z

        # ── Observationsbrus ────────────────────────────────────────────
        # sigma2_eps_base är låst till 0: basrörets mätningar antas exakta
        self.ssm["obs_cov"] = np.diag([0.0, s2_ref])

        # ── Processkörsbrus Q (2×2) ─────────────────────────────────────
        Q = np.array([[s2_level, 0.0],
                      [0.0, s2_season]])
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris R (k×2): kopplar chockerna till rätt state ─
        R = np.zeros((k, 2))
        R[0, 0] = 1.0  # η_level  → state 0 (nivå)
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

        #nedan beräknas antalet states med 26 st säsonger blir det 26+1(nivå) = 27 states
        k_states = 1 + nseason
        # Bara 2 chockkällor (nivå + säsong) — resten av states är deterministiska skift
        k_posdef = 2

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        # param_names sätts dynamiskt: 2 process noise + 4 parametrar per referensrör
        # sparas som _param_names eftersom MLEModel redan har param_names som property
        self._param_names = ["sigma2_eta_level", "sigma2_eta_season"]
        for rid in self.ref_ids:
            self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}",
                                  f"alpha_{rid}", f"gamma_{rid}"]

        # Diffus initiering för nivå (okänt startvärde)
        self.ssm.initialize_approximate_diffuse()

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        import statsmodels.api as sm

        y1 = self.endog[:, 0]  # basröret 22W102

        # Steg 1-2: Anpassa univariat strukturmodell till basröret
        # Detta ger oss databaserade startgissningar för varianser och säsong
        mod = sm.tsa.UnobservedComponents(
            y1, level="local level", seasonal=self.nseason
        )
        res = mod.fit(disp=False, method="nm", maxiter=500)

        sigma2_eta_level  = float(res.params[1])  # process noise nivå
        sigma2_eta_season = float(res.params[2])  # process noise säsong

        params = [sigma2_eta_level, sigma2_eta_season]

        level  = res.level.smoothed
        season = res.seasonal.smoothed

        # Steg 3: OLS för varje referensrör: y_ref = alpha + beta*level + gamma*season
        for i in range(self.n_refs):
            y_ref = self.endog[:, 1 + i]
            mask  = ~(np.isnan(y_ref) | np.isnan(level) | np.isnan(season))
            X     = np.column_stack([np.ones(mask.sum()), level[mask], season[mask]])
            ols   = np.linalg.lstsq(X, y_ref[mask], rcond=None)
            alpha_i  = float(ols[0][0])
            beta_i   = float(ols[0][1])
            gamma_i  = float(ols[0][2])
            s2_eps_i = float(np.var(y_ref[mask] - X @ ols[0]))
            params  += [max(s2_eps_i, 1e-6), beta_i, alpha_i, gamma_i]

        return np.array(params)

#FORTSÄTT FÖRKLARA DENNA KOD FRÅN OCH MED IMORGON 

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        # Varianser: exp-transform → alltid positiva
        p[0] = np.exp(unconstrained[0])   # sigma2_eta_level
        p[1] = np.exp(unconstrained[1])   # sigma2_eta_season
        for i in range(self.n_refs):
            p[2 + 4*i] = np.exp(unconstrained[2 + 4*i])  # sigma2_eps_ref_i
            # beta, alpha, gamma (index 3+4i, 4+4i, 5+4i) obegränsade
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0])
        p[1] = np.log(constrained[1])
        for i in range(self.n_refs):
            p[2 + 4*i] = np.log(constrained[2 + 4*i])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        s2_level  = params[0]
        s2_season = params[1]
        # Plocka ut (sigma2_eps_i, beta_i, alpha_i, gamma_i) för varje referensrör
        ref_params = [
            (params[2 + 4*i], params[3 + 4*i], params[4 + 4*i], params[5 + 4*i])
            for i in range(self.n_refs)
        ]

        ns      = self.nseason
        k       = self.k_states
        n_endog = 1 + self.n_refs

        # ── Transitionsmatris T ──────────────────────────────────────────
        T = np.zeros((k, k))

        # Nivå: µ(t+1) = µ(t)  (random walk utan trend)
        T[0, 0] = 1.0

        # Säsong med summavillkor:
        #   γ_0(t+1) = -γ_0(t) - γ_1(t) - ... - γ_{s-2}(t)
        #   γ_j(t+1) = γ_{j-1}(t)  för j=1..s-1
        T[1, 1:1+ns] = -1.0
        for j in range(1, ns):
            T[1 + j, 1 + j - 1] = 1.0

        self.ssm["transition"] = T

        # ── Observationsmatris Z ─────────────────────────────────────────
        # y_base  = µ + γ_0                              (direkt nivå + säsong)
        # y_ref_i = alpha_i + beta_i*µ + gamma_i*γ_0     (intercept + nivå + säsong)
        Z = np.zeros((n_endog, k))
        Z[0, 0] = 1.0   # base ← nivå
        Z[0, 1] = 1.0   # base ← säsong
        for i, (_, beta_i, _, gamma_i) in enumerate(ref_params):
            Z[1 + i, 0] = beta_i   # ref_i ← beta_i * nivå
            Z[1 + i, 1] = gamma_i  # ref_i ← gamma_i * säsong

        self.ssm["design"] = Z

        # ── Observationsintercept (d-vektor) ─────────────────────────────
        # alpha_i fångar skillnad i absolut nivå mellan referensrör och basrör
        d = np.zeros((n_endog, 1))
        for i, (_, _, alpha_i, _) in enumerate(ref_params):
            d[1 + i, 0] = alpha_i
        self.ssm["obs_intercept"] = d

        # ── Observationsbrus ────────────────────────────────────────────
        # sigma2_eps_base är låst till 0: basrörets mätningar antas exakta
        obs_cov = np.zeros((n_endog, n_endog))
        for i, (s2_eps_i, _, _, _) in enumerate(ref_params):
            obs_cov[1 + i, 1 + i] = s2_eps_i
        self.ssm["obs_cov"] = obs_cov

        # ── Processkörsbrus Q (2×2) ─────────────────────────────────────
        Q = np.array([[s2_level, 0.0],
                      [0.0, s2_season]])
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris R (k×2): kopplar chockerna till rätt state ─
        R = np.zeros((k, 2))
        R[0, 0] = 1.0  # η_level  → state 0 (nivå)
        R[1, 1] = 1.0  # η_season → state 1 (γ_0)
        self.ssm["selection"] = R


# ─────────────────────────────────────────────
#  ANPASSA OCH PREDIKTERA
# ─────────────────────────────────────────────

def fit_model_baseline(df: pd.DataFrame, nseason: int = 26) -> tuple:
    """Anpassar baslinjemodellen (ett referensrör) och returnerar (result, model)."""
    endog = df[["base", "ref"]].values.astype(float)

    model = GroundwaterSSM(endog, nseason=nseason)

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
    season_var = smoothed.smoothed_state_cov[1, 1]
    total_var  = level_var + season_var
    std        = np.sqrt(np.maximum(total_var, 0))
    pred_mean  = level_smoothed + smoothed.smoothed_state[1]
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
    fcast_idx = pd.date_range(
        last_date + pd.Timedelta("7D"),
        periods=n_forecast,
        freq="7D",
    )

    return {
        "index":          df.index,
        "observed_base":  df["base"].values,
        "level_smoothed": level_smoothed,
        "level_var":      level_var,
        "season_var":     season_var,
        "pred_base":      pred_mean,
        "pred_ci":        pred_ci,
        "fcast_index":    fcast_idx,
        "fcast_base":     fcast_mean,
        "fcast_ci":       fcast_ci,
    }


# ─────────────────────────────────────────────
#  MODELLDIAGNOSTIK / UTVÄRDERING
# ─────────────────────────────────────────────

def evaluate_model(result, out: dict, label: str = "Modell") -> dict:
    """
    Beräknar diagnostik- och utvärderingsmått för en state-space-modell.

    Returnerar en dict med alla mått.
    Hanterar både univariata (baseline) och multivariata modeller.

    Mått som beräknas:
      ── Informationskriterier ──
        Log-likelihood, AIC, BIC, HQIC
      ── Prediktionsfel (basröret, Kalman-filter one-step-ahead) ──
        RMSE, MAE, ME (bias)
      ── Standardiserade innovationer (Kalman-filter) ──
        Medelvärde, Std, Skevhet, Kurtosis  (bör vara ≈ N(0,1))
      ── Restdiagnostik ──
        Ljung-Box Q (lag 10), Durbin-Watson
      ── Prediktionsintervall ──
        95%-täckningsgrad (andel obs inom PI)
      ── Latenta variabler (SSM-specifikt) ──
        Medel osäkerhet i latent nivå µ(t)
        Medel osäkerhet i säsong γ₀(t)
    """
    from scipy import stats as sp_stats

    obs  = out["observed_base"]
    pred = out["pred_base"]
    ci   = out["pred_ci"]

    # ── Mask: bara tidpunkter med faktisk observation ─────────────────
    valid = ~np.isnan(obs)
    o = obs[valid]
    p = pred[valid]

    # ── 1. Informationskriterier ─────────────────────────────────────
    metrics = {
        "Log-likelihood":    result.llf,
        "AIC":               result.aic,
        "BIC":               result.bic,
        "HQIC":              result.hqic,
        "Antal parametrar":  len(result.params),
        "Antal obs (T)":     int(result.nobs),
    }

    # ── 2. One-step-ahead prediktionsfel (Kalman-filtret) ────────────
    #   Filtrets innovation e(t) = y(t) - Z·a(t|t-1) beräknas INNAN
    #   observationen assimileras — ger meningsfull RMSE även med
    #   obs_cov = 0 (smoothade residualer ≡ 0 i det fallet).
    filter_resid = result.filter_results.forecasts_error[0]  # rad 0 = basröret
    fr_valid = filter_resid[valid & ~np.isnan(filter_resid)]
    metrics["RMSE"]       = float(np.sqrt(np.mean(fr_valid**2)))
    metrics["MAE"]        = float(np.mean(np.abs(fr_valid)))
    metrics["ME (bias)"]  = float(np.mean(fr_valid))

    # ── 3. Standardiserade innovationer från Kalman-filtret ──────────
    #   Dessa är e(t) / sqrt(F(t)) — bör vara ≈ N(0,1) om modellen
    #   är korrekt specificerad.  Rad 0 = basröret.
    try:
        std_innov = result.filter_results.standardized_forecasts_error
        si_base = std_innov[0]
        si_valid = si_base[~np.isnan(si_base)]
        if len(si_valid) > 10:
            metrics["Innov. medel"]   = float(np.mean(si_valid))
            metrics["Innov. std"]     = float(np.std(si_valid))
            metrics["Innov. skevhet"] = float(sp_stats.skew(si_valid))
            metrics["Innov. kurtosis"] = float(sp_stats.kurtosis(si_valid, fisher=True))
    except Exception:
        pass

    # ── 4. Ljung-Box test (lag 10) för autokorrelation i innovationer ─
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(fr_valid, lags=[10], return_df=True)
        metrics["Ljung-Box Q(10)"] = float(lb["lb_stat"].values[0])
        metrics["Ljung-Box p"]     = float(lb["lb_pvalue"].values[0])
    except Exception:
        pass

    # ── 5. Durbin-Watson (autokorrelation av ordning 1) ──────────────
    try:
        from statsmodels.stats.stattools import durbin_watson
        metrics["Durbin-Watson"] = float(durbin_watson(fr_valid))
    except Exception:
        pass

    # ── 6. Prediktionsintervallets täckningsgrad ─────────────────────
    ci_lower = ci[valid, 0]
    ci_upper = ci[valid, 1]
    inside   = (o >= ci_lower) & (o <= ci_upper)
    metrics["95%-täckning"] = float(inside.mean())

    # ── 7. Latenta variabler: genomsnittlig osäkerhet ────────────────
    #   Detta är SSM-specifikt — visar hur väl modellen identifierar
    #   de dolda tillstånden.  Lägre = bättre.
    level_std  = np.sqrt(np.maximum(out["level_var"], 0))
    season_std = np.sqrt(np.maximum(out["season_var"], 0))
    metrics["Latent nivå µ — medel-std"]   = float(np.mean(level_std))
    metrics["Latent säsong γ₀ — medel-std"] = float(np.mean(season_std))

    # Genomsnittlig total latent osäkerhet (nivå + säsong)
    total_std = np.sqrt(np.maximum(out["level_var"] + out["season_var"], 0))
    metrics["Latent total — medel-std"]    = float(np.mean(total_std))

    # ── Skriv ut ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIK: {label}")
    print(f"{'='*60}")

    print("\n── Informationskriterier ──")
    for k in ["Log-likelihood", "AIC", "BIC", "HQIC", "Antal parametrar", "Antal obs (T)"]:
        print(f"  {k:<30} = {metrics[k]:>12.4f}" if isinstance(metrics[k], float)
              else f"  {k:<30} = {metrics[k]:>12d}")

    print("\n── Prediktionsfel (basröret, one-step-ahead) ──")
    for k in ["RMSE", "MAE", "ME (bias)"]:
        print(f"  {k:<30} = {metrics[k]:>12.6f}")

    if "Innov. medel" in metrics:
        print("\n── Standardiserade innovationer (bör vara ≈ N(0,1)) ──")
        for k in ["Innov. medel", "Innov. std", "Innov. skevhet", "Innov. kurtosis"]:
            print(f"  {k:<30} = {metrics[k]:>12.4f}")

    if "Ljung-Box Q(10)" in metrics:
        print("\n── Autokorrelationstest ──")
        for k in ["Ljung-Box Q(10)", "Ljung-Box p", "Durbin-Watson"]:
            if k in metrics:
                print(f"  {k:<30} = {metrics[k]:>12.4f}")

    print("\n── Prediktionsintervall ──")
    print(f"  {'95%-täckning':<30} = {metrics['95%-täckning']:>11.1%}")

    print("\n── Latenta variabler (SSM-specifikt) ──")
    print("  Dessa variabler observeras aldrig direkt — de skattas av")
    print("  Kalman-smoothern.  Lägre osäkerhet = bättre identifiering.")
    for k in ["Latent nivå µ — medel-std", "Latent säsong γ₀ — medel-std",
              "Latent total — medel-std"]:
        print(f"  {k:<35} = {metrics[k]:>10.6f} m")

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
                 ref_label: str = "95_2"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "Grundvatten State Space Model\n"
        f"Basobjekt: {title_base} | Referens: {ref_label}",
        fontsize=14, fontweight="bold",
    )

    idx      = out["index"]
    obs      = out["observed_base"]
    pred     = out["pred_base"].flatten()[:len(idx)]
    ci       = out["pred_ci"]
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

    ax.plot(idx, obs, "o", color="navy", ms=3, alpha=0.7, label="Observation (22W102)")
    ax.plot(idx, pred, color="steelblue", lw=1.5, label="Predikterat (in-sample)")
    ax.plot(idx, smoothed, color="darkorange", lw=1.5,
            linestyle="--", label="Kalman-smoothad latent nivå")

    # Prognos
    ax.fill_between(f_idx, f_ci[:, 0], f_ci[:, 1], color="green", alpha=0.15)
    ax.plot(f_idx, f_vals, color="green", lw=2, linestyle="-.",
            label=f"Prognos ({len(f_idx)} veckor)")

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

    # ── STEG 1: ALL DATAHÄMTNING FÖRST ────────────────────────────────────
    # (Samlar all nätverkskommunikation innan tung beräkning för att undvika
    #  att socketpooler/SSL-kontext störs av statsmodels/scipy.)

    print("=== Laddar basröret ===")
    base = load_base_station()
    print(f"Basobjekt 22W102: {len(base)} rader, {base.isna().sum()} saknade värden")

    ref = None
    df_base = None
    if MODE in ("baseline", "both"):
        print("\n--- Hämtar referensrör 95_2 (baslinje) ---")
        ref = load_reference_station("95_2")
        print(f"Referens 95_2: {len(ref)} rader, {ref.isna().sum()} saknade värden")
        df_base = prepare_joint_dataframe_baseline(base, ref, freq="7D")

    refs = None
    df_multi = None
    if MODE in ("multi", "both"):
        print("\n--- Hämtar kandidatrör från SGU API (multivariat) ---")
        refs = load_candidate_stations(base)
        df_multi = prepare_joint_dataframe_multi(base, refs, freq="7D", max_refs=4)

    print("\n✓ All data hämtad — startar modellanpassning\n")

    # Initiera resultatvariabler
    result_base = result_multi = None
    out_base = out_multi = None
    metrics_base = metrics_multi = None

    # ── STEG 2: BASLINJEMODELL ────────────────────────────────────────────
    if MODE in ("baseline", "both"):
        print("="*60)
        print("BASLINJEMODELL — ett referensrör (95_2)")
        print("="*60)
        print(f"\nGemensamt dataset: {len(df_base)} veckor "
              f"({df_base.index[0].date()} – {df_base.index[-1].date()})")
        print(f"Saknade i base: {df_base['base'].isna().sum()}")
        print(f"Saknade i ref:  {df_base['ref'].isna().sum()}")

        # Anpassa modellen
        print("\n=== Anpassar baslinjemodellen ===")
        result_base, model_base = fit_model_baseline(df_base, nseason=26)

        # Kör Kalman smoother + prognos (26 veckor = ~6 månader)
        print("\n=== Kör Kalman smoother + prognos (baslinje) ===")
        out_base = smooth_and_forecast(result_base, df_base, n_forecast=26)

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
            fig_base = plot_results(out_base, anomaly_base, ref_label="95_2")
            fig_base.savefig("groundwater_ssm_baseline.png", dpi=150, bbox_inches="tight")
            print("Plot sparad till: groundwater_ssm_baseline.png")
            plt.close(fig_base)
        except (Exception, KeyboardInterrupt) as e:
            print(f"⚠ Plottning (baslinje) misslyckades: {type(e).__name__}: {e}")
            plt.close("all")

        # Spara imputerade värden
        if not imputed_base.empty:
            imputed_base.to_csv("imputed_values_baseline.csv")
            print("Imputerade värden sparade till: imputed_values_baseline.csv")

        # Skattade parametrar
        print("\n=== Skattade parametrar (baslinje) ===")
        params_base = dict(zip(model_base.param_names, result_base.params))
        for k, v in params_base.items():
            print(f"  {k:<28} = {v:.6f}")


    # ── STEG 3: MULTIVARIAT MODELL ────────────────────────────────────────
    if MODE in ("multi", "both"):
        print("\n" + "="*60)
        print("MULTIVARIAT MODELL — referensrör filtrerade på akvifer+jordart")
        print("="*60)
        # ref_ids hämtas från df_multi efter max_refs-begränsning (inte från refs-listan)
        ref_ids = [col for col in df_multi.columns if col != "base"]
        print(f"\nAnvänder {len(ref_ids)} referensrör: {ref_ids}")
        print(f"Gemensamt dataset: {len(df_multi)} veckor "
              f"({df_multi.index[0].date()} – {df_multi.index[-1].date()})")
        print(f"Saknade i base: {df_multi['base'].isna().sum()}")
        for rid in ref_ids:
            print(f"Saknade i {rid}: {df_multi[rid].isna().sum()}")

        # Anpassa modellen
        print("\n=== Anpassar multivariat modell ===")
        result_multi, model_multi = fit_model_multi(df_multi, nseason=26)

        # Kör Kalman smoother + prognos (26 veckor = ~6 månader)
        print("\n=== Kör Kalman smoother + prognos (multivariat) ===")
        out_multi = smooth_and_forecast(result_multi, df_multi, n_forecast=26)

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
                                     ref_label=", ".join(ref_ids))
            fig_multi.savefig("groundwater_ssm_multi.png", dpi=150, bbox_inches="tight")
            print("Plot sparad till: groundwater_ssm_multi.png")
            plt.close(fig_multi)
        except (Exception, KeyboardInterrupt) as e:
            print(f"⚠ Plottning (multi) misslyckades: {type(e).__name__}: {e}")
            plt.close("all")

        # Spara imputerade värden
        if not imputed_multi.empty:
            imputed_multi.to_csv("imputed_values_multi.csv")
            print("Imputerade värden sparade till: imputed_values_multi.csv")

        # Skattade parametrar
        print("\n=== Skattade parametrar (multivariat) ===")
        params_multi = dict(zip(model_multi.param_names, result_multi.params))
        for k, v in params_multi.items():
            print(f"  {k:<40} = {v:.6f}")


    # ── UTVÄRDERING & JÄMFÖRELSE ──────────────────────────────────────
    if MODE in ("baseline", "both") and result_base is not None:
        metrics_base = evaluate_model(result_base, out_base,
                                      label="Baslinje (95_2)")
    if MODE in ("multi", "both") and result_multi is not None:
        metrics_multi = evaluate_model(result_multi, out_multi,
                                       label="Multivariat")

    if MODE == "both":
        print("\n" + "="*60)
        print("JÄMFÖRELSE: Baslinje vs Multivariat")
        print("="*60)

        # Alla mått som finns i båda modellerna
        comparison_keys = [
            ("── Informationskriterier ──", None),
            ("Log-likelihood", "{:>12.2f}"),
            ("AIC", "{:>12.2f}"),
            ("BIC", "{:>12.2f}"),
            ("HQIC", "{:>12.2f}"),
            ("Antal parametrar", "{:>12.0f}"),
            ("── Prediktionsfel ──", None),
            ("RMSE", "{:>12.6f}"),
            ("MAE", "{:>12.6f}"),
            ("ME (bias)", "{:>12.6f}"),
            ("── Innovationer ──", None),
            ("Innov. medel", "{:>12.4f}"),
            ("Innov. std", "{:>12.4f}"),
            ("Innov. skevhet", "{:>12.4f}"),
            ("Innov. kurtosis", "{:>12.4f}"),
            ("── Autokorrelation ──", None),
            ("Ljung-Box Q(10)", "{:>12.4f}"),
            ("Ljung-Box p", "{:>12.4f}"),
            ("Durbin-Watson", "{:>12.4f}"),
            ("── PI-täckning ──", None),
            ("95%-täckning", "{:>11.1%}"),
            ("── Latenta variabler ──", None),
            ("Latent nivå µ — medel-std", "{:>12.6f}"),
            ("Latent säsong γ₀ — medel-std", "{:>12.6f}"),
            ("Latent total — medel-std", "{:>12.6f}"),
        ]

        print(f"\n  {'Mått':<35} {'Baslinje':>14} {'Multivariat':>14}  Bäst")
        print(f"  {'-'*75}")

        # Mått där lägre är bättre
        lower_is_better = {"AIC", "BIC", "HQIC", "RMSE", "MAE",
                           "Ljung-Box Q(10)",
                           "Latent nivå µ — medel-std",
                           "Latent säsong γ₀ — medel-std",
                           "Latent total — medel-std"}
        # Mått där högre är bättre
        higher_is_better = {"Log-likelihood", "95%-täckning",
                            "Ljung-Box p", "Antal parametrar"}
        # Mått där närmare 0 / 1 / 2 är bättre
        target_value = {"ME (bias)": 0.0, "Innov. medel": 0.0,
                        "Innov. std": 1.0, "Innov. skevhet": 0.0,
                        "Innov. kurtosis": 0.0, "Durbin-Watson": 2.0}

        for key, fmt in comparison_keys:
            if fmt is None:
                # Rubrikrad
                print(f"\n  {key}")
                continue
            vb = metrics_base.get(key)
            vm = metrics_multi.get(key)
            if vb is None or vm is None:
                continue
            sb = fmt.format(vb)
            sm = fmt.format(vm)

            # Bestäm vilken modell som "vinner"
            winner = ""
            if key in lower_is_better:
                winner = "← bas" if vb < vm else "multi →" if vm < vb else "="
            elif key in higher_is_better:
                winner = "← bas" if vb > vm else "multi →" if vm > vb else "="
            elif key in target_value:
                t = target_value[key]
                winner = "← bas" if abs(vb - t) < abs(vm - t) else \
                         "multi →" if abs(vm - t) < abs(vb - t) else "="

            print(f"  {key:<35} {sb:>14} {sm:>14}  {winner}")