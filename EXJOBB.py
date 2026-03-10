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


basror_22W102 = Path("/home/angelica/EXJOBB/22W102.csv")
metadata_path = Path("/home/angelica/Hämtningar/stationer_metadata.csv")


# ─────────────────────────────────────────────
#  LADDA DATA
# ─────────────────────────────────────────────

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
    with SGUClient() as client:
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
    from sgu_client import SGUClient

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
                                     "limit": 500, "f": "json"}, timeout=30)
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

    with SGUClient() as client:
        for sid in candidates:
            try:
                meas      = client.levels.observed.get_measurements_by_name(station_id=sid)
                df_r      = meas.to_dataframe()
                date_col  = next(c for c in df_r.columns
                                 if "date" in c.lower() or "time" in c.lower())
                level_col = next(c for c in df_r.columns
                                 if "level" in c.lower() or "water" in c.lower())
                df_r[date_col] = pd.to_datetime(df_r[date_col])
                s = df_r.set_index(date_col)[level_col].sort_index()
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                s       = pd.to_numeric(s, errors="coerce")
                overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna()
                if len(overlap) >= min_overlap_weeks:
                    result.append((sid, s))
                    print(f"  ✓ {sid}: {len(overlap)} överlappande obs")
                else:
                    print(f"  ✗ {sid}: för få obs ({len(overlap)} < {min_overlap_weeks})")
            except Exception as e:
                print(f"  ✗ {sid}: fel vid hämtning ({e})")

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
    # väljer de rör med flest överlappande obs (bäst täckning)
    max_refs: int = 10,
) -> pd.DataFrame:
    # Ta bort tidszon om den finns
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)

    # Begränsa till max_refs rör — väljer de med flest obs (redan sorterade i load_candidate_stations)
    if len(refs) > max_refs:
        print(f"Begränsar från {len(refs)} till {max_refs} referensrör (max_refs={max_refs})")
        refs = refs[:max_refs]

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
        k_posdef = k_states

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

#nedan sparas var i tillståndsekvationen som varje komponent sitter
        self.idx_level  = 0
        self.idx_season = slice(1, 1 + nseason)

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

    @property
    def param_bounds(self):
        return [
            (1e-6, None),    # sigma2_eta_level
            (1e-6, None),    # sigma2_eta_season
            (1e-6, None),    # sigma2_eps_ref
            (None, None),    # beta_ref
        ]

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

        # ── Processkörsbrus Q ───────────────────────────────────────────
        Q = np.zeros((k, k))
        Q[0, 0] = s2_level
        Q[1, 1] = s2_season  # brus enbart på den aktuella säsongskomponenten
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris = identitetsmatris ─────────────────────────
        self.ssm["selection"] = np.eye(k)


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
        k_posdef = k_states

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        # param_names sätts dynamiskt: 2 process noise + 2 parametrar per referensrör
        # sparas som _param_names eftersom MLEModel redan har param_names som property
        self._param_names = ["sigma2_eta_level", "sigma2_eta_season"]
        for rid in self.ref_ids:
            self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}"]

#nedan sparas var i tillståndsekvationen som varje komponent sitter
        self.idx_level  = 0
        self.idx_season = slice(1, 1 + nseason)

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

        # Steg 3: OLS för varje referensrör för att skatta beta och mätfelet
        for i in range(self.n_refs):
            y_ref    = self.endog[:, 1 + i]
            y_deseas = y_ref - season
            mask     = ~(np.isnan(y_deseas) | np.isnan(level))
            X        = level[mask].reshape(-1, 1)
            ols      = np.linalg.lstsq(X, y_deseas[mask], rcond=None)
            beta_i   = float(ols[0][0])
            s2_eps_i = float(np.var(y_deseas[mask] - X.flatten() * beta_i))
            params  += [max(s2_eps_i, 1e-6), beta_i]

        return np.array(params)

    @property
    def param_bounds(self):
        bounds = [
            (1e-6, None),   # sigma2_eta_level
            (1e-6, None),   # sigma2_eta_season
        ]
        for _ in self.ref_ids:
            bounds += [
                (1e-6, None),   # sigma2_eps_ref_i
                (None, None),   # beta_ref_i
            ]
        return bounds

#FORTSÄTT FÖRKLARA DENNA KOD FRÅN OCH MED IMORGON 

    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        # Varianser: exp-transform → alltid positiva
        p[0] = np.exp(unconstrained[0])   # sigma2_eta_level
        p[1] = np.exp(unconstrained[1])   # sigma2_eta_season
        for i in range(self.n_refs):
            p[2 + 2*i] = np.exp(unconstrained[2 + 2*i])  # sigma2_eps_ref_i
            # beta_ref_i (index 3+2i) är obegränsad — ingen transform
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0])
        p[1] = np.log(constrained[1])
        for i in range(self.n_refs):
            p[2 + 2*i] = np.log(constrained[2 + 2*i])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        s2_level  = params[0]
        s2_season = params[1]
        # Plocka ut (sigma2_eps_i, beta_i) för varje referensrör
        ref_params = [
            (params[2 + 2*i], params[3 + 2*i])
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
        # y_base  = µ + γ_0      (direkt nivå + säsong)
        # y_ref_i = beta_i * µ   (varje referens kopplad via nivå)
        Z = np.zeros((n_endog, k))
        Z[0, 0] = 1.0   # base ← nivå
        Z[0, 1] = 1.0   # base ← säsong
        for i, (_, beta_i) in enumerate(ref_params):
            Z[1 + i, 0] = beta_i  # ref_i ← beta_i * nivå

        self.ssm["design"] = Z

        # ── Observationsbrus ────────────────────────────────────────────
        # sigma2_eps_base är låst till 0: basrörets mätningar antas exakta
        obs_cov = np.zeros((n_endog, n_endog))
        for i, (s2_eps_i, _) in enumerate(ref_params):
            obs_cov[1 + i, 1 + i] = s2_eps_i
        self.ssm["obs_cov"] = obs_cov

        # ── Processkörsbrus Q ───────────────────────────────────────────
        Q = np.zeros((k, k))
        Q[0, 0] = s2_level
        Q[1, 1] = s2_season  # brus enbart på den aktuella säsongskomponenten
        self.ssm["state_cov"] = Q

        # ── Selektionsmatris = identitetsmatris ─────────────────────────
        self.ssm["selection"] = np.eye(k)


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
    result = model.fit(
        method="lbfgs",
        maxiter=2000,
        disp=True,
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
        "pred_base":      pred_mean,
        "pred_ci":        pred_ci,
        "fcast_index":    fcast_idx,
        "fcast_base":     fcast_mean,
        "fcast_ci":       fcast_ci,
    }


# ─────────────────────────────────────────────
#  AVVIKELSEDETEKTERING (á la Akvifär)
# ─────────────────────────────────────────────

def detect_anomalies(
    observed: np.ndarray,
    predicted: np.ndarray,
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

    # 1. Ladda basröret (gemensamt för båda modellerna)
    print("=== Laddar basröret ===")
    base = load_base_station()
    print(f"Basobjekt 22W102: {len(base)} rader, {base.isna().sum()} saknade värden")


    # ── BASLINJEMODELL ────────────────────────────────────────────────────
    if MODE in ("baseline", "both"):
        print("\n" + "="*60)
        print("BASLINJEMODELL — ett referensrör (95_2)")
        print("="*60)

        # Ladda referensröret 95_2
        ref = load_reference_station("95_2")
        print(f"Referens 95_2: {len(ref)} rader, {ref.isna().sum()} saknade värden")

        # Bygg gemensamt datumindex
        df_base = prepare_joint_dataframe_baseline(base, ref, freq="7D")
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
            predicted=out_base["pred_base"],
            pred_ci=out_base["pred_ci"],
            min_consecutive=3,
        )
        n_anom = anomaly_base.sum()
        print(f"\nAvvikelsedetektering (baslinje): {n_anom} tidpunkter flaggade "
              f"({n_anom/len(anomaly_base)*100:.1f}%)")

        # Plotta och spara
        print("\n=== Genererar plot (baslinje) ===")
        fig_base = plot_results(out_base, anomaly_base, ref_label="95_2")
        fig_base.savefig("groundwater_ssm_baseline.png", dpi=150, bbox_inches="tight")
        print("Plot sparad till: groundwater_ssm_baseline.png")
        plt.show()

        # Spara imputerade värden
        if not imputed_base.empty:
            imputed_base.to_csv("imputed_values_baseline.csv")
            print("Imputerade värden sparade till: imputed_values_baseline.csv")

        # Skattade parametrar
        print("\n=== Skattade parametrar (baslinje) ===")
        params_base = dict(zip(model_base.param_names, result_base.params))
        for k, v in params_base.items():
            print(f"  {k:<28} = {v:.6f}")


    # ── MULTIVARIAT MODELL ────────────────────────────────────────────────
    if MODE in ("multi", "both"):
        print("\n" + "="*60)
        print("MULTIVARIAT MODELL — referensrör filtrerade på akvifer+jordart")
        print("="*60)

        # Hämta kandidatrör från SGU API filtrerat på akvifer och jordart
        print("\n=== Hämtar referensrör från SGU API ===")
        refs    = load_candidate_stations(base)
        # Bygg gemensamt datumindex — max 10 referensrör (de med bäst täckning väljs)
        df_multi = prepare_joint_dataframe_multi(base, refs, freq="7D", max_refs=10)
        # ref_ids hämtas från df_multi efter max_refs-begränsning (inte från refs-listan)
        ref_ids = [col for col in df_multi.columns if col != "base"]
        print(f"\nAnvänder {len(ref_ids)} referensrör: {ref_ids}")
        print(f"\nGemensamt dataset: {len(df_multi)} veckor "
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
            predicted=out_multi["pred_base"],
            pred_ci=out_multi["pred_ci"],
            min_consecutive=3,
        )
        n_anom = anomaly_multi.sum()
        print(f"\nAvvikelsedetektering (multivariat): {n_anom} tidpunkter flaggade "
              f"({n_anom/len(anomaly_multi)*100:.1f}%)")

        # Plotta och spara
        print("\n=== Genererar plot (multivariat) ===")
        fig_multi = plot_results(out_multi, anomaly_multi,
                                 ref_label=", ".join(ref_ids))
        fig_multi.savefig("groundwater_ssm_multi.png", dpi=150, bbox_inches="tight")
        print("Plot sparad till: groundwater_ssm_multi.png")
        plt.show()

        # Spara imputerade värden
        if not imputed_multi.empty:
            imputed_multi.to_csv("imputed_values_multi.csv")
            print("Imputerade värden sparade till: imputed_values_multi.csv")

        # Skattade parametrar
        print("\n=== Skattade parametrar (multivariat) ===")
        params_multi = dict(zip(model_multi.param_names, result_multi.params))
        for k, v in params_multi.items():
            print(f"  {k:<40} = {v:.6f}")


    # ── JÄMFÖRELSE AIC/BIC ───────────────────────────────────────────────
    if MODE == "both":
        print("\n" + "="*60)
        print("JÄMFÖRELSE: Baslinje vs Multivariat")
        print("="*60)
        print(f"  {'Modell':<20} {'Log-lik':>10} {'AIC':>10} {'BIC':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Baslinje (95_2)':<20} "
              f"{result_base.llf:>10.2f} "
              f"{result_base.aic:>10.2f} "
              f"{result_base.bic:>10.2f}")
        print(f"  {'Multivariat':<20} "
              f"{result_multi.llf:>10.2f} "
              f"{result_multi.aic:>10.2f} "
              f"{result_multi.bic:>10.2f}")