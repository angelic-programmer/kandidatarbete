import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as sp_stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

#Importerar statmodels för att kunna använda deras ramverk för state space-modeller 
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.stattools import acf as sm_acf


#Path(__file__).parent gör att filen hittas i samma map som skriptet oavsett dator, därför är det inte en hårdkodad filepath
basror_22W102  = Path(__file__).parent / "22W102.csv"
basror_17XX01U = Path(__file__).parent / "17XX01U.csv"
basror_G1101   = Path(__file__).parent / "G1101.csv"
metadata_path  = Path(__file__).parent / "stationer_metadata.csv"

#Vi skapar en dataframe i pandas av basröret
def load_base_station(filepath: Path = basror_G1101) -> pd.Series:
    #Här är en funktion som tar en filsökväg som argument, basror_22W102 är defaultvärdet
    #anger separator automatiskt (semikolon eller komma) eftersom CSV filerna har olika några har punkt och andr har komma
    with open(filepath, encoding="utf-8-sig") as f:
        first_line = f.readline()
    sep = ";" if ";" in first_line else ","
    #Detektera om filen har rubrikrad eftersom ibland är första raden en rubrik och ibland börjar datan direkt
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

    #Nedan läser vi in csv-filen till en DataFrame med kolumnerna date och level
    for fmt in ["%Y-%m-%d", "%y/%m/%d", "%d/%m/%Y"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            break
        except (ValueError, TypeError):
            continue
    df["level"] = (
        df["level"]
        .astype(str)
          #byt ut komma mot punkt 
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



# SGU OGC API — basadress för stationslistan
_SGU_API_URL = ("https://api.sgu.se/oppnadata/grundvattennivaer-observerade"
                "/ogc/features/v1/collections/stationer/items")


def _strip_tz(s: pd.Series) -> pd.Series:
    """Tar bort tidszon från ett index (SGU returnerar UTC, basröret saknar tz)."""
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def _parse_cached_series(records: dict) -> pd.Series:
    """Konverterar en dict {datum_str: float} från ref_cache.json till pd.Series."""
    s = pd.Series({pd.Timestamp(k): v for k, v in records.items()}, dtype=float)
    return _strip_tz(pd.to_numeric(s.sort_index(), errors="coerce"))


def _query_sgu_api(akvifer: str, jordart: str, exclude_id: str) -> list[str]:
    """Hämtar stationsnamn från SGU OGC API filtrerat på akvifer+jordart.
    verify=False behövs pga saknat root-certifikat i lokala Python-installationen."""
    import requests, urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    resp = requests.get(_SGU_API_URL,
                        params={"akvifer": akvifer, "jordart": jordart,
                                "limit": 500, "f": "json"},
                        timeout=30, verify=False)
    resp.raise_for_status()
    return [f["properties"]["platsbeteckning"]
            for f in resp.json().get("features", [])
            if f["properties"].get("platsbeteckning") not in (None, exclude_id)]


# Hämtar referensrör via SGUClient
def load_reference_station(station_id: str) -> pd.Series:
    from sgu_client import SGUClient
    with SGUClient() as client:
        df_ref = client.levels.observed.get_measurements_by_name(
            station_id=station_id
        ).to_dataframe()

    # Identifiera datum- och nivåkolumn automatiskt (kolumnnamnen varierar)
    date_col  = next(c for c in df_ref.columns if "date" in c.lower() or "time" in c.lower())
    level_col = next(c for c in df_ref.columns if "level" in c.lower() or "water" in c.lower())
    df_ref[date_col] = pd.to_datetime(df_ref[date_col])
    ref = df_ref.set_index(date_col)[level_col].sort_index()
    ref = pd.to_numeric(_strip_tz(ref), errors="coerce")
    print(f"✓ Referensdata hämtad från SGUClient ({station_id})")
    return ref


# Hämtar kandidatrör från SGU API med samma akvifer och jordart som basröret
# används av både envariata (topp-1) och multivariata (topp-k) modellen
def load_candidate_stations(
    base_series: pd.Series,
    base_station_id: str = "G1101",
    meta_path: Path = metadata_path,
    ignore_geology: bool = False,
) -> list:
    


    """
    1. Läser basrörets akvifer+jordart från metadata-CSV
    2. Hämtar matchande stationer baserat på punkt 1 
    3. Filtrerar på tidsöverlappning med basröret
    4. Sorterar efter antal överlappande observationer
    """

    #Steg 1: Läs basrörets geologiska egenskaper från metadata 
    meta = pd.read_csv(meta_path)
    base_row = meta[meta["station_id"] == base_station_id]
    if base_row.empty:
        raise ValueError(f"Basröret {base_station_id} saknas i {meta_path}")
    target_akvifer = base_row.iloc[0]["akvifer"]
    target_jordart = base_row.iloc[0]["jordart"]
    print(f"Basröret {base_station_id}: akvifer={target_akvifer}, jordart={target_jordart}")

    #Tidsperiod som basröret täcker (för överlappningskontroll)
    base_start = base_series.dropna().index.min()
    base_end   = base_series.dropna().index.max()
    result     = []
    max_download = 10

    # Steg 2: Hämta kandidatrör 
    cache_path = Path(__file__).parent / "ref_cache.json"

    if cache_path.exists():
        import json
        print(f"  Läser cache: {cache_path}")
        with open(cache_path) as f:
            all_data = json.load(f)

        # Bestäm vilka stationer i cachen som matchar geologin
        if ignore_geology:
            print("  ignore_geology=True: använder hela cachen")
            candidates = list(all_data.keys())
        else:
            candidates = _query_sgu_api(target_akvifer, target_jordart, base_station_id)
            print(f"SGU API returnerade {len(candidates)} kandidatrör "
                  f"(akvifer={target_akvifer}, jordart={target_jordart})")

        # Bygg tidsserier från cache och filtrera på överlappning
        for sid in candidates:
            if sid == base_station_id or sid not in all_data:
                continue
            records = all_data[sid]
            if "__error__" in records:
                continue
            s = _parse_cached_series(records)
            n_overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna().size
            if n_overlap > 0:
                result.append((sid, s, n_overlap))
                print(f"  ✓ {sid}: {n_overlap} överlappande obs (cache)")
            if len(result) >= max_download:
                break
    else:
        #  Strategi B: direkt API-hämtning (långsam, kan hänga pga SSL) 
        print(" Ingen cache hittad — kör: py fetch_candidates.py")
        candidates = _query_sgu_api(
            target_akvifer if not ignore_geology else "",
            target_jordart if not ignore_geology else "",
            base_station_id)
        import time
        for sid in candidates[:50]:
            if len(result) >= max_download:
                break
            try:
                s = load_reference_station(sid)
                n_overlap = s[(s.index >= base_start) & (s.index <= base_end)].dropna().size
                if n_overlap > 0:
                    result.append((sid, s, n_overlap))
                    print(f"  ✓ {sid}: {n_overlap} överlappande obs")
                else:
                    print(f"  ✗ {sid}: ingen överlappning")
            except Exception as e:
                print(f"  ✗ {sid}: fel ({e})")
            time.sleep(2.0)

    if not result:
        print("\nVarning: inga kandidatrör hittades!")
        return []

    # Steg 3: Sortera efter flest överlappande obs 
    result.sort(key=lambda x: x[2], reverse=True)
    print(f"\nRören sorterade efter överlappning. Topp 5: {[sid for sid, _, _ in result[:5]]}")

    # Returnera utan overlap-räknaren (behövs inte längre)
    return [(sid, s) for sid, s, _ in result]


#Här görs tidserien om till veckovisa medelvärden
def resample_to_weekly(s: pd.Series, freq: str = "7D") -> pd.Series:
    #Ta bort tidszon om den finns
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    # Slår ihop flera mätningar samma dag till medelvärde
    s = s.groupby(s.index.normalize()).mean()
    s = s.resample(freq).mean()
    return s

# Nedan skapas en gemensam dataframe av basröret och flera referensrör
# det används av den multivariata modellen
def dataframe_multi(
    base: pd.Series,
    refs: list,
    # frekvensen sätts veckovis
    freq: str = "7D",
    max_refs: int = 5,
) -> pd.DataFrame:
    # Ta bort tidszon om den finns
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)


    #Rankar referensrör efter Pearson-korrelation med basröret 
    base_w = resample_to_weekly(base, freq=freq)
    base_w.index = base_w.index.to_period("W").to_timestamp()
    corr_list = []
    for sid, s in refs:
        s_w = resample_to_weekly(s, freq=freq)
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
    refs_weekly = []
    for sid, s in refs:
        s_w = resample_to_weekly(s, freq=freq)
        refs_weekly.append((sid, s_w))

    # Använd hela perioden från basröret
    start = base.index.min()
    end   = base.index.max()

    # Skapar ett regelbundet datumindex med en rad per vecka, från start till slut
    idx = pd.date_range(start, end, freq=freq)

    # Basröret: resampla till vecka
    base_weekly = base.groupby(base.index.normalize()).mean().resample(freq).mean()
    df = pd.DataFrame(index=idx)
    df["base"] = base_weekly.reindex(idx)
    # Vi har data veckovist så om det saknas för referensrören fylls det på med närmaste observation inom 4 dagar om det inte finns lämnas det som Nan
    for sid, s in refs_weekly:
        df[sid] = s.reindex(idx, method="nearest", tolerance=pd.Timedelta("4D"))

    return df


"""
─────────────────────────────────────────────
UNIVARIABEL MODELL (ett referensrör) 
─────────────────────────────────────────────

Modellen är en local level-modell utan trend och utan säsong.
Nivåkomponenten följer en random walk, vilket innebär att σ²_η
skattas fritt via MLE, om σ²_η > 0 är nivån stokastisk,
om MLE skattar σ²_η nära noll beter sig nivån deterministiskt.

Tillståndsekvation (random walk):
α(t+1) = α(t) + η(t+1),   η(t+1) ~ N(0, σ²_η)

Observationsekvationer:
y_base(t) = α(t)                           α(t) är grundvattennivån i meter över havet för basröret, notation från boken
y_ref(t)  = β · α(t)  + ε_ref(t),             ε_ref(t) ~ N(0, σ²_ε,ref)


Parametrar som skattas via MLE (3 st):
σ²_η      varians för vad boken kallar disturbance  η(t+1)
σ²_ε,ref  observationsvarians för referensröret
β         koefecienten mellan latent nivå och referensröret


För vår modell med två observationsserier och en skalär latent nivå α_t specificeras systemmatriserna som:
Designmatris Z ∈ ℝ^{2×1}:
# är en 2 × 1-matris med en rad per observationsserie
och en kolumn för det enda tillståndet αt. Första raden är alltid 1 (basröret observerar nivån direkt)
medan den andra raden innehåller koefficienten β som uppdateras med det aktuella värdet vid varje
steg i optimeringen
Z = [1; β]

Observationskovarians H ∈ ℝ^{2×2}:
H = diag(0, σ²_ε,ref)
"""


class GroundwaterSSM(MLEModel): #MLE model är från statmodels

    def __init__(self, endog, **extra):
        super().__init__(
            endog,
            k_states=1,
            k_posdef=1,
            **extra,
        )

        self._param_names = [
            "sigma2_eta_level",
            "sigma2_eps_ref",
            "beta_ref",
        ]
        self.ssm.initialize_diffuse()


    #klassen attribute skapas den används för att definiera namnen på de parametrar som modellen ska skatta 
    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self): #start_params ger startvärden för de parametrar som MLE ska skatta  
        import statsmodels.api as sm

        y1 = self.endog[:, 0]  #basröret
        y2 = self.endog[:, 1]  # referensröret

        # för σ²_η variansen för nivån görs en local level modell i statmodels för att få startvärden
        #här används kalmanfiltret det oliga är att local level också behöver startvärden men det sköter
        #UnobservedComponents som är från statmodels så det behöver vi inte tänka på
        mod = sm.tsa.UnobservedComponents(y1, level="local level")
        res = mod.fit(disp=False, method="nm", maxiter=500)
        sigma2_eta_level = float(res.params[1])
        level = res.level.smoothed

        # för β och σ²_ε används OLS-regression för att få fram startgissningar för referensröret
        mask = ~(np.isnan(y2) | np.isnan(level))
        X    = level[mask].reshape(-1, 1)
        ols  = np.linalg.lstsq(X, y2[mask], rcond=None)
        beta_ref       = float(ols[0][0])
        sigma2_eps_ref = float(np.var(y2[mask] - X.flatten() * beta_ref))

        return np.array([sigma2_eta_level, sigma2_eps_ref, beta_ref])


    #Nedan transformeras varianserna eftersom de måste vara större än noll, det görs genom att ta exp(av talet som MLE skattade)
    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        p[0] = np.exp(unconstrained[0])  # σ²_η > 0
        p[1] = np.exp(unconstrained[1])  # σ²_ε,ref > 0
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0])
        p[1] = np.log(constrained[1])
        return p


    def update(self, params, **extra):
        # update av MLE varje gång den provar nya parametervärden.
        params = super().update(params, **extra)
        sigma2_eta_level, s2_ref, beta_ref = params

        #Transitionsmatris beskriver hur tillståndet övergår från en tidpunkt till nästa
        self.ssm["transition"] = np.array([[1.0]])

        #Observationsmatris Z ∈ ℝ^{2×1} 
        #Kopplar den latenta nivån α(t) till de två observationerna:
        #y_base(t) = 1· α(t)   ← basröret mäter nivån direkt
        #y_ref(t)  = β· α(t)   ← referensröret skalat med β
        # β skattas fritt av MLE 
        self.ssm["design"] = np.array([[1.0],
                        [beta_ref]])

        #H ∈ ℝ^{2×2} 
        #Mätosäkerheten per observationsserie:
        #H = diag(0, σ²_ε,ref)
        self.ssm["obs_cov"] = np.diag([0.0, s2_ref])

        #Q ∈ ℝ^{1×1} jag har testat för detta i en annan fil
        # Variansen i random walk-steget
        # per tidssteg:
        #   σ²_η stor → nivån hoppar mycket (stokastisk)
        #   σ²_η ≈ 0  → nivån är nästan konstant (deterministisk)
        # σ²_η skattas fritt av MLE.
        self.ssm["state_cov"] = np.array([[sigma2_eta_level]])

        #Selektionsmatris R ∈ ℝ^{1×1} 
        # Jag sätter den bara till 1 den är 0 per default så det enda den för nu är att multiplicerar nivåkomponenten med 1 men den behövs för paketet
        self.ssm["selection"] = np.array([[1.0]])


"""
─────────────────────────────────────────────
  MULTIVARIAT MODELL (flera referensrör)
────────────────────────────────────────────
"""

#vi skapar en ny "klass" i python GroundwaterSSM_Multi
class GroundwaterSSM_Multi(MLEModel):

#ref_ids är en lista med station_id för referensrören, n_refs är antalet referensrör
    def __init__(self, endog, ref_ids=None, **extra):
        self.ref_ids = ref_ids if ref_ids is not None else []
        self.n_refs  = len(self.ref_ids)
        self.nseason = 0

        #super().__init__() anropar föräldraklassens konstruktor altså MLEModel.__init__()
        super().__init__(
            endog, #datan heter endog och den heter så i ramverket som jag följer så för att det inte ska bli någon krock behåller jag det 
            k_states=1,
            k_posdef=1,
            **extra,
        )

        #self._param_names skapar en lista med namn på de parametrar som modellen ska skatta den börjar med sigma2_eta_level 
        #sen loopar den egenom resten
        self._param_names = ["sigma2_eta_level"]
        for rid in self.ref_ids:
            self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}",
                                  f"alpha_{rid}"]
        self.ssm.initialize_diffuse()

    #property gör att metoden anropas som ett attribut och då behövs inga paranteser
    #returnerar en lista med parameeternamnen som skapades innan
    @property
    def param_names(self):
        return self._param_names

    #vi hämtar de skattade startparametrarna 
    @property
    def start_params(self):
        import statsmodels.api as sm
        #hämtar basrörets tidserie
        y1 = self.endog[:, 0]  

        #sm.tsa.UnobservedComponents skapar en enkel local level-modell i statsmodels för att få startvärden för sigma2_eta_level (variansen för grundvattennivån)
        mod = sm.tsa.UnobservedComponents(y1, level="local level")
        #kör MLE på local-level modellen och sparar resultatet i res 
        res = mod.fit(disp=False, method="nm", maxiter=500)
        #vi plockar ut sigma2_eta_level
        sigma2_eta_level = float(res.params[1])
        level = res.level.smoothed
        params = [sigma2_eta_level]



        #Här skiljer sig den univariata med den multivariata
        #loopar egenom varje referensrör och kör OLS skattningar separat för varje rör
        for i in range(self.n_refs):
            #hämtar referensrörets tidserie 1+i endog eftersom första kolumnen är basröret
            y_ref = self.endog[:, 1 + i]
            #tar bort tidpunkter där referensröret eller basröret är Nan
            mask  = ~(np.isnan(y_ref) | np.isnan(level))
             #x är designmatrisen: första kolonnen är ettor
             #andra kolonnen är β
            X     = np.column_stack([np.ones(mask.sum()), level[mask]])
            #kör OLS
            ols   = np.linalg.lstsq(X, y_ref[mask], rcond=None)
            #plocjar ut parametrarna
            alpha_i  = float(ols[0][0])
            beta_i   = float(ols[0][1])
            s2_eps_i = float(np.var(y_ref[mask] - X @ ols[0]))
            params += [max(s2_eps_i, 1e-6), beta_i, alpha_i]

        return np.array(params)




    def transform_params(self, unconstrained):
        #kopierar parametervektorn 
        p = unconstrained.copy()
        #transoformerar sigma2_eta_level så att den är större än noll genom att ta exp av den
        p[0] = np.exp(unconstrained[0])   
        #loopar över varje referensrör och transformerar sigma2_eps så att den är större än noll genom att ta exp av den
        for i in range(self.n_refs):
            p[1 + 3*i] = np.exp(unconstrained[1 + 3*i])  
        return p


    #samma som ovan fast vi tar log ist för exp
    def untransform_params(self, constrained):
        p = constrained.copy()
        p[0] = np.log(constrained[0])
        for i in range(self.n_refs):
            p[1 + 3*i] = np.log(constrained[1 + 3*i])
        return p


    #metoden update skapas med argumenten self som är det aktuela modellobjektet det ger tillgång till allt som skapades i _init_
    #**extra är alla extra argument som statsmodels skickar in internt
    def update(self, params, **extra):
        #super() anropar föräldraklassens update-metod som hanterar transformering av parametrar 
        params = super().update(params, **extra)
        #n _endog är antalet observationsserier, det är 1 (basröret) + n_refs (antalet referensrör)
        n_endog = 1 + self.n_refs
        sigma2_eta_level = params[0]

        # Plocka ut (sigma2_eps, beta, alpha) per referensrör
        ref_params = [
            (params[1 + 3*i], params[1 + 3*i + 1], params[1 + 3*i + 2])
            for i in range(self.n_refs)
        ]

        #Transitionsmatris 
        self.ssm["transition"] = np.array([[1.0]])

        #Observationsmatris 
        Z = np.zeros((n_endog, 1))
        Z[0, 0] = 1.0   
        for i, (_, beta_i, _) in enumerate(ref_params):
            Z[1 + i, 0] = beta_i
        self.ssm["design"] = Z

        #Observationsintercept d ∈ ℝ^{(1+n_refs)×1}
        d = np.zeros((n_endog, 1))
        for i, (_, _, alpha_i) in enumerate(ref_params):
            d[1 + i, 0] = alpha_i
        self.ssm["obs_intercept"] = d

        #Observationsbrus H ∈ ℝ^{(1+n_refs)×(1+n_refs)} 
        obs_cov = np.zeros((n_endog, n_endog))
        for i, (s2_eps_i, _, _) in enumerate(ref_params):
            obs_cov[1 + i, 1 + i] = s2_eps_i
        self.ssm["obs_cov"] = obs_cov

        #Processbrus Q ∈ ℝ^{1×1}: σ²_η 
        self.ssm["state_cov"] = np.array([[sigma2_eta_level]])

        #Selektionsmatris R ∈ ℝ^{1×1} 
        self.ssm["selection"] = np.array([[1.0]])



"""
─────────────────────────────────────────────
ANPASSA OCH PREDIKTERA
─────────────────────────────────────────────
"""
#vi defierar funktionen fit_model_univariate som anpassar den univariata modellen den returnerar en tuple med resultatet av anpassningen och själva modellen
def fit_model_univariate(df: pd.DataFrame) -> tuple:
    endog = df[["base", "ref"]].values.astype(float)
    model = GroundwaterSSM(endog)
    result = model.fit(
        method="lbfgs", #lbfgs är metoden som används för att hitta de värden som maximerar liklihoodfunktionen
        #det är defaultmetoden i statsmodels och var därför jag valde den
        maxiter=1000, #hur måga gånger lbfgs får testa nya värden 
        disp=False,  #gör att den inte skriver ut massa i terminalen
    )
    return result, model


#samma princip fast multivariat
def fit_model_multi(df: pd.DataFrame) -> tuple:
    #den plockar uta alla id:n utom fär de som heter base
    ref_ids = [col for col in df.columns if col != "base"]
    endog   = df[["base"] + ref_ids].values.astype(float)
    model = GroundwaterSSM_Multi(endog, ref_ids=ref_ids)
    print(f"Anpassar multivariat modell med {len(ref_ids)} referensrör: {ref_ids}...")
    result = model.fit(
        method="lbfgs",
        maxiter=2000,
        disp=False,
        cov_type="none", 
    )
    return result, model


def smooth_and_forecast(result, df: pd.DataFrame, n_forecast: int = 26) -> dict:
    #hämtar kalman smootherns resultat från den skattade modellen
    smoothed = result.smoother_results
    #smoothern skattar data med hela tidserien och kalmanfiltret skattar med bara tidigare data
    level_smoothed = smoothed.smoothed_state[0]
    level_var      = smoothed.smoothed_state_cov[0, 0]

    #konfidensintervall för kalman smoother
    total_var = level_var
    pred_mean = level_smoothed
    std = np.sqrt(np.maximum(total_var, 0))
    pred_ci    = np.column_stack([
        pred_mean - 1.96 * std,
        pred_mean + 1.96 * std,
    ])

    # Smoother-baserad prognos: konstant nivå + växande osäkerhet från random walk
    sigma2_eta_level   = result.params[0]
    last_std   = std[-1]
    fcast_mean = np.full(n_forecast, pred_mean[-1])
    fcast_std  = np.sqrt(last_std**2 + np.arange(1, n_forecast + 1) * sigma2_eta_level)
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

    # One-step-ahead filter-prediktioner (basröret är rad 0)
    filter_pred_base = result.filter_results.forecasts[0]
    filter_pred_var  = result.filter_results.forecasts_error_cov[0, 0]
    filter_pred_std  = np.sqrt(np.maximum(filter_pred_var, 0))
    filter_pred_ci   = np.column_stack([
        filter_pred_base - 1.96 * filter_pred_std,
        filter_pred_base + 1.96 * filter_pred_std,
    ])

    return {
    # tidsindex används som x-axel i alla grafer
    "index":            df.index,
    # basrörets faktiska mätvärden jämförs mot prediktioner i utvärdering och plottar
    "observed_base":    df["base"].values,
    # kalman-smoothad nivå används för imputation av saknade värden
    "level_smoothed":   level_smoothed,
    # smoothad prediktionskurva ritas som utjämnad linje i graferna
    "pred_base":        pred_mean,
    # smoother-baserat 95%-konfidensintervall anänder all data och ger ett smalare intervall används för att hitta när det är utanför 95%ki
    "pred_ci":          pred_ci,
    # kalmanfiltrets one-step-ahead prediktioner för MAE, RMSE och ACF
    "filter_pred_base": filter_pred_base,
    # filter-baserat 95%-konfidensintervall använder bara data fram till t används för utvärdering  och i graferna
    "filter_pred_ci":   filter_pred_ci,
    # datum för prognosperioden — används som x-axel för prognosen i grafen
    "fcast_index":      fcast_idx,
    # prognostiserade nivåvärden — ritas som prognoslinje i grafen
    "fcast_base":       fcast_mean,
    # 95%-konfidensintervall men för prognosen frammåt
    "fcast_ci":         fcast_ci,
    }

"""
─────────────────────────────────────────────
MODELLDIAGNOSTIK / UTVÄRDERING
─────────────────────────────────────────────

Utvärderar de tre antaganden som krävs för SSM (Kap. 17, Applied Time Series Analysis, 2019):

Antagande 1 (§17.1): Initialtillstånd α₀ har E(α₀)=a₀, V(α₀)=P₀
Antagande 2 (§17.1): Feltermerna εₜ och ηₜ är okorrelerade med
                        varandra i alla tidsperioder samt okorrelerade
                        med initialtillståndet
Antagande 3 (§17.9): Feltermerna och initialtillståndet är
                        normalfördelade (krävs för Kalmanfiltrets
                        optimalitet)
"""



def evaluate_model(result, out: dict, label: str = "Modell", station_id: str = "station") -> dict:
    obs  = out["observed_base"]
    #Värdena som kalmanfiltret predikterade
    pred = out["filter_pred_base"]
    ci   = out["filter_pred_ci"]

    #valid är där vi har basrör data och predikterad data för att kunna utvärdera hur bra modellen har predikterat
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o = obs[valid]
    p = pred[valid]
    #de faktiska värdena i basröret - de predikterade
    resid_eval = o - p

    metrics = {}#en tom dictionary som fylls på med utvärderingsmåtten
    metrics["n"] = int(np.sum(~np.isnan(obs)))
    metrics["MAE"] = float(np.mean(np.abs(resid_eval)))
    metrics["RMSE"] = float(np.sqrt(np.mean(resid_eval**2)))

    # Prediktionsintervallets täckningsgrad
    ci_lower = ci[valid, 0]
    ci_upper = ci[valid, 1]
    inside   = (o >= ci_lower) & (o <= ci_upper)
    metrics["95%-täckning"] = float(inside.mean())

    # Durbin-Watson
    try:
        from statsmodels.stats.stattools import durbin_watson
        metrics["Durbin-Watson"] = float(durbin_watson(resid_eval))
    except Exception:
        pass

    return metrics


"""
─────────────────────────────────────────────
VISUALISERING
─────────────────────────────────────────────
"""

def plot_results(out: dict, anomaly: pd.Series, title_base: str = "22W102",
                 ref_label: str = "", model_label: str = ""):
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
    #Använd filter-CI (one-step-ahead) för visuellt band — synligt över hela serien
    ci       = out["filter_pred_ci"][:len(idx)]
    f_idx    = out["fcast_index"]
    f_vals   = out["fcast_base"].flatten()[:len(f_idx)]
    f_ci     = out["fcast_ci"][:len(f_idx)]
    smoothed = out["level_smoothed"].flatten()[:len(idx)]


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

    # Residualer & avvikelsedetektering
    ax2 = axes[1]
    ax2.set_title("Residualer (observation – prediktion) & anomalidetektion", fontsize=11)

    residuals = obs - pred
    pi_const  = (ci[:, 1] - ci[:, 0]) / 2  

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


def plot(out: dict, anomaly: pd.Series,
                       title_base: str, ref_label: str,
                       model_label: str = ""):
    
    """
    Observationer, prediktioner och osäkerhetsband
    för hela perioden. Kalmanfiltret uppdateras
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

"""
─────────────────────────────────────────────
Prediktionen
─────────────────────────────────────────────
"""

def imputation_report(df: pd.DataFrame, out: dict) -> pd.DataFrame:
    #skapar en boolean arrey med true för varje tidpunkt där basröret är NaN
    mask = df["base"].isna()
    #skapar en tabelll ed fyra kolumner fr de tidpunkter där basröret är NaN: datum, imputerad nivå, nedre och övre gräns för 95% ki
    imputed = pd.DataFrame({
        "date":          df.index[mask],
        "imputed_level": out["level_smoothed"][mask],
        "lower_95":      out["pred_ci"][mask, 0],
        "upper_95":      out["pred_ci"][mask, 1],
    })
    #sätter datum som index
    imputed = imputed.set_index("date")
    # Hoppa över de två första observationerna — Kalman-filtret
    # ger ofta orimliga värden i början
    imputed = imputed.iloc[2:]
    print(f"\nImputation för {len(imputed)} saknade mätpunkter:")
    print(imputed.round(4).to_string())
    return imputed


if __name__ == "__main__":
    # Alla basrör som ska köras
    STATIONS = [
        (basror_22W102,  "22W102"),
        (basror_17XX01U, "17XX01U"),
        (basror_G1101,   "G1101"),
    ]

    #Här loopar vi egenom stationerna och kör en i taget
    for station_path, station_id in STATIONS:
        print(f"# KÖR STATION: {station_id}")

        #vi anropar load_base_station som hämtar basröret och sparar det i variabeln base som är en pandas serie
        base = load_base_station(station_path)
        #sen skriver den ut stations id, totalt antal rader och antal saknade värden i basröret
        print(f"Basobjekt {station_id}: {len(base)} rader, {base.isna().sum()} saknade värden")





        #Välj tidsfrekvens automatiskt baserat på tätheten på observationerna.
        #Glesa stationer (typ månadsvis) → "MS" (månad)
        #Täta stationer (typ veckovis)   → "7D"  (vecka)
        span_days = (base.index.max() - base.index.min()).days
        n_obs     = base.notna().sum()
        # medel antal dagar mellan obseervationer
        avg_gap   = span_days / max(n_obs - 1, 1)  
        #glesare än var tredje vecka → månadsfrekvens 
        if avg_gap > 21:          
            freq = "MS"
            #12 månader framåt OBS BEROENDE PÅ SYFTET KANSKE PROGNOSEN SKA TAS BORT
            n_forecast = 12   
            print(f"  Gles data (snitt {avg_gap:.0f} dagar mellan obs) → månadsfrekvens")
        else:
            freq = "7D"
            #26 veckor, 1/2 år frammåt OBS BEROENDE PÅ SYFTET KANSKE PROGNOSEN SKA TAS BORT
            n_forecast = 26   
            print(f"  Tät data (snitt {avg_gap:.0f} dagar mellan obs) → veckofrekvens")


        #Hämtar kandidatrör (samma akvifer+jordart)
        print("\n Hämtar kandidatrör från SGU API")
        refs = load_candidate_stations(base, base_station_id=station_id, ignore_geology=False)
        #vi sätter df_base och univariate_ref_id till None så att de finns och kan användas senare
        # Välj topp 1 via Pearson-korrelation 
        df_top1 = dataframe_multi(base, refs, freq=freq, max_refs=1)
        univariate_ref_id = [c for c in df_top1.columns if c != "base"][0]
        # Bygg enkel dataframe med kolumnerna "base" och "ref"
        df_base = df_top1.rename(columns={univariate_ref_id: "ref"})[["base", "ref"]]
        df_multi = dataframe_multi(base, refs, freq=freq, max_refs=5)

        print("\n All data hämtad -startar modellanpassning\n")



    


        """
        ─────────────────────────────────────────────
        ENVARIAT MODELL 
        ─────────────────────────────────────────────
        """

        print(f"ENVARIAT MODELL - ett referensrör ({univariate_ref_id})")
        tidstyp = "månader" if freq == "MS" else "veckor"
        print(f"\nGemensamt dataset: {len(df_base)} {tidstyp} "
              f"({df_base.index[0].date()} – {df_base.index[-1].date()})")
        print(f"Saknade i base: {df_base['base'].isna().sum()}")
        print(f"Saknade i ref:  {df_base['ref'].isna().sum()}")

        #Anpassa modellen
        print("\n Anpassar envariata modellen")
        result_base, model_base = fit_model_univariate(df_base)

        #Kör Kalman smoother + prognos
        print("Kör Kalman smoother")
        out_base = smooth_and_forecast(result_base, df_base, n_forecast=n_forecast)
        imputed_base = imputation_report(df_base, out_base)

        #Oobservationer utanför prediktionsintervallet används för plottarna
        obs_b = out_base["observed_base"]
        ci_b  = out_base["pred_ci"]
        anomaly_base = pd.Series(
            ~np.isnan(obs_b) & ((obs_b < ci_b[:, 0]) | (obs_b > ci_b[:, 1]))
        )

        # Plotta och spara
        fig_base = plot_results(out_base, anomaly_base, title_base=station_id, ref_label=univariate_ref_id, model_label="Envariat")
        fig_base.savefig(f"groundwater_ssm_univariate_{station_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_base)

        fig_akv = plot(
            out_base, anomaly_base,
            title_base=station_id,
            ref_label=univariate_ref_id,
            model_label="State Space"
        )
        fig_akv.savefig(f"plot_univariate_{station_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_akv)

        fig_acf = plot_acf_residuals(out_base, station_id=station_id, model_label="State Space")
        if fig_acf is not None:
            fig_acf.savefig(f"acf_residuals_univariate_{station_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_acf)

        #Spara imputerade värden
        if not imputed_base.empty:
            imputed_base.to_csv(f"imputed_values_univariate_{station_id}.csv")
            print(f"Imputerade värden sparade till: imputed_values_univariate_{station_id}.csv")

        #Skattade parametrar
        print("\n=== Skattade parametrar (envariat) ===")
        params_base = dict(zip(model_base.param_names, result_base.params))
        for k, v in params_base.items():
            print(f"  {k:<28} = {v:.6f}")


        """
        ─────────────────────────────────────────────
        MULTIVARIAT MODELL 
        ─────────────────────────────────────────────
        """

        print("MULTIVARIAT MODELL - referensrör filtrerade på akvifer+jordart")
        #plockar ut id:n på alla referensrör 
        ref_ids = [col for col in df_multi.columns if col != "base"]
        for rid in ref_ids:
            print(f"Saknade i {rid}: {df_multi[rid].isna().sum()}")

        #Anpassa modellen
        print("\n Anpassar multivariat modell")
        result_multi, model_multi = fit_model_multi(df_multi)

        #Kör Kalman smoother 
        print("\n Kör Kalman smoother + prognos (multivariat)")
        out_multi = smooth_and_forecast(result_multi, df_multi, n_forecast=n_forecast)

        #Imputation-rapport
        imputed_multi = imputation_report(df_multi, out_multi)

        #observationer utanför prediktionsintervallet används för att plotta)
        obs_m = out_multi["observed_base"]
        ci_m  = out_multi["pred_ci"]
        anomaly_multi = pd.Series(
            ~np.isnan(obs_m) & ((obs_m < ci_m[:, 0]) | (obs_m > ci_m[:, 1]))
        )


        #Plotta 
        fig_multi = plot_results(out_multi, anomaly_multi,
                                 title_base=station_id, ref_label=", ".join(ref_ids), model_label="Multivariat")
        fig_multi.savefig(f"groundwater_ssm_multi_{station_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_multi)

        fig_akv_multi = plot(
            out_multi, anomaly_multi,
            title_base=station_id,
            ref_label=", ".join(ref_ids),
            model_label="State Space Multivariat"
        )
        fig_akv_multi.savefig(f"plot_multi_{station_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_akv_multi)

        fig_acf_multi = plot_acf_residuals(out_multi, station_id=station_id, model_label="State Space Multivariat")
        if fig_acf_multi is not None:
            fig_acf_multi.savefig(f"acf_residuals_multi_{station_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig_acf_multi)

        #Spara imputerade värden
        if not imputed_multi.empty:
            imputed_multi.to_csv(f"imputed_values_multi_{station_id}.csv")
            print(f"Imputerade värden sparade till: imputed_values_multi_{station_id}.csv")

        #Skattade parametrar
        print("\n Skattade parametrar (multivariat)")
        params_multi = dict(zip(model_multi.param_names, result_multi.params))
        for k, v in params_multi.items():
            print(f"  {k:<40} = {v:.6f}")

        """
        ─────────────────────────────────────────────
        UTVÄRDERING 
        ─────────────────────────────────────────────
        """

        metrics_base = evaluate_model(result_base, out_base,
                                      label="Envariat", station_id=station_id)
        metrics_multi = evaluate_model(result_multi, out_multi,
                                       label="Multivariat", station_id=station_id)

        print(f"\n{'='*60}")
        print(f"JÄMFÖRELSE — {station_id}")
        print(f"{'='*60}")
        print(f"  {'Mått':<20} {'Envariat':>12} {'Multi':>12}")
        print(f"  {'-'*46}")

        for key, fmt in [
            ("n",              "{:>12d}"),
            ("RMSE",           "{:>12.6f}"),
            ("MAE",            "{:>12.6f}"),
            ("95%-täckning",   "{:>11.1%}"),
            ("Durbin-Watson",  "{:>12.4f}"),
        ]:
            v1 = metrics_base.get(key)
            v2 = metrics_multi.get(key)
            s1 = fmt.format(v1) if v1 is not None else "N/A"
            s2 = fmt.format(v2) if v2 is not None else "N/A"
            print(f"  {key:<20} {s1:>12} {s2:>12}")