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


# Hämtar referensrör 95_2 via SGUClient
def load_reference_station() -> pd.Series:
    from sgu_client import SGUClient
    with SGUClient() as client:
        meas = client.levels.observed.get_measurements_by_name(
            station_id="95_2"
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
    ref = pd.to_numeric(ref, errors="coerce")
    print("✓ Referensdata hämtad från SGUClient (95_2)")
    return ref


# Nedan skapas en gemensam dataframe av basröret och referensröret
def prepare_joint_dataframe(
    base: pd.Series,
    ref: pd.Series,
    # frekvensen sätts veckovis
    freq: str = "7D",  
) -> pd.DataFrame:
    # Ta bort tidszon (SGU returnerar UTC och basröret har ingen tidszon)
    if ref.index.tz is not None:
        ref = ref.copy()
        ref.index = ref.index.tz_localize(None)
    if base.index.tz is not None:
        base = base.copy()
        base.index = base.index.tz_localize(None)

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
        beta_ref = float(ols[0][0])
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
#  ANPASSA OCH PREDIKTERA
# ─────────────────────────────────────────────

def fit_model(df: pd.DataFrame, nseason: int = 26) -> tuple:
    """Anpassar SSM och returnerar (result, model)."""
    endog = df[["base", "ref"]].values.astype(float)

    model = GroundwaterSSM(endog, nseason=nseason)

    print("Anpassar modellen (MLE med Kalman filter)...")
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
    s2_level       = result.params[0]
    last_std       = std[-1]
    fcast_mean     = np.full(n_forecast, pred_mean[-1])
    fcast_std      = np.sqrt(last_std**2 + np.arange(1, n_forecast + 1) * s2_level)
    fcast_ci       = np.column_stack([
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
        "observed_ref":   df["ref"].values,
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
    in_streak = 0
    streak_start = None
    for i, val in enumerate(outside):
        if val:
            if in_streak == 0:
                streak_start = i
            in_streak += 1
        else:
            if in_streak >= min_consecutive:
                anomaly.iloc[streak_start:i] = True
            in_streak = 0
            streak_start = None
    if in_streak >= min_consecutive:
        anomaly.iloc[streak_start:] = True
    return anomaly


# ─────────────────────────────────────────────
#  VISUALISERING
# ─────────────────────────────────────────────

def plot_results(out: dict, anomaly: pd.Series, title_base: str = "22W102"):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        "Grundvatten State Space Model\n"
        f"Basobjekt: {title_base} | Referens: 95_2",
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

    # 1. Ladda data
    print("=== Laddar data ===")
    base = load_base_station()
    ref  = load_reference_station()
    print(f"Basobjekt 22W102: {len(base)} rader, {base.isna().sum()} saknade värden")
    print(f"Referens 95_2:    {len(ref)} rader, {ref.isna().sum()} saknade värden")

    # 2. Bygg gemensamt datumindex
    df = prepare_joint_dataframe(base, ref, freq="7D")
    print(f"\nGemensamt dataset: {len(df)} veckor "
          f"({df.index[0].date()} – {df.index[-1].date()})")
    print(f"Saknade i base: {df['base'].isna().sum()}")
    print(f"Saknade i ref:  {df['ref'].isna().sum()}")

    # 3. Anpassa modellen
    print("\n=== Anpassar State Space Model ===")
    result, model = fit_model(df, nseason=26)

    # 4. Kör Kalman smoother + prognos (26 veckor = ~6 månader)
    print("\n=== Kör Kalman smoother + prognos ===")
    out = smooth_and_forecast(result, df, n_forecast=26)

    # 5. Imputation-rapport
    imputed = imputation_report(df, out)

    # 6. Avvikelsedetektering
    anomaly = detect_anomalies(
        observed=out["observed_base"],
        predicted=out["pred_base"],
        pred_ci=out["pred_ci"],
        min_consecutive=3,
    )
    n_anom = anomaly.sum()
    print(f"\nAvvikelsedetektering: {n_anom} tidpunkter flaggade "
          f"({n_anom/len(anomaly)*100:.1f}%)")

    # 7. Plotta och spara
    print("\n=== Genererar plot ===")
    fig = plot_results(out, anomaly)
    fig.savefig("groundwater_ssm_results.png", dpi=150, bbox_inches="tight")
    print("Plot sparad till: groundwater_ssm_results.png")
    plt.show()

    # 8. Spara imputerade värden
    if not imputed.empty:
        imputed.to_csv("imputed_values.csv")
        print("Imputerade värden sparade till: imputed_values.csv")

    # 9. Skattade parametrar
    print("\n=== Skattade parametrar ===")
    params = dict(zip(model.param_names, result.params))
    for k, v in params.items():
        print(f"  {k:<28} = {v:.6f}")