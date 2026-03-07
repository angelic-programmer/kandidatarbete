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
    """
    Custom Kalman Filter för grundvattennivå med referensrör.

    Tillståndsvektor (k_states = 2 + nseason + 1):
        [0] µ          – lokal nivå (m ö.h.)
        [1] ν          – lokal trend (m/vecka)
        [2] γ_0        – aktuell säsongskomponent
        [3..s-1]       – laggade säsongskomponenter
        [s]  α         – latent aquifer-komponent

    Observationsvektor (k_endog = 2):
        y[0] = basobjekt  (22W102)   – direkt nivåmätning
        y[1] = referensobjekt (95_2) – korrelerad nivåmätning
    """

    param_names = [
        "sigma2_eta_level",   # process noise: nivå
        "sigma2_eta_trend",   # process noise: trend
        "sigma2_eta_season",  # process noise: säsong
        "sigma2_eta_aquifer", # process noise: latent aquifer
        "sigma2_eps_base",    # obs noise: basobjekt
        "sigma2_eps_ref",     # obs noise: referensobjekt
        "phi_trend",          # AR(1) för trend (dämpning)
        "rho_aquifer",        # AR(1) för latent aquifer
        "beta_ref",           # koppling nivå → referens
        "gamma_ref",          # koppling aquifer → referens
    ]

    def __init__(self, endog, nseason=26, **kwargs):
        self.nseason = nseason
        k_states = 2 + nseason + 1  # nivå, trend, nseason säsongslägen, aquifer
        k_posdef = k_states

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            **kwargs,
        )

        self.idx_level   = 0
        self.idx_trend   = 1
        self.idx_season  = slice(2, 2 + nseason)
        self.idx_aquifer = 2 + nseason

        # Diffus initiering för nivå/trend (okänt startvärde)
        self.ssm.initialize_approximate_diffuse()

    @property
    def start_params(self):
        y_std = np.nanstd(self.endog[:, 0])
        return np.array([
            (0.02 * y_std) ** 2,  # sigma2_eta_level
            (0.005 * y_std) ** 2, # sigma2_eta_trend
            (0.01 * y_std) ** 2,  # sigma2_eta_season
            (0.05 * y_std) ** 2,  # sigma2_eta_aquifer
            (0.1 * y_std) ** 2,   # sigma2_eps_base
            (0.15 * y_std) ** 2,  # sigma2_eps_ref
            0.95,                 # phi_trend
            0.90,                 # rho_aquifer
            0.85,                 # beta_ref
            0.30,                 # gamma_ref
        ])

    @property
    def param_bounds(self):
        return [
            (1e-6, None),    # sigma2_eta_level
            (1e-6, None),    # sigma2_eta_trend
            (1e-6, None),    # sigma2_eta_season
            (1e-6, None),    # sigma2_eta_aquifer
            (1e-6, None),    # sigma2_eps_base
            (1e-6, None),    # sigma2_eps_ref
            (-0.999, 0.999), # phi_trend
            (-0.95, 0.95),   # rho_aquifer (begränsat för stabilitet)
            (None, None),    # beta_ref
            (None, None),    # gamma_ref
        ]

#CLAUDE FÖRKLARAR HIT



    def transform_params(self, unconstrained):
        p = unconstrained.copy()
        # Varianser: exp-transform → alltid positiva
        p[:6] = np.exp(unconstrained[:6])
        # AR-koefficienter: tanh → (-1, 1)
        p[6] = np.tanh(unconstrained[6])
        p[7] = np.tanh(unconstrained[7])
        return p

    def untransform_params(self, constrained):
        p = constrained.copy()
        p[:6] = np.log(constrained[:6])
        p[6] = np.arctanh(constrained[6])
        p[7] = np.arctanh(constrained[7])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)

        (s2_level, s2_trend, s2_season, s2_aquifer,
         s2_base, s2_ref, phi_trend, rho_aquifer,
         beta_ref, gamma_ref) = params

        ns = self.nseason
        k  = self.k_states

        # ── Transitionsmatris T ──────────────────────────────────────────
        T = np.zeros((k, k))

        # Nivå: µ(t+1) = µ(t) + ν(t)
        T[0, 0] = 1.0
        T[0, 1] = 1.0

        # Trend: ν(t+1) = phi * ν(t)
        T[1, 1] = phi_trend

        # Säsong med summavillkor:
        #   γ_0(t+1) = -γ_0(t) - γ_1(t) - ... - γ_{s-2}(t)
        #   γ_j(t+1) = γ_{j-1}(t)  för j=1..s-1
        T[2, 2:2+ns] = -1.0
        for j in range(1, ns):
            T[2 + j, 2 + j - 1] = 1.0

        # Latent aquifer: α(t+1) = rho * α(t)
        T[k-1, k-1] = rho_aquifer

        self.ssm["transition"] = T

        # ── Observationsmatris Z ─────────────────────────────────────────
        # y_base = µ + γ_0          (direkt nivå + säsong)
        # y_ref  = beta*µ + gamma*α (referens kopplad via nivå & aquifer)
        Z = np.zeros((2, k))
        Z[0, 0] = 1.0          # base ← nivå
        Z[0, 2] = 1.0          # base ← säsong
        Z[1, 0] = beta_ref     # ref  ← nivå
        Z[1, k-1] = gamma_ref  # ref  ← latent aquifer

        self.ssm["design"] = Z

        # ── Observationsbrus ────────────────────────────────────────────
        self.ssm["obs_cov"] = np.diag([s2_base, s2_ref])

        # ── Processkörsbrus Q ───────────────────────────────────────────
        Q = np.zeros((k, k))
        Q[0, 0] = s2_level
        Q[1, 1] = s2_trend
        Q[2, 2] = s2_season  # brus enbart på den aktuella säsongskomponenten
        Q[k-1, k-1] = s2_aquifer
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
        method="nm",
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

    pred        = result.get_prediction()
    pred_ci_arr = np.array(pred.conf_int(alpha=0.05))
    # Väljer kolumn 0 (base lower) och kolumn 2 (base upper) – ignorerar ref-intervallet
    pred_ci     = np.column_stack([pred_ci_arr[:, 0], pred_ci_arr[:, 2]])
    pred_mean   = np.array(pred.predicted_mean).flatten()

    forecast     = result.get_forecast(steps=n_forecast)
    fcast_ci_arr = np.array(forecast.conf_int(alpha=0.05))
    fcast_ci     = np.column_stack([fcast_ci_arr[:, 0], fcast_ci_arr[:, 2]])
    fcast_mean   = np.array(forecast.predicted_mean).flatten()

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
        "latent_aquifer": smoothed.smoothed_state[-1],
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
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False)
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
    latent   = out["latent_aquifer"].flatten()[:len(idx)]

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

    # ── Panel 3: Latent aquifer-komponent (marktyp/jordart) ──────────────
    ax3 = axes[2]
    ax3.set_title("Latent aquifer-komponent α(t)  [marktyp / jordart]", fontsize=11)
    ax3.plot(idx, latent, color="saddlebrown", lw=1.5)
    ax3.axhline(0, color="black", lw=0.5, linestyle="--")
    ax3.set_ylabel("α (dimensionslös)")
    ax3.set_xlabel("Datum")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
    ax3.grid(alpha=0.3)

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