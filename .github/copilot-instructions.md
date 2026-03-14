# Copilot Instructions — Predictive Maintenance (Groundwater)

## Project Overview

Thesis project replacing Akvifär's regression-based groundwater reference method with a **state-space model** (Kalman filter/smoother) for anomaly detection. Base well **22W102** is jointly modelled with reference wells from SGU. Two variants compared via AIC/BIC: **baseline** (single ref 95_2) and **multivariate** (top-4 correlated refs filtered by aquifer+soil type). Code and comments are in **Swedish**.

## Key Files — What to Edit

- **`projekt_nivaer.py`** — The canonical script. All changes go here. Contains the full pipeline: data loading → SSM classes → fitting → forecasting → anomaly detection → visualization. Set `MODE = "baseline" | "multi" | "both"` in `__main__`.
- **`fetch_candidates.py`** — Pre-fetches SGU reference data to `ref_cache.json`. **Must be run separately** before `projekt_nivaer.py` multivariate mode (`py fetch_candidates.py`) because SGU API hangs when statsmodels/scipy are loaded in the same process.
- **`EXJOBB.py`** — Archived copy of `projekt_nivaer.py`. **Never edit.**
- **`main.py`** — Placeholder (no-op).

## Running the Project

```bash
uv sync                        # Python ≥ 3.12, deps in pyproject.toml
py fetch_candidates.py         # pre-fetch reference data (run once / when cache is stale)
python projekt_nivaer.py       # run pipeline (set MODE in __main__)
```

Outputs: `groundwater_ssm_*.png` (plots), `imputed_values_*.csv`, `diagnostik_*.json`.

## State-Space Model Architecture

Both `GroundwaterSSM` and `GroundwaterSSM_Multi` extend `statsmodels.tsa.statespace.mlemodel.MLEModel`.

- **State vector** (27 states): `[µ, γ_0, …, γ_25]` — latent level (random walk, no trend) + seasonal (nseason=26, half-year, sum-to-zero)
- **Observation equations**:
  - Base well: `y_base = µ + γ_0` (obs noise locked to 0 — base measurements treated as exact)
  - Baseline ref: `y_ref = beta · µ`
  - Multi refs: `y_ref_i = alpha_i + beta_i · µ + gamma_i · γ_0` (intercept via `obs_intercept` d-vector + seasonal loading)
- **Parameter transforms**: variance params use exp/log; beta, alpha, gamma unconstrained
- **Start params**: derived from univariate `UnobservedComponents` fit + OLS regression
- **Optimisation**: baseline uses `method="lbfgs"`; multi uses `method="powell"` with `cov_type="none"`

## Data Handling Conventions

- CSV input: `;`-separated, Swedish decimal commas (`","` → `"."`)
- All time series aligned to **weekly 7D DatetimeIndex** via `_resample_to_weekly()`
- Alignment: `reindex(method="nearest", tolerance=pd.Timedelta("4D"))`
- Short gaps (≤4 weeks): `interpolate(method="time", limit=4)`; longer gaps stay NaN
- Multi-model ref selection: `prepare_joint_dataframe_multi()` ranks by Pearson correlation, picks top-k (default 4)
- Functions return `pd.Series` (single well) or `pd.DataFrame` (column `"base"` + reference station IDs)

## Coding Conventions

- **Swedish** for all comments, variable names (`basrör`, `akvifer`, `jordart`), and print output
- Pipeline order in `__main__`: all network I/O first, then model fitting (avoids SSL/scipy conflicts)
- Reference data cached in `ref_cache.json` (dict of `{station_id: {date_str: float}}`, entries with `"__error__"` key are failures)

## Known Gotchas

- **SSL hangs**: SGU API calls hang after ~8-10 requests when statsmodels/scipy are loaded. Workaround: use `fetch_candidates.py` in a separate process.
- **`cov_type="none"`**: Multi model skips standard error computation. Consider `cov_type="approx"` if SEs needed.
- **sigma2_eta_season ≈ 0**: Seasonal process noise collapses — may need reparametrisation.
- **Negative betas**: Some refs show negative beta (inverse hydraulic connection) — verify physical plausibility before removing.
