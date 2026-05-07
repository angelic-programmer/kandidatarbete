import numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kandidatuppsats_endast_basror import load_base_station, detect_frequency, prepare_series, fit_model

base_dir = Path(__file__).parent
for f, name in [("22W102","Rör A"),("G1101","Rör B"),("17XX01U","Rör C")]:
    base = load_base_station(base_dir / f"{f}.csv")
    freq = detect_frequency(base)
    y = prepare_series(base, freq)
    out = fit_model(y)
    p = np.array(out["results"].params)
    pn = list(out["results"].param_names)
    print(f"{name} ({f}): sigma2_eps={p[0]:.8f}, sigma2_eta={p[1]:.8f}")
