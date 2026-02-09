import numpy as np
import pandas as pd

from clear_day_analysis.ashrae_clear_day import fit_ashrae_clear_day


def test_fit_ashrae_clear_day_synthetic_envelope():
    rng = np.random.default_rng(123)

    E0_true = 1100.0
    beta_true = 0.12

    n = 2000
    alpha_deg = rng.uniform(6.0, 75.0, size=n)
    sin_alpha = np.sin(np.deg2rad(alpha_deg))

    dni_clear = E0_true * np.exp(-beta_true / sin_alpha)

    # Clear-ish noise in log-space
    noise_log = rng.normal(0.0, 0.03, size=n)
    dni_meas = dni_clear * np.exp(noise_log)

    # Inject cloudy attenuation on 60% of points
    cloudy = rng.random(n) < 0.60
    attenuation = rng.uniform(0.05, 0.7, size=n)
    dni_meas[cloudy] *= attenuation[cloudy]

    dni_meas = np.clip(dni_meas, 1e-6, None)

    df = pd.DataFrame({"DNI": dni_meas, "sun_elevation_deg": alpha_deg})

    res = fit_ashrae_clear_day(
        df,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=5.0,
        confidence=0.95,
        outlier_mode="lower",
        max_iter=50,
        min_points=200,
        eps_a=1e-4,
        eps_b=1e-4,
        min_outliers_to_continue=2,
        enforce_envelope=True,
        envelope_quantile=0.98,
    )

    assert res.converged is True
    assert np.isfinite(res.E0) and res.E0 > 0
    assert np.isfinite(res.beta) and res.beta > 0
    assert abs(res.beta - beta_true) < 0.03

    dni_fit = res.E0 * np.exp(-res.beta / sin_alpha)
    ratio = dni_meas / dni_fit

    # With envelope enforcement at 0.98, only a small fraction should exceed 1
    assert float(np.mean(ratio > 1.0)) < 0.10