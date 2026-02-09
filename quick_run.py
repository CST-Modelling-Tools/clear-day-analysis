from clear_day_analysis.tmy_reader import read_nsrdb_tmy_csv
from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.ashrae_clear_day import fit_ashrae_clear_day
from clear_day_analysis.day_classification import (
    add_clear_dni_model,
    daily_dni_integral_ratio,
    classify_days_by_ratio,
)

# --- 1) Load TMY ---
path = r"C:\Users\manue_6t240gh\Dropbox\247Solar\01 - Ongoing\Hyder_Arizona\295968_33.01_-113.38_tmy-2024.csv"
df, md = read_nsrdb_tmy_csv(path)

print("Metadata:", md)
print("First datetime:", df["datetime"].iloc[0])
print("Rows:", len(df))

# --- 2) Compute sun position for ALL rows ---
df = compute_sun_position_columns(
    df,
    datetime_col="datetime",
    lat_deg=md.latitude,
    lon_deg=md.longitude,
    daylight_elevation_deg=2.0,
)

df["date"] = df["datetime"].dt.date
df["doy"] = df["datetime"].dt.dayofyear

print("\nSample with sun columns:")
print(df[["datetime", "DNI", "sun_elevation_deg", "sun_azimuth_deg", "sun_is_daylight"]].head(12))

# --- 3) Fit ASHRAE clear-day model (iterative OLS + t corridor) ---
# Enable envelope enforcement so daily integral ratios behave (ratio <= ~1).
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

print("\n=== ASHRAE clear-day fit result ===")
print(f"E0 = {res.E0:.3f}")
print(f"beta = {res.beta:.6f}")
print(f"converged = {res.converged}")
print(f"n_final = {res.n_final}")
print(f"iterations = {len(res.history)}")
print(f"enforce_envelope = {res.enforce_envelope}")
print(f"envelope_quantile = {res.envelope_quantile}")
print(f"delta_a (log shift) = {res.delta_a:.6f}  -> E0 multiplied by exp(delta_a)")

print("\nLast iteration (OLS line before envelope shift):")
h = res.history[-1]
print(
    f"it={h.iteration:02d}  n_in={h.n_in:6d}  outliers={h.n_outliers:6d}  kept={h.n_kept:6d}  "
    f"a={h.a: .6f}  b={h.b: .6f}  rmse={h.rmse: .6f}  tcrit={h.tcrit: .4f}"
)

# --- 4) Build clear-day DNI model per timestamp and compute daily integral ratios ---
df = add_clear_dni_model(
    df,
    E0=res.E0,
    beta=res.beta,
    dni_col="DNI",
    elevation_col="sun_elevation_deg",
    alpha_min_deg=5.0,
    clear_col="dni_clear_model",
)

daily = daily_dni_integral_ratio(
    df,
    datetime_col="datetime",
    dni_col="DNI",
    clear_col="dni_clear_model",
    alpha_min_deg=5.0,
    elevation_col="sun_elevation_deg",
)

# --- 5) Classify days based on integral ratio ---
daily_cls = classify_days_by_ratio(daily)

print("\n=== Day classification counts ===")
print(daily_cls["class"].value_counts())

# Diagnostics: check how many days still exceed 1.0 after envelope enforcement
n_gt_1 = int((daily_cls["ratio"] > 1.0).sum())
max_ratio = float(daily_cls["ratio"].max())
print(f"\nDays with ratio > 1.0: {n_gt_1} / {len(daily_cls)}   (max ratio = {max_ratio:.4f})")

print("\nTop 10 clearest days by ratio:")
print(daily_cls.sort_values("ratio", ascending=False).head(10)[["date", "ratio", "H_dni", "H_dni_clear", "n_points"]])

print("\nBottom 10 cloudiest days by ratio:")
print(daily_cls.sort_values("ratio", ascending=True).head(10)[["date", "ratio", "H_dni", "H_dni_clear", "n_points"]])

out_csv = "daily_classification.csv"
daily_cls.to_csv(out_csv, index=False)
print(f"\nWrote: {out_csv}")