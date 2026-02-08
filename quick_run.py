from clear_day_analysis.tmy_reader import read_nsrdb_tmy_csv
from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.ashrae_clear_day import fit_ashrae_clear_day

# --- 1) Load TMY ---
path = r"C:\Users\manue_6t240gh\Dropbox\247Solar\01 - Ongoing\Hyder_Arizona\295968_33.01_-113.38_tmy-2024.csv"
df, md = read_nsrdb_tmy_csv(path)

print("Metadata:", md)
print("First datetime:", df["datetime"].iloc[0])
print("Rows:", len(df))

# --- 2) Compute sun position for ALL rows ---
# Use a small daylight threshold to avoid horizon numerical noise
df = compute_sun_position_columns(
    df,
    datetime_col="datetime",
    lat_deg=md.latitude,
    lon_deg=md.longitude,
    daylight_elevation_deg=2.0,
)

# Add day keys (useful for later steps)
df["date"] = df["datetime"].dt.date
df["doy"] = df["datetime"].dt.dayofyear

print("\nSample with sun columns:")
print(df[["datetime", "DNI", "sun_elevation_deg", "sun_azimuth_deg", "sun_is_daylight"]].head(12))

# --- 3) Fit ASHRAE clear-day model using iterative OLS + t-Student corridor ---
# Important:
# - dni_col must match the CSV column name ("DNI")
# - elevation_col must match the column created by compute_sun_position_columns ("sun_elevation_deg")
#
# alpha_min_deg: use 5° by default to avoid low-sun effects and heteroscedasticity near horizon
# confidence: corridor confidence level
# outlier_mode:
#   "lower"     -> remove only points below the lower corridor (physically typical: clouds reduce DNI)
#   "two_sided" -> remove both above and below
#
res = fit_ashrae_clear_day(
    df,
    dni_col="DNI",
    elevation_col="sun_elevation_deg",
    alpha_min_deg=5.0,
    confidence=0.95,
    outlier_mode="lower",
    max_iter=25,
    min_points=200,
)

print("\n=== ASHRAE clear-day fit result ===")
print(f"E0 = {res.E0:.3f}")
print(f"beta = {res.beta:.6f}")
print(f"converged = {res.converged}")
print(f"n_final = {res.n_final}")
print(f"iterations = {len(res.history)}")
print(f"alpha_min_deg = {res.alpha_min_deg}")
print(f"confidence = {res.confidence}")
print(f"outlier_mode = {res.outlier_mode}")

print("\nIteration history (last 10 iterations or fewer):")
for h in res.history[-10:]:
    print(
        f"it={h.iteration:02d}  n_in={h.n_in:6d}  outliers={h.n_outliers:6d}  kept={h.n_kept:6d}  "
        f"a={h.a: .6f}  b={h.b: .6f}  rmse={h.rmse: .6f}  tcrit={h.tcrit: .4f}"
    )