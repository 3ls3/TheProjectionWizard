# berlin_rent_mock.py
import numpy as np
import pandas as pd

# ------------------------------- parameters -------------------------------
N_ROWS = 2_000              # change at will
DUPLICATE_FRAC = 0.02
MISSING_FRAC   = 0.03
RNG_SEED = 42
# --------------------------------------------------------------------------

rng = np.random.default_rng(RNG_SEED)

# Districts with €/m² multipliers (vs. city median 15.83 €)
district_info = {
    "Tiergarten":        1.51,
    "Friedrichshain":    1.32,
    "Mitte":             1.32,
    "Charlottenburg":    1.24,
    "Prenzlauer Berg":   1.20,
    "Schöneberg":        1.15,
    "Westend":           1.04,
    "Wedding":           0.97,
    "Lichtenberg":       0.92
}
districts, multipliers = zip(*district_info.items())
district_probs = np.array([0.07,0.12,0.08,0.11,0.12,0.10,0.08,0.17,0.15])  # pop-weighted

# helper distributions
def sample_size(n):
    # log-normal centred on 65 m²
    return np.clip(rng.lognormal(mean=np.log(65), sigma=0.4, size=n), 20, 150)

def rooms_from_size(size):
    # 1 room per 40 m², clipped 1-5
    noise = rng.normal(0.0, 0.3, size.shape)
    return np.clip(np.round(size / 40 + noise).astype(int), 1, 5)

def sample_year_built(n):
    buckets = [rng.choice(range(1880,1919)),
               rng.choice(range(1950,1991)),
               rng.choice(range(1991,2011)),
               rng.choice(range(2011,2025))]
    probs   = [0.32, 0.28, 0.25, 0.15]
    return rng.choice(buckets, p=probs, size=n)

def floor_from_year(yb):
    # Altbau → max 5 floors, Neubau → 6-7
    max_floor = np.where(yb>=1991, rng.integers(6,8), 6)
    return rng.integers(0, max_floor)

def sample_binary(p, n): return rng.random(n) < p

def sample_energy_class(n):
    # 1 (A+) … 8 (H), skewed toward mid-range
    return rng.choice(range(1,9), p=[0.03,0.05,0.08,0.15,0.25,0.2,0.15,0.09], size=n)

def distance_to_station(n):
    # kneecapped exponential: median ≈ 350 m, tail to 1500 m
    return np.minimum(rng.gamma(shape=2.0, scale=175, size=n), 1500).round()

# ---------------------------- generation pass -----------------------------
n = N_ROWS
df = pd.DataFrame({
    "district":          rng.choice(districts, p=district_probs, size=n),
    "size_sqm":          sample_size(n).round(1),
})

df["rooms"]              = rooms_from_size(df["size_sqm"])
df["year_built"]         = sample_year_built(n)
df["floor"]              = floor_from_year(df["year_built"])
df["balcony"]            = sample_binary(0.65, n).astype(int)
df["rooftop_terrace"]    = sample_binary(0.07, n).astype(int)
df["elevator"]           = np.where(df["year_built"]>=1991,
                                    sample_binary(0.9, n),
                                    sample_binary(0.35, n)).astype(int)
df["energy_class"]       = sample_energy_class(n)
df["distance_to_U_Sbahn_m"] = distance_to_station(n)

# --------------------------- rent calculation -----------------------------
base_rate = 15.83  # €/m² median
district_mult = df["district"].map(district_info)

# vintage premium/discount
vintage_adj = np.select(
    [df["year_built"]>=2011,
     df["year_built"]>=1991,
     df["year_built"]>=1950],
    [1.15, 1.05, 0.90],  # Neubau↑, 90s↑, Plattenbau↓
    default=1.00)

feature_adj = (
    1
    + 0.03*df["balcony"]
    + 0.06*df["rooftop_terrace"]
    + 0.02*df["elevator"]
    - 0.04*(df["energy_class"]-1)  # -4 % per efficiency step down
)

df["monthly_rent_eur"] = (
    df["size_sqm"]
    * base_rate
    * district_mult
    * vintage_adj
    * feature_adj
    * rng.lognormal(mean=0.0, sigma=0.07, size=n)   # noise
).round(0)

# ---------------------- inject duplicates & missing -----------------------
# duplicates
dup_rows = df.sample(frac=DUPLICATE_FRAC, random_state=RNG_SEED)
df = pd.concat([df, dup_rows], ignore_index=True)

# missing values (exclude target & district)
feature_cols = [c for c in df.columns if c not in ("district","monthly_rent_eur")]
n_missing = int(MISSING_FRAC * df[feature_cols].size)
for row_idx, col_idx in zip(
        rng.integers(len(df), size=n_missing),
        rng.integers(len(feature_cols), size=n_missing)):
    df.at[row_idx, feature_cols[col_idx]] = np.nan

# ----------------------------- export -------------------------------------
df.to_csv("berlin_flats_mock.csv", index=False)
print("Saved berlin_flats_mock.csv with", len(df), "rows.")