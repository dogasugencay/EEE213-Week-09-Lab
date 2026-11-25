"""EEE213 Week 09 lab - machine learning on the CCPP power-plant data.

Reproduces every number reported in Sections 2.5 and 3-5 of the form.
Run from the repository folder:   python lab/w09.py
Expects the data at data/ccpp.csv (relative to the repository folder).

Split: training = first 7,654 rows, test = last 1,914 rows, no shuffling.
Nothing is random, so no seed is used. Lines PE = a*x + b are least-squares
fits (numpy.polyfit, degree 1) on the training set; every MAE is on the test set.
"""
import platform
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "ccpp.csv"
N_TRAIN = 7654
UNITS = {"AT": "degC", "V": "cm Hg", "AP": "mbar", "RH": "%", "PE": "MW"}


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_line(train, feat):
    a, b = np.polyfit(train[feat], train["PE"], 1)
    return a, b


df = pd.read_csv(DATA)
train, test = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]

print(f"Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}")
print(f"Data: {len(df)} rows, train {len(train)}, test {len(test)}")

# ---------------- Section 3: data checks ----------------
print("\n[3.1] columns:", ", ".join(f"{c} ({UNITS[c]})" for c in df.columns))
print("[3.1] missing values:", int(df.isna().sum().sum()))
print("[3.1] exact duplicate rows (repeats after first):", int(df.duplicated().sum()),
      "| rows involved:", int(df.duplicated(keep=False).sum()))
print("[3.2] min - max:")
for c in df.columns:
    print(f"      {c}: {df[c].min():.2f} - {df[c].max():.2f} {UNITS[c]}")
atm = df["AP"] * 0.0750062  # mbar -> cm Hg
print(f"[3.2] RH > 100 %: {int((df['RH'] > 100).sum())} rows | "
      f"atmospheric {atm.min():.2f}-{atm.max():.2f} cm Hg | "
      f"V > own atmospheric: {int((df['V'] > atm).sum())} rows")
train_rows = set(map(tuple, train.values))
print(f"[3.3] repeats in train: {int(train.duplicated().sum())}, in test: "
      f"{int(test.duplicated().sum())}, test rows also in train: "
      f"{sum(tuple(r) in train_rows for r in test.values)}")
dedup = df.drop_duplicates()  # first occurrence kept, order preserved
tr_d, te_d = dedup[dedup.index < N_TRAIN], dedup[dedup.index >= N_TRAIN]
print(f"[3.3] after removing repeats: train {len(tr_d)}, test {len(te_d)}")

# ---------------- Section 4: statistics (all 9,568 rows, sample std) ----------------
for c in ["AT", "PE"]:
    s = df[c]
    print(f"[4.1] {c} mean / median / std: {s.mean():.2f} / {s.median():.2f} / "
          f"{s.std():.2f} {UNITS[c]}")
r = df.corr()["PE"].drop("PE")
print("[4.2] r with PE:", "  ".join(f"{c} {v:.3f}" for c, v in r.items()),
      f"| strongest: {r.abs().idxmax()}")

# ---------------- Section 5: baseline and lines ----------------
mean_pe = train["PE"].mean()
base = mae(test["PE"], mean_pe)
print(f"\n[5.1] training mean PE {mean_pe:.2f} MW | baseline test MAE {base:.2f} MW")
lines = {}
for feat in ["AT", "V", "AP", "RH"]:
    a, b = fit_line(train, feat)
    lines[feat] = (a, b, mae(test["PE"], a * test[feat] + b))
    print(f"[5.2] line from {feat}: a {a:.4f} MW per {UNITS[feat]}, b {b:.2f} MW, "
          f"test MAE {lines[feat][2]:.2f} MW")
print(f"[5.3] test MAE (MW): baseline {base:.2f} | AT {lines['AT'][2]:.2f} | "
      f"V {lines['V'][2]:.2f}")

a, b, _ = lines["V"]
i = N_TRAIN            # first test row, 0-based
x, y = df.loc[i, "V"], df.loc[i, "PE"]
pred = a * x + b
print(f"[5.5] file line {i + 2}: V {x} cm Hg -> predicted {pred:.2f} MW, "
      f"true {y:.2f} MW, abs error {abs(y - pred):.2f} MW")

# ---------------- Separate experiment (3.3): duplicates removed ----------------
base_d = mae(te_d["PE"], tr_d["PE"].mean())
mae_d = {f: mae(te_d["PE"], np.polyval(fit_line(tr_d, f), te_d[f])) for f in ["AT", "V"]}
print(f"\n[dedup] test MAE (MW): baseline {base_d:.2f} | AT {mae_d['AT']:.2f} | "
      f"V {mae_d['V']:.2f}")
