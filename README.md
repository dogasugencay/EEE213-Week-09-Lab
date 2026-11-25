# EEE213 — Week 09 lab: machine learning on the CCPP data

Run from the repository folder:

```
python lab/w09.py
```

Requirements: Python 3, NumPy, pandas (`pip install numpy pandas`).

- `data/ccpp.csv` — Combined Cycle Power Plant data (9,568 rows; AT °C, V cm Hg, AP mbar, RH %, PE MW).
- `lab/w09.py` — reproduces every number in Sections 2.5 and 3–5 of the Week 09 form.
- `lab/w09_output.txt` — complete console output of the script.

Split: training = first 7,654 rows, test = last 1,914 rows, no shuffling; the code is deterministic, so no random seed is used.
