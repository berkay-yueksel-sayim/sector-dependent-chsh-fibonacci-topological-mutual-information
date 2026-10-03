#!/usr/bin/env python3
"""Aggregation and BIC model comparison for the finite-size scaling run.

This script regenerates stufe2_analyse.json from the raw run stufe2_results.json.
Together they are the generating chain behind Table~\\ref{tab:z2_scaling} (the
per-cell mutual information) and Table~\\ref{tab:z2_bic} (the model comparison).

WHAT IT DOES

  1. Aggregates the 100 individual runs by (engine, L). For each cell it reports
     the mean of I_corr over the 10 seeds, its standard error, the same two
     quantities for the sector-shuffled surrogate, and the standardized contrast
     between them.

  2. Fits three pre-registered models to the surrogate-subtracted signal, jointly
     over L in {8, 16, 32, 48, 64}, per engine:

        M2:    I ~ a * L^b            (pure power law, 2 parameters)
        M2c:   I ~ a * L^b + c        (power law with residual offset, 3)
        Mexp:  I ~ a * exp(-b*L) + c  (exponential with residual offset, 3)

     Weighted least squares with absolute_sigma, then BIC = chi2 + k*ln(n).

ON THE FIELD NAME sig_minus_scrambled. The paper calls this null hypothesis the
permutation null (Sec. II E) and writes I_shuffled for the surrogate; this file
and the deposited JSON call the same quantity scrambled. The three words denote
one and the same object: the sector labels are randomly permuted across
configurations, 2000 shuffles per seed. The field name is kept as it is because
stufe2_analyse.json was published with this key; renaming it here would break
the key of a record that is already archived. See the README for the mapping.

THE JSON KEYS ARE THE PUBLISHED ONES. summary_per_cell, fits, and every field
below reproduce the deposited file exactly. Run with --check to verify that
against the copy in this directory instead of overwriting it.

Requires NumPy and SciPy (scipy.optimize.curve_fit). The plotting step of the
original working script is deliberately omitted here: it writes an image that is
not part of this record and has no effect on the JSON.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "stufe2_results.json")
OUT = os.path.join(HERE, "stufe2_analyse.json")

ENGINES = ["v05", "vec"]
SIZES = [8, 16, 32, 48, 64]


def model_M2(L, a, b):
    return a * L ** b


def model_M2c(L, a, b, c):
    return a * L ** b + c


def model_Mexp(L, a, b, c):
    return a * np.exp(-b * L) + c


def fit_model(model_fn, L_arr, y_arr, sem_arr, p0):
    """Weighted least squares fit, returns parameters plus BIC."""
    try:
        popt, pcov = curve_fit(model_fn, L_arr, y_arr,
                               p0=p0, sigma=sem_arr,
                               absolute_sigma=True, maxfev=10000)
    except Exception:
        return None
    pred = model_fn(L_arr, *popt)
    residuals = (y_arr - pred) / sem_arr
    chi2 = float(np.sum(residuals ** 2))
    k = len(popt)
    n = len(L_arr)
    bic = chi2 + k * np.log(n)
    perr = np.sqrt(np.diag(pcov))
    return {"params": popt.tolist(), "perr": perr.tolist(),
            "chi2": chi2, "bic": float(bic), "k": k, "n": n,
            "predictions": pred.tolist()}


def aggregate(data):
    """Mean, standard error and surrogate contrast per (engine, L) cell."""
    cells = defaultdict(list)
    scrambled = defaultdict(list)
    for r in data["results"]:
        cells[(r["engine"], r["L"])].append(r["mi_full"]["I_corr"])
        scrambled[(r["engine"], r["L"])].append(r["mi_scrambled"]["I_corr"])

    summary = {}
    for engine in ENGINES:
        for L in SIZES:
            I = np.array(cells[(engine, L)])
            S = np.array(scrambled[(engine, L)])
            I_mean = float(I.mean())
            I_sem = float(I.std(ddof=1) / np.sqrt(len(I)))
            S_mean = float(S.mean())
            S_sem = float(S.std(ddof=1) / np.sqrt(len(S)))
            sig = (I_mean - S_mean) / np.sqrt(I_sem ** 2 + S_sem ** 2)
            summary[(engine, L)] = {
                "I_mean": I_mean, "I_sem": I_sem,
                "scrambled_mean": S_mean, "scrambled_sem": S_sem,
                "sig_minus_scrambled": float(sig),
                "n_seeds": len(I),
            }
    return summary


def fit_all(summary):
    fits = {}
    for engine in ENGINES:
        L_arr = np.array(SIZES, dtype=float)
        y_arr = np.array([summary[(engine, L)]["I_mean"]
                          - summary[(engine, L)]["scrambled_mean"] for L in SIZES])
        sem_arr = np.array([math.sqrt(summary[(engine, L)]["I_sem"] ** 2
                                      + summary[(engine, L)]["scrambled_sem"] ** 2)
                            for L in SIZES])
        f_M2 = fit_model(model_M2, L_arr, y_arr, sem_arr, p0=[10.0, -2.0])
        f_M2c = fit_model(model_M2c, L_arr, y_arr, sem_arr, p0=[10.0, -2.0, 0.0])
        f_exp = fit_model(model_Mexp, L_arr, y_arr, sem_arr, p0=[0.1, 0.1, 0.0])
        eng = {"M2": f_M2, "M2c": f_M2c, "Mexp": f_exp,
               "L_values": L_arr.tolist(),
               "y_values": y_arr.tolist(),
               "sem_values": sem_arr.tolist()}
        bics = {k: v["bic"] for k, v in (("M2", f_M2), ("M2c", f_M2c), ("Mexp", f_exp)) if v}
        if bics:
            best = min(bics, key=bics.get)
            ordered = sorted(bics.values())
            dbic = (ordered[1] - ordered[0]) if len(ordered) > 1 else float("inf")
            eng["best_model"] = best
            eng["dBIC_best_vs_second"] = float(dbic)
        fits[engine] = eng
    return fits


def vergleich(a, b, pfad="", diffs=None):
    """Elementwise comparison; returns the list of differing paths."""
    if diffs is None:
        diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append("%s: keys %s vs %s" % (pfad, sorted(a), sorted(b)))
            return diffs
        for k in sorted(a):
            vergleich(a[k], b[k], pfad + "/" + str(k), diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append("%s: length %d vs %d" % (pfad, len(a), len(b)))
            return diffs
        for i, (x, y) in enumerate(zip(a, b)):
            vergleich(x, y, pfad + "[%d]" % i, diffs)
    elif isinstance(a, float) and isinstance(b, float):
        if not (a == b or (abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b)))):
            diffs.append("%s: %.17g vs %.17g" % (pfad, a, b))
    elif a != b:
        diffs.append("%s: %r vs %r" % (pfad, a, b))
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="compare against the deposited JSON instead of writing it")
    args = ap.parse_args()

    with open(RAW, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = aggregate(data)
    fits = fit_all(summary)

    print("%4s %3s %13s %10s %13s %10s %8s"
          % ("eng", "L", "I_mean", "sem", "scrambled", "sem", "sigma"))
    for engine in ENGINES:
        for L in SIZES:
            v = summary[(engine, L)]
            print("%4s %3d %+13.6f %10.6f %+13.6f %10.6f %+8.1f"
                  % (engine, L, v["I_mean"], v["I_sem"],
                     v["scrambled_mean"], v["scrambled_sem"],
                     v["sig_minus_scrambled"]))
    for engine in ENGINES:
        e = fits[engine]
        print("%s: best %s, dBIC to next %.2f"
              % (engine, e.get("best_model"), e.get("dBIC_best_vs_second", float("nan"))))

    out = {"summary_per_cell": {"%s_L%d" % (e, L): v for (e, L), v in summary.items()},
           "fits": fits}

    if args.check:
        with open(OUT, "r", encoding="utf-8") as f:
            deposited = json.load(f)
        diffs = vergleich(out, deposited)
        print("\ncheck against %s: %d difference(s)"
              % (os.path.basename(OUT), len(diffs)))
        for d in diffs[:20]:
            print("   %s" % d)
        raise SystemExit(0 if not diffs else 1)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n-> %s" % os.path.basename(OUT))


if __name__ == "__main__":
    main()
