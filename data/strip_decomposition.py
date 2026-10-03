#!/usr/bin/env python3
"""Decomposition of the gauge-invariant plaquette-strip measurement.

The main text reports that the association between the topological sector
label and the CHSH value is carried by the excitation-number component of the
label and not by its Wilson-loop component.  This script produces the numbers
behind that statement, so that each of them has a field in the deposited
record rather than living only in a working file.

Two quantities are produced:

1.  The decomposition of the beta = 2.0 strip measurement into

        V       1 if at least two anyons are present, else 0
        wilson  (W_x, W_y)                 -- the topological sector alone
        anyons  min(n_anyons // 2, 10)     -- the excitation-number component
        full    (W_x, W_y, anyons)         -- the label used in the main text

    over the same 21 seeds, with the circular-shift null.  The `full` variant
    must reproduce the deposited strip result; that is checked below and the
    script stops if it does not.

2.  The aggregate of `tau_int_anyons` over seeds, for all five couplings.
    The runner stores this quantity per seed but not as an aggregate, unlike
    the three other autocorrelation times it records.  The aggregation rule is
    not assumed: it is verified against the three aggregates that are present
    (they are the arithmetic mean over seeds, reproduced to 1e-14) and only
    then applied to the fourth.

The label series are not reconstructed here.  `null_calibration.strip_series`
is called unchanged and the four component series are derived from the full
label it returns, so this script cannot drift from the measurement it
decomposes.

Draws are keyed to the identity of the cell, so the output does not depend on
how many other quantities were computed in the same run.  The reported numbers
are all corrected against the circular-shift null, which is exhaustive over
all offsets and draws nothing.
"""
import sys

sys.dont_write_bytecode = True

import importlib.util
import io
import json
import os
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HIER = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(os.path.dirname(HIER), "sim")
for p in (HIER, SIM):
    if p not in sys.path:
        sys.path.insert(0, p)


def lade(name, pfad):
    spec = importlib.util.spec_from_file_location(name, pfad)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


nc = lade("nc", os.path.join(HIER, "null_calibration.py"))


def engine_class():
    """The sampler, loaded only when PART 2 needs it.  It is not part of the
    distributed package; null_calibration.require_engine says what runs
    without it.  Loading it at import time stopped the script before PART 1,
    which needs nothing but the deposited record."""
    nc.require_engine("the strip decomposition")
    return lade("z2_vec_rep", os.path.join(SIM, "z2_vec_rep.py")).Z2GaugeVecRep

SEEDS = [42] + list(range(10, 30))
BETAS = ("1.0", "1.5", "2.0", "2.5", "3.0")
RECORD = os.path.join(HIER, "null_calibration_results_vecrep_v2.json")
AUS = os.path.join(HIER, "strip_decomposition_results.json")

# The deposited strip result the `full` variant has to reproduce.
STRIP_SOLL = {"corrected_mean": 0.017257465836938317,
              "ci95": [0.015889396213472885, 0.018660999229081137],
              "sigma_mean": 10.472477806865939,
              "n_gt3": 21}

print("=" * 92)
print("PART 1  tau_int_anyons aggregate")
print("=" * 92)
rec = json.load(io.open(RECORD, encoding="utf-8"))
BEKANNT = ("tau_int_labels", "tau_wilson_joint_match", "tau_wilson_max_of_marginals")


def mittel_ueber_seeds(knoten, feld):
    ps = knoten["per_seed"]
    w = [ps[s][feld] for s in ps if feld in ps[s]]
    return (sum(w) / len(w), len(w)) if w else (None, 0)


# The rule is verified before it is used.
schlimmste = 0.0
for b in BETAS:
    for feld in BEKANNT:
        m, n = mittel_ueber_seeds(rec["decomposition"][b], feld)
        soll = rec["decomposition"][b]["aggregate_all_seeds"][feld]
        schlimmste = max(schlimmste, abs(m - soll))
print("  rule check: aggregate == mean over seeds for the three recorded")
print("              autocorrelation times, 15 cells, worst deviation %.1e" % schlimmste)
if schlimmste > 1e-9:
    print("  STOP: the aggregation rule does not hold; do not aggregate.")
    sys.exit(1)

tau_anyons = {}
for b in BETAS:
    m, n = mittel_ueber_seeds(rec["decomposition"][b], "tau_int_anyons")
    ps = rec["decomposition"][b]["per_seed"]
    w = sorted(ps[s]["tau_int_anyons"] for s in ps if "tau_int_anyons" in ps[s])
    tau_anyons[b] = {"mean": m, "n_seeds": n, "min": w[0], "max": w[-1]}
    print("  beta=%-4s n=%2d  mean %11.6f   min %10.6f  max %10.6f" % (b, n, m, w[0], w[-1]))

# The spread across all four definitions, at the coupling where it matters.
vier = {f: rec["decomposition"]["3.0"]["aggregate_all_seeds"][f] for f in BEKANNT}
vier["tau_int_anyons"] = tau_anyons["3.0"]["mean"]
lo, hi = min(vier.values()), max(vier.values())
print("  spread of the four definitions at beta = 3.0: %.2f to %.2f, factor %.2f"
      % (lo, hi, hi / lo))

print()
print("=" * 92)
print("PART 2  strip decomposition at beta = 2.0, %d seeds" % len(SEEDS))
print("=" * 92)
ENGINE = engine_class()
daten = {k: {"I": [], "korr": [], "sigma": []} for k in ("V", "wilson", "anyons", "full")}
t0 = time.time()
for seed in SEEDS:
    # The deposited routine, called unchanged.
    labels, sb, S_best = nc.strip_series(seed, ENGINE)
    reihen = {
        "full": labels,
        "wilson": [(w[0], w[1]) for w in labels],
        "anyons": [w[2] for w in labels],
        "V": [1 if w[2] >= 1 else 0 for w in labels],
    }
    for name in ("V", "wilson", "anyons", "full"):
        r = nc.both_nulls(reihen[name], sb, nc.rng_for("stripdecomp", seed, name))
        daten[name]["I"].append(r["I_raw"])
        daten[name]["korr"].append(r["I_raw"] - r["E_circ"])
        daten[name]["sigma"].append(r["sigma_circ"])
print("  %d seeds x 4 variants in %.1f min" % (len(SEEDS), (time.time() - t0) / 60.0))

print()
print("  variant  | I_raw    | I - E_circ | 95%% CI                 | sigma_circ | >3sig")
erg = {}
for name in ("V", "wilson", "anyons", "full"):
    I = np.array(daten[name]["I"])
    k = np.array(daten[name]["korr"])
    s = np.array(daten[name]["sigma"])
    loc, hic = nc.bootstrap_ci(k)
    print("  %-8s | %.6f | %+.6f  | [%+.6f, %+.6f] | %9.2f | %2d/%d"
          % (name, I.mean(), k.mean(), loc, hic, s.mean(), int((s > 3).sum()), len(SEEDS)))
    erg[name] = {"I_raw_mean": float(I.mean()), "corrected_mean": float(k.mean()),
                 "ci95": [loc, hic], "sigma_circ_mean": float(s.mean()),
                 "n_gt3sigma_circ": int((s > 3).sum()), "n_seeds": len(SEEDS)}

print()
print("=" * 92)
print("CHECKS")
print("=" * 92)
ok = True
for feld, soll in (("corrected_mean", STRIP_SOLL["corrected_mean"]),
                   ("sigma_circ_mean", STRIP_SOLL["sigma_mean"])):
    d = abs(erg["full"][feld] - soll)
    print("  full/%-16s %.12f  vs deposited %.12f  delta %.2e %s"
          % (feld, erg["full"][feld], soll, d, "ok" if d < 1e-9 else "MISMATCH"))
    ok = ok and d < 1e-9
for i, soll in enumerate(STRIP_SOLL["ci95"]):
    d = abs(erg["full"]["ci95"][i] - soll)
    print("  full/ci95[%d]            %.12f  vs deposited %.12f  delta %.2e %s"
          % (i, erg["full"]["ci95"][i], soll, d, "ok" if d < 1e-9 else "MISMATCH"))
    ok = ok and d < 1e-9
print("  full/n_gt3sigma_circ     %d  vs deposited %d  %s"
      % (erg["full"]["n_gt3sigma_circ"], STRIP_SOLL["n_gt3"],
         "ok" if erg["full"]["n_gt3sigma_circ"] == STRIP_SOLL["n_gt3"] else "MISMATCH"))
ok = ok and erg["full"]["n_gt3sigma_circ"] == STRIP_SOLL["n_gt3"]

# A check that only fires if the components really are different series.
gleich = (erg["wilson"]["corrected_mean"] == erg["anyons"]["corrected_mean"])
print("  components are distinct series: %s" % ("NO -- suspect" if gleich else "yes"))
ok = ok and not gleich

if not ok:
    print()
    print("  STOP: the full variant does not reproduce the deposited strip result.")
    print("  Nothing written.")
    sys.exit(1)

aus = {"description": "Decomposition of the beta = 2.0 gauge-invariant strip "
                      "measurement into the components of the sector label, and "
                      "the seed aggregate of tau_int_anyons for all couplings.",
       "source_record": os.path.basename(RECORD),
       "null_model": "circular shift, exhaustive over all offsets",
       "beta": 2.0, "seeds": SEEDS,
       "components": erg,
       "tau_int_anyons_aggregate": tau_anyons,
       "tau_definitions_at_beta_3.0": vier,
       "runtime_min": round((time.time() - t0) / 60.0, 1)}
io.open(AUS, "w", encoding="utf-8", newline="\n").write(json.dumps(aus, indent=2) + "\n")
print()
print("  written: %s" % os.path.basename(AUS))
