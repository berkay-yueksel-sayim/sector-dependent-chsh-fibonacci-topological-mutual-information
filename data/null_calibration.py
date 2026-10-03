#!/usr/bin/env python3
"""Null-model calibration and engine re-measurement for the Z2 analysis.

WHAT THIS SCRIPT IS FOR
-----------------------
Two things had to be corrected in this analysis, and this script produces
every number that follows from both.

(1) The Monte Carlo engine.  `z2_sim_v05`, which produced all simulation
    results of v2.0-v2.3, does not sample the Boltzmann distribution.  Its
    update schedule -- all horizontal links in row-major order, then all
    vertical links, with deterministic acceptance at dE = 0 -- is reducible:
    time averages depend on the initial configuration.  Measured against the
    exact single-plaquette expectation on the torus (see bp_exact_constrained;
    tanh(beta) is its infinite-size limit and is not the right reference at
    small excitation number).  Deviations below are z-scores over the 20
    independent seeds of the validation gate, so the denominator is the
    scatter of seed means and carries no autocorrelation:

        beta = 1, L = 16, 4000 measurements per seed, 20 seeds
        z2_sim_v05                          not admissible, reducible
        z2_vec                              z = +14.2
        z2_vec with randomized
        acceptance at dE = 0 (used here)    z = -1.2

    All results here use the last of these (`z2_vec_rep.Z2GaugeVecRep`), and
    the four validation cells are written to engine_validation.json before the
    first mutual information is computed.

(2) The null model.  Significance was previously assessed against a
    permutation null: the sector-label sequence is randomly permuted while
    the CHSH sequence is held fixed.  That destroys the temporal structure of
    a Markov chain.  Where sector transitions are rare, the permuted
    surrogate is far narrower than the real fluctuation and the resulting
    z-score is not calibrated.  This script measures both that null and a
    circular-shift null, which displaces one record against the other and is
    evaluated over *all* n-1 offsets, so it carries no free parameter.  For
    both it also measures the false-positive level directly, by pairing
    records that cannot be related by construction.

OUTPUT
------
null_calibration_results.json:

  validation      the engine gate: <B_p> against the exact expectation
  decomposition   per (beta, label, seed): I_raw, E_perm, sigma_perm,
                  E_circ, sigma_circ; per (beta, label) the bootstrap
                  interval for I - E_circ and the significance counts
  false_alarm_*   the measured false-positive level of both nulls, from all
                  ordered pairs of independent seeds
  strip           the 21-seed gauge-invariant plaquette-strip route
  scaling         the L-dependence, in the configuration of the deposited
                  record, with tau_int and N_eff per cell
  long_run        the run-length control: 40000 measurements per seed

PROVENANCE
----------
The sampler, the CHSH settings rule and the mutual-information estimator are
imported read-only.  `z2_sim_v05` is imported for correlation_matrix /
best_chsh / chsh_per_config only -- pure analysis helpers that exist in no
other module and contain no sampling.  A guard installed at start-up makes
that statement enforceable: constructing the superseded sampler raises.  The
guard is lifted only in the regression mode (--engine v05), whose sole
purpose is to show that the analysis path itself is unchanged.

Usage:
    python null_calibration.py                 # the full re-measurement
    python null_calibration.py --tracer        # one cell, for verification
    python null_calibration.py --engine v05    # regression on the old engine
"""
import sys

sys.dont_write_bytecode = True          # leave no bytecode in the record

import hashlib
import io
import itertools
import json
import math
import os
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(HERE)
for p in (HERE, os.path.join(SANDBOX, "sim")):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# z2_sim_v05 is imported for correlation_matrix / best_chsh / chsh_per_config
# only.  These are pure analysis helpers with no sampling, and they exist in
# no other module.  The sampler in that file is superseded: it does not sample
# the Boltzmann distribution (<B_p> = 0.8610 against tanh(1) = 0.7616).  The
# guard installed below makes that statement enforceable rather than merely
# documented -- instantiating the legacy sampler raises.  The regression mode
# (--engine v05) lifts it on purpose.
import z2_sim_v05 as analysis
import crossover_decomposition_validator as dec
import gauge_inv_plaquette_strip_recompute as strip

# The sampler is imported optionally, and on purpose.  It is not part of the
# distributed package; what the package carries is the per-seed record, the
# analysis that recomputes every reported number from it, the engine
# validation, and the exact reference below, against which any sampler can be
# calibrated:
#
#     <B_p> = 1 - (1 - t)(1 - t^(n-1)) / (1 + t^n),  t = tanh(beta), n = L^2
#
# Importing it at module level would mean the script could not even be loaded
# without the sampler -- the analysis, the null models, the autocorrelation
# times and the layer decomposition would all be unreachable for a reader who
# has the record.  They are reachable; only re-simulation is not.
try:
    import z2_vec_rep as vecrep
except ImportError:
    vecrep = None


def require_engine(was):
    if vecrep is None:
        raise SystemExit(
            "%s needs the sampler (z2_vec_rep), which is not part of the "
            "distributed package. Everything that does not re-simulate -- the "
            "estimator, both null models, tau_int, the layer decomposition and "
            "the exact reference -- runs without it on the deposited per-seed "
            "record." % was)
    return vecrep


def check_modules_are_deposited():
    """Every module this script runs on must come from inside the deposit.

    An earlier version reached the sampler through a sys.path entry into a
    working directory outside the record.  The script ran, the numbers were
    right, and none of it was reproducible from the deposit alone.  The check
    below states where each module was actually loaded from, so that the same
    thing cannot happen silently again: it compares the resolved file of every
    imported module against this directory tree."""
    wurzel = os.path.realpath(SANDBOX)

    def herkunft(m):
        p = os.path.realpath(getattr(m, "__file__", "") or "")
        return p, p.startswith(wurzel + os.sep)

    aus = []
    paare = [("z2_sim_v05", analysis),
             ("crossover_decomposition_validator", dec),
             ("gauge_inv_plaquette_strip_recompute", strip)]
    if vecrep is not None:
        paare.append(("z2_vec_rep", vecrep))
    for name, m in paare:
        p, drin = herkunft(m)
        aus.append((name, p, drin))
    # counter-control: a module that is certainly outside the deposit must be
    # reported as outside.  Without it the check could be one that never turns
    # red -- which reads exactly like a check that always passes.
    p_np, drin_np = herkunft(np)
    aus.append(("numpy (counter-control, must be OUTSIDE)", p_np, not drin_np))
    return aus, wurzel

# ---------------------------------------------------------------- constants
L_DEFAULT = dec.L
BETAS = dec.BETAS
SEEDS_TABLE = list(dec.SEEDS)                     # the five seeds of Table II
SEEDS_WIDE = [42] + list(range(10, 20))           # eleven, for the null level
SEEDS_STRIP = [42] + list(range(10, 30))          # twenty-one, strip route
SEEDS_SCALING = list(range(10, 20))               # ten per (engine, L) cell
SEEDS_LONGRUN = [42, 10, 11, 12, 13, 14]          # six, run-length control
SEEDS_VALIDATION = list(range(100, 120))          # twenty, engine gate
N_BINS = dec.N_BINS
N_SHUFFLE = dec.N_SHUFFLE
N_THERM, N_MEAS = dec.N_THERM, dec.N_MEAS
N_BOOT = 20000
BOOT_SEED = 20260908
GATE_SE = 2.0

# the configuration of the deposited scaling record (stufe2_results.json).
# Run lengths grow with L because the autocorrelation time does; holding N
# fixed would give the largest lattice the worst statistics of the series.
# tau_int and N_eff are reported per cell so that a reader can see whether
# the L-dependence follows the physics or the statistics.
SCALING_CONFIG = {8: (2000, 4000), 16: (10000, 4000), 32: (10000, 4000),
                  48: (22500, 8430), 64: (40000, 26641)}
LONGRUN = {"n_therm": 2000, "n_meas": 40000, "betas": [2.0, 2.5]}


# =================================================== guard on the old sampler
class LegacyEngineUsed(RuntimeError):
    """Raised when the superseded sampler is constructed in a production run."""


def guard_legacy_sampler():
    """Forbid constructing z2_sim_v05.Z2GaugeTheory; return an undo callable.

    The patch is applied to the class object rather than to a module
    attribute.  gauge_inv_plaquette_strip_recompute loads z2_sim_v05 through
    importlib, so it holds a *separate* module object with its own name for
    the class; patching one module name would leave the strip route
    unguarded while appearing to succeed.  Patching __init__ on each loaded
    class object covers every reference, run_sim() included.
    """
    klassen = []
    for modul in (analysis, strip):
        k = getattr(modul, "Z2GaugeTheory", None)
        if k is not None and k not in klassen:
            klassen.append(k)
    originale = [(k, k.__init__) for k in klassen]

    def gesperrt(self, *args, **kwargs):
        raise LegacyEngineUsed(
            "z2_sim_v05.Z2GaugeTheory was constructed. This sampler does not "
            "sample the Boltzmann distribution and must not produce results. "
            "It is imported only for correlation_matrix / best_chsh / "
            "chsh_per_config, which contain no sampling. Use --engine v05 for "
            "the regression check, which lifts this guard deliberately.")

    for k in klassen:
        k.__init__ = gesperrt

    def undo():
        for k, orig in originale:
            k.__init__ = orig

    return undo, len(klassen)


# =============================================================== estimator
def codes(seq):
    """Map a sequence of hashable labels onto 0..k-1.  Mutual information is
    invariant under relabelling, so any bijection will do."""
    order = {}
    out = np.empty(len(seq), dtype=np.int64)
    for i, v in enumerate(seq):
        if v not in order:
            order[v] = len(order)
        out[i] = order[v]
    return out, len(order)


def mi_vec(cx, nx, cy, ny):
    """Plug-in mutual information in bits, vectorised.

    Identical in value to the deposited mi_estimator; the loop form there is
    too slow for the exhaustive circular-shift null (n-1 evaluations per
    record).  check_estimator_agreement() verifies the two agree on every run.
    """
    n = cx.size
    joint = np.bincount(cx * ny + cy, minlength=nx * ny).reshape(nx, ny)
    pj = joint / n
    px = pj.sum(axis=1, keepdims=True)
    py = pj.sum(axis=0, keepdims=True)
    nz = pj > 0
    return float((pj[nz] * np.log2(pj[nz] / (px @ py)[nz])).sum())


def check_estimator_agreement(labels, sb, tol=1e-12):
    cx, nx = codes(labels)
    cy, ny = codes(sb)
    a = mi_vec(cx, nx, cy, ny)
    b = dec.mi_estimator(labels, sb)
    if abs(a - b) > tol:
        raise SystemExit("estimator disagreement: %.17g vs %.17g" % (a, b))
    return abs(a - b)


# =================================================================== nulls
def circ_null(cx, nx, cy, ny):
    """Exhaustive circular-shift null: every offset 1..n-1, no sampling."""
    n = cx.size
    v = np.array([mi_vec(np.roll(cx, k), nx, cy, ny) for k in range(1, n)])
    return float(v.mean()), float(v.std()), int(n - 1)


def perm_null(cx, nx, cy, ny, n_shuffle, rng):
    v = np.array([mi_vec(rng.permutation(cx), nx, cy, ny)
                  for _ in range(n_shuffle)])
    return float(v.mean()), float(v.std()), int(n_shuffle)


def both_nulls(labels, sb, rng, n_shuffle=N_SHUFFLE):
    cx, nx = codes(labels)
    cy, ny = codes(sb)
    I = mi_vec(cx, nx, cy, ny)
    Ep, Sp, np_ = perm_null(cx, nx, cy, ny, n_shuffle, rng)
    Ec, Sc, nc = circ_null(cx, nx, cy, ny)
    return {"I_raw": I,
            "E_perm": Ep, "sigma_perm": (I - Ep) / Sp if Sp > 0 else 0.0,
            "E_circ": Ec, "sigma_circ": (I - Ec) / Sc if Sc > 0 else 0.0,
            "n_shuffle": np_, "n_offsets": nc}


def rng_for(*parts):
    """A generator whose stream depends only on what it is for.

    The permutation null draws random numbers.  Taken from one running stream,
    those draws depend on how many calls preceded them, so adding any quantity
    to the run silently shifts every permutation-derived number in the record
    -- the difference is within the null's own sampling error, but it is not
    reproducible, and a reader cannot tell the two apart.  Deriving the stream
    from the identity of the cell makes the whole file reproducible: the same
    script on the same input returns the same bytes, whatever else was
    computed alongside.  The circular-shift null needs none of this; it is
    exhaustive and was reproducible from the start."""
    h = int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)
    return np.random.default_rng([BOOT_SEED, h])


def bootstrap_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    b = np.array([v[rng.integers(0, v.size, v.size)].mean()
                  for _ in range(n_boot)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


# ============================================== integrated autocorrelation
def tau_int(x):
    """Integrated autocorrelation time with Sokal's automatic window.

    tau = 0.5 + sum_k rho(k), truncated at the first k with k >= 5*tau.
    White noise gives 0.5; an AR(1) chain with parameter phi gives
    (1+phi)/(2(1-phi)).  Both are checked in check_tau()."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if n < 4 or x.std() == 0:
        return 0.5
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    if acf[0] <= 0:
        return 0.5
    rho = acf / acf[0]
    t = 0.5
    for k in range(1, n):
        t += rho[k]
        if k >= 5.0 * t:
            break
    return float(max(t, 0.5))


def tau_match(c):
    """Integrated autocorrelation time of a CATEGORICAL series.

    tau_int computed on integer codes is not well defined for a categorical
    variable: renumbering the categories changes the autocorrelation of the
    number sequence, and with it tau.  The match autocorrelation

        rho(k) = [P(Z_t = Z_{t+k}) - sum_j p_j^2] / [1 - sum_j p_j^2]

    is invariant under any relabelling and decays exactly as a^k on a
    two-state Markov chain.  Both properties are checked in
    check_tau_match().  For a binary series the two definitions agree, so the
    distinction only matters for the four-valued Wilson pair.
    """
    c = np.asarray(c)
    n = c.size
    p = np.array(list(Counter(c.tolist()).values()), dtype=float) / n
    basis = 1.0 - float((p ** 2).sum())
    if basis <= 1e-12:
        return float("inf")
    t = 0.5
    for k in range(1, n):
        rho = (float((c[:-k] == c[k:]).mean()) - (1.0 - basis)) / basis
        t += rho
        if k >= 5.0 * t or k > 4000:
            break
    return float(max(t, 0.5))


def check_tau_match():
    """The categorical tau against chains with a known answer, and its
    invariance against a relabelling."""
    rng = np.random.default_rng(7)
    iid = rng.integers(0, 4, 100000)
    t_iid = tau_match(iid)
    q = 0.2
    x = np.empty(100000, dtype=np.int64)
    x[0] = 0
    u = rng.random(100000)
    for i in range(1, x.size):
        x[i] = 1 - x[i - 1] if u[i] < q else x[i - 1]
    t_mk = tau_match(x)
    soll = 0.5 + (1 - 2 * q) / (2 * q)
    um = {0: 3, 1: 0, 2: 1, 3: 2}
    t_perm = tau_match(np.array([um[int(v)] for v in iid]))
    return t_iid, t_mk, soll, abs(t_iid - t_perm)


def entropy_bits(seq):
    p = np.array(list(Counter(list(seq)).values()), dtype=float) / len(seq)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def switches(c):
    c = np.asarray(c)
    return int((c[1:] != c[:-1]).sum())


def layer_decomposition(lab, sb, anyons):
    """Chain rule split of I(Z;S) by excitation layer.

    V = 1{n >= 2} is a function of Z, so I(Z;S) = I(V;S) + I(Z;S|V), and the
    conditional term splits by layer weight.  The three parts must add up to
    the total; the residual is reported and checked by the caller.  Anyons
    occur in pairs (the torus constraint forces an even count), so V = 0 is
    the vacuum layer.
    """
    cz, nz = codes(lab)
    cs, ns = codes(sb)
    V = (np.asarray(anyons) >= 2).astype(np.int64)
    I_total = mi_vec(cz, nz, cs, ns)
    I_VS = mi_vec(V, 2, cs, ns)
    teil, gew = {}, {}
    for v in (0, 1):
        m = np.nonzero(V == v)[0]
        gew[v] = float(m.size) / V.size
        if m.size < 2:
            teil[v] = 0.0
            continue
        ca, nca = codes([lab[i] for i in m])
        cb, ncb = codes([sb[i] for i in m])
        teil[v] = mi_vec(ca, nca, cb, ncb)
    summe = I_VS + gew[0] * teil[0] + gew[1] * teil[1]
    return {"I_total": I_total, "I_layer_membership": I_VS,
            "p_vacuum": gew[0], "p_excited": gew[1],
            "I_given_vacuum": teil[0], "I_given_excited": teil[1],
            "contribution_vacuum": gew[0] * teil[0],
            "contribution_excited": gew[1] * teil[1],
            "residual": I_total - summe}


def layered_null(lab, sb, anyons):
    """Circular shift restricted to the excited layer.

    The global circular-shift null destroys the association everywhere.  This
    one displaces the labels only among the configurations that carry
    excitations, leaving the vacuum layer and the layer membership itself
    untouched -- so it tests I(Z;S|V=1) alone.  Exhaustive over all m-1
    offsets, so it has no free parameter either.
    """
    cz, nz = codes(lab)
    cs, ns = codes(sb)
    idx = np.nonzero(np.asarray(anyons) >= 2)[0]
    m = idx.size
    if m < 10:
        return None
    I = mi_vec(cz, nz, cs, ns)
    basis = cz.copy()
    teil = cz[idx]
    v = np.empty(m - 1)
    for k in range(1, m):
        basis[idx] = np.roll(teil, k)
        v[k - 1] = mi_vec(basis, nz, cs, ns)
    sd = float(v.std())
    return {"I": I, "E": float(v.mean()), "sd": sd,
            "sigma": (I - float(v.mean())) / sd if sd > 0 else 0.0,
            "n_offsets": int(m - 1), "n_excited": int(m)}


def check_tau():
    """tau_int against two chains with a known answer."""
    rng = np.random.default_rng(5)
    weiss = float(tau_int(rng.normal(size=200000)))
    phi = 0.9
    x = np.empty(200000)
    x[0] = 0.0
    e = rng.normal(size=200000)
    for i in range(1, x.size):
        x[i] = phi * x[i - 1] + e[i]
    ar1 = float(tau_int(x))
    soll = (1.0 + phi) / (2.0 * (1.0 - phi))
    return weiss, ar1, soll


# ================================================================== series
def series_for(beta, seed, engine, L=None, n_therm=None, n_meas=None):
    """The three label series and the binned CHSH series of one run.

    Label definitions are those of the deposited decompose(); they are
    repeated here because that function returns summary numbers, not the
    series the null models need."""
    L = L_DEFAULT if L is None else L
    n_therm = N_THERM if n_therm is None else n_therm
    n_meas = N_MEAS if n_meas is None else n_meas
    data = engine(L=L, beta=beta, n_therm=n_therm, n_meas=n_meas, seed=seed)
    C = analysis.correlation_matrix(data)
    S_best, (a1, a2, b1, b2) = analysis.best_chsh(C)
    S_vals = analysis.chsh_per_config(data, a1, a2, b1, b2)
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, N_BINS + 1)
    sb = list(np.digitize(S_vals, edges))
    lab = {"full": [(r["Wx"], r["Wy"], min(r["n_anyons"] // 2, 10)) for r in data],
           "wilson": [(r["Wx"], r["Wy"]) for r in data],
           "anyons": [(min(r["n_anyons"] // 2, 10),) for r in data]}
    anyons = np.array([r["n_anyons"] for r in data], dtype=float)
    return lab, sb, float(S_best), anyons


def strip_series(seed, engine_class, L=16, beta=2.0, n_therm=2000, n_meas=4000):
    """Sector and CHSH series of one plaquette-strip run.

    run_one_seed() of the deposited script returns summary numbers only; the
    series are formed inside its mutual_info().  The loop below calls the same
    deposited building blocks in the same order."""
    tc = engine_class(L, beta, seed)
    det_A = (L // 4, L // 4)
    det_B = (3 * L // 4, 3 * L // 4)
    path_len = L // 4
    for _ in range(n_therm):
        tc.sweep()
    data = []
    for _ in range(n_meas):
        tc.sweep()
        Bp = tc.plaquette_array()
        data.append({
            "Wx": tc.topological_sector()[0], "Wy": tc.topological_sector()[1],
            "n_anyons": tc.anyon_count(),
            "A_p": strip.measure_detector_plaquette(Bp, det_A[0], det_A[1], path_len),
            "B_p": strip.measure_detector_plaquette(Bp, det_B[0], det_B[1], path_len)})
    C = strip.correlation_matrix(data, "A_p", "B_p", n_dir=strip.N_DIRECTIONS)
    S_best, combo = strip.best_chsh(C)
    labels = [(r["Wx"], r["Wy"], min(r["n_anyons"] // 2, 10)) for r in data]
    S_vals = strip.chsh_per_config(data, "A_p", "B_p", *combo)
    edges = np.linspace(S_vals.min() - 0.01, S_vals.max() + 0.01, N_BINS + 1)
    return labels, list(np.digitize(S_vals, edges)), float(S_best)


# =================================================================== parts
def bp_exact_constrained(L, beta):
    """<B_p> on a torus of L x L plaquettes, with the parity constraint.

    After tree gauge fixing the plaquettes are independent except for one
    global condition: their product is +1, so the number of excited
    plaquettes is even.  A configuration with k excitations carries weight
    C(n, k) * exp(-2 beta k), n = L^2.

    tanh(beta) is the unconstrained result and is what the constrained sum
    approaches once many plaquettes are excited.  It is *not* the right
    reference when few are: at L = 16, beta = 2.5 the mean excitation number
    is about 1.7, and the parity condition raises <B_p> by 0.084 % -- four
    standard errors at twenty seeds.  The sum is evaluated in logarithms
    because C(256, 128) overflows a float.
    """
    n = L * L
    ks = np.arange(0, n + 1, 2, dtype=float)
    logc = (math.lgamma(n + 1)
            - np.array([math.lgamma(k + 1) for k in ks])
            - np.array([math.lgamma(n - k + 1) for k in ks]))
    logw = logc - 2.0 * beta * ks
    w = np.exp(logw - logw.max())
    return 1.0 - 2.0 * float((w * ks).sum() / w.sum()) / n


def check_reference():
    """The constrained reference against cases with a known answer.

    The first comparison is against the closed form in the sampler module and
    is skipped when that module is absent; the other two are self-contained
    and are the ones that matter for a reader of the record: the correction
    must vanish where the constraint is irrelevant and must not vanish where
    it is not."""
    a = (abs(bp_exact_constrained(4, 2.0) - vecrep.bp_exakt(4, 2.0))
         if vecrep is not None else None)
    b = abs(bp_exact_constrained(64, 0.5) - math.tanh(0.5))
    c = abs(bp_exact_constrained(16, 2.5) - math.tanh(2.5))
    return a, b, c


def part_validation(engine_class, log):
    """The engine gate: <B_p> against the exact expectation, 20 seeds.

    Cells: L = 16 at three temperatures against tanh(beta), and L = 4 against
    the exact value *with* the torus constraint, where the test is sharpest.
    The counter-control is the same engine without the randomized acceptance:
    it must fail, otherwise the threshold does not bite."""
    require_engine("the engine validation")
    zellen = [(16, 1.0), (16, 2.0), (16, 2.5), (4, 2.0)]
    a, b, c = check_reference()
    log("    [PK-reference] against the small-L form of the sampler module:")
    log("                   residual %s   tolerance %.3e"
        % ("%.3e" % a if a is not None else "skipped (sampler not present)", 1e-12))
    log("                   constraint must vanish (L=64, beta=0.5): "
        "residual %.3e   tolerance %.3e" % (b, 1e-9))
    log("                   constraint must NOT vanish (L=16, beta=2.5): "
        "correction %.3e   lower bound %.3e" % (c, 1e-5))
    if (a is not None and a > 1e-12) or b > 1e-9 or c < 1e-5:
        raise SystemExit("reference check failed -- no run")
    res = {"threshold_se": GATE_SE, "seeds": SEEDS_VALIDATION,
           "reference": "exact torus value with the parity constraint; "
                        "tanh(beta) is the unconstrained limit and is not the "
                        "right reference at L=16, beta>=2.5",
           "cells": {}}
    alle_ok = True
    for L, beta in zellen:
        ex = bp_exact_constrained(L, beta)
        v = [vecrep._mittel_bp(L, beta, s, muenzwurf=True) for s in SEEDS_VALIDATION]
        m, se = float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))
        z = (m - ex) / se if se > 0 else float("inf")
        w = [vecrep._mittel_bp(L, beta, s, muenzwurf=False) for s in SEEDS_VALIDATION[:10]]
        mw = float(np.mean(w))
        sew = float(np.std(w, ddof=1) / math.sqrt(len(w)))
        zw = (mw - ex) / sew if sew > 0 else float("inf")
        ok = abs(z) < GATE_SE
        alle_ok = alle_ok and ok
        res["cells"]["L%d_beta%.1f" % (L, beta)] = {
            "exact": ex, "mean": m, "sem": se, "z": z, "passes": ok,
            "counter_control_no_coin_flip": {"mean": mw, "z": zw,
                                             "fails_as_required": abs(zw) >= GATE_SE}}
        log("    L=%-3d beta=%.1f  <B_p> %.6f +- %.6f  exact %.6f  %+5.1f SE  %s"
            "   | control %+6.1f SE %s"
            % (L, beta, m, se, ex, z, "OK" if ok else "FAIL",
               zw, "(fails, good)" if abs(zw) >= GATE_SE else "(!! does not fail)"))
    res["passes"] = alle_ok
    if not alle_ok:
        raise SystemExit("engine validation failed -- no run")
    return res


def part_decomposition(betas, seeds, seeds_wide, engine, log, seeds_all=None):
    """The beta dependence, decomposed by label component and by layer.

    Four label variants are measured on the same runs:
        full     (Wx, Wy, min(n//2, 10))   the label the table prints
        wilson   (Wx, Wy)                  the topological sector alone
        anyons   min(n//2, 10)             the excitation number alone
        V        1{n >= 2}                  the mere presence of excitations
    The point of the split is that the first is a claim about a topological
    quantity and the third and fourth are not; without it a nonzero value of
    the first cannot be attributed.

    Per beta the record also carries the resolvability of the Wilson sector --
    entropy, number of sector changes, and effective sample size in two
    definitions -- because "carries nothing" and "cannot be resolved" are
    different statements and only these numbers separate them.
    """
    rng = np.random.default_rng(BOOT_SEED)
    seeds_all = seeds_wide if seeds_all is None else seeds_all
    out = {"per_beta": {}, "false_alarm": {}}
    for beta in betas:
        reihen, cell = {}, {"per_seed": {}}
        for seed in seeds_all:
            t0 = time.time()
            lab, sb, S_best, anyons = series_for(beta, seed, engine)
            lab["V"] = [(int(x),) for x in (np.asarray(anyons) >= 2).astype(int)]
            if seed in seeds_wide:
                reihen[seed] = (lab, sb)
            cw, _ = codes(lab["wilson"])
            wx = np.array([t[0] for t in lab["wilson"]], dtype=float)
            wy = np.array([t[1] for t in lab["wilson"]], dtype=float)
            tau_joint = tau_match(cw)
            tau_marg_max = max(tau_int(wx), tau_int(wy))
            r = {"S_best": S_best, "anyons_mean": float(anyons.mean()),
                 "tau_int_anyons": tau_int(anyons),
                 "tau_int_labels": tau_int(codes(lab["full"])[0]),
                 # two definitions, both reported: the joint one measures the
                 # decorrelation of the pair directly, the max of the two
                 # marginals bounds it from above (it waits for the slower
                 # component).  They differ, and each is quoted with its own
                 # definition rather than silently merged.
                 "tau_wilson_joint_match": tau_joint,
                 "tau_wilson_max_of_marginals": tau_marg_max,
                 "N_eff_wilson_joint": len(sb) / (2.0 * max(tau_joint, 0.5)),
                 "N_eff_wilson_max_marginal": len(sb) / (2.0 * max(tau_marg_max, 0.5)),
                 "H_wilson_bits": entropy_bits(lab["wilson"]),
                 "sector_switches": switches(cw),
                 "p_vacuum": float((np.asarray(anyons) == 0).mean()),
                 "n_vacuum_sectors": len(set(
                     (t[0], t[1]) for t, a in zip(lab["full"], anyons) if a == 0)),
                 "layer": layer_decomposition(lab["full"], sb, anyons),
                 "layered_null": layered_null(lab["full"], sb, anyons)}
            r["N_eff_labels"] = len(sb) / (2.0 * max(r["tau_int_labels"], 0.5))
            # The expectation values of the two Wilson loops.  A symmetry
            # forces them to vanish at any finite temperature on a finite
            # lattice; measuring them is therefore a check on the sampler,
            # not on the physics -- and where a single run fails to realise
            # the vanishing, that run has not sampled the sectors.
            r["W_x_mean"] = float(wx.mean())
            r["W_y_mean"] = float(wy.mean())
            for name in ("full", "wilson", "anyons", "V"):
                r[name] = both_nulls(lab[name], sb,
                                     rng_for("decomp", beta, seed, name))
            cell["per_seed"][str(seed)] = r
            if seed == seeds_all[0]:
                log("  beta=%.1f seed=%-3s %5.1fs  I_full=%.5f  s_circ=%6.2f  "
                    "wilson s_circ=%6.2f  H=%.3f  N_eff(w)=%.0f  tau=%.1f"
                    % (beta, seed, time.time() - t0, r["full"]["I_raw"],
                       r["full"]["sigma_circ"], r["wilson"]["sigma_circ"],
                       r["H_wilson_bits"], r["N_eff_wilson_joint"], tau_joint))
        log("  beta=%.1f: %d seeds done" % (beta, len(seeds_all)))

        # per-beta aggregates that the paper prints or cites
        agg = {}
        for feld in ("H_wilson_bits", "sector_switches", "tau_wilson_joint_match",
                     "tau_wilson_max_of_marginals", "N_eff_wilson_joint",
                     "N_eff_wilson_max_marginal", "tau_int_labels",
                     "N_eff_labels", "p_vacuum", "anyons_mean",
                     "W_x_mean", "W_y_mean"):
            agg[feld] = float(np.mean([cell["per_seed"][str(s)][feld]
                                       for s in seeds_all]))
        for feld in ("I_layer_membership", "contribution_vacuum",
                     "contribution_excited", "I_total"):
            agg[feld] = float(np.mean([cell["per_seed"][str(s)]["layer"][feld]
                                       for s in seeds_all]))
        agg["layer_residual_max"] = float(max(
            abs(cell["per_seed"][str(s)]["layer"]["residual"]) for s in seeds_all))
        agg["sigma_layered_null_mean"] = float(np.mean(
            [cell["per_seed"][str(s)]["layered_null"]["sigma"] for s in seeds_all
             if cell["per_seed"][str(s)]["layered_null"]]))
        agg["n_gt3_layered_null"] = int(sum(
            1 for s in seeds_all if cell["per_seed"][str(s)]["layered_null"]
            and cell["per_seed"][str(s)]["layered_null"]["sigma"] > 3))
        agg["W_abs_max_single_run"] = float(max(
            max(abs(cell["per_seed"][str(s)]["W_x_mean"]),
                abs(cell["per_seed"][str(s)]["W_y_mean"])) for s in seeds_all))
        agg["n_seeds"] = len(seeds_all)
        cell["aggregate_all_seeds"] = agg
        log("    residual of the layer split (max over seeds): %.3e   "
            "tolerance %.3e   %s"
            % (agg["layer_residual_max"], 1e-12,
               "OK" if agg["layer_residual_max"] < 1e-12 else "FAIL"))
        if agg["layer_residual_max"] >= 1e-12:
            raise SystemExit("layer decomposition does not add up to the total")

        # component summaries over all seeds, and over the five of Table II
        cell["components"] = {}
        for menge, marke in ((seeds_all, "all_seeds"), (seeds, "table_seeds")):
            z = {}
            for name in ("full", "wilson", "anyons", "V"):
                k = [cell["per_seed"][str(s)][name]["I_raw"]
                     - cell["per_seed"][str(s)][name]["E_circ"] for s in menge]
                lo, hi = bootstrap_ci(k)
                sg = [cell["per_seed"][str(s)][name]["sigma_circ"] for s in menge]
                z[name] = {
                    "n_seeds": len(menge),
                    "I_raw_mean": float(np.mean(
                        [cell["per_seed"][str(s)][name]["I_raw"] for s in menge])),
                    "E_circ_mean": float(np.mean(
                        [cell["per_seed"][str(s)][name]["E_circ"] for s in menge])),
                    "corrected_mean": float(np.mean(k)), "ci95": [lo, hi],
                    "sigma_circ_mean": float(np.mean(sg)),
                    "n_gt3sigma_circ": int(sum(1 for v in sg if v > 3)),
                    "n_gt2sigma_circ": int(sum(1 for v in sg if v > 2))}
            cell["components"][marke] = z
        log("    wilson over %d seeds: I-E_circ %+.3e  CI [%+.3e, %+.3e]  "
            "%d/%d above 3 sigma"
            % (len(seeds_all), cell["components"]["all_seeds"]["wilson"]["corrected_mean"],
               cell["components"]["all_seeds"]["wilson"]["ci95"][0],
               cell["components"]["all_seeds"]["wilson"]["ci95"][1],
               cell["components"]["all_seeds"]["wilson"]["n_gt3sigma_circ"],
               len(seeds_all)))

        for name in ("full", "wilson", "anyons"):
            k = [cell["per_seed"][str(s)][name]["I_raw"]
                 - cell["per_seed"][str(s)][name]["E_circ"] for s in seeds]
            lo, hi = bootstrap_ci(k)
            cell.setdefault("summary", {})[name] = {
                "seeds": seeds,
                "I_raw_mean": float(np.mean([cell["per_seed"][str(s)][name]["I_raw"]
                                             for s in seeds])),
                "E_circ_mean": float(np.mean([cell["per_seed"][str(s)][name]["E_circ"]
                                              for s in seeds])),
                "E_perm_mean": float(np.mean([cell["per_seed"][str(s)][name]["E_perm"]
                                              for s in seeds])),
                "corrected_mean": float(np.mean(k)), "ci95": [lo, hi],
                "n_gt2sigma_circ": int(sum(1 for s in seeds
                                           if cell["per_seed"][str(s)][name]["sigma_circ"] > 2)),
                "n_gt2sigma_perm": int(sum(1 for s in seeds
                                           if cell["per_seed"][str(s)][name]["sigma_perm"] > 2))}
        out["per_beta"][str(beta)] = cell
        out["false_alarm"][str(beta)] = false_alarm(reihen, seeds_wide, rng, log,
                                                    beta)
    return out


def false_alarm(reihen, seeds, rng, log, beta):
    """Pair the sector series of one seed with the CHSH series of another.

    Both records are real, so both carry their own autocorrelation; they are
    independent by construction, so every significant result is a false one."""
    res = {}
    paare = [(i, j) for i, j in itertools.permutations(seeds, 2)]
    for name in ("full", "wilson", "anyons"):
        sp, sc = [], []
        for i, j in paare:
            r = both_nulls(reihen[i][0][name], reihen[j][1],
                           rng_for("fa", beta, name, i, j))
            sp.append(r["sigma_perm"])
            sc.append(r["sigma_circ"])
        sp, sc = np.array(sp), np.array(sc)
        res[name] = {"n_pairs": len(paare)}
        for tag, v in (("perm", sp), ("circ", sc)):
            f3, f2 = float((v > 3).mean()), float((v > 2).mean())
            res[name][tag] = {
                "mean": float(v.mean()), "sd": float(v.std()),
                "frac_gt2": f2, "frac_gt3": f3, "max": float(v.max()),
                "level_per_5_gt2": 5.0 * f2,
                "se_level_per_5_gt2": 5.0 * float(np.sqrt(f2 * (1 - f2) / len(v)))}
        log("  beta=%.1f %-7s false alarm >2sigma: perm %.2f/5  circ %.2f/5  "
            "(%d pairs)" % (beta, name, res[name]["perm"]["level_per_5_gt2"],
                            res[name]["circ"]["level_per_5_gt2"], len(paare)))
    return res


def part_strip(seeds, engine_class, log, n_meas=4000, beta=2.0, tag="strip"):
    rng = np.random.default_rng(BOOT_SEED)
    per, reihen = {}, {}
    for seed in seeds:
        t0 = time.time()
        lab, sb, S_best = strip_series(seed, engine_class, beta=beta,
                                       n_meas=n_meas)
        reihen[seed] = (lab, sb)
        per[str(seed)] = both_nulls(lab, sb,
                                    rng_for(tag, beta, n_meas, seed),
                                    n_shuffle=2000)
        per[str(seed)]["S_best"] = S_best
        per[str(seed)]["tau_int_labels"] = tau_int(codes(lab)[0])
        log("  %s seed=%-3s beta=%.1f n=%d %5.1fs  I=%.5f  s_perm=%6.2f  "
            "s_circ=%6.2f" % (tag, seed, beta, n_meas, time.time() - t0,
                              per[str(seed)]["I_raw"],
                              per[str(seed)]["sigma_perm"],
                              per[str(seed)]["sigma_circ"]))
    k = [per[str(s)]["I_raw"] - per[str(s)]["E_circ"] for s in seeds]
    lo, hi = bootstrap_ci(k)
    sp = np.array([per[str(s)]["sigma_perm"] for s in seeds])
    sc = np.array([per[str(s)]["sigma_circ"] for s in seeds])
    Ir = np.array([per[str(s)]["I_raw"] for s in seeds])
    return {"per_seed": per, "seeds": seeds, "beta": beta, "n_meas": n_meas,
            "I_raw_mean": float(Ir.mean()), "I_raw_sd_pop": float(Ir.std(ddof=0)),
            "corrected_mean": float(np.mean(k)), "ci95": [lo, hi],
            "sigma_perm_mean": float(sp.mean()),
            "sigma_perm_sd_pop": float(sp.std(ddof=0)),
            "sigma_circ_mean": float(sc.mean()),
            "sigma_circ_sd_pop": float(sc.std(ddof=0)),
            "n_gt3sigma_perm": int((sp > 3).sum()),
            "n_gt3sigma_circ": int((sc > 3).sum())}, reihen


def part_scaling(engine, log, seeds=SEEDS_SCALING):
    """The L-dependence, in the configuration of the deposited record.

    Run lengths grow with L because tau does; tau_int and N_eff are reported
    per cell so that a reader can tell whether the L-dependence follows the
    physics or the statistics."""
    rng = np.random.default_rng(BOOT_SEED)
    out = {"config": {str(L): {"n_therm": v[0], "n_meas": v[1]}
                      for L, v in SCALING_CONFIG.items()}, "cells": {}}
    for L in sorted(SCALING_CONFIG):
        nt, nm = SCALING_CONFIG[L]
        per = {}
        for seed in seeds:
            t0 = time.time()
            lab, sb, S_best, anyons = series_for(2.0, seed, engine, L=L,
                                                 n_therm=nt, n_meas=nm)
            r = both_nulls(lab["full"], sb, rng_for("scaling", L, seed))
            r["tau_int_anyons"] = tau_int(anyons)
            r["tau_int_labels"] = tau_int(codes(lab["full"])[0])
            r["N_eff"] = nm / (2.0 * max(r["tau_int_labels"], 0.5))
            r["anyons_mean"] = float(anyons.mean())
            per[str(seed)] = r
            log("  L=%-3d seed=%-3s %6.1fs  I=%.6f  s_circ=%6.2f  tau=%.1f  "
                "N_eff=%.0f" % (L, seed, time.time() - t0, r["I_raw"],
                                r["sigma_circ"], r["tau_int_labels"], r["N_eff"]))
        k = [per[str(s)]["I_raw"] - per[str(s)]["E_circ"] for s in seeds]
        lo, hi = bootstrap_ci(k)
        Ir = np.array([per[str(s)]["I_raw"] for s in seeds])
        out["cells"][str(L)] = {
            "per_seed": per, "seeds": seeds,
            "I_raw_mean": float(Ir.mean()),
            "I_raw_sem": float(Ir.std(ddof=1) / math.sqrt(len(Ir))),
            "corrected_mean": float(np.mean(k)), "ci95": [lo, hi],
            "tau_int_labels_mean": float(np.mean([per[str(s)]["tau_int_labels"]
                                                  for s in seeds])),
            "N_eff_mean": float(np.mean([per[str(s)]["N_eff"] for s in seeds]))}
    return out


# ====================================================================== main
def main():
    tracer = "--tracer" in sys.argv
    regression = "--engine" in sys.argv and "v05" in sys.argv
    t0 = time.time()

    def log(s):
        print(s, flush=True)

    log("=" * 78)
    log("null_calibration -- %s" % ("REGRESSION on z2_sim_v05" if regression
                                    else ("TRACER" if tracer else "full run")))
    log("=" * 78)

    # ---- provenance: is every module inside the deposit? ---------------
    module, wurzel = check_modules_are_deposited()
    log("\n[provenance] every module must resolve inside %s" % wurzel)
    for name, pfad, drin in module:
        log("    [%s] %-36s %s" % ("OK " if drin else "!!!", name,
                                   os.path.relpath(pfad, wurzel) if drin else pfad))
    if not all(d for _, _, d in module):
        raise SystemExit(
            "a module was loaded from outside the deposit -- the numbers "
            "would not be reproducible from the record alone")

    # ---- guard --------------------------------------------------------
    undo, n_guarded = guard_legacy_sampler()
    log("\n[guard] legacy sampler locked on %d loaded class object(s)" % n_guarded)
    ausgeloest = False
    try:
        analysis.Z2GaugeTheory(4, 1.0, 1)
    except LegacyEngineUsed:
        ausgeloest = True
    try:
        strip.Z2GaugeTheory(4, 1.0, 1)
    except LegacyEngineUsed:
        pass
    else:
        raise SystemExit("guard does not cover the strip route -- no run")
    if not ausgeloest:
        raise SystemExit("guard did not fire when tried -- no run")
    # PK-plus at the same place: firing is only half the evidence. A guard
    # that also blocks the analysis helpers would pass the test above and
    # still be wrong -- it would be too broad rather than too narrow.
    try:
        probe = [{"Wx": 1, "Wy": -1, "n_anyons": 2,
                  "A": [1, -1, 1, -1], "B": [-1, 1, -1, 1]},
                 {"Wx": -1, "Wy": 1, "n_anyons": 4,
                  "A": [-1, 1, 1, -1], "B": [1, 1, -1, -1]}]
        S_probe, combo_probe = analysis.best_chsh(
            analysis.correlation_matrix(probe))
        analysis.chsh_per_config(probe, *combo_probe)
    except LegacyEngineUsed:
        raise SystemExit(
            "guard is too broad: it blocks correlation_matrix / best_chsh / "
            "chsh_per_config, which contain no sampling and are the only "
            "reason z2_sim_v05 is imported -- no run")
    log("        PK-minus: tried on both routes, it fires."
        "  PK-plus: the analysis\n        helpers still run (S = %.3f), so it is "
        "not too broad. A guard that is\n        never tried is "
        "indistinguishable from no guard; one that is only tried\n        with "
        "bad input is indistinguishable from one that blocks everything."
        % S_probe)
    if regression:
        undo()
        log("        --engine v05: guard lifted on purpose (regression only)")

    if regression:
        engine, engine_class, engine_name = (
            lambda **kw: analysis.run_sim(verbose=False, **kw),
            analysis.Z2GaugeTheory, "z2_sim_v05 (superseded, regression only)")
    else:
        require_engine("a measurement run")
        engine, engine_class, engine_name = (
            vecrep.run_sim_vec_rep, vecrep.Z2GaugeVecRep,
            "z2_vec_rep.Z2GaugeVecRep")
    log("[engine] %s" % engine_name)

    # ---- PK: estimator -------------------------------------------------
    # Every verdict line below prints the residual AND the tolerance it is
    # judged against, in %.3e.  A verdict without its tolerance cannot be read
    # for headroom -- a residual of 1e-16 against a tolerance of 1e-6 is a
    # different statement from the same residual against 1e-15.  The format
    # matters too: %.10f would print 1.2e-16 as 0.0000000000 and invite a
    # false report of non-reproduction.
    log("\n[PK-estimator] vectorised form against the deposited one")
    rng0 = np.random.default_rng(1)
    a = [(int(x), int(y)) for x, y in rng0.integers(0, 3, (600, 2))]
    b = list(rng0.integers(0, 20, 600))
    for marke, reihe in (("random series      ", a),
                         ("single-label series", [(0, 0)] * 600),
                         ("dependent series   ", [(int(x),) for x in b])):
        d_ = check_estimator_agreement(reihe, b)
        log("    %s: residual %.3e   tolerance %.3e   headroom %.1e"
            % (marke, d_, 1e-12, 1e-12 / d_ if d_ > 0 else float("inf")))

    # ---- PK: tau -------------------------------------------------------
    w, ar1, soll = check_tau()
    log("\n[PK-tau] integrated autocorrelation time against known chains")
    log("    white noise    : %.4f   expected %.4f   residual %.3e   "
        "tolerance %.3e" % (w, 0.5, abs(w - 0.5), 0.1))
    log("    AR(1) phi=0.9  : %.4f   expected %.4f   residual %.3e   "
        "tolerance %.3e (relative %.2f)"
        % (ar1, soll, abs(ar1 - soll), 0.25 * soll, 0.25))
    if abs(w - 0.5) > 0.1 or abs(ar1 - soll) / soll > 0.25:
        raise SystemExit("tau_int is not calibrated -- no run")

    t_iid, t_mk, t_soll, inv = check_tau_match()
    log("\n[PK-tau-categorical] the coding-invariant form")
    log("    iid, four values  : %.4f   expected %.4f   residual %.3e   "
        "tolerance %.3e" % (t_iid, 0.5, abs(t_iid - 0.5), 0.05))
    log("    Markov q=0.2      : %.4f   expected %.4f   residual %.3e   "
        "tolerance %.3e" % (t_mk, t_soll, abs(t_mk - t_soll), 0.15 * t_soll))
    log("    invariance under a relabelling: residual %.3e   tolerance %.3e"
        % (inv, 1e-9))
    if abs(t_iid - 0.5) > 0.05 or abs(t_mk - t_soll) / t_soll > 0.15 or inv > 1e-9:
        raise SystemExit("tau_match is not calibrated -- no run")

    # ---- PK: bit identity of the copied update methods ------------------
    if not regression:
        log("\n[PK-bitidentity] z2_vec_rep without the coin flip must equal z2_vec")
        import z2_vec
        schlecht = []
        for L, beta, seed in ((16, 1.0, 7), (16, 2.5, 11), (4, 2.0, 3),
                              (8, 0.5, 99), (16, 3.0, 42)):
            p = z2_vec.Z2GaugeVec(L, beta, seed)
            q = vecrep.Z2GaugeVecRep(L, beta, seed)
            q.muenzwurf = False
            for _ in range(300):
                p.sweep()
                q.sweep()
            if not (np.array_equal(p.sh, q.sh) and np.array_equal(p.sv, q.sv)):
                schlecht.append((L, beta, seed))
        p = z2_vec.Z2GaugeVec(16, 1.0, 7)
        q = vecrep.Z2GaugeVecRep(16, 1.0, 7)
        for _ in range(300):
            p.sweep()
            q.sweep()
        kann_rot = not (np.array_equal(p.sh, q.sh) and np.array_equal(p.sv, q.sv))
        log("    5 cells bit-identical: %s   | with coin flip they differ: %s"
            % ("yes" if not schlecht else "NO %s" % schlecht,
               "yes" if kann_rot else "NO -- the test is blind"))
        if schlecht or not kann_rot:
            raise SystemExit("bit-identity check failed -- no run")

    # ---- validation gate ------------------------------------------------
    validation = None
    if not regression:
        log("\n[gate] engine validation, %d seeds" % len(SEEDS_VALIDATION))
        validation = part_validation(vecrep.Z2GaugeVecRep, log)
        io.open(os.path.join(HERE, "engine_validation.json"), "w",
                encoding="utf-8", newline="\n").write(
            json.dumps(validation, indent=2) + "\n")
        log("    wrote engine_validation.json (before the first MI number)")

    betas = [2.0] if tracer else BETAS
    seeds = SEEDS_TABLE[:2] if tracer else SEEDS_TABLE
    wide = SEEDS_TABLE[:2] if tracer else SEEDS_WIDE
    strip_seeds = SEEDS_STRIP[:2] if tracer else SEEDS_STRIP

    log("\n[1/5] decomposition route -- four label components, 21 seeds")
    d = part_decomposition(betas, seeds, wide, engine, log,
                           seeds_all=(SEEDS_TABLE[:2] if tracer else SEEDS_STRIP))

    log("\n[2/5] plaquette-strip route")
    s, strip_reihen = part_strip(strip_seeds, engine_class, log)

    # regression: the raw values must reproduce the deposited record
    if regression:
        dep = {r["seed"]: r for r in json.load(io.open(os.path.join(
            HERE, "gauge_inv_plaquette_strip_recompute_v2_expanded21_results.json"),
            encoding="utf-8"))["per_seed"]}
        n_ok = 0
        for seed in strip_seeds:
            if seed in dep and abs(s["per_seed"][str(seed)]["I_raw"]
                                   - dep[seed]["I"]) < 1e-9:
                n_ok += 1
        log("    [PK-regression] %d/%d seeds reproduce the deposited I"
            % (n_ok, len(strip_seeds)))
        if n_ok != len(strip_seeds):
            raise SystemExit("analysis path has changed -- investigate")

    log("\n[3/5] strip false-alarm level")
    rng = np.random.default_rng(BOOT_SEED)
    paare = [(i, j) for i, j in itertools.permutations(strip_seeds, 2)]
    sp, sc = [], []
    for i, j in paare:
        r = both_nulls(strip_reihen[i][0], strip_reihen[j][1],
                       rng_for("fa_strip", i, j))
        sp.append(r["sigma_perm"])
        sc.append(r["sigma_circ"])
    sp, sc = np.array(sp), np.array(sc)
    fa = {"n_pairs": len(paare)}
    for tag, v in (("perm", sp), ("circ", sc)):
        f3 = float((v > 3).mean())
        fa[tag] = {"mean": float(v.mean()), "sd": float(v.std()), "frac_gt3": f3,
                   "max": float(v.max()), "level_per_21_gt3": 21.0 * f3,
                   "se_level_per_21_gt3":
                       21.0 * float(np.sqrt(f3 * (1 - f3) / len(v)))}
    log("  strip false alarm >3sigma: perm %.1f+-%.1f / 21   circ %.1f+-%.1f / 21"
        % (fa["perm"]["level_per_21_gt3"], fa["perm"]["se_level_per_21_gt3"],
           fa["circ"]["level_per_21_gt3"], fa["circ"]["se_level_per_21_gt3"]))

    scaling = long_run = None
    if not (tracer or regression):
        log("\n[4/5] scaling, deposit configuration, tau_int and N_eff per cell")
        scaling = part_scaling(engine, log)

        log("\n[5/5] run-length control: %d measurements per seed"
            % LONGRUN["n_meas"])
        long_run = {}
        for beta in LONGRUN["betas"]:
            r, _ = part_strip(SEEDS_LONGRUN, engine_class, log,
                              n_meas=LONGRUN["n_meas"], beta=beta, tag="long")
            long_run["beta%.1f" % beta] = r

    res = {"parameters": {
               "engine": engine_name, "L": L_DEFAULT, "betas": betas,
               "n_therm": N_THERM, "n_meas": N_MEAS, "n_bins": N_BINS,
               "n_shuffle_permutation_null": N_SHUFFLE,
               "circular_shift_null": "exhaustive over all n-1 offsets",
               "seeds_table": seeds, "seeds_null_level": wide,
               "seeds_strip": strip_seeds, "seeds_scaling": SEEDS_SCALING,
               "seeds_long_run": SEEDS_LONGRUN, "n_bootstrap": N_BOOT,
               "sd_convention": "population (N in the denominator)",
               "tracer": tracer, "regression": regression},
           "validation": validation,
           "decomposition": d["per_beta"],
           "false_alarm_decomposition": d["false_alarm"],
           "strip": s, "false_alarm_strip": fa,
           "scaling": scaling, "long_run": long_run,
           "runtime_min": round((time.time() - t0) / 60.0, 1)}
    # The output name is explicit so that a run never silently overwrites a
    # record someone is already quoting.  Permutation-derived numbers move
    # when the draw scheme changes; the file they moved in has to be a
    # different file, and the two compared, not one replaced.
    ausdruecklich = None
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            ausdruecklich = sys.argv[i + 1]
    name = (ausdruecklich if ausdruecklich else
            "null_calibration_tracer.json" if tracer else
            "null_calibration_regression_v05.json" if regression else
            "null_calibration_results_vecrep.json")
    io.open(os.path.join(HERE, name), "w", encoding="utf-8",
            newline="\n").write(json.dumps(res, indent=2) + "\n")
    log("\nwrote %s  (%.1f min)" % (name, res["runtime_min"]))


if __name__ == "__main__":
    main()
