#!/usr/bin/env python3
"""Exhaustive 2^L enumeration of braiding sequences in the d_1b encoding.

This script regenerates fibonacci_enumeration_results.json, which supplies the
values of Table I (Fibonacci anyon CHSH violation vs. sequence length).

WHAT IT COMPUTES, per length L = 3 ... 12:

  delta0     the optimum over all 2^L sequences at delta = 0, the point that
             realizes the standard Fibonacci modular tensor category;
  delta_opt  the optimum over sequence AND delta jointly, on the 50-point
             grid linspace(0, 2*pi, 50, endpoint=False);

and, for both, how many of the 2^L sequences exceed the classical CHSH bound.

TWO ROUTES, ON PURPOSE. Route 1 is the chain deposited with the companion
record: apply the gates, take the concurrence, apply the Horodecki bound
|S| = 2*sqrt(1 + C^2). Route 2 is independent: it builds the correlation
matrix T_ij = <sigma_i x sigma_j>, forms M = T^T T, and takes
|S|_max = 2*sqrt(l1 + l2) from its two largest eigenvalues. Route 2 touches
neither concurrence() nor chsh_horodecki(). Their largest disagreement is
recorded per length as max_route_deviation; it is at the level of floating
point noise.

THE GATES ARE NOT RE-TYPED HERE. They are imported read-only from
d4_s_delta_sweep_validation.py of the companion record (Zenodo concept DOI
10.5281/zenodo.19601352). Re-implementing a published definition is how two
versions of it start to drift; importing it means this script and the
companion record cannot disagree by construction. Point --engine at that file,
or place the two records side by side (see find_engine below).

TOLERANCE. No bare threshold comparisons: a sequence counts as violating when
S > 2 + TOL with TOL = 1e-10. Sequences with |S - 2| <= TOL are counted
separately as edge cases and are never silently assigned to either side. Every
nonviolating sequence found is such an edge case -- no sequence falls below
the classical bound.

DETERMINISM. No timestamps in the output, sequences enumerated in
lexicographic order, keys sorted. Running this twice yields a byte-identical
JSON file.
"""
import sys

sys.dont_write_bytecode = True   # must precede every import: loading the
                                 # engine would otherwise drop a __pycache__
                                 # directory next to the companion record

import argparse
import importlib.util
import io
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fibonacci_enumeration_results.json")

ENGINE_NAME = "d4_s_delta_sweep_validation.py"
COMPANION_DOI = "10.5281/zenodo.19601352"

LENGTHS = list(range(3, 13))
TOL = 1e-10
N_DELTA = 50


def find_engine(explicit=None):
    """Locate the companion record's gate module.

    Searched in order: an explicit --engine path, then a few placements that
    arise when both records are unpacked together. If none matches, the error
    names the file and the DOI instead of failing with a bare ImportError.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    up = os.path.abspath(os.path.join(HERE, ".."))
    up2 = os.path.abspath(os.path.join(up, ".."))
    for base in (HERE, up, up2):
        candidates.append(os.path.join(base, ENGINE_NAME))
        for sibling in ("paper1b", "companion", "fibonacci"):
            candidates.append(os.path.join(base, sibling, ENGINE_NAME))
    if os.path.isdir(up2):
        for entry in sorted(os.listdir(up2)):
            candidates.append(os.path.join(up2, entry, ENGINE_NAME))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise SystemExit(
        "Could not find %s.\n"
        "It is deposited with the companion record, Zenodo concept DOI %s.\n"
        "Download that record and either place it beside this one or run:\n"
        "    python %s --engine /path/to/%s"
        % (ENGINE_NAME, COMPANION_DOI, os.path.basename(__file__), ENGINE_NAME))


def load_engine(path):
    spec = importlib.util.spec_from_file_location("companion_gates", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["companion_gates"] = mod
    spec.loader.exec_module(mod)
    for needed in ("gate_AM_d1b", "gate_MB_d1b", "concurrence", "chsh_horodecki"):
        if not hasattr(mod, needed):
            raise SystemExit("%s does not define %s -- wrong file?"
                             % (os.path.basename(path), needed))
    return mod


# --------------------------------------------------------------------- route 2
_PAULI = [np.array([[0, 1], [1, 0]], dtype=complex),
          np.array([[0, -1j], [1j, 0]], dtype=complex),
          np.array([[1, 0], [0, -1]], dtype=complex)]


def chsh_correlation_route(psi):
    """Independent route: Horodecki criterion via the correlation matrix.

    rho = |psi><psi| ; T_ij = tr(rho sigma_i x sigma_j) ; M = T^T T ;
    |S|_max = 2 sqrt(l1 + l2) from the two largest eigenvalues of M.
    Touches neither concurrence() nor chsh_horodecki().
    """
    rho = np.outer(psi, psi.conj())
    T = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = np.trace(rho @ np.kron(_PAULI[i], _PAULI[j])).real
    ev = np.linalg.eigvalsh(T.T @ T)
    return 2.0 * math.sqrt(max(ev[-1] + ev[-2], 0.0))


# ----------------------------------------------------------------- enumeration
def sequences(L):
    """All 2^L sequences over {A, B}, in lexicographic order."""
    for bits in itertools.product("AB", repeat=L):
        yield "".join(bits)


def apply_with_gates(seq, G_A, G_B):
    """Same arithmetic as the companion's apply_sequence_d1b, with the two
    generators built once per delta instead of once per sequence."""
    psi = np.array([1, 0, 0, 0], dtype=complex)
    for ch in seq:
        psi = (G_A if ch == "A" else G_B) @ psi
    n = np.linalg.norm(psi)
    return psi / n if n > 1e-15 else psi


def enumerate_at_delta(engine, L, delta, with_route2=False):
    """Enumerate all 2^L sequences at fixed delta."""
    G_A = engine.gate_AM_d1b(delta)
    G_B = engine.gate_MB_d1b(delta)
    best_S, best_seq, best_C = -1.0, None, None
    n_viol = n_edge = 0
    all_S = []
    max_route_dev = 0.0
    for seq in sequences(L):
        psi = apply_with_gates(seq, G_A, G_B)
        C = engine.concurrence(psi)
        S = engine.chsh_horodecki(C)
        all_S.append(S)
        if with_route2:
            dev = abs(S - chsh_correlation_route(psi))
            if dev > max_route_dev:
                max_route_dev = dev
        if S > 2.0 + TOL:
            n_viol += 1
        elif abs(S - 2.0) <= TOL:
            n_edge += 1
        if S > best_S:
            best_S, best_seq, best_C = S, seq, C
    arr = np.array(all_S)
    return {
        "n_sequences": len(all_S),
        "S_max": best_S,
        "C_at_max": best_C,
        "winner": best_seq,
        "n_violating": n_viol,
        "n_edge_within_tol": n_edge,
        "frac_violating": n_viol / len(all_S),
        "S_min": float(arr.min()),
        "S_median": float(np.median(arr)),
        "S_mean": float(arr.mean()),
        "max_route_deviation": max_route_dev,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", default=None,
                    help="path to %s of the companion record" % ENGINE_NAME)
    args = ap.parse_args()

    engine_path = find_engine(args.engine)
    engine = load_engine(engine_path)
    print("engine: %s" % os.path.basename(engine_path))

    deltas = np.linspace(0, 2 * math.pi, N_DELTA, endpoint=False)
    results = {}
    for L in LENGTHS:
        at0 = enumerate_at_delta(engine, L, 0.0, with_route2=True)
        best = {"S_max": -1.0}
        for i, dlt in enumerate(deltas):
            r = enumerate_at_delta(engine, L, float(dlt))
            if r["S_max"] > best["S_max"]:
                best = dict(r)
                best["delta_star"] = float(dlt)
                best["delta_index"] = i
        results[str(L)] = {"delta0": at0, "delta_opt": best}
        print("  L=%-2d  delta=0: S=%.9f (%s)  violating=%d/%d  |  "
              "delta-opt: S=%.9f (%s) at delta=%.6f"
              % (L, at0["S_max"], at0["winner"], at0["n_violating"],
                 at0["n_sequences"], best["S_max"], best["winner"],
                 best["delta_star"]), flush=True)

    doc = {
        "provenance": {
            "script": os.path.basename(__file__),
            "encoding": "d_1b (two generators A = AM, B = MB), start state |0> "
                        "in the 4D fusion space",
            "route_1": "companion chain: gate_AM_d1b / gate_MB_d1b -> concurrence "
                       "-> chsh_horodecki (|S| = 2*sqrt(1 + C^2))",
            "route_2": "independent: correlation matrix T_ij = <sigma_i x sigma_j>, "
                       "|S|_max = 2*sqrt(l1 + l2) from the two largest eigenvalues "
                       "of T^T T",
            "source_module": "%s, imported read-only from the companion record "
                             "(concept DOI %s); the gates and the Horodecki chain "
                             "are not re-typed here" % (ENGINE_NAME, COMPANION_DOI),
            "violation_tolerance": TOL,
            "violation_rule": "S > 2 + tol counts as violating; |S - 2| <= tol is "
                              "counted separately as an edge case, never silently "
                              "assigned to either side",
            "delta_grid": "numpy.linspace(0, 2*pi, %d, endpoint=False) -- the grid "
                          "of s_delta_sweeps.json in the companion record" % N_DELTA,
            "lengths": LENGTHS,
            "determinism": "no timestamps in content; sequences enumerated in "
                           "lexicographic order; sorted keys",
        },
        "per_length": results,
    }
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print("-> %s" % os.path.basename(OUT))


if __name__ == "__main__":
    main()
