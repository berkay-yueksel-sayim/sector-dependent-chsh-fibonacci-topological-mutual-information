"""
Vectorized Z_2 lattice gauge theory on an L x L torus.

STATUS: second, independent engine of this record -- used for the
finite-size-scaling column, the independent crossover route and the
frozen-regime characterization. It is NOT the anchor engine:
Tables I-II report the v05 branch (see below).

Finding from an independent serial-vs-vectorized cross-check:
  At L=4 with very long thermalization (50000 sweeps): vec and v05
  both converge to <B_p> ~ 0.997 -- identical physics.
  At L=16, beta=2.0 with the paper's thermalization (2000 sweeps):
  vec equilibrates faster and reaches the 'true' Boltzmann
  distribution with more anyons (<any> ~ 4.3 over the deposited
  seed set), while v05 remains on the not-yet-converged
  low-temperature branch (<any> ~ 2.7), which is the branch the
  paper's Tab. II reports.
  The paper's results are therefore the v05-specific mixing
  equilibrium, not full Boltzmann equilibrium. To reproduce the
  paper's anchor values, v05 MUST be used; this vectorized code will
  not reproduce them under the paper's thermalization settings.

Identical physics to z2_sim_v05.py (same Hamiltonian, same Metropolis
rule, same topology definitions), but the sweep is vectorized (numpy)
in 4 checkerboard sub-passes instead of L^2 serial Python loops.
Speedup ~7x at L=16.

Goalposts unchanged: Hamiltonian H = -sum(B_p), single-site
Metropolis, same topology sectors (Wx, Wy), same anyon definition
(B_p = -1), same tree gauge fixing.

Checkerboard scheme:
  Phase 1: horizontal edges with (y+x) even
  Phase 2: horizontal edges with (y+x) odd
  Phase 3: vertical edges with (y+x) even
  Phase 4: vertical edges with (y+x) odd

Rationale: sh[y,x] touches plaquettes (y,x) and (y-1,x); the OTHER
horizontal edges appearing in their Delta-E are sh[(y+1)%L,x] and
sh[(y-1)%L,x], which have the opposite y-parity, hence lie in the
other checkerboard class and are NOT flipped within the same phase.
Detailed balance is preserved.
"""

from __future__ import annotations

import numpy as np


class Z2GaugeVec:
    """Vektorisierte Z_2-LGT-Engine. Identisch in Physik zu v05.Z2GaugeTheory."""

    def __init__(self, L: int, beta: float, seed: int):
        self.L = L
        self.beta = float(beta)
        self.rng = np.random.default_rng(seed)
        self.sh = np.ones((L, L), dtype=np.int8)  # horizontale Edges
        self.sv = np.ones((L, L), dtype=np.int8)  # vertikale Edges
        # Checkerboard-Masken vorgerechnet
        yy, xx = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        self._mask_even = ((yy + xx) % 2 == 0)
        self._mask_odd = ~self._mask_even

    # ------------------------------------------------------------
    # Plaketten
    # ------------------------------------------------------------
    def plaquette_array(self) -> np.ndarray:
        """B_p[y,x] = sh[y,x] * sv[y,(x+1)%L] * sh[(y+1)%L,x] * sv[y,x] (CCW)."""
        return (
            self.sh
            * np.roll(self.sv, -1, axis=1)
            * np.roll(self.sh, -1, axis=0)
            * self.sv
        )

    def energy(self) -> int:
        return -int(np.sum(self.plaquette_array()))

    # ------------------------------------------------------------
    # Sweep (Metropolis, Checkerboard)
    # ------------------------------------------------------------
    def _update_h(self, mask: np.ndarray) -> None:
        """Update alle horizontalen Edges, wo mask True ist."""
        # Aktuelle Plakettenwerte (vor Flip)
        bp1 = (
            self.sh
            * np.roll(self.sv, -1, axis=1)
            * np.roll(self.sh, -1, axis=0)
            * self.sv
        )
        # B_p at ((y-1)%L, x) - die "obere" Plakette dieses Edge
        bp2 = (
            np.roll(self.sh, 1, axis=0)
            * np.roll(np.roll(self.sv, -1, axis=1), 1, axis=0)
            * self.sh
            * np.roll(self.sv, 1, axis=0)
        )
        dE = 2.0 * (bp1 + bp2)
        # Akzeptanz-Wahrscheinlichkeit: 1 wenn dE<=0, sonst exp(-beta*dE)
        prob = np.where(dE <= 0, 1.0, np.exp(-self.beta * dE))
        rand = self.rng.random(size=(self.L, self.L))
        flip = mask & (rand < prob)
        # In-place Flip ueber where (np.where erhaelt dtype)
        self.sh = np.where(flip, -self.sh, self.sh).astype(np.int8)

    def _update_v(self, mask: np.ndarray) -> None:
        """Update alle vertikalen Edges, wo mask True ist."""
        bp1 = (
            self.sh
            * np.roll(self.sv, -1, axis=1)
            * np.roll(self.sh, -1, axis=0)
            * self.sv
        )
        # B_p at (y, (x-1)%L) - die "linke" Plakette dieses Edge
        bp2 = (
            np.roll(self.sh, 1, axis=1)
            * self.sv
            * np.roll(np.roll(self.sh, -1, axis=0), 1, axis=1)
            * np.roll(self.sv, 1, axis=1)
        )
        dE = 2.0 * (bp1 + bp2)
        prob = np.where(dE <= 0, 1.0, np.exp(-self.beta * dE))
        rand = self.rng.random(size=(self.L, self.L))
        flip = mask & (rand < prob)
        self.sv = np.where(flip, -self.sv, self.sv).astype(np.int8)

    def sweep(self) -> None:
        """Ein voller Sweep = 4 Checkerboard-Sub-Paesse."""
        self._update_h(self._mask_even)
        self._update_h(self._mask_odd)
        self._update_v(self._mask_even)
        self._update_v(self._mask_odd)

    # ------------------------------------------------------------
    # Topologie & Anyonen (identisch zu v05)
    # ------------------------------------------------------------
    def wilson_x(self, y: int = 0) -> int:
        return int(np.prod(self.sh[y, :]))

    def wilson_y(self, x: int = 0) -> int:
        return int(np.prod(self.sv[:, x]))

    def topological_sector(self) -> tuple[int, int]:
        return (self.wilson_x(0), self.wilson_y(0))

    def anyon_count(self) -> int:
        return int(np.sum(self.plaquette_array() == -1))

    # ------------------------------------------------------------
    # Tree-Gauge-Fix (identisch zu v05.gauge_fix_tree)
    # ------------------------------------------------------------
    def gauge_fix_tree(self) -> tuple[np.ndarray, np.ndarray]:
        L = self.L
        sh = self.sh.copy()
        sv = self.sv.copy()
        for x in range(L):
            for y in range(L - 1):
                if sv[y, x] == -1:
                    self._gauge_transform(sh, sv, (y + 1) % L, x)
        for x in range(L - 1):
            if sh[0, x] == -1:
                self._gauge_transform(sh, sv, 0, (x + 1) % L)
        return sh, sv

    @staticmethod
    def _gauge_transform(sh: np.ndarray, sv: np.ndarray, y: int, x: int) -> None:
        L = sh.shape[0]
        sh[y, x] *= -1
        sh[y, (x - 1) % L] *= -1
        sv[y, x] *= -1
        sv[(y - 1) % L, x] *= -1

    # ------------------------------------------------------------
    # Detektor-Wilson-Linien (identisch zu v05.measure_detector)
    # ------------------------------------------------------------
    @staticmethod
    def wilson_line(
        sh: np.ndarray, sv: np.ndarray, y: int, x: int, direction: int, length: int
    ) -> int:
        L = sh.shape[0]
        prod = 1
        cy, cx = y, x
        for _ in range(length):
            if direction == 0:  # right
                prod *= sh[cy, cx]
                cx = (cx + 1) % L
            elif direction == 1:  # up
                cy = (cy - 1) % L
                prod *= sv[cy, cx]
            elif direction == 2:  # left
                cx = (cx - 1) % L
                prod *= sh[cy, cx]
            elif direction == 3:  # down
                prod *= sv[cy, cx]
                cy = (cy + 1) % L
        return int(prod)

    def measure_detector(
        self, sh: np.ndarray, sv: np.ndarray, y: int, x: int, path_len: int
    ) -> dict[int, int]:
        return {
            d: Z2GaugeVec.wilson_line(sh, sv, y, x, d, path_len)
            for d in range(4)
        }


# ============================================================
# Mess-Lauf (analog v05.run_sim)
# ============================================================
def run_sim_vec(
    L: int,
    beta: float,
    n_therm: int,
    n_meas: int,
    seed: int,
) -> list[dict]:
    tc = Z2GaugeVec(L, beta, seed)
    det_A = (L // 4, L // 4)
    det_B = (3 * L // 4, 3 * L // 4)
    path_len = L // 4

    for _ in range(n_therm):
        tc.sweep()

    data = []
    for _ in range(n_meas):
        tc.sweep()
        Wx, Wy = tc.topological_sector()
        n_any = tc.anyon_count()
        sh_gf, sv_gf = tc.gauge_fix_tree()
        A = tc.measure_detector(sh_gf, sv_gf, det_A[0], det_A[1], path_len)
        B = tc.measure_detector(sh_gf, sv_gf, det_B[0], det_B[1], path_len)
        Bp = tc.plaquette_array()
        bpA = int(Bp[det_A[0], det_A[1]])
        bpB = int(Bp[det_B[0], det_B[1]])
        data.append({
            "Wx": Wx, "Wy": Wy, "n_anyons": n_any,
            "A": A, "B": B,
            "bpA": bpA, "bpB": bpB,
        })
    return data
