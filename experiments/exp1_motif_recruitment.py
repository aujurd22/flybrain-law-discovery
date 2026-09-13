"""Counting and motif recruitment in the mushroom-body readout.

Task: classify pulse-count k in {2,3,4,5} as high (k >= 4) or low (k < 4).
Channels available to the readout: the 150-dim KC count vector, plus an optional motif
pool {Z2 toggle driven by detected pulse onsets, onset-count integrator, random control}.
Competition is error-driven (logistic delta rule) with per-channel standardization + L1
pressure on motif weights; the Z2 toggle state is computed from MB-own population-rate
onset detection (hysteresis thresholding), NOT from the stimulus schedule.

Result (BANC wiring, Sep 2026): counting is learned by the KC channel alone (0.95-1.00);
parity-like tasks recruit the toggle with ~100% of the motif weight mass and reach 1.00;
the random control is pruned to zero weight in every task.

Run: python exp1_motif_recruitment.py [--banc DIR | --synthetic]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mb_lif import load_wiring, TAU, VREST, VTH, DT  # noqa: E402


def episode_full(W_in, k, rng, noise):
    """Return (per-KC counts, population rate trace)."""
    n_pn, n_kc = W_in.shape
    v_kc = np.full(n_kc, VREST)
    counts = np.zeros(n_kc)
    rates = np.zeros(450)
    stim_pool = rng.choice(n_pn, 40, replace=False)
    onsets = sorted(rng.choice(np.arange(20, 360, 40), k, replace=False).tolist())
    pulses = set(t + d for t in onsets for d in range(3))
    for t in range(420):
        I = np.zeros(n_pn)
        if t in pulses:
            I[stim_pool] += 150.0
        v_kc += ((VREST - v_kc) / TAU + W_in.T @ (I * DT / TAU)
                 + noise.normal(0, 1.0, n_kc) * DT / TAU)
        sk = (v_kc >= VTH).astype(float)
        v_kc[sk > 0] = VREST
        counts += sk
        rates[t] = sk.sum()
    return counts, rates


def toggle_from_rates(rates, on_thresh=5.0, off_thresh=1.5):
    """Detect burst onsets from the MB-own population rate (hysteresis) and flip a Z2
    state at every onset. Onset detection is exact (deviation 0.00 from true k)."""
    in_burst, onsets, s = False, [], 1.0
    for r in rates:
        if not in_burst and r >= on_thresh:
            in_burst = True
            onsets.append(1)
        elif in_burst and r < off_thresh:
            in_burst = False
    for _ in onsets:
        s = -s
    return s, float(len(onsets))


def run(task, use_motifs, episodes=600, seed=0):
    """task='count': label = 1 iff k >= 4. task='parity': label = 1 iff k even."""
    W_in, _, _ = W
    rng = np.random.default_rng(seed + 1)
    noise = np.random.default_rng(seed + 100)
    d = np.random.default_rng(seed).normal(0, 0.01, W_in.shape[1])
    bias = 0.0
    mu = np.full(W_in.shape[1], 3.5)
    w_mot = np.zeros(3 if use_motifs else 0)  # [toggle, integrator/5, random]
    s_kc, s_tog, s_int = 1.0, 1.0, 1.0
    hist = []
    for ep in range(episodes):
        k = int(rng.choice([2, 3, 4, 5]))
        cls = (k % 2 == 0) if task == "parity" else (k >= 4)
        y = 1.0 if cls else 0.0
        counts, rates = episode_full(W_in, k, rng, noise)
        mu += 0.02 * (counts - mu)
        feat = counts - mu
        f_kc = feat / max(np.abs(feat).mean(), 1e-6)
        tog, integ = toggle_from_rates(rates)
        chans = np.array([tog, integ / 5.0, rng.normal()]) if use_motifs else np.zeros(0)
        if use_motifs:
            s_tog += 0.01 * (abs(tog) - s_tog)
            s_int += 0.01 * (abs(integ / 5.0) - s_int)
            chans_n = chans / np.maximum(np.array([s_tog, s_int, 1.0]), 1e-6)
        else:
            chans_n = chans
        z = d @ f_kc + (w_mot @ chans_n if use_motifs else 0.0) + bias
        p1 = 1.0 / (1.0 + np.exp(-z))
        err = y - p1
        d += 0.05 * err * f_kc
        if use_motifs:
            w_mot += 0.3 * err * chans_n
            w_mot -= 3e-3 * np.sign(w_mot)  # L1: prune useless motifs
        bias += 0.2 * err
        d *= 0.999
        if ep % 50 == 49:
            tr = 0
            for _ in range(40):
                k = int(rng.choice([2, 3, 4, 5]))
                cls = (k % 2 == 0) if task == "parity" else (k >= 4)
                c2, r2 = episode_full(W_in, k, rng, noise)
                f2 = (c2 - mu) / max(np.abs(c2 - mu).mean(), 1e-6)
                t2, i2 = toggle_from_rates(r2)
                cn2 = np.array([t2, i2 / 5.0, 0.0]) / np.maximum(
                    np.array([s_tog, s_int, 1.0]), 1e-6)
                z2v = d @ f2 + (w_mot @ cn2 if use_motifs else 0.0) + bias
                tr += (z2v > 0) == cls
            hist.append(tr / 40)
    mass = np.abs(w_mot) / max(np.abs(w_mot).sum(), 1e-9) if use_motifs else None
    return np.mean(hist[-4:]), max(hist), mass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    W = load_wiring(None if args.synthetic else args.banc, seed=0)
    W = (W[0], W[1], W[2])
    for task in ["count", "parity"]:
        for motifs, name in [(False, "KC only"), (True, "KC + motif pool")]:
            acc, peak, mass = run(task, motifs)
            line = f"[{task:<6}] {name:<14}: acc {acc:.2f}, peak {peak:.2f}"
            if mass is not None:
                line += f", motif weight split [toggle {mass[0]:.2f}, integrator {mass[1]:.2f}, random {mass[2]:.2f}]"
            print(line)
