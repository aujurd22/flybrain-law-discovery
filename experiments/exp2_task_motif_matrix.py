"""Task x motif matrix: the learnability algebra table.

Tasks over pulse counts k in {2..7}: threshold (k>=5), parity (Z2), mod-3 == 0 (Z3),
carry (k>=6). Motif channels: none / Z2 toggle / Z3 ring (cos, sin) / integrator / all.
All tasks share one precomputed episode stream (fair comparison); channels are
variance-standardized (fair competition).

Predicted and observed (BANC wiring, Sep 2026):

| task            | none | Z2   | Z3   | integ | all  |
|-----------------|------|------|------|-------|------|
| thr5 (k>=5)     | 0.95 | 0.99 | 0.97 | 0.99  | 1.00 |
| parity (Z2)     | 0.61 | 1.00 | 0.70 | 0.55  | 1.00 |
| mod3z (Z3)      | 0.59 | 0.65 | 1.00 | 0.60  | 1.00 |
| carry (k>=6)    | 0.97 | 0.95 | 0.99 | 0.99  | 1.00 |

The task's symmetry group selects exactly the coordinate it needs.

Run: python exp2_task_motif_matrix.py [--banc DIR | --synthetic]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mb_lif import load_wiring, episode_counts  # noqa: E402

TASKS = {
    "thr5 (k>=5)": lambda k: 1 if k >= 5 else 0,
    "parity (Z2)": lambda k: 1 if k % 2 == 0 else 0,
    "mod3z (Z3)": lambda k: 1 if k % 3 == 0 else 0,
    "carry (k>=6)": lambda k: 1 if k >= 6 else 0,
}


def motif_channels(n_on, kind):
    if kind == "none":
        return np.zeros(0)
    if kind == "Z2":
        return np.array([1.0 if n_on % 2 == 0 else -1.0])
    if kind == "Z3":
        ang = 2 * np.pi * (n_on % 3) / 3
        return np.array([np.cos(ang), np.sin(ang)])
    if kind == "integ":
        return np.array([float(n_on) / 7.0])
    if kind == "all":
        ang = 2 * np.pi * (n_on % 3) / 3
        return np.array([1.0 if n_on % 2 == 0 else -1.0,
                         np.cos(ang), np.sin(ang), float(n_on) / 7.0])


def run(task_fn, mkind, stream, mu, episodes=800, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 0.01, stream[0][0].shape[0])
    wm_len = {"none": 0, "Z2": 1, "Z3": 2, "integ": 1, "all": 4}[mkind]
    wm = np.zeros(wm_len)
    bias, s_kc = 0.0, 1.0
    s_m = np.ones(max(wm_len, 1))
    hist = []
    for ep in range(episodes):
        counts, n_on, k = stream[ep]
        cls = task_fn(k)
        feat = counts - mu
        mch = motif_channels(n_on, mkind)
        ch_kc = (d @ feat) / max(s_kc, 1e-6)
        ch_m = (wm @ (mch / np.maximum(s_m, 1e-6))) if wm_len else 0.0
        z = ch_kc + ch_m + bias
        p1 = 1.0 / (1.0 + np.exp(-z))
        err = cls - p1
        d += 0.05 * err * feat / max(s_kc, 1e-6)
        d *= 0.999
        s_kc += 0.01 * (abs(d @ feat) - s_kc)
        if wm_len:
            wm += 0.3 * err * (mch / np.maximum(s_m, 1e-6))
            s_m += 0.01 * (np.abs(mch) - s_m)
        bias += 0.2 * err
        if ep % 100 == 99:
            trc = 0
            for j in range(50):
                counts2, n_on2, k2 = stream[(ep + j) % len(stream)]
                mch2 = motif_channels(n_on2, mkind)
                zz = ((d @ (counts2 - mu)) / max(s_kc, 1e-6) + bias)
                if wm_len:
                    zz += wm @ (mch2 / np.maximum(s_m, 1e-6))
                trc += (zz > 0) == (task_fn(k2) == 1)
            hist.append(trc / 50)
    return np.mean(hist[-3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    W_in, _, _ = load_wiring(None if args.synthetic else args.banc, seed=0)
    rng, noise = np.random.default_rng(7), np.random.default_rng(8)
    stream = []
    for _ in range(800):
        k = int(rng.choice([2, 3, 4, 5, 6, 7]))
        counts = episode_counts(W_in, k, rng, noise)
        stream.append((counts, k, k))  # onset count == k by construction here
    mu = np.full(W_in.shape[1], 4.5)

    header = f"{'task':<14}" + "".join(f"{mk:>8}" for mk in
                                       ["none", "Z2", "Z3", "integ", "all"])
    print(header)
    for tname, tf in TASKS.items():
        row = f"{tname:<14}"
        for mk in ["none", "Z2", "Z3", "integ", "all"]:
            row += f"{run(tf, mk, stream, mu):>8.2f}"
        print(row)


if __name__ == "__main__":
    main()
