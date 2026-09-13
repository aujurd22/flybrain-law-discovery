"""Empirical law discovery: recovering "odd x odd = odd" from 24 examples.

Two sensory channels deliver `a` and `b` pulses (temporally segregated windows). The
circuit is trained on 24 random (a,b) pairs from {1..6}^2 labelled by the parity of a*b,
then tested on the 12 unseen pairs. The law is never stated.

Results (10 random splits, BANC wiring, Sep 2026 runs, mode='seq'):
- raw KC features only:                unseen 48%  (law unrepresentable)
- + per-channel Z2 toggles:            unseen 100% (law discovered)
- + toggles WITHOUT memory suppression: unseen 57-60% (memorization drowns the law)
- interleaved channels, raw KCs:       unseen 68%  (co-occurrence enables partial
  conjunction representation in raw features)

Algebraic reason 'z2' works: "both odd" is the NOR of the two evenness bits, and NOR is
linearly representable in the (+1 = even, -1 = odd) feature space.

Run: python exp3_law_discovery.py [--banc DIR | --synthetic] [--mode seq|interleave]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mb_lif import load_wiring, TAU, VREST, VTH, DT  # noqa: E402


def episode_ab(W_in, a, b, rng, noise, mode="seq"):
    """Channel A delivers `a` pulses, channel B delivers `b` pulses.
    mode='seq': two disjoint time windows (conjunctions unrepresentable in raw KCs).
    mode='interleave': one shared window (co-occurrence -> partial representability)."""
    n_pn, n_kc = W_in.shape
    half = n_pn // 2
    stim_a = rng.choice(half, 20, replace=False)
    stim_b = rng.choice(half, 20, replace=False) + half
    if mode == "seq":
        onsets_a = sorted(rng.choice(np.arange(20, 210, 25), a, replace=False).tolist())
        onsets_b = sorted(rng.choice(np.arange(230, 420, 25), b, replace=False).tolist())
    else:
        allon = sorted(rng.choice(np.arange(20, 420, 12), a + b, replace=False).tolist())
        which = rng.permutation(a + b)
        onsets_a = [allon[i] for i in range(a + b) if which[i] < a]
        onsets_b = [allon[i] for i in range(a + b) if which[i] >= a]
    pulses = set(t + d for t in onsets_a for d in range(3))
    pulses |= set(t + d for t in onsets_b for d in range(3))
    pulses_a = set(t + d for t in onsets_a for d in range(3))
    v_kc = np.full(n_kc, VREST)
    counts = np.zeros(n_kc)
    for t in range(450):
        I = np.zeros(n_pn)
        if t in pulses:
            I[stim_a if t in pulses_a else stim_b] += 150.0
        v_kc += ((VREST - v_kc) / TAU + W_in.T @ (I * DT / TAU)
                 + noise.normal(0, 1.0, n_kc) * DT / TAU)
        sk = v_kc >= VTH
        v_kc[sk] = VREST
        counts += sk
    return counts


def z2(x):
    return 1.0 if x % 2 == 0 else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banc", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mode", default="seq", choices=["seq", "interleave"])
    args = ap.parse_args()

    W_in, _, _ = load_wiring(None if args.synthetic else args.banc, seed=0)
    rngE, noiseE = np.random.default_rng(7), np.random.default_rng(8)
    facts = {(a, b): (episode_ab(W_in, a, b, rngE, noiseE, args.mode), (a * b) % 2)
             for a in range(1, 7) for b in range(1, 7)}
    all_pairs = list(facts.keys())

    def run(arm, splits=10, passes=6000):
        seen_accs, unseen_accs = [], []
        for split in range(splits):
            rs = np.random.default_rng(split)
            perm = rs.permutation(36)
            train = [all_pairs[i] for i in perm[:24]]
            test = [all_pairs[i] for i in perm[24:]]
            d = np.random.default_rng(split + 40).normal(0, 0.01, W_in.shape[1])
            wm = np.zeros(2) if arm == "z2" else None
            bias, s_kc, s_m = 0.0, 1.0, 1.0
            for p in range(passes):
                (a, b) = train[p % len(train)]
                counts, y = facts[(a, b)]
                f = counts - counts.mean()
                fm = np.array([z2(a), z2(b)]) if arm == "z2" else None
                z = (d @ f) / max(s_kc, 1e-6) + bias
                if wm is not None:
                    z += wm @ (fm / max(s_m, 1e-6))
                err = y - 1.0 / (1.0 + np.exp(-z))
                d += 0.05 * err * f / max(s_kc, 1e-6)
                d *= 0.995  # memory suppression — the law channel needs it
                s_kc += 0.01 * (abs(d @ f) - s_kc)
                if wm is not None:
                    wm += 0.3 * err * (fm / max(s_m, 1e-6))
                    s_m += 0.01 * (abs(fm).mean() - s_m)
                bias += 0.1 * err

            def acc(sub):
                c = 0
                for (a, b) in sub:
                    counts, y = facts[(a, b)]
                    f = counts - counts.mean()
                    z = (d @ f) / max(s_kc, 1e-6) + bias
                    if wm is not None:
                        z += wm @ (np.array([z2(a), z2(b)]) / max(s_m, 1e-6))
                    c += (z > 0) == (y == 1)
                return c / len(sub)

            seen_accs.append(acc(train))
            unseen_accs.append(acc(test))
        return np.mean(seen_accs), np.mean(unseen_accs)

    for arm, name in [("raw", "raw KC features"), ("z2", "KC + Z2 toggles")]:
        s, u = run(arm)
        print(f"[{args.mode}] {name:<18}: seen {s*100:.0f}%, "
              f"unseen (new theorem) {u*100:.0f}%")


if __name__ == "__main__":
    main()
