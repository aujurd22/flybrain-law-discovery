"""Automated coordinate synthesis (the automated-mathematician loop).

Train a readout on first-order parity coordinates only; if residual errors remain, scan the
monomial grammar (products of coordinate subsets) for the monomial the residuals concentrate
on (separation score = |mean on errors - mean on correct|), synthesize it, retrain, repeat.

On XOR3 (label = 1 iff an odd number of odd factors) the loop starts at 32/64 errors,
automatically diagnoses "residuals concentrate on the f1*f2*f3 monomial (separation 2.00)",
synthesizes it, and reaches 0 errors. On NOR3 it correctly synthesizes nothing
(first-order coordinates already represent the law).

Pure Python/numpy; no connectome data required. Run: python exp4_auto_synthesis.py
"""
import numpy as np
from itertools import product, combinations


def z2(x):
    return 1.0 if x % 2 == 0 else -1.0


def mval(sig, ks):
    v = 1.0
    for i in sig:
        v *= z2(ks[i])
    return v


def batch_fit(task_fn, coords, allk, epochs=4000, lr=0.5):
    """Full-batch logistic on the current monomial coordinate set.

    Full-batch is essential: on symmetric data, SGD drift produces phantom error
    structure that defeats the corner diagnosis ("the diagnostician needs a stable
    student"). With full-batch, first-order gradients vanish exactly and the residual
    set is the clean parity half the algebra predicts.
    """
    V = np.array([[mval(sg, ks) for sg in coords] for ks in allk])
    Y = np.array([task_fn(*ks) for ks in allk], float)
    w, bias = np.zeros(len(coords)), 0.0
    for _ in range(epochs):
        z = V @ w + bias
        err = Y - 1.0 / (1.0 + np.exp(-z))
        w += lr * (err @ V) / len(V)
        bias += lr * err.mean()
    return w, bias


def run_auto(task_fn, K, allk, max_rounds=4, epochs=4000, sep_threshold=0.15):
    coords = [tuple([i]) for i in range(K)]
    history, log = [], []
    for _ in range(max_rounds):
        w, bias = batch_fit(task_fn, coords, allk, epochs=epochs)
        errs = [ks for ks in allk
                if (w @ np.array([mval(sg, ks) for sg in coords]) + bias > 0)
                != (task_fn(*ks) == 1)]
        history.append(len(errs))
        if not errs:
            return True, coords, history, log
        corr = [ks for ks in allk if ks not in errs]
        best, best_sep = None, sep_threshold
        for r in range(2, K + 1):
            for sig in combinations(range(K), r):
                if sig in coords:
                    continue
                e_m = np.mean([mval(sig, ks) for ks in errs])
                c_m = np.mean([mval(sig, ks) for ks in corr])
                if abs(e_m - c_m) > best_sep:
                    best_sep, best = abs(e_m - c_m), sig
        if best is None:
            log.append("residuals carry no monomial structure "
                       "(a coordinate outside the product grammar is needed)")
            break
        name = "x".join(f"f{i+1}" for i in best)
        log.append(f"residuals concentrate on monomial {name} "
                   f"(separation {best_sep:.2f}) -> synthesizing")
        coords.append(best)
    return False, coords, history, log


def main():
    values = [1, 2, 3, 4]
    allk3 = list(product(values, repeat=3))
    tasks = {
        "NOR3 (a*b*c odd)": lambda a, b, c: 1 if (a * b * c) % 2 == 1 else 0,
        "XOR3 (odd # of odd factors)": lambda a, b, c: 1 if (a + b + c) % 2 == 1 else 0,
    }
    for name, tf in tasks.items():
        ok, coords, hist, log = run_auto(tf, 3, allk3)
        pretty = ["x".join(f"f{i+1}" for i in sg) for sg in coords]
        print(f"[{name}] residual trace {hist}")
        for line in log:
            print(f"   {line}")
        print(f"   coordinates: {pretty} -> "
              f"{'law fully recovered' if ok else 'NOT recovered'}")


if __name__ == "__main__":
    main()
