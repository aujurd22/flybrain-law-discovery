"""Core LIF mushroom-body model on BANC wiring (or synthetic fallback).

Conventions (established across the Sep 2026 campaign, see results/RESULTS.md):

- Stimuli are pulse trains: `k` onsets, each onset driving a sensory sub-pool for 3 ms.
- Pulse amplitude must exceed ~13 mV/step of leak to fire KCs at all (drives of 70 were
  silently sub-threshold; we use 150).
- The KC->MBON readout must be a pure integrator over the episode: with 20 ms membrane
  leak and >= 40 ms pulse spacing, any leaky readout loses the counting signal.
- Learning is the delta rule on raw per-KC spike counts with a running baseline
  (homeostatic scaling) plus an explicit bias. Channel standardization (running |channel|
  average) is REQUIRED whenever motif channels compete with the 150-dim KC channel,
  otherwise KC noise drowns the motif signal.
- Binary classification uses a single decision direction d with bias (softmax two-column
  parameterizations have a conserved quantity that collapses to constant-class guessing).
"""

import numpy as np
import pandas as pd

TAU, VREST, VTH, DT = 20.0, -70.0, -55.0, 1.0
PULSE_AMP = 150.0


def load_wiring(banc_dir, n_kc=150, n_pn=100, seed=0):
    """Select n_pn antennal-lobe projection neurons and n_kc Kenyon cells from BANC,
    build the PN->KC weight matrix from real synapse counts (sqrt-scaled), top up every
    KC to at least 5 inputs with random 0.6-weight edges.

    Returns (W_in of shape (n_pn, n_kc), kc_ids, pn_ids).
    Falls back to a random sparse wiring if banc_dir is None or files are missing.
    """
    rng = np.random.default_rng(seed)
    if banc_dir is not None:
        try:
            meta = pd.read_feather(f"{banc_dir}/meta.feather")
            edges = pd.read_feather(f"{banc_dir}/edgelist_simple_v3.feather")
            kc_all = meta.loc[meta["cell_class"] == "kenyon_cell", "root_id"].to_numpy()
            pn_all = meta.loc[meta["cell_class"] == "antennal_lobe_projection_neuron",
                              "root_id"].to_numpy()
            KCs = list(rng.choice(kc_all, n_kc, replace=False))
            PNs = list(rng.choice(pn_all, n_pn, replace=False))
            kc_ix = {k: i for i, k in enumerate(KCs)}
            pn_ix = {p: i for i, p in enumerate(PNs)}
            E = edges[edges["pre"].isin(pn_ix) & edges["post"].isin(kc_ix)]
            W_in = np.zeros((n_pn, n_kc))
            for _, r in E.iterrows():
                W_in[pn_ix[r["pre"]], kc_ix[r["post"]]] = np.sqrt(r["count"]) * 0.3
        except (FileNotFoundError, OSError):
            W_in, KCs, PNs = None, None, None
    else:
        W_in = KCs = PNs = None
    if W_in is None:  # synthetic fallback
        W_in = np.zeros((n_pn, n_kc))
        for j in range(n_kc):
            for i in rng.choice(n_pn, 6, replace=False):
                W_in[i, j] = 0.6
        KCs = [f"kc{j}" for j in range(n_kc)]
        PNs = [f"pn{i}" for i in range(n_pn)]
        return W_in, KCs, PNs
    for j in range(n_kc):
        idx = np.where(W_in[:, j] > 0)[0]
        if len(idx) < 5:
            extra = rng.choice([i for i in range(n_pn) if i not in idx], 5 - len(idx),
                               replace=False)
            W_in[extra, j] = 0.6
    return W_in, KCs, PNs


def episode_counts(W_in, k, rng, noise, pool_frac=0.4, amp=PULSE_AMP,
                   onsets=None, n_steps=450):
    """One episode: k pulse onsets drive a fixed sensory sub-pool; return per-KC spike
    counts over the whole episode (the counting code)."""
    n_pn, n_kc = W_in.shape
    v_kc = np.full(n_kc, VREST)
    counts = np.zeros(n_kc)
    stim_pool = rng.choice(n_pn, int(n_pn * pool_frac), replace=False)
    if onsets is None:
        onsets = sorted(rng.choice(np.arange(20, n_steps - 60, 40), k, replace=False).tolist())
    pulses = set(t + d for t in onsets for d in range(3))
    for t in range(n_steps):
        I = np.zeros(n_pn)
        if t in pulses:
            I[stim_pool] += amp
        v_kc += ((VREST - v_kc) / TAU + W_in.T @ (I * DT / TAU)
                 + noise.normal(0, 1.0, n_kc) * DT / TAU)
        sk = v_kc >= VTH
        v_kc[sk] = VREST
        counts += sk
    return counts


def train_logistic(feat_fn, examples, passes=4000, lr_d=0.05, lr_bias=0.2, decay=0.999,
                   standardize_kc=True, seed=0, dim_kc=150):
    """Delta-rule logistic readout with running standardization of the KC channel.

    feat_fn(k) -> (kc_counts_feature_vector or None, motif_feature_vector or None)
    examples: list of (k, label). Returns (d, wm, bias).
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 0.01, dim_kc) if dim_kc else None
    wm_len = len(feat_fn(examples[0][0])[1]) if feat_fn(examples[0][0])[1] is not None else 0
    wm = np.zeros(wm_len)
    bias = 0.0
    s_kc, s_m = 1.0, np.ones(max(wm_len, 1))
    for p in range(passes):
        k, y = examples[p % len(examples)]
        f_kc, f_m = feat_fn(k)
        ch_kc = 0.0
        if d is not None and f_kc is not None:
            ch_kc = (d @ f_kc) / max(s_kc, 1e-6)
        ch_m = (wm @ (f_m / np.maximum(s_m, 1e-6))) if wm_len else 0.0
        z = ch_kc + ch_m + bias
        err = y - 1.0 / (1.0 + np.exp(-z))
        if d is not None and f_kc is not None:
            d += lr_d * err * f_kc / max(s_kc, 1e-6)
            d *= decay
            s_kc += 0.01 * (abs(d @ f_kc) - s_kc)
        if wm_len:
            wm += 0.3 * err * (f_m / np.maximum(s_m, 1e-6))
            s_m += 0.01 * (np.abs(f_m) - s_m)
        bias += lr_bias * err
    return d, wm, bias


def eval_logistic(d, wm, bias, feat_fn, examples, standardize_kc=True, s_kc=1.0):
    correct = 0
    for k, y in examples:
        f_kc, f_m = feat_fn(k)
        z = 0.0
        if d is not None and f_kc is not None:
            z += (d @ f_kc) / max(s_kc, 1e-6)
        if wm is not None and f_m is not None:
            z += wm @ f_m
        z += bias
        correct += (z > 0) == (y == 1)
    return correct / len(examples)
