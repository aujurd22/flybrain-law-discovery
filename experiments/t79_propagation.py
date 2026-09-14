"""T79: Propagation-regime memory kernel — where wiring becomes visible.

Design change vs T77/T78 (which were sensory-dominated: wiring invisible):
- token = brief volley (direct depolarization of the char's sensory pool, 2 steps)
- avalanche propagates through the brain via slow synapses
- state = s_slow trace at the NEXT token's boundary (pure propagated footprint,
  no direct drive contamination)

Phase 1: gain calibration — volley → avalanche size vs gain
  subcritical (avalanche < seed) / propagation (>> seed, bounded) / runaway (sustained)
Phase 2: at the propagation operating point, decode lag 0-3 (real vs shuffled).

Run: python t79_propagation.py  (GPU)
"""
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import Ridge
import scipy.sparse as sp
from collections import Counter

CORPUS = (
    "the fruit fly brain contains about one hundred and forty thousand neurons. "
    "scientists have mapped all of its wires. a neuron is a cell that carries signals. "
    "the fly can see, smell, taste, and move. its brain is small but powerful. "
    "we study the fly to learn how brains work in general. "
    "memory forms when neurons connect in new ways. a thought is a pattern of activity. "
    "every idea lives in the wiring of cells. the brain computes with spikes and chemicals. "
    "no single neuron knows the answer; the pattern holds it. "
)

dev = "cuda"
print("loading...", flush=True)
meta = pd.read_feather("meta.feather")
edges = pd.read_feather("edgelist_simple_v3.feather")
uniq = np.unique(np.concatenate([meta["root_id"].to_numpy(), edges["pre"].to_numpy(),
                                 edges["post"].to_numpy()]))
N = len(uniq)
id_ix = {c: i for i, c in enumerate(uniq)}
pre = edges["pre"].map(id_ix).to_numpy()
post = edges["post"].map(id_ix).to_numpy()
cnt = edges["count"].to_numpy().astype(float)
sc_map = dict(zip(meta["root_id"].astype(str), meta["super_class"].astype(str)))
sensory_flags = np.array(["sensory" in sc_map.get(str(c), "").lower() for c in uniq])
sensory_idx = np.where(sensory_flags)[0]

indices = torch.tensor(np.vstack([pre, post]), dtype=torch.long, device=dev)
values = torch.tensor(np.sqrt(cnt), dtype=torch.float32, device=dev)
values = values / values.max()
W_real = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
perm = torch.randperm(W_real._nnz(), device=dev)
W_shuf = torch.sparse_coo_tensor(indices[:, perm], values[perm], (N, N)).coalesce()
del meta, edges
print(f"neurons {N}, sensory {len(sensory_idx)}", flush=True)

alphabet = sorted(set(CORPUS))
ch_ix = {c: i for i, c in enumerate(alphabet)}
A = len(alphabet)
n_per_char = max(len(sensory_idx) // A, 1)
rng0 = np.random.default_rng(0)
si = rng0.permutation(sensory_idx)
char_channels = {c: torch.tensor(si[i*n_per_char:(i+1)*n_per_char],
                                 dtype=torch.long, device=dev)
                 for i, c in enumerate(alphabet)}

TAU, VREST, VTH, DT = 20.0, -70.0, -55.0, 1.0
TAU_S = 400.0
VOLLEY_STEPS = 2
PROP_STEPS = 140          # 传播期 (至下一 token), s_slow τ=400ms 保留 ~70% 上一痕迹
TOKEN_STEPS = VOLLEY_STEPS + PROP_STEPS

def volley_stream(W, corpus, gain, seed, seed_frac=0.03, record_states=True):
    """token = 短阵发性去极化 (char 池, v←−35) + 传播期; state = 传播期结束时的 s_slow"""
    g = torch.Generator(device=dev); g.manual_seed(seed)
    v = torch.full((N,), VREST + 3.0, device=dev)
    spikes = torch.zeros(N, dtype=torch.bool, device=dev)
    s_slow = torch.zeros(N, device=dev)
    states = []
    total_free = 0
    runaway_steps = 0
    for ti, c in enumerate(corpus):
        chan = char_channels[c]
        # volley: 2 步直接去极化
        for _ in range(VOLLEY_STEPS):
            v[chan] = torch.maximum(v[chan], torch.tensor(-35.0, device=dev))
            I = torch.randn(N, device=dev, generator=g) * 2.0
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, (s_slow/10.0).unsqueeze(1)).squeeze(1) * gain * DT/TAU
                  + I*DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
        # 传播期
        active_steps = 0
        for t in range(PROP_STEPS):
            I = torch.randn(N, device=dev, generator=g) * 2.0
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, (s_slow/10.0).unsqueeze(1)).squeeze(1) * gain * DT/TAU
                  + I*DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
            if spk.any(): active_steps += 1
        total_free += int(spikes.sum())
        if active_steps > PROP_STEPS*0.5: runaway_steps += 1
        if record_states:
            states.append((s_slow/ (s_slow.max()+1e-9) * 10.0).half().cpu())
    return (torch.stack(states) if record_states else None), total_free, runaway_steps

# ---------- Phase 2: 传播域记忆核 (gain 5.8, 全语料, 真实 vs 打乱) ----------
from sklearn.linear_model import Ridge

GAIN_OP = 5.8

def decode_lag(states, corpus, lag):
    X = states.float().numpy()
    T = len(corpus)
    if lag >= T - 10: return 0.0
    Xs, ys = X[lag:], []
    for t in range(lag, T):
        ys.append(ch_ix[corpus[t-lag]])
    ys = np.array(ys)
    tr = int(len(ys)*0.8)
    mu = Xs[:tr].mean(0)
    Xc = sp.csr_matrix(Xs - mu)
    Y = np.eye(A)[ys]
    model = Ridge(alpha=10.0, solver="sparse_cg", fit_intercept=True)
    model.fit(Xc[:tr], Y[:tr])
    pred = model.predict(Xc[tr:])
    return float(np.mean(pred.argmax(1) == ys[tr:]))

print(f"=== Phase 2: 传播域记忆核 (gain {GAIN_OP}) ===", flush=True)
results = {}
for wname, W in [("real", W_real), ("shuf", W_shuf)]:
    print(f"streaming {wname}...", flush=True)
    st, tot, run = volley_stream(W, CORPUS, GAIN_OP, 42, record_states=True)
    row = {}
    for lag in [0, 1, 2, 3]:
        row[lag] = decode_lag(st, CORPUS, lag)
    results[wname] = row
    print(f"  {wname}: " + " ".join(f"lag{l}={row[l]*100:.1f}%" for l in [0,1,2,3]), flush=True)
print("\n=== 传播域记忆核 ===")
print(f"{'条件':<10}" + "".join(f"lag{l:>8}" for l in [0,1,2,3]))
for wname, row in results.items():
    print(f"{wname:<10}" + "".join(f"{row[l]*100:>7.1f}%" for l in [0,1,2,3]))
import json
json.dump(results, open("T79_propagation_kernel.json", "w"))
print("saved T79_propagation_kernel.json")
