"""T78: Memory kernel measurement — how long does the connectome reservoir remember?

Decode char[t-lag] from the neural state at token t, for lag = 0, 1, 2, 3.
Four conditions: {fast, slow} synapse x {real, shuffled} wiring.
This maps the memory trace of the connectome reservoir — the key parameter
for any sequence learning (T77 found: fast synapses = no context = wiring-blind).

Run: python t78_memory_kernel.py  (GPU, ~1-2h: 4 streams x 2500 tokens)
"""
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import Ridge
import scipy.sparse as sp

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
print(f"neurons {N}", flush=True)

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
STEPS_TOK = 250
REC_WIN = 100
GAIN = 3.0
TAU_S = 400.0

def stream(W, corpus, seed, slow):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    v = torch.full((N,), VREST + 3.0, device=dev)
    spikes = torch.zeros(N, dtype=torch.bool, device=dev)
    s_slow = torch.zeros(N, device=dev)
    states = []
    for c in corpus:
        chan = char_channels[c]
        acc = torch.zeros(N, device=dev)
        for t in range(STEPS_TOK):
            I = torch.randn(N, device=dev, generator=g) * 3.0
            I[chan] += 30.0
            if slow:
                s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
                drive = spikes.float() + 0.5*s_slow/10.0
            else:
                drive = spikes.float()
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, drive.unsqueeze(1)).squeeze(1) * GAIN * DT/TAU
                  + I * DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            if t >= STEPS_TOK - REC_WIN:
                acc += spikes.float()
        states.append((acc/REC_WIN).half().cpu())
    return torch.stack(states)

def decode_lag(states, corpus, lag, ridge=10.0):
    """从 state[t] 解码 char[t-lag]"""
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

corpus = CORPUS
results = {}
for slow in [False, True]:
    for wname, W in [("real", W_real), ("shuf", W_shuf)]:
        tag = f"{'slow' if slow else 'fast'}/{wname}"
        print(f"streaming {tag}...", flush=True)
        st = stream(W, corpus, 42, slow)
        row = {}
        for lag in [0, 1, 2, 3]:
            row[lag] = decode_lag(st, corpus, lag)
        results[tag] = row
        print(f"  {tag}: " + " ".join(f"lag{l}={row[l]*100:.0f}%" for l in [0,1,2,3]), flush=True)

print("\n=== 记忆核矩阵 (next/lag 可解码性) ===")
print(f"{'条件':<12}" + "".join(f"lag{l:>7}" for l in [0,1,2,3]))
for tag, row in results.items():
    print(f"{tag:<12}" + "".join(f"{row[l]*100:>8.1f}%" for l in [0,1,2,3]))
print(f"chance: {1/A*100:.1f}%")
import json
json.dump({k: v for k, v in results.items()}, open("T78_memory_kernel.json", "w"))
print("saved T78_memory_kernel.json")
