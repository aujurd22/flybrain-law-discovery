"""T80: Nonlinear probe on propagation-regime states.

T79 showed linear probes extract identical info from real vs shuffled wiring.
Question: does a small MLP find MORE in the real wiring's states? If yes, the
information exists in nonlinear population modes; if no, the linear-invisibility
theorem extends to nonlinear probes at this scale.

Also: record states during T79-style streaming (need fresh stream since states
weren't saved — rerun with save).
Run: python t80_nonlinear_probe.py  (GPU; streams 2x ~1.5h then trains MLP)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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
PROP_STEPS = 140
GAIN_OP = 5.8

def volley_stream(W, corpus, seed):
    g = torch.Generator(device=dev); g.manual_seed(seed)
    v = torch.full((N,), VREST + 3.0, device=dev)
    spikes = torch.zeros(N, dtype=torch.bool, device=dev)
    s_slow = torch.zeros(N, device=dev)
    states = []
    for c in corpus:
        chan = char_channels[c]
        for _ in range(VOLLEY_STEPS):
            v[chan] = torch.maximum(v[chan], torch.tensor(-35.0, device=dev))
            I = torch.randn(N, device=dev, generator=g) * 2.0
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, (s_slow/10.0).unsqueeze(1)).squeeze(1) * GAIN_OP * DT/TAU
                  + I*DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
        for t in range(PROP_STEPS):
            I = torch.randn(N, device=dev, generator=g) * 2.0
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, (s_slow/10.0).unsqueeze(1)).squeeze(1) * GAIN_OP * DT/TAU
                  + I*DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
        # state: 拓扑 PCA 式压缩不可行; 用随机投影到 4096 维 (保非线性结构, MLP 可训)
        states.append((s_slow/ (s_slow.max()+1e-9) * 10.0).half().cpu())
    return torch.stack(states)

# 随机投影 204k → 4096 (Johnson-Lindenstrauss, 保距)
gproj = torch.Generator(device=dev); gproj.manual_seed(7)
proj = (torch.randn(4096, N, device=dev, generator=gproj) / 64.0).half().cpu()

def project(states):
    """states (T,N) fp16 cpu → (T,4096) via chunked matmul on GPU"""
    out = []
    pj = proj.to(dev).float()
    for i in range(0, len(states), 64):
        chunk = states[i:i+64].to(dev).float()
        out.append((chunk @ pj.T).half().cpu())
    return torch.cat(out)

class MLP(nn.Module):
    def __init__(self, d_in, n_hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, n_hidden), nn.GELU(),
            nn.Linear(n_hidden, A))
    def forward(self, x):
        return self.net(x)

def mlp_eval(states_fp, corpus, lag, epochs=30):
    X = project(states_fp).float().numpy()
    T = len(corpus)
    if lag >= T - 10: return 0.0
    Xs = X[lag:]
    ys = np.array([ch_ix[corpus[t-lag]] for t in range(lag, T)])
    tr = int(len(ys)*0.8)
    mu = Xs[:tr].mean(0); sd = Xs[:tr].std(0) + 1e-6
    Xtr = (Xs[:tr]-mu)/sd; Xte = (Xs[tr:]-mu)/sd
    Ytr = ys[:tr]; Yte = ys[tr:]
    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr, device=dev)
    Yt = torch.tensor(Ytr, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]
            loss = lossf(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xev = torch.tensor((Xs[tr:]-mu)/sd, device=dev)
        pred = model(Xev).argmax(1).cpu().numpy()
    return float(np.mean(pred == Yte))

results = {}
for wname, W in [("real", W_real), ("shuf", W_shuf)]:
    print(f"streaming {wname}...", flush=True)
    st = volley_stream(W, CORPUS, 42)
    row = {}
    for lag in [0, 1, 2]:
        row[lag] = mlp_eval(st, CORPUS, lag)
        print(f"  {wname} MLP lag{lag}: {row[lag]*100:.1f}%", flush=True)
    results[wname] = row

print("\n=== MLP 探针 vs 线性探针 (lag) ===")
print(f"{'条件':<10}" + "".join(f"lag{l:>8}" for l in [0,1,2]))
for wname, row in results.items():
    print(f"{wname:<10}" + "".join(f"{row[l]*100:>7.1f}%" for l in [0,1,2]))
print("线性探针 (T79): real = shuf = 23.3/16.5/14.7")
import json
json.dump(results, open("T80_mlp_probe.json", "w"))
print("saved")
