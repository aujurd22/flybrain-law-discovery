"""T77: The fly brain reads — connectome-constrained sequence learning.

Stream a character corpus through the FULL BANC connectome (204k neurons, GPU LIF,
real sensory injection), record neural states at token boundaries, train a next-char
readout, and test whether the connectome's recurrence adds contextual information
beyond the current token.

Conditions: real wiring vs shuffled wiring (same density).
Baselines: chance, current-char-only readout (no brain), bigram table.

Run: python t77_flyreads.py  (GPU, ~30-60 min)
"""
import pandas as pd
import numpy as np
import torch

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
print("loading connectome...", flush=True)
meta = pd.read_feather("meta.feather")
edges = pd.read_feather("edgelist_simple_v3.feather")
uniq = np.unique(np.concatenate([meta["root_id"].to_numpy(), edges["pre"].to_numpy(),
                                 edges["post"].to_numpy()]))
N = len(uniq)
id_ix = {c: i for i, c in enumerate(uniq)}
pre = edges["pre"].map(id_ix).to_numpy()
post = edges["post"].map(id_ix).to_numpy()
cnt = edges["count"].to_numpy().astype(float)

# 感觉神经元 (注射通道) — 字典映射 (uniq 含 meta 之外的碎片 ID)
sc_map = dict(zip(meta["root_id"].astype(str), meta["super_class"].astype(str)))
sensory_flags = np.array(["sensory" in sc_map.get(str(c), "").lower() for c in uniq])
sensory_idx = np.where(sensory_flags)[0]
print(f"neurons {N}, sensory {len(sensory_idx)}", flush=True)

indices = torch.tensor(np.vstack([pre, post]), dtype=torch.long, device=dev)
values = torch.tensor(np.sqrt(cnt), dtype=torch.float32, device=dev)
values = values / values.max()
W_real = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
# 打乱权重对照: nnz 值随机重排 (同密度同度序无关)
perm = torch.randperm(W_real._nnz(), device=dev)
W_shuf = torch.sparse_coo_tensor(indices[:, perm], values[perm], (N, N)).coalesce()

# 字符表与感觉编码
alphabet = sorted(set(CORPUS))
ch_ix = {c: i for i, c in enumerate(alphabet)}
A = len(alphabet)
n_per_char = max(len(sensory_idx) // A, 1)
rng0 = np.random.default_rng(0)
char_channels = {}
si = rng0.permutation(sensory_idx)
for i, c in enumerate(alphabet):
    char_channels[c] = torch.tensor(si[i*n_per_char:(i+1)*n_per_char],
                                    dtype=torch.long, device=dev)
print(f"corpus {len(CORPUS)} chars, alphabet {A}, {n_per_char} sensory/ch", flush=True)

TAU, VREST, VTH, DT = 20.0, -70.0, -55.0, 1.0
STEPS_TOK = 250
REC_WIN = 100
GAIN = 3.0

def stream(W, corpus, seed=42, record=True, slow=True):
    """slow=True: 加 τ=400ms 慢突触通道 — 上下文跨 token 保持的关键.
    无慢通道时状态只含当前 token (T74 时间尺度教训), 真实=打乱."""
    g = torch.Generator(device=dev); g.manual_seed(seed)
    v = torch.full((N,), VREST + 3.0, device=dev)
    spikes = torch.zeros(N, dtype=torch.bool, device=dev)
    s_slow = torch.zeros(N, device=dev)
    TAU_S = 400.0
    states = []
    for ti, c in enumerate(corpus):
        chan = char_channels[c]
        acc = torch.zeros(N, device=dev)
        for t in range(STEPS_TOK):
            I = torch.randn(N, device=dev, generator=g) * 3.0
            I[chan] += 30.0
            s_slow += (0 - s_slow)*DT/TAU_S + spikes.float()
            drive = spikes.float() + 0.5 * s_slow/10.0
            v += ((VREST - v)/TAU
                  + torch.sparse.mm(W, drive.unsqueeze(1)).squeeze(1) * GAIN * DT/TAU
                  + I * DT/TAU)
            spk = v >= VTH
            v[spk] = VREST
            spikes = spk
            if t >= STEPS_TOK - REC_WIN:
                acc += spikes.float()
        if record:
            states.append((acc / REC_WIN).half().cpu())
    return torch.stack(states) if record else None

def readout_eval(states, corpus, train_frac=0.8, ridge=10.0):
    """states: (T, N) float16 cpu → next-char prediction (sparse-CG ridge, 内存安全)"""
    from sklearn.linear_model import Ridge
    import scipy.sparse as sp
    X = states.float().numpy()
    y = np.array([ch_ix[c] for c in corpus[1:]] + [ch_ix[corpus[-1]]])
    X, y = X[:-1], y[:-1]
    T = len(y)
    tr = int(T*train_frac)
    mu = X[:tr].mean(0)
    Xc = X - mu
    Y = np.eye(A)[y]
    model = Ridge(alpha=ridge, solver="sparse_cg", fit_intercept=True)
    model.fit(sp.csr_matrix(Xc[:tr]), Y[:tr])
    pred = model.predict(sp.csr_matrix(Xc))
    acc = np.mean(pred.argmax(1) == y)
    return acc

def bigram_acc(corpus):
    from collections import Counter, defaultdict
    ctx = defaultdict(Counter)
    for a, b in zip(corpus, corpus[1:]):
        ctx[a][b] += 1
    ok = sum(max(ctx[a], key=ctx.get) == b for a, b in zip(corpus, corpus[1:]))
    return ok/(len(corpus)-1)

print("streaming real wiring...", flush=True)
import time
t0 = time.time()
states_real = stream(W_real, CORPUS)
print(f"  {time.time()-t0:.0f}s", flush=True)
acc_real = readout_eval(states_real, CORPUS)
print(f"真实接线 next-char 精度: {acc_real*100:.1f}%", flush=True)

# 基线
chance = 1.0/A
cur_acc = np.mean([max((c == d) for d in set(CORPUS)) for c in CORPUS[1:]])
big = bigram_acc(CORPUS)
print(f"基线: chance {chance*100:.1f}%, 当前字符 {cur_acc*100:.1f}%, bigram {big*100:.1f}%", flush=True)

print("streaming shuffled wiring...", flush=True)
states_shuf = stream(W_shuf, CORPUS)
acc_shuf = readout_eval(states_shuf, CORPUS)
print(f"打乱接线 next-char 精度: {acc_shuf*100:.1f}%", flush=True)

np.save("T77_states_real.npy", states_real.numpy())
print(f"\n总结: 真实 {acc_real*100:.1f}% vs 打乱 {acc_shuf*100:.1f}% vs bigram {big*100:.1f}% vs chance {chance*100:.1f}%")
