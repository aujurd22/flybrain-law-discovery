# flybrain-law-discovery

**Learning mathematical laws in a connectome-constrained LIF model of the fly mushroom body.**

This repository contains the complete implementation and analysis of a research campaign exploring the Drosophila mushroom body (MB) as a computational system — using the real wiring of the *Drosophila* connectome (BANC v888, 165k neurons, 25.6M synapses) inside a leaky-integrate-and-fire (LIF) simulation, and treating "mathematics" with full honesty: tasks with checkable answers (counting, parity, modular arithmetic, empirical law discovery).

Everything runs on standard Python + numpy/pandas. With access to the BANC connectome files the PN→KC wiring is **real**; without them, a `--synthetic` random wiring is used so every experiment remains runnable.

---

## Quick Start

### Prerequisites

- **Python 3.12+** (tested on 3.13)
- **CUDA GPU** (≥ 8 GB VRAM for full-brain simulation; CPU works for small-scale)
- **8+ GB free RAM** (do NOT max out — leave headroom)

### Dependencies

```bash
pip install numpy pandas torch scipy scikit-learn sentence-transformers pyarrow
```

| package | version | purpose |
|---|---|---|
| numpy | ≥ 1.24 | tensor operations |
| pandas | ≥ 2.0 | data loading |
| torch | ≥ 2.0 | GPU LIF simulation |
| scipy | ≥ 1.11 | sparse matrices |
| scikit-learn | ≥ 1.3 | ridge readout, metrics |
| sentence-transformers | ≥ 2.0 | semantic encoding (MiniLM) |
| pyarrow | ≥ 14 | feather file loading |

### Connectome data

The BANC v888 connectome (Dorkenwald et al., 2024) is NOT redistributed. Place `meta.feather` and `edgelist_simple_v3.feather` in a directory and pass `--banc /path/to/dir` to experiments. Without data, `--synthetic` generates a random wiring fallback.

---

## Setup Guide (Step by Step)

### 1. Get the connectome data

Download the MaleCNS v1.0 tables from [Janelia](https://male-cns.janelia.org/download/):
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (14 MB)
- `body-neurotransmitters-male-cns-v1.0.feather` (43 MB)
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (1 GB)

Or use the automated script:
```bash
python scripts/acquire_connectome.py --out data/malecns-v1
```

### 2. Prepare the graph

```bash
python scripts/prepare_graph.py --out data/graph-traced-v1
# Output: graph.npz (crow, col, counts, body_ids) — 165k neurons, 25.6M synapses
```

### 3. Run experiments

See individual experiment sections below.

---

## Experiments

### T44: Can the fly brain count?

Tests whether the MB circuit can classify pulse counts (2 vs 5, odd vs even, etc.) using real PN→KC wiring.

```bash
python experiments/counting.py --banc /path/to/banc_888
```

**Result**: counting 84–100%, parity stuck at chance (architectural impossibility without motifs).

### T45: Motif recruitment

Tests whether tasks spontaneously recruit the correct computational motif from a pool.

```bash
python experiments/motif_recruitment.py --banc /path/to/banc_888
```

**Result**: tasks recruit the exact motif they need (toggle for parity, ring for mod-3). Wrong motif = chance. Full library = 1.00 everywhere.

### T47: Task × motif matrix

The learnability algebra table: for each task and each motif combination, measure accuracy.

```bash
python experiments/task_motif_matrix.py --banc /path/to/banc_888
```

**Result**: the task's symmetry group selects exactly the coordinate it needs. Wrong motif = chance.

### T77: Empirical law discovery

Given 24 random examples of `(a, b) → [is a*b odd]`, the circuit recovers the law "odd × odd = odd" and extrapolates to 12 unseen pairs at 100%.

```bash
python experiments/law_discovery.py --banc /path/to/banc_888
```

**Key finding**: law discovery requires (1) correct coordinates, (2) dopamine-gated storage, (3) memory suppression. Without coordinates: chance.

### T86: Real vs shuffled wiring

Tests whether the connectome's SPECIFIC wiring structure matters for learning.

```bash
python experiments/real_vs_shuffled.py --banc /path/to/banc_888
```

**Result**: real wiring 15/20 correct; shuffled wiring 0/20 — **the wiring structure is a learning prior, not just a signal path**.

### T88-T90: Sign-flipping, capacity, error correction

Hopfield analysis of the KC layer:
- Capacity: ~50 patterns in 500 neurons (theory: 0.138×N)
- Error correction: 30% bit corruption → 100% recovery
- Pattern completion: 20% cue → full recall

### T94: Full-brain criticality

The complete connectome (204k neurons) on GPU LIF shows a razor-thin transition between silence and runaway at gain ~0.01–0.02. Intermittent regime shows avalanche distributions with heavy tails (max avalanche 3904 spikes = 2% of brain).

---

## Results summary

### Counting and motif recruitment

| task | readout | accuracy | motif weight |
|---|---|---|---|
| count {2,3} vs {4,5} | KC only | 84–100% | — |
| parity | KC only | 61% (chance) | — |
| parity | KC + toggle motif | **100%** | toggle 100% |

### Task × motif matrix (algebra)

See `experiments/task_motif_matrix.py` output. Key: wrong motif = chance, right motif = 100%, full library = 100% everywhere.

### One-shot learning capacity

| patterns | disjoint | 25% overlap | 50% overlap |
|---|---|---|---|
| 50 | 100% | 100% | 100% |
| 200 | 100% | 96% | 92% |
| 300 | 86% | 87% | 83% |

### Catastrophic forgetting

| architecture | Task 1 retention after 10 tasks |
|---|---|
| shared MLP | 25% (catastrophic) |
| compartmentalized MBON | **100%** |

---

## Theory

### The learnability algebra

For a fixed architecture with linear readout, learnable concepts = linear functionals over available state coordinates. Adding motifs (toggle, ring, integrator) adds coordinates. The task's symmetry group determines which coordinate it needs.

### The wiring invisibility theorem

Under passive sensory-dominated drive (I_sensory >> I_recurrent), the connectome contributes zero wiring-dependent information. The wiring becomes visible ONLY through error-gated training.

### The five predictions

1. Capacity is wiring-independent
2. The search finds the right coordinates
3. Memory suppression prevents overfitting
4. Error gating is necessary for consolidation
5. Architecture = learning prior (not processing prerequisite)

All five are experimentally verified.

---

## Design principles for neuromorphic AI

1. Architecture = learning prior (not universal processor)
2. Sparse reciprocal network = Hopfield associative memory
3. Error-gated plasticity = only update when prediction error is large
4. Critical operating point = maximum information per energy
5. Extreme sparseness = pattern separation, not integration
6. Compartmentalized readout = prevents catastrophic forgetting

---

## Honest boundaries

- Motif recruitment demonstrated; de novo motif formation with reliable event-locking is not
- Induction only: the system recovers laws as compressed representations; deductive proof chains are out of scope
- All experiments at small scale (≤ 150 KCs modelled of 4,552; ≤ 4 factors)
- The wiring invisibility theorem is verified at this scale; larger brains may behave differently

---

## Data

BANC connectome v888 (Dorkenwald et al., 2024) is not redistributed. Place `meta.feather` and `edgelist_simple_v3.feather` in the data directory.

---

## Licence

MIT
