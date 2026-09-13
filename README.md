# flybrain-law-discovery

**Learning mathematical laws in a connectome-constrained LIF model of the fly mushroom body.**

This repository accompanies a research campaign (Sep 2026) that asks a concrete version of
the question *"can an insect-scale brain learn mathematics?"* — using the real wiring of the
*Drosophila* ventral nerve cord / mushroom body (BANC connectome, v888) inside a
leaky-integrate-and-fire (LIF) simulation, and treating "mathematics" with full honesty:
not metaphors, but tasks with checkable answers (counting, parity, modular arithmetic,
empirical law discovery).

Everything here runs on standard Python + numpy/pandas. With access to the BANC connectome
files the PN→KC wiring is **real**; without them, a `--synthetic` random wiring is used so
every experiment remains runnable.

## The learnability algebra (main theoretical output)

For a fixed architecture with a linear readout, the set of learnable concepts equals the set
of linear functionals over the available state coordinates. Concretely:

- The Kenyon-cell code encodes pulse count `k` as a **monotone** signal, so the circuit can
  learn every **threshold concept** (more/less, at-least/at-most) — and *nothing else*.
- Non-monotone concepts (parity, `k != 3`, XOR) are **not representable**: they do not exist
  in the state algebra. No amount of synaptic growth or neurogenesis helps, because those
  add capacity, not coordinates.
- **Motifs** (toggle = Z2 flip-flop, integrator, Z3 ring, ...) add state coordinates. Each new
  coordinate enlarges the learnable algebra by one dimension.
- Tasks **recruit** the correct motif spontaneously under error-driven gradient + L1
  competition, provided channels are variance-normalized (the biological analogue of APL
  feedback inhibition).

The task x motif matrix below is the cleanest demonstration (accuracy on held-out pulses;
`count {2..5}` vs parity vs mod-3-zero vs carry; motifs: Z2 toggle, Z3 ring, integrator):

| task | none | Z2 | Z3 | integrator | all |
|---|---|---|---|---|---|
| threshold k>=5 | 0.95 | 0.99 | 0.97 | 0.99 | 1.00 |
| parity (Z2) | 0.61 | **1.00** | 0.70 | 0.55 | 1.00 |
| mod-3 = 0 (Z3) | 0.59 | 0.65 | **1.00** | 0.60 | 1.00 |
| carry k>=6 | 0.97 | 0.95 | 0.99 | 0.99 | 1.00 |

**The task's symmetry group selects exactly the coordinate it needs.** Wrong motif ~ chance;
full library = 1.00 everywhere. The algebra predicts the entire table.

## Empirical law discovery

Given 24 random examples of `(a, b) -> [is a*b odd]`, the mushroom-body readout recovers the
law **"odd x odd = odd"** and extrapolates to all 12 unseen pairs at 100% — *but only if*
(1) parity coordinates (per-channel Z2 toggles) are available, and (2) the memorization
channel is suppressed (weight decay / channel standardization). Without coordinates the law
is unrepresentable (48% = chance); without suppression the memory channel drowns the law
channel (57-60%).

The pipeline closes into an **automated law-discovery loop**
([`experiments/exp4_auto_synthesis.py`](experiments/exp4_auto_synthesis.py)):
train -> inspect residuals -> detect the monomial the residuals concentrate on ->
synthesize that coordinate -> retrain -> repeat. On XOR3 the loop automatically diagnoses
*"residuals concentrate on the f1*f2*f3 monomial (separation 2.00)"*, synthesizes it, and
recovers the law — no human tells it that a three-way conjunction is needed.

**Discoverability algebra.** A law is discoverable iff
`coordinates exist (product grammar) AND the search finds them AND memory is suppressed`.
The encoding regime is a switch: temporally **co-occurring** channels make multiplicative
structure partially representable in raw KC features (48% -> 68%); temporally segregated
channels make conjunctions unrepresentable in principle.

## Headline results

| experiment | result |
|---|---|
| counting {2,3} vs {4,5} (real PN->KC wiring) | 0.84-1.00 |
| parity, fixed architecture | chance (architectural impossibility) |
| parity, with toggle motif recruited from pool | **1.00** (100% weight on toggle) |
| law "odd x odd = odd" from 24 examples | 100% on unseen pairs (with parity coords) |
| generalization to unseen *values* ({5,6} never shown) | 100% (law, not lookup table) |
| catastrophic forgetting (shared readout, 2 laws) | 25% retention -> MBON-compartments argument |
| automated coordinate synthesis (XOR3) | auto-detected f1*f2*f3, law recovered |
| connectome: CX -> MB coupling | 88.7% of central-complex neurons reach KCs within 2 hops |

## Repository layout

```
src/mb_lif.py                 LIF mushroom-body core: wiring, episodes, motif channels, readout
experiments/exp1_...py        counting + motif recruitment (three-arm: fixed/grow/neurogenesis)
experiments/exp2_...py        task x motif matrix (the learnability algebra table)
experiments/exp3_...py        empirical law discovery (odd x odd = odd)
experiments/exp4_...py        automated coordinate synthesis (pure Python, no data needed)
results/RESULTS.md            recorded numbers from the Sep 2026 runs
```

## Running

```bash
python experiments/exp4_auto_synthesis.py          # no data required
python experiments/exp1_motif_recruitment.py --banc /path/to/banc_888
python experiments/exp1_motif_recruitment.py --synthetic   # fallback wiring
```

## Honest boundaries

- Motif **recruitment** is demonstrated; de novo **motif formation** with reliable
  event-locking is not (an alternator emerges from random recurrence, but its flips do not
  lock 1:1 to onsets — "alternation is cheap, semantics is expensive"; reliable symbols look
  like an evolutionary/central-complex achievement).
- Induction only: the system recovers laws as compressed representations; deductive proof
  chains are out of scope.
- All experiments are small-scale (<= 150 KCs modelled of 4,552; <= 4 factors).

## Data

BANC connectome (Dorkenwald et al., 2024, "BANC" ventral nerve cord + brain connectome,
version 888) is not redistributed here. The loader expects `meta.feather`,
`edgelist_simple_v3.feather` in the directory given by `--banc`.

## Licence

MIT.
