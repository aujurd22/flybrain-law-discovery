# Recorded results (Sep 2026 campaign)

All numbers produced by the code in `experiments/` on the BANC v888 wiring
(150 KCs modelled, 100 PNs, real synapse-count weights) unless noted.

## Counting and motif recruitment (exp1)

| task | readout | accuracy | motif weight split |
|---|---|---|---|
| count {2,3} vs {4,5} | KC only | 0.84-1.00 | - |
| count {2,3} vs {4,5} | KC + motif pool | 0.99 | KC 0.76 / toggle 0.23 / random 0.00 |
| parity {2,4} vs {3,5} | KC only | 0.61 (chance) | - |
| parity {2,4} vs {3,5} | KC + motif pool | **1.00** | toggle 1.00 / integrator 0.00 / random 0.00 |

Pulse-onset detection from the MB-own population rate: exact (mean deviation 0.00 from k).
Without channel standardization the KC channel drowns motif channels (parity peaks at 0.58-0.70
instead of 1.00) — fair competition is a precondition, not a detail.

## Task x motif matrix (exp2)

See README table. Key reading: wrong motif ~ chance, right motif = 1.00, full library =
1.00 everywhere. The learnability algebra predicts the whole table.

## Empirical law discovery (exp3)

Law: odd x odd = odd (never stated). Train 24 pairs from {1..6}^2, test the other 12
(10 splits):

| readout channels | seen | unseen (new theorem) |
|---|---|---|
| raw KC counts, sequential channels | 100% | 48% (chance — law unrepresentable) |
| raw KC counts, interleaved channels | 100% | 68% (co-occurrence enables partial conjunction) |
| KC + per-channel Z2 toggles, with memory suppression | 100% | **100%** |
| KC + Z2, WITHOUT suppression | 100% | 57-60% (memorization drowns the law) |

Generalization to unseen values: training on {1..4}^2 only, testing pairs involving 5 and 6
(numbers never exemplified): KC+Z2 = 100% (the law, not a lookup table); raw = 75%.

## Catastrophic forgetting and compartments

Same 36-pair domain, shared readout: law1 (product parity) to 100%, then law2 (sum parity)
to 100% — law1 retention drops to **25%** (below chance: active overwrite). Both laws are
linearly representable in the same coordinates; interference lives in the readout layer,
arguing for output-compartment allocation (the fly MB has ~34 MBONs).

## Automated coordinate synthesis (exp4)

XOR3 from first-order coordinates: residual trace [32] -> auto-diagnosis
"residuals concentrate on f1*f2*f3 (separation 2.00)" -> synthesize -> [0].
NOR3: correctly synthesizes nothing (first-order suffices).
XOR3 with full coordinates: **8 training examples -> 56 unseen pairs at 100%.**

## Connectome findings feeding this model

- Resting neuron positions are NTIL-like (collinearity-avoidant, 0.04x uniform);
  active assemblies are anti-NTIL (8-15x collinearity, line loads 100+, direction-concentrated
  bundles) — structure layer vs function layer.
- Central complex -> mushroom body coupling exists anatomically: 88.7% of CX neurons reach
  KCs within 2 hops (67% at exactly 2), plus 51 direct CX->KC synapses.
- The toggle/ring hardware exists: CX intrinsic mutual-excitation pairs are 10.7x enriched
  over degree-preserving random controls (2,327 pairs), consistent with the known
  ring-attractor architecture (recurrent excitation inside CX, inhibition via external D7).

## Known pitfalls (for reproduction)

1. Pulse amplitude 70 is sub-threshold: KCs never fire (need ~150 with 3 ms pulses).
2. A leaky readout (tau = 20 ms) with 40 ms pulse spacing destroys counting — integrate over
   the full episode.
3. Softmax two-column readouts have a conserved quantity (W column sums) that collapses to
   constant-class prediction; use a single decision direction with bias.
4. Episode-centered features delete the counting signal (it lives in the mean); use a
   running baseline across episodes + explicit bias.
5. Standardize every channel (running |value| average) before letting them compete.


## Beyond-NTIL extensions (Sep 14 night session)

### Similarity decorrelation / novelty detection (t66)
k-WTA feedback inhibition (top 5% KC) compresses input overlap super-quadratically on the
real wiring: rho 0.2 -> 0.02, 0.4 -> 0.08, 0.8 -> 0.42. Orthogonal novel inputs activate a
fresh KC set (overlap 0.10). **The MB is a distance-squaring machine; novelty detection is
the geometric side effect of sparsification.**

### One-shot learning capacity (t73)
Each pattern presented exactly once (3 delta-rule passes on that example only):
- disjoint patterns: 100% accuracy up to N=200, 86% at N=300
- 25%/50% overlap: 96%/92% at N=200
Capacity ~200-300 one-shot associations; interference = overlap x sparseness.
(Metrology: single-pass lr=0.5 explodes the weights; normalize features, use low lr.)

### Decision dynamics (t74, scalar-evidence version)
Two leaky accumulators (approach/avoid MBONs, mutual inhibition) fed by weak per-step KC
projection evidence (disjoint PN pools required - with mixed pools the prototype overlap
is 0.96 and the two "odors" are unresolvable):
- flip-rate psychometric: 98% / 92% / 85% / 82% at 0/10/20/30% evidence corruption
- speed-accuracy tradeoff: threshold 0.5 -> 80% (instant); 2.5 -> 95% (4 steps)

### Full-connectome criticality (t75 series, GPU) ★
The ENTIRE BANC (204,257 neurons, 13.6M connections) with slow synapses (tau=150ms):
- the silence-to-runaway transition occurs within a razor-thin gain band [0.005, 0.05]
- in the intermittent regime (gain 0.01-0.02): heavy-tailed avalanche distributions,
  max avalanche 3,904 spikes (~2% of the brain), log-log slope -0.78..-0.86
- metrology: an avalanche counter must distinguish two causes of "zero closed avalanches"
  (silence vs never-closing runaway) - the first measurement mislabeled supercritical
  global activation as network death

### T41-A pure-math extension (cross-linked from the NTIL repo)
Exhaustive verification extended to m=5..12: 7,565 coarse solutions, 0 confined-lift
survivors. Complete kill taxonomy (diagonal multiset + center-diagonal) matches the brute
force lift test per-solution at m=4..7 (see the NTIL repo
`research/FOUR_TIGHT_EXCEPTION.md` and the night FINDINGS).

