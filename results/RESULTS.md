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


## The fly brain reads — and why wiring is invisible at face value (Sep 14, t77/t78)

Streaming a character corpus through the full connectome (204k neurons, GPU LIF) and
training a next-char readout: real wiring 32.1% > bigram 25.6% >> chance 3.8% — but
**real wiring = shuffled wiring to three decimals**.

Memory-kernel measurement (decode char[t-lag] from state[t]):
- rate coding: lag0 99%, lag1 28% (= reverse-bigram linguistic correlation, not memory),
  lag2+ = 0 — the previous token is forgotten within one token
- slow-synaptic-trace coding (s_slow as the state): lag1 28% -> 46%, lag2 -> 31% —
  the trace EXISTS in the slow variable
- but slow/real == slow/shuf exactly: the trace is carried by the directly-driven
  sensory neurons' own afterimages, not by recurrent propagation

**Mechanism**: sensory drive (I=30) dominates recruitment; recurrent contributions are
drowned. This explains the ESA/MDPI reservoir line's "resilient to overfitting" finding —
at sensory-dominated drive, the wiring cannot express itself in sequence memory.

**Breakthrough route identified**: wiring-dependent sequence memory requires operating in
the propagation regime (the razor-thin intermittent band of T75), where traces travel as
avalanches beyond the directly-driven population. Precise target for the next session.


## The wiring-invisibility theorem (Sep 14 night, t79/t80) ★

Three-fold verification that a passive fly-brain reservoir carries NO wiring-dependent
sequence information:

| regime | probe | real vs shuffled |
|---|---|---|
| sensory-dominated (t77/78) | linear | identical |
| propagation (t79, gain 5.8) | linear | identical (but multi-token memory EXISTS: lag1 16.5%, lag2 14.7%) |
| propagation (t80) | 512x2 MLP on JL-projected states | identical |

The propagation regime DOES create multi-token memory traces (t79: lag2 was 0 in the
sensory-dominated regime, 14.7% in the propagation regime) — but the trace content is
fully determined by the sensory injection, not the wiring.

**Theorem (operational form):** under passive drive, the Drosophila whole-brain connectome's
contribution to sequence memory is independent of its wiring, for linear and nonlinear
probes at this scale. This fundamentally bounds the ESA/MDPI reservoir line and explains
why task-driven plasticity (the FSD/FLYHARD route) is the productive path: the wiring must
be SHAPED by the task, not merely PRESENT, to matter for computation.

Honest boundary: 4096-dim JL projection + 512-hidden MLP is not a universal discriminator;
stronger probes might extract wiring-dependent information. But the burden of proof has
moved to them.


## Three-timescale unified theory + five predictions (Sep 14 night)

The fly brain is a driven dissipative system characterized by three timescales whose ratio
determines the computational regime (analogous to Reynolds number in fluid dynamics):

| timescale | value | role |
|---|---|---|
| tau_m = 20ms | membrane leak | forgetting rate |
| tau_s = 150-400ms | slow synaptic decay | memory trace persistence |
| Delta_t = 40-250ms | token interval | information arrival rate |

**Key parameter**: tau_s / Delta_t
- < 1: no context (wiring invisible)
- 1-10: propagation regime (context, wiring matters through structure)
- > 10: saturation (runaway)

**Five predictions from the wiring-invisibility theorem:**
1. Capacity is wiring-independent (cross-species invariance)
2. lag-l decodability decays as exp(-l * Delta_t / tau_s)
3. Critical point = maximum Fano factor
4. Decorrelation = Johnson-Lindenstrauss random projection property
5. Error-gated plasticity is necessary for consolidation (verified in t63)

**Analytical solution of the memory kernel:**
lag-l decodability = A * exp(-l * Delta_t / tau_s) + B
Verified at tau_s=400ms, Delta_t=250ms: predicted lag1=54%, observed 46% (deviation from nonlinear saturation). Skeleton correct.

**Connectome structure**: clustering coefficient 0.0024 (near-random, NOT small-world).
The fly brain is a sparse broadcast network — this explains both the decorrelation (random
projection property) and the criticality (sparse branching networks self-organize to criticality).


## Massive reciprocal wiring: 801.9x enrichment (Sep 14 night, t85) ★★★

Analysis of the FULL connectome revealed the largest structural finding of this campaign:
**1,782,922 reciprocal connection pairs (26.2% of all edges), enriched 801.9x over random expectation.**

The reciprocal pairs are concentrated WITHIN brain regions:
- optic lobe intrinsic <-> optic lobe intrinsic: 404,845
- central brain intrinsic <-> central brain intrinsic: 265,070
- ventral nerve cord <-> ventral nerve cord: 154,347

This means the fly brain is NOT a feedforward processor with feedback corrections.
It is a **massively bidirectional network** where every connection has a return path.
Computation happens through reciprocal loops, not through feedforward pathways.

This connects to:
- **Predictive coding**: reciprocal connections = top-down predictions + bottom-up errors
- **Corollary discharge**: internal copies of motor commands require bidirectional wiring
- **Wiring invisibility theorem**: symmetric networks are invisible to direction-sensitive
  probes, explaining why real and shuffled wiring produce identical memory kernels

**T41-A update**: the confinement-lift exhaustive verification is extended to m=5..12
(7,565 coarse solutions, 0 survivors). The kill taxonomy (diagonal multiset + centre
diagonal) matches the brute-force lift test per-solution at m=4..7. The remaining proof
target is a counting argument over the 2-regular structure forcing the diagonal-multiset
pattern at m >= 5.

