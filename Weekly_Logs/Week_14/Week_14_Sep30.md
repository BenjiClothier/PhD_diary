# Weekly Log: [30 Sept - 7 Oct]

## 🎯 Focus of the Week
Act on the outcome of the 30 Sept supervisor meeting. First close out the
open question from Week 13 (how to combine the local and global scorers
without using future data), then turn to the meeting's to-do list, with
writing Hankel_AD up as a conference paper as the main thread.

Source of truth for numbers: `Hankel_AD/docs/PROGRESS.md`. Protocol for
every VUS-PR figure unless stated otherwise: TSB-AD-U, Tuning split (48
datasets) for development, Eva split (350) for single final runs,
`sliding_window` derived per dataset.

---

## 📝 To-Do List

**Carried over from Week 13:**
- [x] Try the "simple step" for combining scorers: only trust the local
  scorer's alarms where the global scorer is at least mildly suspicious.
  It failed (see below), so the whole combination question is written up
  in `Hankel_AD/docs/local_global_combination_report.md` and set aside.

**From the 30 Sept meeting:**
- [ ] Write up Hankel Anomaly Detection as a paper ready to submit to a
  high-level conference.
- [ ] Can Hankel AD be used for volcano anomaly detection?
- [x] Why does the Hankel method work? (see Progress)
- [x] Explain in detail, with examples, why local works for OPPORTUNITY
  and not for others. (see Progress)
- [ ] Can we pre-normalise the data so we don't need MAD?
- [ ] Why is curvature not working?
- [ ] Hankel offers low complexity: is there an application in distributed
  anomaly detection?

---

## 🔬 Progress & Experiments

**Gated combination ("simple step"), 30 Sept:**
- **Objective:** combine the local and global scorers without future
  data by trusting local's alarms only where global's robust z-score
  is above 1.
- **Setup:** training-only normalisation, gate fixed at 1 in advance,
  Tuning-48. Pass criteria stated before running: no significant loss
  vs. global; both Tuning OPPORTUNITY datasets ≥ 0.9; mean ≥ 0.5177.
- **Result:** mean 0.5012 vs. global 0.5295 (Wilcoxon p = 0.73);
  OPPORTUNITY 0.430 / 0.172 — fails, as predicted in advance (global is
  inverted on OPPORTUNITY, so it is rarely suspicious where the anomalies
  are). It did fix one false-alarm case (`386_UCR_id_84` 0.092 → 0.914).
- **Report:** `Hankel_AD/docs/local_global_combination_report.md` covers
  all five combination variants tried (four normalisations plus gating),
  each failing a different pre-stated criterion, and why: drift and
  anomalies look the same to any scaling rule, and `max()` lets one
  scorer's false alarms through.

**Why the Hankel method works, and when local wins, 30 Sept:**
- **Objective:** answer the two meeting questions with evidence, not
  assertion.
- **Setup:** a controlled synthetic test of four mechanisms (known ground
  truth, fixed L and d), an Eva family breakdown, and three real examples
  with figures.
- **Result:** write-up in `Hankel_AD/docs/why_global_and_local_work.md`.
  Global generalises within the one linear structure it learns (healthy
  windows at an unseen amplitude barely change its score, ×1.02 vs. ×2.51
  for local), and fails when the healthy data is several regimes: in a
  synthetic two-regime case an anomaly mixing the regimes lies inside
  the fitted subspace, VUS-PR 0.08 global vs. 1.00 local, the OPPORTUNITY
  pattern. Local remembers every training window, so it fails on any
  healthy data it hasn't seen: drift (`560_YAHOO`), a new level after an
  anomaly (`142_MSL`), chaotic dynamics. One prediction was refuted:
  chaos does not favour local (Lorenz: global 0.971 vs. local 0.529).

**Reflection: is this novel, and what is the narrative? (30 Sept)**

*The honest assessment.* The Hankel_AD work so far is largely piecemeal
engineering. The detector itself is not novel: a subspace residual on
delay vectors, close to singular spectrum analysis and to Sub-PCA on the
TSB-AD leaderboard. The (L, d) selection is a stack of heuristics. The
last two weeks were reactive — ensemble, then five normalisations, then
gating — each patching the previous failure. That sequence is a
well-documented negative result, not a contribution. It will probably
make a thesis chapter rather than a stand-alone paper.

*What is not piecemeal.* Read past the engineering and the analysis keeps
pointing at one idea. Takens' theorem guarantees that healthy delay
vectors lie on a manifold; each detector is a different approximation of
that manifold, and where an anomaly lies relative to it decides who can
see it:

| where the anomaly lies | global (linear span) | local (training samples) | example |
| :--- | :---: | :---: | :--- |
| outside the span | ✓ | ✓ | spikes; synthetic C1 |
| inside the span, off the manifold | ✗ | ✓ | OPPORTUNITY; MGAB; synthetic C3, C4b |
| healthy but outside training coverage (false alarm) | ✓ tolerates | ✗ flags | drift (`560_YAHOO`); new level (`142_MSL`); synthetic C2 |

The failure of every normalisation fix is the same fact from the scaling
side: from scores alone, "healthy but not yet covered" and "anomalous"
look identical.

*Candidate narrative.* "What a linear delay-embedding detector can and
cannot see, and how evaluation choices hide it." In the thesis, this
chapter becomes the linear approximation and exactly where it breaks;
the following chapters (local tangent geometry, Hankel_CDC_AD; learned
geometry, DG_AD) model the manifold between the two extremes.

*What could be genuinely novel (all untested):*
1. **One scale parameter linking global and local.** The Tweedie posterior
   covariance (DG_AD) has a noise scale ε: at large ε it averages over all
   the data (like the global scorer), at small ε it collapses to the
   nearest training points (like local). If so, "which scorer wins"
   becomes "which scale suits this data", and the drift/anomaly
   ambiguity is a statement about scale. May be partly known (kernel
   and diffusion methods have scale parameters) — needs checking.
2. **The geometric taxonomy above,** validated by predicting each dataset
   family's winner in advance and then checking.
3. **Evaluation pitfalls, quantified across methods:** whole-series
   normalisation (swings individual results by up to 0.9), test-period
   score scaling (the ensemble's significance came from it), anomalies
   inside the "healthy" training prefix (34/350 Eva datasets). Benchmark
   critique has precedent; what would be new is measuring how these
   choices change leaderboard rankings for several existing methods.

*Analysis plan, in order:*
1. Literature check on (1)–(3) before building anything.
2. Scale sweep of the posterior-covariance detector on Tuning: does it
   reproduce global at large ε and local at small ε, and does the best ε
   track something measurable (coverage, number of regimes)?
3. Taxonomy test on Eva: predict each family's winner from measurable
   properties before looking at scores.
4. Protocol sensitivity for three or four leaderboard methods (e.g.
   Sub-PCA, k-NN, Isolation Forest): training-only vs. whole-series
   normalisation, and with contaminated datasets excluded.

**Literature check on novelty (30 Sept):** full write-up in
`Hankel_AD/docs/literature_check_novelty.md`; key claims re-checked
against the sources.
- **The detector is already known.** Moskvina & Zhigljavsky (2003) is
  essentially its core (subspace of the uncentred trajectory-matrix
  covariance, distance to it), used for change points. It must be cited
  as the direct antecedent. TSB-AD's Sub-PCA is a different, weaker
  method, so beating it doesn't show novelty.
- **The scale idea is partly known.** Hoffmann (2007), kernel PCA for
  novelty detection: small kernel width behaves like a Parzen density,
  large width approaches standard PCA. DTE (ICLR 2024) links small noise
  to kNN. It becomes a bridge to DG_AD, not a headline.
- **The geometric taxonomy (span / manifold / coverage) was not found**
  in the main time-series anomaly taxonomies. With a stated-in-advance
  prediction of which detector wins per dataset family, this is the
  strongest remaining novel angle.
- **Evaluation pitfalls are mixed.** Contaminated training prefixes are
  already known (a TSB-AD GitHub issue; by design, all at the minimum
  prefix of 500). Whole-series normalisation is a stated convention whose
  effect is unmeasured, and TSB-AD's own OCSVM wrapper scales train and
  test differently. Whole-series score standardisation looks new.
- **Correction:** MGAB anomalies are splices (a matching segment removed
  and the ends joined), not changes of dynamics.

**Criticism of the taxonomy (30 Sept): the embedding isn't guaranteed to
be good.** Takens' embedding is sensitive to L and τ; τ is fixed at 1 to
avoid a parameter, and v4_3 picks L for detection power, not embedding
quality. So attractor-level claims (e.g. MGAB splices "off the
attractor") are about an idealised embedding, not verified for ours. The
taxonomy's categories remain well-defined for the point cloud actually
built, and the pre-registered test measures that same point cloud, so it
stands; the narrative should say "geometry of the delay vectors as
built", with Takens as motivation. Planned checks on Tuning: sweep τ
(1, 2, 4) at fixed L and at fixed window span, and compare the chosen
embedding with standard diagnostics (mutual information for τ, false
nearest neighbours for dimension).

**Pre-registered blind test of the geometric taxonomy (30 Sept): passes.**
Rule, set and criteria fixed in advance
(`Hankel_AD/docs/preregistration_geometry_predicts_winner.md`); 472
datasets never scored by the local scorer; predictions fingerprinted
before scoring. The rule compares, on the training prefix only, how far
new healthy windows are from the fitted subspace vs. from the nearest
earlier window, and predicts the detector with the smaller gap.
- Spearman with (global − local) VUS-PR: **0.596** (needed ≥ 0.37).
- Accuracy on 296 clear-winner datasets: **0.855** vs. 0.784 for always
  predicting global (p = 0.001).
- Caveat: within families the ordering holds (YAHOO +0.44, WSD +0.59, UCR
  +0.29), but the fixed cutoff doesn't beat each family's own majority
  winner — part of the result is between-family. One revision to the
  rule was made before any blind data was touched (logged).

**Maths guide: the same detector with two beliefs (30 Sept).**
`Hankel_AD/docs/two_beliefs_maths_guide.md`. Starting from Tweedie's
formula, both scorers are "guess the healthy window, measure the gap"
(a denoising residual). The global scorer is exactly this under the
belief "healthy data is one Gaussian" (at noise γ²); the local scorer is
this under "healthy data is exactly the training windows", at small
noise. Turning the noise dial never crosses from one to the other; both
meet at "distance from the average window" at high noise. Mixtures of
Gaussians run between the two beliefs.

---

## 📊 Results & Insights

---

## ⏭️ Next Steps
