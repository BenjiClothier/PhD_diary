# Weekly Log: [16 - 23 Sept]

## 🎯 Focus of the Week
Simplify the (L, d) parameter-selection pipeline down to a single, defensible,
citable method (the closed-form Hankel orthogonal-residual detector, `v4_3`),
fix a significant evaluation-protocol bug, and produce an honest paper
write-up. Deliberate pivot away from the more elaborate, convoluted prior
attempts (Week_11) towards something simple enough to fully verify end to end.

---

## 📝 To-Do List
- [x] Audit and simplify the `(L, d)` estimator: remove dead fields
  (`K_harmonics`/`spec_entropy` that nothing consumed), remove an unused
  generic registry class, make `v4_3` self-contained (it previously called
  into `v2`'s candidate-search code for its held-out diagnostic; now has its
  own).
- [x] Derive `v4_3`'s window-length rule from citable theory: SSA
  separability licenses an integer-multiple-of-period window; Cohen's (1988)
  two-sample detection-power formula derives the multiplier `m*`. Replaces
  an earlier, unexplained constant multiplier.
- [x] Find and fix a real evaluation-protocol bug: this project's VUS-PR had
  been computed with a fixed `sliding_window=100` for its entire life. TSB-AD's
  actual official protocol derives this per dataset via `find_length_rank`.
  Fixed, and every affected number was re-run, not just the headline one.
- [x] Re-tune `delta` (target detection effect size) and the rank cap
  `max_d` from scratch under the corrected protocol.
- [x] Re-run the six-version (`v2`–`v4_3`) full-870-dataset diagnostic under
  the corrected protocol to check whether the earlier version comparison
  still held.
- [x] Citation-integrity check on the SSA separability claim, prompted by a
  direct question about how it was actually licensed.
- [x] Write the full paper draft set: Introduction, Background, Method,
  Results, Discussion, Abstract, Conclusion.
- [x] Build talk materials: a flow diagram of the full `(L,d)` pipeline plus
  two slide decks (a sparse talk deck and a full-derivation companion deck)
  — [Slides/](Slides/): [presentation.pdf](Slides/presentation.pdf) (talk),
  [companion.pdf](Slides/companion.pdf) (full derivations + intuition,
  not for the audience), [ld_estimation_flow.pdf](Slides/ld_estimation_flow.pdf)
  (the standalone pipeline diagram both decks embed).
- [x] Initial scaffolding for the Week 13 pivot (see Next Steps) started.

---

## 🔬 Progress & Experiments

**Protocol fix and re-tuning cascade:**
- **Objective:** find out whether the project's VUS-PR numbers matched
  TSB-AD's actual official evaluation protocol, after noticing the codebase
  asserted "fixed `sliding_window=100`" as a TSB-AD convention without that
  ever having been checked against the primary source.
- **Setup:** read `benchmark_exp/Run_Detector_U.py` directly. It derives
  `sliding_window` per dataset via `find_length_rank` (an ACF-based period
  estimate on the raw signal — confirmed unsupervised, no labels involved).
  Not a fixed value.
- **Execution:** fixed `src/util/metrics.py`/`src/scorer.py` to derive it
  the same way; re-ran the full tuning cascade in sequence: `detection_effect_size`
  (13-point fine grid on the 48-dataset Tuning split, not the original
  7-point coarse grid), a test of raising the window ceiling `max_L`
  256→512 (motivated by real clipping evidence — 37.5% of Tuning datasets
  were hitting the old ceiling), and a re-confirmation of the rank cap
  `max_d`.

**Six-version diagnostic re-run:**
- **Objective:** check whether the original justification for multi-cycle
  windows (an earlier comparison showing the single-period version, `v4_2`,
  had the worst mean VUS-PR of six versions) still held under the corrected
  protocol.
- **Setup:** `v2`, `v3`, `v4`, `v4_1`, `v4_2`, `v4_3`, full 870-dataset union,
  8 workers.
- **Execution:** `utils/run_benchmark.py` per version, cleared stale caches
  first (the resume-cache was keyed on version but not on the protocol
  change, which would have silently served stale results otherwise).

---

## 📊 Results & Insights

**Final controlled result** (official TSB-AD-U Eva split, 350 datasets,
tuned only on the disjoint 48-dataset Tuning split): `v4_3`
(`delta=1.9`, `max_d=14`, `max_L=256`) scores mean VUS-PR **0.5673**,
median **0.6647**, against the current leaderboard entry TimeRCD-MAFT's
0.5856/0.6511.
- Behind on mean (−0.0183), ahead on median (+0.0136). A **mixed** result,
  not a win on both metrics — this corrects an earlier, wrong-protocol
  version of this comparison that had claimed to exceed TimeRCD-MAFT on
  both.

**Every VUS-PR number in the project dropped once the protocol was
corrected**, not just the headline figure — confirms the old fixed-100
convention was inflating scores project-wide, not a one-off.

**The window ceiling test (`max_L` 256→512) made things worse, not
better.** Raised specifically because 256 was clipping the derived window
length for over a third of the Tuning split — but re-running the full
delta ablation under 512 gave a *lower* mean VUS-PR at every single one of
13 tested delta values. Reverted to 256. Consistent with this project's own
"expressive slack" argument (a larger window gives the healthy subspace
more room to also explain away anomalies) cutting the other way from the
motivating hypothesis. Recorded as a real negative result, not discarded.

**The six-version diagnostic reversed, not just shifted.** Under the old
protocol, `v4` (multi-cycle, clamped rank) had the best mean of six
versions and `v4_2` (single-period) the worst. Under the corrected
protocol, `v4_2` now has the **best mean and median of all six** —
undermining the *aggregate* case for multi-cycle windows. What still
holds: `v4_2`'s catastrophic-collapse failure mode is real and unchanged
(still ~15% of the 870-dataset union, worst case a ~0.97 VUS-PR drop on one
dataset), so multi-cycle windows remain a legitimate robustness argument —
just not an aggregate-performance one, on this specific diagnostic.

**Citation-integrity finding.** While preparing the talk, a direct question
about how the window-length rule was "licensed by SSA separability theory
(Golyandina & Zhigljavsky, 2011)" led to actually checking that citation —
it does not exist. Confirmed by reading the real table of contents of the
named journal issue directly (the actual articles at those page numbers are
on unrelated topics — Einstein–Yang–Mills–Higgs equations and harmonic
analysis). The underlying claim is still plausible general SSA theory
(most likely traceable to the Golyandina/Nekrutkin/Zhigljavsky 2001
textbook, already correctly cited elsewhere in the project), but that has
not itself been independently confirmed, and the fabricated citation had
been asserted as "checked live" in three separate places before this. Fixed
everywhere it appeared — 6 files, 8 locations, including the slides — and
logged as a finding in its own right, not quietly edited away.

**Net honest takeaway:** a materially more defensible, more honestly
reported method than the previous version — every free constant now has a
stated rationale and every number has a stated protocol — but a genuinely
mixed empirical result, not a state-of-the-art claim. Both the strengths
(median win, domain-appropriate accuracy on quasi-periodic telemetry) and
the weaknesses (mean loss to TimeRCD-MAFT, the reversed six-version
diagnostic, blindness to nonlinear anomalies on HumanActivity data) are
written into the paper drafts as found, not smoothed over.

---

## ⏭️ Next Steps
Week 13 (starting today) pivots to an analytical method built on
**Diffusion Geometry and the Carré du Champ operator**, applied directly
to the point cloud in the `(L, d)`-derived embedding space — not a further
projection into a reduced Hankel-SVD latent subspace. Reuses this week's
`(L, d)` front end (the derivation itself, not the scoring mechanism built
on top of it) under the same principles: closed-form and checkable where
possible, every literature claim verified against a primary source before
it's trusted (not repeating this week's citation lesson), and mixed or
negative results reported as plainly as positive ones. Initial scaffolding
(a fresh, self-contained project combining the ported `(L,d)` estimator
with a from-scratch Carré du Champ scorer) was already started today; a
real scalability issue was found and fixed early (naive pairwise-distance
computation didn't scale past ~10k training windows without a GPU — replaced
with a proper spatial index).
