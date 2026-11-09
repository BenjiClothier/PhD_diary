# Weekly Log: [5 Nov – 11 Nov]

*Started on 4 Nov, after the meeting. The plan below answers the meeting's questions, cheapest first. Nothing in it has
been run yet. Project log: `SG_AD/docs/PROGRESS.md`; results website: http://cuda16.ecs.vuw.ac.nz:8000/.*

## 🎯 Focus of the Week

1. Answer the questions from the 4 Nov meeting (below).
2. Keep the project's question in view: **which score detects which anomaly, and why?**

---

## 🗒️ Notes from the meeting (4 Nov)

- **"4d seems small for real data."** This refers to week 17's local-dimension check: at a larger neighbourhood scale,
  the median Tuning series' healthy windows lie near a set of about 4 dimensions.
  - What is the variance over the datasets?
  - How much noise breaks the estimate?
- **Why is the 1-D CNN better than the MLP?** Make the MLP act as a CNN: "copy the kernel into the weights".
- **What geometry lens works best for volcano and earthquake data?**
- **What is the measure of accuracy for the manifold view of the data?**
- **Make a synthetic test to show that large eigenvalues are regime shifts ("island hopping").**

---

## 📝 To-Do List

From this meeting:
- [ ] 1. The dimension estimate: its spread over datasets, its bias at realistic sample sizes, and how much noise breaks
  it.
- [ ] 2. Island hopping: a synthetic test of whether eigenvalues above 1 mark regime shifts.
- [ ] 3. A measure of how accurate the learned manifold is, without labels.
- [ ] 4. The MLP made to act as a CNN by copying the kernels into its weights.
- [ ] 5. Volcano and earthquake data: which geometry lens fits (survey first).

Carried over:
- [ ] Paper title (open since week 16).
- [ ] Results by period group (week 17).
- [ ] Fix the DG_AD paper's "our $L$ is the period" (week 17).
- [ ] Exp and Log maps from the learned tangent space: the known-answer check (owner, 3 Nov).
- [ ] When to spend the third Eva look (needs a frozen method).

---

## 🔬 Plan of action

Every item states the question, what is measured, the cheapest checks that the code does what the theory says, and what
it is compared with. No target numbers are set in advance.

### 1. Is "about 4 dimensions" real? (CPU, under an hour; mostly saved results)

**Question.** The 4 came from the Levina–Bickel nearest-neighbour estimate on the healthy training stretches
(`Wavelet_AD/experiments/check_local_dimension.py`, 28 Oct). Three things could make it read small:
- the series differ a lot, and the median hides that;
- the estimator reads low at our sample sizes;
- noise changes it.

**(a) Spread over datasets** (saved results, no new run).
- The per-series estimate at each neighbourhood scale and window length, by dataset family and period group.
- Quartiles and range.
- A bootstrap interval per series (resampling the query windows), so the spread between series can be compared with
  the uncertainty within a series.

**(b) Bias at realistic sample sizes.**
- Synthetic sets of known dimension (1, 2, 4, 8 and 16; for example products of circles) with the same number of
  windows as the Tuning series (median 1,915), estimated exactly as on Tuning.
- This shows whether a true 8 or 16 would read as about 4.

**(c) How much noise breaks the estimate.**
- The same known sets, and s01's five-parameter family, with noise added at a ladder of levels.
- The estimate against noise level, at each neighbourhood scale.
- Each Tuning series is then placed on that curve using its own noise level, estimated two ways: Donoho's estimate,
  and the denoiser's held-out residual (Donoho's estimate is unreliable on many series).

**Known answers.** Noise-free circles and tori read their true dimension. The shuffled-values reference, which fills
the space, keeps reading as in week 17.

**Compared with.** The week 17 estimates (median about 4; shuffled values growing ×4.9 with window length).

### 2. Island hopping: do eigenvalues above 1 mark regime shifts? (synthetic; a closed form first, then a short GPU run)

**Hypothesis** (supervisor). Eigenvalues of the learned $2\Gamma$ above 1 appear when a window lies between separate
healthy regimes ("islands"), not just far from the data.

**Theory to check first.** For healthy data made of two islands, the spread of possible clean windows around a point
between them is the within-island spread plus a between-island term: (weight × (1 − weight)) times the outer product of
the vector between the island centres, divided by $\sigma^2$. That term gives an eigenvalue above 1 along the line
between the islands once they are farther apart than the noise. This can be computed exactly for Gaussian islands
before any training.

**Test.**
- Healthy data with two or three islands: the same periodic signal at two levels, or at two periods. Training contains
  both.
- Train the denoiser and RMM as in s12.
- Score test windows in three places:
  - inside each island;
  - along the path from one island to the other (a regime switch, such as a level change halfway);
  - far from both islands, in a direction pointing to neither.
- **Measured:** the largest eigenvalue, its direction (its cosine with the direction between the islands), and "excess
  above 1" along the path and in each place.

**What would tell the readings apart.** A peak mid-path with an eigenvector along the island direction supports
"island hopping". Large eigenvalues only in the far place support "out of the training data".

**Known answers.** The closed-form eigenvalues for Gaussian islands. Inside an island, the learned dimension should
match that island's true dimension.

**Compared with.** The Tuning anomalies with large eigenvalues (s14 groups 4 and 5; DG_AD's 852_OPPORTUNITY). For
those, check afterwards whether the top eigenvector points towards another healthy regime of the same series.

### 3. A measure of accuracy for the manifold view (synthetic first, then Tuning from saved models)

**Question.** On real series there is no true manifold to compare with. Which label-free measures tell us whether the
learned geometry is right?

**Candidates.** All are computed on held-out healthy windows only.
- **Denoising fit:** the held-out denoising loss against the noise floor.
- **Idempotence:** denoising a clean estimate again should change it little, as for a projection.
- **Plane consistency:** the planes of neighbouring windows. Healthy motion should stay in the plane, apart from a
  curvature term; the sine gives the known amount.
- **Stability:**
  - the learned dimension and the angle between the planes of two models trained on the two halves of the training
    stretch, or with two seeds;
  - the model's own null: the healthy residual's size against what a set of the learned dimension predicts (tangent
    use 2.10).

**Test.**
- On the synthetic testbed, where the true projector is known, keep the measures that track the true projector error
  across training sizes and noise levels.
- Then report them per Tuning series, and read detection results by geometric accuracy.

**Known answers.** On the synthetic testbed the true error is known for every model.

**Compared with.** The true projector error (synthetic), and the learned dimension of s12 Part B and s17 (Tuning).

### 4. Make the MLP act as a CNN: copy the kernel into the weights (synthetic; GPU)

**Question.** Is the CNN's advantage something an MLP can represent but not find (then an MLP started from the CNN's
weights keeps the geometry), or something only weight sharing maintains during training (then it drifts away)?
- s13's untied CNN tested a version of this. It was locally connected, with the kernel copied to every position, and
  its geometry fell to MLP levels.
- A dense copy also removes locality: the weights outside each kernel's band start at zero and are free to grow.

**Design.**
1. Write each convolution of a trained CNN as a dense matrix (block-banded, zeros outside the band). The result is an
   MLP that computes exactly the same function.
2. Train it further with the same recipe, and track the s09/s13 geometry measures through training: projector error,
   tangent cosine, shift consistency and dimension count.
3. Compare with:
   - the same dense network from a random start;
   - the CNN trained for the same extra steps.

A dense copy of SG_AD's full CNN is very large (64 channels × 256 samples per layer). So this uses a narrower CNN, for
example 8 channels, after checking that the narrow CNN still learns the geometry.

**Known answers.** At step 0 the copy reproduces the CNN's outputs to float precision. The CNN's own geometry matches
s09.

**Compared with.** s09's CNN and MLP, and s13's untied CNN.

### 5. Volcano and earthquake data: which geometry lens? (desk work this week)

**Question.** Which of our lenses suits seismic and volcanic monitoring?
- **The residual at a short time resolution** (s15, s16): short transients such as event arrivals.
- **Phase, speed and tangent mismatch** (s04, s17a): tremor whose frequency drifts.
- **Eigenvalues above 1** (island hopping): switches between regimes, such as quiet and active states.

**This week:**
- **A literature and data survey:** open labelled datasets of seismic and volcanic signals, including New Zealand's
  GeoNet, with what their anomalies look like, the labels available, and how others evaluate.
- **A first look at one dataset:** what an "anomaly" is there, and which of our anomaly groups it resembles.

The s14-style map on that data comes next week, once the data and labels are understood.

**Compared with.** The literature's own reported numbers, labelled as theirs.

---

## Order for the week

1. Item 1(a), from saved results. Then 1(b) and 1(c) (CPU).
2. Item 2's closed form, then its synthetic run (a short GPU job).
3. Item 5's survey, run in the background throughout.
4. Item 3 on the synthetic testbed, then on Tuning from saved models.
5. Item 4 (GPU, the largest), once the narrow CNN is checked.

GPU work will be named (hosts and GPUs) before it starts, using at most 6 GPUs.

## 🔬 Progress & Experiments

*(to be filled in during the week)*

## 📊 Results & Insights

*(to be filled in during the week)*

## ⏭️ Next Steps

*(to be filled in at the end of the week)*
