# Weekly Log: 24 June 2026

## 🎯 Focus of the Week
Investigating the structural limitations of K-Nearest Neighbours (K-NN) in high-dimensional spaces, establishing rigid anomaly detection baselines, and proving that generative manifold drifting (via the Carré-du-Champ operator) can generalise sparse topologies.

---

## 📝 To-Do List
- [x] **Evaluate CDC Tangential Metric:** Implement the Carré-du-Champ mathematical operator as an evaluation metric to explicitly calculate tangential vs. orthogonal flow vectors on baseline models.
- [ ] **Geometric Score Function:** Understand and document the mathematical derivation for the geometric interpretation of the score function.
- [ ] **Investigate K-NN Breakdown:** Analyse the exact mathematical point at which K-NN fails by systematically analysing the ratio of feature dimensionality to the number of available data points.
- [ ] **Establish rigid K-NN Baseline:** Finalise a solid baseline for Anomaly Detection (AD) that relies purely on K-NN (e.g., Euclidean distance tracking), ensuring we have a strictly defined performance threshold to beat.
- [ ] **Develop Sub-Manifold Generalisation:** Train a generative neural network that structurally outperforms the K-NN baseline by successfully learning to interpolate and generalise the continuous sub-manifold geometry *between* sparse data points.

---

## 🔬 Progress & Experiments

**Geometric Score Function:** ### Breakdown of Appendix B.1: A General Framework for the Convergence of the Limiting Distribution

Instead of directly analysing the complex neural network scores right away, the authors first build a "general framework". They analyse how any probability distribution of a specific mathematical form behaves as a scaling parameter (like noise) shrinks to zero. 

#### 1. The Setup: The Target Density Function
The authors consider a probability density that looks like this: $\exp(-(f_{\theta}(x))/\theta)$. 

Here, $\theta$ is a small parameter that will eventually go to zero (in the main text, this is related to the noise level $\sigma$). The function $f_{\theta}(x)$ is broken down into a Taylor-like expansion:

$$
f_{\theta}(x) = f_0(x) + \theta f_1(x) + \hat{f}(x,\theta)
$$

* **$f_0(x)$**: The leading-order term. Crucially, its absolute minimum sits exactly on the data manifold $\mathcal{M}$. This term represents the *geometry*.
* **$f_1(x)$**: The next-order term. This term will eventually encode the actual *data distribution* on the manifold. 
* **$\hat{f}(x,\theta)$**: A tiny mathematical remainder (a perturbation) that shrinks faster than $\theta$, so it won't affect the final limit.

Because the density will eventually concentrate tightly around the manifold, they stop using standard ambient coordinates and switch to **local coordinates $(u, r)$**.
* **$u$**: The coordinate moving tangentially to the manifold.
* **$r$**: The coordinate moving orthogonally to the manifold.

---

#### 2. Assumption B.1: The Ground Rules
Before doing calculus, they need to ensure the maths won't break. Assumption B.1 sets these rules:
* The manifold $\mathcal{M}$ is compact, smooth ($C^4$), and has no boundaries.
* The leading term $f_0$ is completely minimised exactly on the manifold.
* The functions are smooth enough to take multiple derivatives ($f_0$ is $C^3$, $f_1$ is $C^1$).
* **The Curvature Condition:** The "second derivative" (Hessian) of $f_0$ in the perpendicular $r$ direction is strictly positive. This means $f_0$ curves sharply upward as you move away from the manifold, forming a steep "valley" that traps the probability mass.

---

#### 3. Corollary B.1 & Lemma B.1: Laplace's Method
The authors need to figure out what happens to the probability mass in the perpendicular $r$ direction as the valley gets infinitely steep ($\theta \rightarrow 0$). To do this, they rely on **Laplace's Method for Integrals**. 

* **Corollary B.1:** This is an adaptation of an existing mathematical theorem (Łapiński, 2019) that provides strict error bounds when approximating complex integrals of exponential functions. 
* **Lemma B.1:** They apply Laplace's method specifically to "integrate out" the normal coordinate $r$. By integrating over $r$, they are effectively squashing the probability cloud flat onto the surface of the manifold. 

The result of Lemma B.1 shows that integrating out $r$ yields a formula dominated by $\exp(-f_1(u, 0))$ and divided by the square root of the Hessian of $f_0$.

---

#### 4. Lemma B.2: Concentration of Measure
This lemma proves that as $\theta$ approaches zero, the probability of finding a data point outside the manifold drops to exactly zero. 

It formally proves that the density completely concentrates within the tubular neighbourhood and ultimately collapses onto the set of minimisers for $f_0$. If the resulting density converges to a stable distribution, this lemma guarantees that the support of that final distribution is strictly contained within the manifold $\mathcal{M}$.

---

#### 5. Theorem B.1: The Grand Conclusion
This is the payoff for Appendix B.1. It ties all the previous lemmas together to provide the exact formula for the limiting distribution on the manifold.

As $\theta \rightarrow 0$, the probability distribution $\pi_{\theta}(x)$ converges weakly to a distribution $\pi(u)$ on the manifold, defined as:

$$
\pi(u) = \frac{\exp(-f_1(u, 0)) \left| \frac{\partial^2 f_0(u, 0)}{\partial r^2} \right|^{-1/2} d\mathcal{M}(u)/du}{Z}
$$

*(Note: $Z$ represents the normalising integral in the denominator to ensure the probabilities sum to 1.)*

**What this formula means:**
1. **$\exp(-f_1(u, 0))$**: The density on the manifold is entirely dictated by the higher-order term $f_1$. The leading term $f_0$ just forced the points onto the manifold, but $f_1$ decides how they are spread out across the surface.
2. **The Hessian Term $\left| \frac{\partial^2 f_0(u, 0)}{\partial r^2} \right|^{-1/2}$**: This is a geometric correction factor. It adjusts the probability density based on how sharply the space curves away from the manifold at that specific point $u$. 
3. **$d\mathcal{M}(u)/du$**: This is the intrinsic Riemannian volume measure, ensuring the maths respects the natural, non-Euclidean curvature of the manifold. 

***

### Line-by-Line Breakdown of Corollary B.1

At its core, Corollary B.1 is a rigorous statement of **Laplace’s Method**. Laplace's method is a mathematical technique used to approximate integrals of the form $\int g(x) e^{-F(x)/\theta} dx$ when $\theta$ becomes very small. As $\theta \rightarrow 0$, the function $e^{-F(x)/\theta}$ creates an incredibly sharp, needle-like peak precisely at the minimum of $F(x)$. Because the peak is so sharp, almost all the "mass" (the value of the integral) comes from the immediate neighbourhood around that minimum.

Here is how the authors formalise this to track the exact error bounds.

#### 1. Setting the Stage: The Space and the Functions
> **Line:** "Let $\Omega\subset\mathbb{R}^{m}$ be an open set and let $\Omega^{\prime}\subset\Omega$ be a closed ball. Let $c_{1}:=Vol(\Omega^{\prime})$."

* **The Maths:** $\Omega$ is the entire mathematical space (or domain) we are integrating over. $\Omega^{\prime}$ is a smaller, tightly defined "bubble" (a closed ball) inside that space.
* **The Concept:** We know the action happens at the minimum of our function. $\Omega^{\prime}$ is defined as the specific neighbourhood immediately surrounding that minimum. We record its volume as $c_1$ so we can use it to bound our error later.

> **Line:** "Let $F,g:\Omega\rightarrow\mathbb{R}$ with the following assumptions:"

* **The Concept:** We are dealing with two functions. $F(x)$ sits in the exponent and drives the sharp peak. $g(x)$ sits outside the exponent and modulates the height or amplitude of the function.

#### 2. Assumptions on $F(x)$ (The Exponent)
> **Line:** "$F|_{\Omega^{\prime}}\in C^{3}(\Omega^{\prime})$ and $F\ge0$ on $\Omega$. There is a unique minimiser $x^{*}\in int(\Omega^{\prime})$ of $F$ on $\Omega$."

* **The Maths:** Inside our bubble $\Omega^{\prime}$, $F$ is three-times continuously differentiable ($C^3$). $F$ is never negative, and it hits its bottom value at exactly one point, $x^*$, which is safely inside the bubble.
* **The Concept:** We need $F$ to be highly smooth near the minimum so we can approximate it with a parabola (a Taylor expansion up to the second derivative). 

> **Line:** "Define $m_{1}:=\inf_{x\in\Omega\backslash\Omega^{\prime}}\{F(x)-F(x^{*})\}>0,$ $m_{2}:=\inf_{x\in\Omega^{\prime}}\lambda_{\min}(\nabla^{2}F(x))>0$." *(Note: The text briefly contains a typo calling the first term $n_1$, but later refers to it correctly as $m_1$)*.

* **The Maths ($m_1$):** This looks at all points *outside* our bubble ($\Omega\backslash\Omega^{\prime}$). It states that the value of $F(x)$ everywhere outside the bubble is strictly greater than the minimum $F(x^*)$ by at least a gap of $m_1$. This guarantees there are no "competing" peaks elsewhere in the domain that could mess up the integral.
* **The Maths ($m_2$):** Inside the bubble, we look at the Hessian matrix $\nabla^{2}F(x)$ (the matrix of second derivatives, which measures curvature). We require its smallest eigenvalue ($\lambda_{\min}$) to be bounded away from zero by at least $m_2$. This guarantees the minimum is strictly "sharp" or convex—it looks like a steep bowl, not a flat trough.

> **Line:** "Let $c_{2}:=\sup_{x\in\Omega^{\prime}}||\nabla^{2}F(x)|| , c_{3}:=\sup_{x\in\Omega^{\prime}}||\nabla^{3}F(x)||$"

* **The Concept:** The authors are just taking inventory. They record the maximum possible values (the supremum) of the second ($c_2$) and third ($c_3$) derivatives inside the bubble. By capping how wildly $F$ can curve, they can strictly bound the error of their approximation.

#### 3. Assumptions on $g(x)$ (The Amplitude)
> **Line:** "$g|_{\Omega^{\prime}}\in C^{1}(\Omega^{\prime})$ and $\int_{\Omega}|g(x)|dx<\infty$"

* **The Maths:** Inside the bubble, $g$ must be smooth (differentiable once). Across the *entire* space, the absolute area under $g(x)$ must be finite. 
* **The Concept:** If $g(x)$ were allowed to explode to infinity somewhere outside the bubble, it could overpower the exponential decay of $F(x)$ and ruin the integral. 

> **Line:** "Let $c_{4}:=\sup_{x\in\Omega^{\prime}}|g(x)| , c_{5}:=\sup_{x\in\Omega^{\prime}}||\nabla g(x)|| , c_{6}:=\int_{\Omega}|g(x)|dx.$"

* **The Concept:** Again, taking inventory. They record the maximum height of $g$ ($c_4$), the maximum slope of $g$ ($c_5$) inside the bubble, and the total area of $g$ across the whole space ($c_6$). 

#### 4. The Grand Result: The Integral Approximation
> **Line:** "Then, for every $\theta>0.$ $\int_{\Omega}g(x)e^{-F(x)/\theta}dx=\exp(-F(x^{*})/\theta)\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^{2}F(x^{*})|}}(g(x^{*})+h(\theta)).$"

This is the core equation of the corollary. Because the peak is so sharp around $x^*$, we can evaluate the integral by effectively replacing $F(x)$ with a parabolic Taylor expansion around $x^*$, turning the integrand into a standard Gaussian curve. Here are the pieces:
1. **$\exp(-F(x^{*})/\theta)$:** The raw height of the exponential function exactly at its peak.
2. **$\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^{2}F(x^{*})|}}$:** This is the volume or "width" of the peak. When you integrate a Gaussian, the area depends on the variance. Here, $\theta$ acts like the variance. The term $|\nabla^{2}F(x^{*})|$ is the determinant of the Hessian (the product of the curvatures in all dimensions). The sharper the bowl, the smaller the volume under the peak. 
3. **$g(x^{*})$:** Because the peak is infinitesimally narrow, the integral essentially only "sees" the function $g(x)$ at the exact point $x^*$.
4. **$h(\theta)$:** The mathematical error of this approximation.

#### 5. Bounding the Error
> **Line:** "where $|h(\theta)|$ can be upper bounded by a function of $(c_{1},...,c_{6},m_{1},m_{2})$. Moreover, $h(\theta)=O(\sqrt{\theta})$ as $\theta\rightarrow0.$"

* **The Maths:** The error $h(\theta)$ shrinks proportionally to $\sqrt{\theta}$ as $\theta$ gets closer to zero. 
* **The Concept:** Because they carefully logged all those boundaries ($c_1$ through $c_6$, and $m_1, m_2$), they can guarantee that the error won't randomly blow up. It is strictly controlled by those specific parameters.

> **Line:** "The $O(\sqrt{\theta})$ is uniform over any class of pairs $(F,g)$ for which $c_{1},...,c_{6}$ are bounded above and $m_{1}, m_{2}$ are bounded below by strictly positive constants uniformly over the class."

* **The Concept:** This ensures that this approximation works consistently (uniformly) for *any* functions $F$ and $g$ you might swap in, as long as they obey the minimum and maximum boundaries established earlier. In the context of the paper, this is vital because the authors want to apply this theorem to complex, shifting probability distributions around a data manifold.

---

### Line-by-Line Breakdown of Lemma B.1

**Goal:** The primary purpose of Lemma B.1 is to take a high-dimensional probability distribution that is concentrated near a data manifold and "flatten" it onto the surface of that manifold. It does this mathematically by integrating out the normal (perpendicular) coordinates.

---

#### 1. The Setup and the Goal

> **Line:** "Assume Assumption B.1, and let $h(x):\mathbb{R}^{d}\rightarrow\mathbb{R}$ be $C^{1}$ and uniformly bounded in $T_{\mathcal{M}}(\epsilon)$. Define $h(u,r):=h(\Phi(u,r))$."

* **The Maths:** We are bringing in the safety rules from Assumption B.1 (smoothness, compact manifold, sharp minimums). We also introduce a new function, $h(x)$, which is smooth ($C^{1}$) and doesn't blow up to infinity inside our tubular neighbourhood $T_{\mathcal{M}}(\epsilon)$. We then map it into our local coordinates $(u, r)$.
* **The Concept:** Think of $h(x)$ as a generic "test function" or an arbitrary amplitude. The authors are setting up a general integral so this lemma can be reused flexibly later in the paper.

> **Line:** "Then we have $\int_{||r||<\epsilon}\exp(-\frac{f_{\theta}(u,r)}{\theta})h(u,r)dr = \exp(-\frac{f_{0}(u,0)}{\theta})\exp(-f_{1}(u,0))\frac{(2\pi\theta)^{(d-n)/2}}{\sqrt{|\frac{\partial^{2}f_{0}}{\partial r^{2}}(u,0)|}}(h(u,0)+o(1))$, where the $o(1)$ term is uniform for $u$."

* **The Concept:** This is the destination. They are integrating *only* over $r$ (the normal/perpendicular direction) within the tiny distance $\epsilon$ from the manifold. Because the probability forms a sharp peak right on the manifold (where $r=0$), integrating across the peak effectively collapses the equation down to evaluate everything exactly at $r=0$. The result is a scaling factor, the volume of the Gaussian peak in the $d-n$ normal dimensions, and an error term $o(1)$ that uniformly vanishes as $\theta \rightarrow 0$.

---

#### 2. The Proof: Step 1 - Unpacking the Exponent

> **Line:** "We have that $\int_{||r||<\epsilon}\exp(-\frac{f_{\theta}(u,r)}{\theta})h(u,r)dr = \int_{||r||<\epsilon}\exp(-\frac{f_{0}(u,r)}{\theta})\exp(-f_{1}(u,r))h(u,r)(\exp(-\frac{\hat{f}(u,r,\theta)}{\theta}))dr$"

* **The Maths:** They take the target function $f_{\theta}$ and expand it into its three known parts: $f_{0} + \theta f_{1} + \hat{f}$. Because these parts are added inside an exponent, they can be separated into multiplied exponential terms using the rule $e^{A+B+C} = e^{A} e^{B} e^{C}$.
* **The Concept:** They are isolating the dominant geometric term ($f_{0}$), the density term ($f_{1}$), and the annoying perturbation/error term ($\hat{f}$).

---

#### 3. The Proof: Step 2 - The "+1 / -1" Trick

> **Line:** "$= \int_{||r||<\epsilon}\exp(-\frac{f_{0}(u,r)}{\theta})\exp(-f_{1}(u,r))h(u,r)dr + \int_{||r||<\epsilon}\exp(-\frac{f_{0}(u,r)}{\theta})\exp(-f_{1}(u,r))h(u,r)(\exp(-\frac{\hat{f}(u,r,\theta)}{\theta})-1)dr.$"

* **The Maths:** Look at the last term from the previous step: $\exp(-\frac{\hat{f}}{\theta})$. They rewrite this as $(1 + [\exp(-\frac{\hat{f}}{\theta}) - 1])$. They then multiply this through the integral to split it into two separate integrals.
* **The Concept:** The first integral is now completely "clean"—it only contains $f_{0}$, $f_{1}$, and $h$. The second integral acts as a "bin" that captures all the messy error associated with $\hat{f}$. The rest of the proof is just solving the clean integral and proving the bin integral goes to zero.

---

#### 4. The Proof: Step 3 - Solving the "Clean" Integral

> **Line:** "For the first term, we can directly apply Corollary B.1 with $F(r)=f_{0}(u,r)$, $g(r)=\exp(-f_{1}(u,r))h(u,r)$, and $\Omega^{\prime}$ being the ball $\{r | ||r||\le\hat{\epsilon}\}$."

* **The Concept:** They match the pieces of their clean integral to the variables required by Laplace's method (Corollary B.1). The geometric bowl $f_{0}$ becomes the exponent $F$, and the remaining functions merge to become the amplitude $g$.

> **
