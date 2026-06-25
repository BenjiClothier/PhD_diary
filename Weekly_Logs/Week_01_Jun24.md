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
**Geometric Score Function:** 
# Breakdown of Appendix B.1: A General Framework for the Convergence of the Limiting Distribution

Instead of directly analyzing the complex neural network scores right away, the authors first build a "general framework". They analyze how any probability distribution of a specific mathematical form behaves as a scaling parameter (like noise) shrinks to zero. 

## 1. The Setup: The Target Density Function
The authors consider a probability density that looks like this: $exp(-(f_{\theta}(x))/\theta)$. 

Here, $\theta$ is a small parameter that will eventually go to zero (in the main text, this is related to the noise level $\sigma$). The function $f_{\theta}(x)$ is broken down into a Taylor-like expansion:

$$f_{\theta}(x) = f_0(x) + \theta f_1(x) + \hat{f}(x,\theta)$$

* **$f_0(x)$**: The leading-order term. Crucially, its absolute minimum sits exactly on the data manifold $\mathcal{M}$. This term represents the *geometry*.
* **$f_1(x)$**: The next-order term. This term will eventually encode the actual *data distribution* on the manifold. 
* **$\hat{f}(x,\theta)$**: A tiny mathematical remainder (a perturbation) that shrinks faster than $\theta$, so it won't affect the final limit.

Because the density will eventually concentrate tightly around the manifold, they stop using standard ambient coordinates and switch to **local coordinates $(u, r)$**.
* **$u$**: The coordinate moving tangentially to the manifold
* **$r$**: The coordinate moving orthogonally to the manifold.

---

## 2. Assumption B.1: The Ground Rules
Before doing calculus, they need to ensure the math won't break. Assumption B.1 sets these rules:
* The manifold $\mathcal{M}$ is compact, smooth ($C^4$), and has no boundaries.
* The leading term $f_0$ is completely minimized exactly on the manifold.
* The functions are smooth enough to take multiple derivatives ($f_0$ is $C^3$, $f_1$ is $C^1$).
* **The Curvature Condition:** The "second derivative" (Hessian) of $f_0$ in the perpendicular $r$ direction is strictly positive. This means $f_0$ curves sharply upward as you move away from the manifold, forming a steep "valley" that traps the probability mass.

---

## 3. Corollary B.1 & Lemma B.1: Laplace's Method
The authors need to figure out what happens to the probability mass in the perpendicular $r$ direction as the valley gets infinitely steep ($\theta \rightarrow 0$). To do this, they rely on **Laplace's Method for Integrals**. 

* **Corollary B.1:** This is an adaptation of an existing mathematical theorem (Łapiński, 2019) that provides strict error bounds when approximating complex integrals of exponential functions. 
* **Lemma B.1:** They apply Laplace's method specifically to "integrate out" the normal coordinate $r$. By integrating over $r$, they are effectively squashing the probability cloud flat onto the surface of the manifold. 

The result of Lemma B.1 shows that integrating out $r$ yields a formula dominated by $exp(-f_1(u, 0))$ and divided by the square root of the Hessian of $f_0$.

---

## 4. Lemma B.2: Concentration of Measure
This lemma proves that as $\theta$ approaches zero, the probability of finding a data point outside the manifold drops to exactly zero. 

It formally proves that the density completely concentrates within the tubular neighborhood and ultimately collapses onto the set of minimizers for $f_0$. If the resulting density converges to a stable distribution, this lemma guarantees that the support of that final distribution is strictly contained within the manifold $\mathcal{M}$.

---

## 5. Theorem B.1: The Grand Conclusion
This is the payoff for Appendix B.1. It ties all the previous lemmas together to provide the exact formula for the limiting distribution on the manifold.

As $\theta \rightarrow 0$, the probability distribution $\pi_{\theta}(x)$ converges weakly to a distribution $\pi(u)$ on the manifold, defined as:

$$\pi(u) = \frac{exp(-f_1(u, 0)) \left| \frac{\partial^2 f_0(u, 0)}{\partial r^2} \right|^{-1/2} d\mathcal{M}(u)/du}{Z}$$

*(Note: $Z$ represents the normalizing integral in the denominator to ensure the probabilities sum to 1.)*

**What this formula means:**
1. **$exp(-f_1(u, 0))$**: The density on the manifold is entirely dictated by the higher-order term $f_1$. The leading term $f_0$ just forced the points onto the manifold, but $f_1$ decides how they are spread out across the surface.
2. **The Hessian Term $\left| \frac{\partial^2 f_0(u, 0)}{\partial r^2} \right|^{-1/2}$**: This is a geometric correction factor. It adjusts the probability density based on how sharply the space curves away from the manifold at that specific point $u$. 
3. **$d\mathcal{M}(u)/du$**: This is the intrinsic Riemannian volume measure, ensuring the math respects the natural, non-Euclidean curvature of the manifold. 


***


# Line-by-Line Breakdown of Corollary B.1

At its core, Corollary B.1 is a rigorous statement of **Laplace’s Method**. Laplace's method is a mathematical technique used to approximate integrals of the form $\int g(x) e^{-F(x)/\theta} dx$ when $\theta$ becomes very small. As $\theta \rightarrow 0$, the function $e^{-F(x)/\theta}$ creates an incredibly sharp, needle-like peak precisely at the minimum of $F(x)$. Because the peak is so sharp, almost all the "mass" (the value of the integral) comes from the immediate neighborhood around that minimum.

Here is how the authors formalize this to track the exact error bounds.

## 1. Setting the Stage: The Space and the Functions
> **Line:** "Let $\Omega\subset\mathbb{R}^{m}$ be an open set and let $\Omega^{\prime}\subset\Omega$ be a closed ball. Let $c_{1}:=Vol(\Omega^{\prime})$."

* **The Math:** $\Omega$ is the entire mathematical space (or domain) we are integrating over. $\Omega^{\prime}$ is a smaller, tightly defined "bubble" (a closed ball) inside that space.
* **The Concept:** We know the action happens at the minimum of our function. $\Omega^{\prime}$ is defined as the specific neighborhood immediately surrounding that minimum. We record its volume as $c_1$ so we can use it to bound our error later.

> **Line:** "Let $F,g:\Omega\rightarrow\mathbb{R}$ with the following assumptions:"

* **The Concept:** We are dealing with two functions. $F(x)$ sits in the exponent and drives the sharp peak. $g(x)$ sits outside the exponent and modulates the height or amplitude of the function.

## 2. Assumptions on $F(x)$ (The Exponent)
> **Line:** "$F|_{\Omega^{\prime}}\in C^{3}(\Omega^{\prime})$ and $F\ge0$ on $\Omega$. There is a unique minimizer $x^{*}\in int(\Omega^{\prime})$ of $F$ on $\Omega$."

* **The Math:** Inside our bubble $\Omega^{\prime}$, $F$ is three-times continuously differentiable ($C^3$). $F$ is never negative, and it hits its bottom value at exactly one point, $x^*$, which is safely inside the bubble.
* **The Concept:** We need $F$ to be highly smooth near the minimum so we can approximate it with a parabola (a Taylor expansion up to the second derivative). 

> **Line:** "Define $m_{1}:=inf_{x\in\Omega\backslash\Omega^{\prime}}\{F(x)-F(x^{*})\}>0,$ $m_{2}:=inf_{x\in\Omega^{\prime}}\lambda_{min}(\nabla^{2}F(x))>0$." *(Note: The text briefly contains a typo calling the first term $n_1$, but later refers to it correctly as $m_1$)*.

* **The Math ($m_1$):** This looks at all points *outside* our bubble ($\Omega\backslash\Omega^{\prime}$). It states that the value of $F(x)$ everywhere outside the bubble is strictly greater than the minimum $F(x^*)$ by at least a gap of $m_1$. This guarantees there are no "competing" peaks elsewhere in the domain that could mess up the integral.
* **The Math ($m_2$):** Inside the bubble, we look at the Hessian matrix $\nabla^{2}F(x)$ (the matrix of second derivatives, which measures curvature). We require its smallest eigenvalue ($\lambda_{min}$) to be bounded away from zero by at least $m_2$. This guarantees the minimum is strictly "sharp" or convex—it looks like a steep bowl, not a flat trough.

> **Line:** "Let $c_{2}:=sup_{x\in\Omega^{\prime}}||\nabla^{2}F(x)|| , c_{3}:=sup_{x\in\Omega^{\prime}}||\nabla^{3}F(x)||$"

* **The Concept:** The authors are just taking inventory. They record the maximum possible values (the supremum) of the second ($c_2$) and third ($c_3$) derivatives inside the bubble. By capping how wildly $F$ can curve, they can strictly bound the error of their approximation.

## 3. Assumptions on $g(x)$ (The Amplitude)
> **Line:** "$g|_{\Omega^{\prime}}\in C^{1}(\Omega^{\prime})$ and $\int_{\Omega}|g(x)|dx<\infty$"

* **The Math:** Inside the bubble, $g$ must be smooth (differentiable once). Across the *entire* space, the absolute area under $g(x)$ must be finite. 
* **The Concept:** If $g(x)$ were allowed to explode to infinity somewhere outside the bubble, it could overpower the exponential decay of $F(x)$ and ruin the integral. 

> **Line:** "Let $c_{4}:=sup_{x\in\Omega^{\prime}}|g(x)| , c_{5}:=sup_{x\in\Omega^{\prime}}||\nabla g(x)|| , c_{6}:=\int_{\Omega}|g(x)|dx.$"

* **The Concept:** Again, taking inventory. They record the maximum height of $g$ ($c_4$), the maximum slope of $g$ ($c_5$) inside the bubble, and the total area of $g$ across the whole space ($c_6$). 

## 4. The Grand Result: The Integral Approximation
> **Line:** "Then, for every $\theta>0.$ $\int_{\Omega}g(x)e^{-F(x)/\theta}dx=exp(-F(x^{*})/\theta)\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^{2}F(x^{*})|}}(g(x^{*})+h(\theta)).$"

This is the core equation of the corollary. Because the peak is so sharp around $x^*$, we can evaluate the integral by effectively replacing $F(x)$ with a parabolic Taylor expansion around $x^*$, turning the integrand into a standard Gaussian curve. Here are the pieces:
1. **$exp(-F(x^{*})/\theta)$:** The raw height of the exponential function exactly at its peak.
2. **$\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^{2}F(x^{*})|}}$:** This is the volume or "width" of the peak. When you integrate a Gaussian, the area depends on the variance. Here, $\theta$ acts like the variance. The term $|\nabla^{2}F(x^{*})|$ is the determinant of the Hessian (the product of the curvatures in all dimensions). The sharper the bowl, the smaller the volume under the peak. 
3. **$g(x^{*})$:** Because the peak is infinitesimally narrow, the integral essentially only "sees" the function $g(x)$ at the exact point $x^*$.
4. **$h(\theta)$:** The mathematical error of this approximation.

## 5. Bounding the Error
> **Line:** "where $|h(\theta)|$ can be upper bounded by a function of $(c_{1},...,c_{6},m_{1},m_{2})$. Moreover, $h(\theta)=O(\sqrt{\theta})$ as $\theta\rightarrow0.$"

* **The Math:** The error $h(\theta)$ shrinks proportionally to $\sqrt{\theta}$ as $\theta$ gets closer to zero. 
* **The Concept:** Because they carefully logged all those boundaries ($c_1$ through $c_6$, and $m_1, m_2$), they can guarantee that the error won't randomly blow up. It is strictly controlled by those specific parameters.

> **Line:** "The $O(\sqrt{\theta})$ is uniform over any class of pairs $(F,g)$ for which $c_{1},...,c_{6}$ are bounded above and $m_{1}, m_{2}$ are bounded below by strictly positive constants uniformly over the class."

* **The Concept:** This ensures that this approximation works consistently (uniformly) for *any* functions $F$ and $g$ you might swap in, as long as they obey the minimum and maximum boundaries established earlier. In the context of the paper, this is vital because the authors want to apply this theorem to complex, shifting probability distributions around a data manifold.

---

# Line-by-Line Breakdown of Lemma B.1

**Goal:** The primary purpose of Lemma B.1 is to take a high-dimensional probability distribution that is concentrated near a data manifold and "flatten" it onto the surface of that manifold. It does this mathematically by integrating out the normal (perpendicular) coordinates.

---

## 1. The Setup and the Goal

> **Line:** "Assume Assumption B.1, and let $h(x):\mathbb{R}^{d}\rightarrow\mathbb{R}$ be $C^{1}$ and uniformly bounded in $T_{\mathcal{M}}(\epsilon)$. Define $h(u,r):=h(\Phi(u,r))$."

* **The Math:** We are bringing in the safety rules from Assumption B.1 (smoothness, compact manifold, sharp minimums). We also introduce a new function, $h(x)$, which is smooth ($C^{1}$) and doesn't blow up to infinity inside our tubular neighborhood $T_{\mathcal{M}}(\epsilon)$. We then map it into our local coordinates $(u, r)$.
* **The Concept:** Think of $h(x)$ as a generic "test function" or an arbitrary amplitude. The authors are setting up a general integral so this lemma can be reused flexibly later in the paper.

> **Line:** "Then we have $\int_{||r||<\epsilon}exp(-\frac{f_{\theta}(u,r)}{\theta})h(u,r)dr = exp(-\frac{f_{0}(u,0)}{\theta})exp(-f_{1}(u,0))\frac{(2\pi\theta)^{(d-n)/2}}{\sqrt{|\frac{\partial^{2}f_{0}}{\partial r^{2}}(u,0)|}}(h(u,0)+o(1))$, where the $o(1)$ term is uniform for $u$."

* **The Concept:** This is the destination. They are integrating *only* over $r$ (the normal/perpendicular direction) within the tiny distance $\epsilon$ from the manifold. Because the probability forms a sharp peak right on the manifold (where $r=0$), integrating across the peak effectively collapses the equation down to evaluate everything exactly at $r=0$. The result is a scaling factor, the volume of the Gaussian peak in the $d-n$ normal dimensions, and an error term $o(1)$ that uniformly vanishes as $\theta \rightarrow 0$.

---

## 2. The Proof: Step 1 - Unpacking the Exponent

> **Line:** "We have that $\int_{||r||<\epsilon}exp(-\frac{f_{\theta}(u,r)}{\theta})h(u,r)dr = \int_{||r||<\epsilon}exp(-\frac{f_{0}(u,r)}{\theta})exp(-f_{1}(u,r))h(u,r)(exp(-\frac{\hat{f}(u,r,\theta)}{\theta}))dr$"

* **The Math:** They take the target function $f_{\theta}$ and expand it into its three known parts: $f_{0} + \theta f_{1} + \hat{f}$. Because these parts are added inside an exponent, they can be separated into multiplied exponential terms using the rule $e^{A+B+C} = e^{A} e^{B} e^{C}$.
* **The Concept:** They are isolating the dominant geometric term ($f_{0}$), the density term ($f_{1}$), and the annoying perturbation/error term ($\hat{f}$).

---

## 3. The Proof: Step 2 - The "+1 / -1" Trick

> **Line:** "$= \int_{||r||<\epsilon}exp(-\frac{f_{0}(u,r)}{\theta})exp(-f_{1}(u,r))h(u,r)dr + \int_{||r||<\epsilon}exp(-\frac{f_{0}(u,r)}{\theta})exp(-f_{1}(u,r))h(u,r)(exp(-\frac{\hat{f}(u,r,\theta)}{\theta})-1)dr.$"

* **The Math:** Look at the last term from the previous step: $exp(-\frac{\hat{f}}{\theta})$. They rewrite this as $(1 + [exp(-\frac{\hat{f}}{\theta}) - 1])$. They then multiply this through the integral to split it into two separate integrals.
* **The Concept:** The first integral is now completely "clean"—it only contains $f_{0}$, $f_{1}$, and $h$. The second integral acts as a "bin" that captures all the messy error associated with $\hat{f}$. The rest of the proof is just solving the clean integral and proving the bin integral goes to zero.

---

## 4. The Proof: Step 3 - Solving the "Clean" Integral

> **Line:** "For the first term, we can directly apply Corollary B.1 with $F(r)=f_{0}(u,r)$, $g(r)=exp(-f_{1}(u,r))h(u,r)$, and $\Omega^{\prime}$ being the ball $\{r | ||r||\le\hat{\epsilon}\}$."

* **The Concept:** They match the pieces of their clean integral to the variables required by Laplace's method (Corollary B.1). The geometric bowl $f_{0}$ becomes the exponent $F$, and the remaining functions merge to become the amplitude $g$.

> **Line:** "Define $J=exp(-\frac{f_{0}(u,0)}{\theta})exp(-f_{1}(u,0))\frac{(2\pi\theta)^{(d-n)/2}}{\sqrt{|\frac{\partial^{2}f_{0}}{\partial r^{2}}(u,0)|}}$. The first term can be approximated as $J(h(u,0)+o(1))$"

* **The Math:** By plugging the pieces directly into the formula from Corollary B.1, they get the exact result they are looking for. To save space, they bundle the bulky exponential and Gaussian volume terms into a single variable, $J$.

---

## 5. The Proof: Step 4 - Trashing the Bin Integral

> **Line:** "The second term can be upper bounded by $sup_{r}|h(u,r)|\cdot sup_{r}|exp(-\frac{\hat{f}(u,r,\theta)}{\theta})-1|\int_{||r||<\epsilon}exp(-\frac{f_{0}(u,r)}{\theta})exp(-f_{1}(u,r))dr$"

* **The Math:** They are trying to find the absolute maximum possible size of the second (bin) integral. They pull the amplitude $h$ and the error factor $(exp(...) - 1)$ outside the integral by replacing them with their maximum possible values (the supremum, or $sup$).

> **Line:** "$=o(1)J(1+o(1))=o(1)J$, where we used Corollary B.1 for the integral. The lower bound can be obtained similarly. The result follows."

* **The Math:** Why does it equal $o(1)J$?
    1.  By Assumption B.1, we know $\hat{f}$ shrinks faster than $\theta$ (it is $o(\theta)$). Therefore, $\frac{\hat{f}}{\theta} \rightarrow 0$. As a result, $exp(0) - 1 = 1 - 1 = 0$. So that entire supremum term becomes vanishingly small ($o(1)$).
    2.  The remaining integral is just another clean integral, which evaluates to $J(1+o(1))$ via Corollary B.1.
    3.  Multiplying a vanishing term by $J$ leaves you with $o(1)J$.
* **The Concept:** They have successfully proven that the messy perturbation $\hat{f}$ from their original function has absolutely zero impact on the final limiting distribution as $\theta \rightarrow 0$. The proof is complete.
---

# Line-by-Line Breakdown of Lemma B.2

**Goal:** This lemma formally proves "Concentration of Measure." It demonstrates that as the noise parameter ($\theta$) shrinks to zero, the probability distribution completely collapses onto the data manifold $\mathcal{M}$. There will be absolutely zero probability mass left anywhere else in the space.

---

## 1. The Setup and Assumptions

> **Line:** "Let $f_{\theta}(x)=f_{0}(x)+\overline{f}(x,\theta)$, such that $exp(-f_{\theta}(x)/\theta)$ is a normalized density function on $\mathbb{R}^{d}$."

* **The Math:** They define the exponent of the probability density as having two parts: a dominant geometric term $f_0(x)$ and an error/perturbation term $\overline{f}(x,\theta)$. 

> **Line:** "Suppose $\mathcal{M}$ is a connected and compact $C^{4}$ manifold without boundary. Assume that:
> 1. $f_{0}(x)$ is continuous with $arg \min_{x \in T_{\mathcal{M}}(\epsilon)} f_{0}(x) = \mathcal{M}$ and $\min_{x \in T_{\mathcal{M}}(\epsilon)} f_{0}(x) = 0$."

* **The Math:** They establish the ground rules for the geometry. Crucially, the dominant function $f_0$ reaches its absolute minimum (which is exactly zero) *only* when evaluated exactly on the manifold $\mathcal{M}$. If you evaluate a point off the manifold, $f_0(x)$ becomes strictly greater than zero.

> **Line:** "2. $\overline{f}(x,\theta)$ is continuous and uniformly $o(1)$ as $\theta\rightarrow0$ for all $x\in\overline{T_{\mathcal{M}}(\epsilon)}$."

* **The Math:** The error term $\overline{f}$ vanishes completely (goes to $o(1)$) as the parameter $\theta$ goes to zero. It does this evenly (uniformly) everywhere inside the tubular neighborhood around the manifold.

> **Line:** "3. The density concentrates in $T_{\mathcal{M}}(\epsilon)$, i.e., $\lim_{\theta\rightarrow0}\int_{T_{\mathcal{M}}(\epsilon)}exp(-\frac{f_{\theta}(x)}{\theta})dx=1$."

* **The Math:** They assume it is already established that 100% of the probability mass will eventually end up somewhere inside the tubular neighborhood ($T_{\mathcal{M}}(\epsilon)$). 

---

## 2. The Core Claim

> **Line:** "For any $\eta>0$, define the set $C_{\eta}=\{x|f_{0}(x)>\eta\}$. Then, $\int_{C_{\eta}\cup T_{\mathcal{M}}(\epsilon)^{c}}exp(-f_{\theta}(x)/\theta)dx\rightarrow0$ as $\theta\rightarrow0$."

* **The Math:** This is the hypothesis they intend to prove. They define a region $C_{\eta}$. This region is everywhere that $f_0(x)$ is greater than some tiny positive number $\eta$—meaning it is everywhere *except* exactly on the manifold. They claim that the probability of finding a data point in this region (or completely outside the tubular neighborhood) shrinks to exactly zero.

> **Line:** "If in addition, $exp(-f_{\theta}(x)/\theta)$ converges weakly to a distribution as $\theta\rightarrow0$, the support of the limiting distribution is contained in $\mathcal{M}$."

* **The Concept:** If the function stabilizes into a final distribution, that distribution will be entirely supported on the manifold $\mathcal{M}$.

---

## 3. The Proof: Bounding the Probability

> **Line:** "Proof. Since we have that $\int_{T_{\mathcal{M}}(\epsilon)}exp(-f_{\theta}(x)/\theta)dx\rightarrow1$ for the first result, it suffices to show that $\int_{T_{\mathcal{M}}(\epsilon)\cap C_{\eta}}exp(-f_{\theta}(x)/\theta)dx\rightarrow0$."

* **The Math:** Because Assumption 3 already established the mass is trapped inside the tubular neighborhood, they only need to evaluate the space *inside* the tube but *off* the manifold ($T_{\mathcal{M}}(\epsilon)\cap C_{\eta}$). If they can prove the integral over that specific area goes to zero, the proof is complete.

> **Line:** "According to the assumptions, we have that for any $\delta>0$, $\exists\theta_{0}$, such that $\forall\theta<\theta_{0},|\overline{f}(x,\theta)|<\delta$. Therefore, we have $\int_{T_{\mathcal{M}}(\epsilon)\cap C_{\eta}}exp(-f_{\theta}(x)/\theta)dx\le\int_{T_{\mathcal{M}}(\epsilon)\cap C_{\eta}}exp((-\eta+\delta)/\theta)dx\le Vol(T_{\mathcal{M}}(\epsilon))exp((-\eta+\delta)/\theta)$."

* **The Math:** Since the error term $\overline{f}$ vanishes, they can bound its maximum absolute value by a tiny number, $\delta$. Because they are evaluating the region where $f_0 > \eta$, they can bound the exponent by its worst-case scenario: $(-\eta + \delta)/\theta$. They pull this exponential term out of the integral, leaving only the finite volume of the tubular neighborhood.

> **Line:** "We choose $\delta=\eta/2$ then the right-hand side goes to zero as $\theta\rightarrow0$."

* **The Concept:** This is the critical step of the proof. If $\delta$ is chosen to be half the size of $\eta$, then $(-\eta + \delta)$ equals $-\eta/2$, which is a strictly negative number. When evaluating the limit of $exp(-\eta/(2\theta))$ as the parameter ($\theta$) shrinks toward zero, the exponent approaches $-\infty$. Since $e^{-\infty}$ approaches $0$, the entire term vanishes.

---

## 4. The Proof: Weak Convergence

> **Line:** "Let the limiting measure be $P,$ and $P_{\theta}$ be the probability measure corresponding to the density $exp(-f_{\theta}(x)/\theta)$. Since $C_{\eta}$ is an open set, we have that $P(C_{\eta})\le \lim \inf_{\theta\rightarrow0}P_{\theta}(C_{\eta})=0$."

* **The Math:** They transition from evaluating densities to evaluating the final probability measure $P$. Using the Portmanteau theorem for weak convergence, the final probability of an open set is less than or equal to the limit infimum of the approaching probabilities. Since they just proved the approaching probabilities go to 0, the final probability must also be exactly 0. 

> **Line:** "Denote $C:=\mathcal{M}^{c}.$ We have that $C=\cup_{m=1}^{\infty}C_{1/m}\cup\overline{T_{\mathcal{M}}(\epsilon)}^{c}$. Then we have $P(C)\le\sum_{m=1}^{\infty}P(C_{1/m})+P(\overline{T_{\mathcal{M}}(\epsilon)}^{c})=0$, which concludes the proof."

* **The Concept:** They define $C$ as the entire complement of the manifold ($\mathcal{M}^{c}$). They represent this space as a countable union of all the bounded regions off the manifold ($C_{1/m}$) plus the space completely outside the tubular neighborhood. Since they proved the probability measure of any of those individual regions is exactly 0, their sum also equals 0. 

The probability of finding a data point off the manifold is 0, concluding the proof.

---

# Line-by-Line Breakdown of Theorem B.1

**Goal:** This theorem represents the culmination of Appendix B.1. It ties together the integration mechanics of Lemma B.1 and the concentration bounds of Lemma B.2 to provide the exact, explicit formula for the limiting probability distribution on the data manifold.

---

## 1. The Setup and Assumptions

> **Line:** "Theorem B.1. Assume Assumption B.1. Define $\pi_{\theta}(x)\propto exp(-\frac{f_{\theta}(x)}{\theta})$ ,"

* **The Math:** The theorem begins by inheriting all the structural rules established in Assumption B.1 (such as a compact $C^4$ manifold and a strictly positive Hessian for $f_0$). It then defines the target probability density $\pi_{\theta}(x)$, which is proportional to the exponential of the expanded function $f_{\theta}(x)$ scaled by the parameter $\theta$.

> **Line:** "Assume that $1-\int_{x\in T_{\mathcal{M}(\epsilon)}}\pi_{\theta}(x)dx\rightarrow0$ as $\theta\rightarrow0$."

* **The Math:** This establishes a prerequisite condition: as $\theta$ approaches zero, the total probability mass strictly outside the tubular neighborhood $T_{\mathcal{M}}(\epsilon)$ drops to zero. This ensures that 100% of the relevant density is concentrated near the manifold, validating the use of local coordinates for the remainder of the analysis.

---

## 2. The Core Claim and Formula

> **Line:** "Then we have that as $\theta\rightarrow0$, $\pi_{\theta}$ converges weakly to the following distribution:"

* **The Concept:** This states the primary result. The high-dimensional, noisy density $\pi_{\theta}$ stabilizes (converges weakly) into a defined distribution strictly supported on the manifold surface.

> **Line:** "$\pi(u)=\frac{exp(-f_{1}(u,0))|\frac{\partial^{2}f_{0}(u,0)}{\partial r^{2}}|^{-1/2}d\mathcal{M}(u)/du}{\int_{\mathcal{M}}exp(-f_{1}(u,0))|\frac{\partial^{2}f_{0}(u,0)}{\partial r^{2}}|^{-1/2}d\mathcal{M}(u)/du},$"

* **The Math:** This is the  formula for the limiting distribution. It consists of a numerator (the unnormalized density at a specific point $u$ on the manifold) and a denominator (the normalization constant ensuring the total probability integrates to 1). 
* **The Breakdown of the Numerator:**
    * **$exp(-f_1(u, 0))$**: The leading geometric term $f_0$ drops out of the relative weighting because it equals 0 everywhere on the manifold. The density is therefore driven by the higher-order term $f_1$.
    * **$|\frac{\partial^{2}f_{0}(u,0)}{\partial r^{2}}|^{-1/2}$**: This is the geometric correction factor derived directly from Lemma B.1. It is the inverse square root of the determinant of the Hessian of $f_0$ in the normal direction. It adjusts the probability density based on how sharply the ambient space curves into the manifold at that specific location.
    * **$d\mathcal{M}(u)/du$**: This ensures the distribution is measured with respect to the manifold's intrinsic geometry rather than standard flat Euclidean space.

> **Line:** "where $d\mathcal{M}$ is the intrinsic measure on the manifold $\mathcal{M}$, i.e., $d\mathcal{M}(u)=|g(u)|^{1/2}du$, and $du$ is the Lebesgue measure on the local parameterization domain $U$."

* **The Concept:** This formally defines the intrinsic measure. Here, $g(u)$ is the Riemannian metric tensor. Multiplying the standard flat volume $du$ by the square root of the determinant of $g(u)$ scales the volume properly to account for the curvature of the manifold.

---

## 3. The Proof Strategy

> **Line:** "Proof. The proof follows the same as the proof in Hwang (1980, Theorem 3.1). The only difference is that we replace the estimate of Hwang (1980, Equation (3.2)) with our Lemma B.1."

* **The Concept:** Rather than rewriting an established mathematical proof, the authors rely on a foundational theorem by Hwang (1980) that handles the complex limits of these types of integrals. The authors simply swap out Hwang's integration step with their own Lemma B.1, which they customized specifically for this manifold setting.

> **Line:** "Note that the $Q$ in Hwang (1980, Theorem 3.1) is assumed as a probability measure, thus $f$ (in his notation) integrates to one. However, the proof technique of Hwang (1980, Theorem 3.1) remains valid even if $f$ is not a probability density, so applying to our case. $\Pi$"

* **The Concept:** The authors address a minor technical discrepancy. Hwang originally assumed his starting function was already fully normalized. The authors are working with an unnormalized density that requires a denominator. However, they assert that the underlying calculus and limit techniques of Hwang's proof are mathematically valid regardless of whether the function is pre-normalized or normalized at the end.

---

# Line-by-Line Breakdown of Theorem 3.1

**Goal:** This theorem captures the central mathematical insight of the entire paper. By expanding the score function (the log-density) as the noise level ($\sigma$) shrinks to zero, it formally proves the "separation of scales"—demonstrating precisely why it is mathematically easier for a model to learn the geometric shape of the manifold than to learn the actual data distribution upon it.

---

## 1. The Setup and Assumptions

> **Line:** "Theorem 3.1 (Informal Theorem B.2). Assume Assumptions 2.1 and 2.2 holds. For any $x\in T_{\mathcal{M}}(\epsilon)$,"

* **The Math:** The theorem grounds itself in the foundational rules established earlier in the paper: the data lives on a smooth, compact manifold (Assumption 2.1) and possesses a smooth, strictly positive probability density (Assumption 2.2). The equation is specifically evaluated for points $x$ that are situated inside the tubular neighborhood ($T_{\mathcal{M}}(\epsilon)$), meaning they are relatively close to the manifold surface.

---

## 2. The Core Equation

> **Line:** "$\log p_{\sigma}(x)=-\frac{1}{\sigma^{2}}d_{\mathcal{M}}(x)+\log p_{data}(\Phi^{-1}(P_{\mathcal{M}}(x)))-\frac{d-n}{2}\log(2\pi\sigma^{2})+H(x)+o(1).$"

This equation deconstructs the log-probability density of the noisy data ($\log p_{\sigma}(x)$) into five distinct components. Let's isolate each term:

* **Term 1: The Geometric Pull ($-\frac{1}{\sigma^{2}}d_{\mathcal{M}}(x)$)**
  Here, $d_{\mathcal{M}}(x)$ is the squared distance from your given point $x$ to the closest point on the manifold. Because it is divided by $\sigma^{2}$, as $\sigma \rightarrow 0$, this term explodes toward $-\infty$ for any point not perfectly resting on the manifold. This acts as a massive mathematical gravity well, forcing the probability mass to concentrate strictly on the geometry of the manifold. Because it scales by $\sigma^{-2}$, it is the "loudest" and most dominant signal for the model to learn.

* **Term 2: The Data Distribution ($\log p_{data}(\Phi^{-1}(P_{\mathcal{M}}(x)))$)**
  This represents the actual density of the original data, evaluated at the specific point on the manifold closest to $x$ (the projection $P_{\mathcal{M}}(x)$ mapped back to local coordinates). Crucially, there is no $\sigma$ in this term. It is $\Theta(1)$, meaning its strength remains constant regardless of the noise level.

* **Term 3: The Perpendicular Volume ($-\frac{d-n}{2}\log(2\pi\sigma^{2})$)**
  This acts as the normalization factor for the Gaussian noise. The quantity $d-n$ represents the number of ambient dimensions that are strictly perpendicular to the manifold. This term accounts for the volume of the noise dispersed out into those off-manifold directions.

* **Term 4: The Curvature ($H(x)$)**
  > **Line:** "where $H(x)$ contains the curvature information of the manifold and $\epsilon$ is some sufficiently small constant; both of them are independent of $\sigma$."
  
  This term captures the intrinsic geometric curvature of the manifold at that location. Like the data density term, it is completely independent of the noise level $\sigma$.

* **Term 5: The Vanishing Error ($o(1)$)**
  > **Line:** "The small $o(1)$ term is uniform for $x\in T_{\mathcal{M}}(\epsilon)$."
  
  This is a mathematically tiny remainder that uniformly shrinks to zero across the entire tubular neighborhood as the noise parameter $\sigma$ approaches zero.

---

## 3. The Grand Implication

This single equation completely reframes the understanding of score-based models like diffusion. 

By comparing the magnitudes of the terms, the geometric distance function ($d_{\mathcal{M}}$) is amplified by an absolutely massive factor: $\sigma^{-2}$. Meanwhile, the actual data distribution ($p_{data}$) is merely multiplied by $1$. 

This establishes a **fundamental rate separation**. Before a neural network can learn *anything* about the specific data distribution $p_{data}$ (which requires extreme precision), it is overwhelmingly forced to first recover the manifold geometry $d_{\mathcal{M}}$. Any minute error in learning the geometry is magnified by that $\sigma^{-2}$ factor, completely drowning out the $\Theta(1)$ distribution details. Consequently, teaching models to learn the uniform geometry requires an easily attainable $o(\sigma^{-2})$ accuracy, while forcing them to learn the exact distribution requires an incredibly difficult $o(1)$ accuracy.
## 📊 Results & Insights
*(To be filled in as the week progresses)*

---

## ⏭️ Next Steps
*(To be filled in as the week progresses)*
