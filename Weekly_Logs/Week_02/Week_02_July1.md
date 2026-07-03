# Weekly Log: 01 July 2026

## 🎯 Focus of the Week
This week focuses on solidifying the theoretical and mathematical foundations of our manifold-aware generative framework. The primary objectives are to rigorously justify our use of the Carré du Champ (CDC) operator, document our anomaly detection experiments, explore the Jordan-Kinderlehrer-Otto (JKO) scheme as a theoretical bridge, and design an annealed orthogonal attraction field to strictly enforce manifold adherence.

---

## 📝 To-Do List
- [ ] **Task 1:** Formally justify our application of the discrete vector CDC operator against continuous differential geometry standards.
- [ ] **Task 2:** Document the experimental implementation of the CDC operator, specifically how we extract the $\nu_{\text{ortho}}$ metric to outperform the PatchCore baseline.
- [ ] **Task 3:** Investigate the JKO (Jordan-Kinderlehrer-Otto) scheme to see if our geometrically restricted drift can be framed as a Wasserstein gradient flow.
- [ ] **Task 4:** Design and implement an annealed orthogonal attraction field to pull wandering generative points back onto the data manifold.

---


## 🔬 Progress & Experiments

### Background: $\Gamma$ -> Carré du Champ (CDC)
For a given infinitesimal generator $L$ (which describes how a process evolves) and two scalar functions $f$ and $h$, the CDC operator ($\Gamma$) is defined as:$$\Gamma(f, h) = \frac{1}{2}(L(fh) - fL(h) - gL(f))$$If we evaluate the operator on a single function ($f=h$), it simplifies to: $$\Gamma(f, f) = \frac{1}{2}(L(f^2) - 2fL(f))$$ $$\Gamma(f,f) = |\nabla f|^2$$

On a continuous Riemannian manifold, $$\Gamma(f,h) = G(df,dh)$$ where $G$ is the Riemannian Metric Tensor, the geometry, and $\Gamma$ is the algebra. If you have a discrete point cloud where computing a continuous differential like $df$ is impossible, you can use the transition matrix to compute $\Gamma(f, h)$  to get the exact same geometric inner product as if you had calculated $G(df, dh)$ on a continuous surface. More on this later. 

### Background: $\Gamma_2$ -> Iterated Carré du Champ
The Iterated Carré du Champ, denoted as $\Gamma_2(f, f)$, is effectively the "geometric curvature" counterpart to the standard Carré du Champ operator ($\Gamma$). If you think of the standard CDC operator ($\Gamma$) as capturing the local slope (tangent space) of your data manifold, you can think of the Iterated CDC ($\Gamma_2$) as capturing how that slope changes, i.e., the local curvature and the stability of the diffusion process.

By applying the CDC operator itself: 
$$\Gamma_2(f, f) = \frac{1}{2} \Delta \Gamma(f,f) - \Gamma(f, \Delta f)$$


### Background: Dirichlet Energy
While the CDC operator measures the geometry at a single specific point, the Dirichlet energy represents the overarching, global consequence of that geometry. In classical physics and mathematics, the Dirichlet energy is a measure of how variable a function is across a space. If you imagine a function as a rubber sheet stretched over a complex terrain, the Dirichlet energy measures the total potential energy stored in the tension of that sheet. A highly wrinkled or sharply spiking sheet has high Dirichlet energy; a smooth, relaxed sheet has low Dirichlet energy. 

Mathematically, for a continuous scalar function $f$, it is defined as the integral of the squared gradient over the entire space:$$\mathcal{E}(f) = \frac{1}{2} \int |\nabla f|^2 dx$$

Because the CDC operator $\Gamma(f, f)$ is the exact mathematical abstraction of the squared gradient ($|\nabla f|^2$), the Dirichlet energy is simply the global integral (or expected value) of the CDC operator across your invariant measure $\mu$: $$\mathcal{E}(f) = \int \Gamma(f, f) d\mu$$ This means that while $\Gamma(x_i)$ tells you how steep the slope is at your specific coordinate $x_i$, the Dirichlet energy $\mathcal{E}$ tells you the total cost of the entire function's variation across the whole manifold.

### Background: The Bochner-Weitzenböck Formula (1923, 1946)
In differential geometry, it is the fundamental equation that connects how a function diffuses (the Laplacian) to the actual physical shape of the space it diffuses across (the curvature).

For a smooth scalar function $f$ defined on a Riemannian manifold, the formula is:$$\frac{1}{2} \Delta (|\nabla f|^2) = \langle \nabla f, \nabla(\Delta f) \rangle + ||\nabla^2(f)||^2 + \text{Ric}(\nabla f, \nabla f)$$

* $\Delta$ (Laplace-Beltrami operator): Measures the diffusion or "spread" of a value. 
* $\nabla f$ (Gradient): The vector field pointing in the direction of the steepest slope of $f$.
* $\nabla^2(f)$ (Hessian): The matrix of second derivatives, measuring the local "bending" of the function itself.
* $\text{Ric}$ (Ricci Curvature Tensor): It measures how much the volume of the manifold deviates from standard flat Euclidean space.



### Background: Substituting $\Gamma_2$ into the Bochner-Weitzenböck Formula
The Bakry-Émery criterion (1984), recognised that the gradient terms in the Bochner-Weitzenbock formula correspond exactly to the Carré du Champ operator. By definition, on a continuous manifold:$$\Gamma(f, f) = |\nabla f|^2$$ $$\Gamma(f, g) = \langle \nabla f, \nabla g \rangle$$

If we substitute $\Delta f$ in for $g$ in the second equation, we get the inner product from the Bochner formula:$$\Gamma(f, \Delta f) = \langle \nabla f, \nabla(\Delta f) \rangle$$

* $\nabla f$ is the vector field showing which way is "uphill"
* $\nabla(\Delta f)$ is the vector field showing which way the diffusion rate is increasing the fastest

If you are standing on a heat map, this term measures whether the direction of the steepest temperature increase ($\nabla f$) points in the same direction as the area where the heat is dissipating the fastest ($\nabla(\Delta f)$). If they point in the exact same direction, this term is large and positive. If they are perpendicular, it is zero.

We can now rewrite the classical Bochner formula entirely in terms of the $\Gamma$ operator. Substituting our new definitions into the original equation yields:$$\frac{1}{2} \Delta \Gamma(f, f) = \Gamma(f, \Delta f) + ||\nabla^2(f)||^2 + \text{Ric}(\nabla f, \nabla f)$$

Next, we rearrange the equation to isolate the geometric terms (Hessian and Ricci curvature) on the right side:$$\frac{1}{2} \Delta \Gamma(f, f) - \Gamma(f, \Delta f) = ||\nabla^2(f)||^2 + \text{Ric}(\nabla f, \nabla f)$$

Let's think of $f$ as describing the altitude of a mountain.
* $\nabla f$ is a river flowing down the steepest part of the mountain.
* $\Gamma(f, f) = |\nabla f|^2$ is the speed of that river.

The left side of the equation, $\frac{1}{2} \Delta \Gamma(f, f) - \Gamma(f, \Delta f)$, measures what happens to the energy of the river over time.
* $\frac{1}{2} \Delta \Gamma(f, f)$ (Energy Diffusion): This asks, "How is the kinetic energy of the river spreading out?" If the water hits a flat, the energy dissipates evenly. 
* $\Gamma(f, \Delta f)$ (Flow vs. Dissipation): This measures how the direction of the river ($\nabla f$) aligns with the areas where the water is pooling or spreading the fastest ($\nabla(\Delta f)$).

If you subtract the second term from the first, you get the "anomalous" change in energy. It tells you exactly how much the river is accelerating, decelerating, or churning in a way that standard, flat-space physics cannot explain.

f the left side tells you that the river's energy is behaving strangely, the right side explains why. There are exactly two physical reasons a flow of water (or heat, or probability) changes its energy:

**1. The Internal Stress of the Flow:** $||\nabla^2 f||^2$
The term $\nabla^2 f$ is the Hessian matrix, which measures the bending of the function $f$ itself.
* **Physical Intuition:** Imagine pouring water down a perfectly flat ramp, but the water itself has been forced into a chaotic, choppy wave pattern. Even though the ramp is flat, the internal collisions, peaks, and valleys of the water will cause it to churn and dissipate energy. This term measures the friction caused by the shape of the function, independent of the space it lives in.

**2. The External Stress of the Universe:** $\text{Ric}(\nabla f, \nabla f)$
* **Physical Intuition (Positive Curvature):** Imagine your river is flowing down from the North Pole of a perfectly smooth sphere (like Earth). As the water flows south, the physical curvature of the planet forces the longitudinal paths to squeeze together. The space itself is physically compressing the water, increasing its density and energy. This is positive Ricci curvature.
* **Physical Intuition (Negative Curvature):** Now imagine the river flowing over a saddle (like a Pringles crisp). The geometry of the space physically pulls the paths apart, causing the water to spread out and lose energy faster than expected. This is negative Ricci curvature.

The left-hand side is the Iterated Carré du Champ, denoted as $\Gamma_2$: $$\Gamma_2(f, f) := \frac{1}{2} \Delta \Gamma(f, f) - \Gamma(f, \Delta f)$$

The final substitution yields the Bochner-Lichnerowicz-Weitzenböck identity:$$\Gamma_2(f, f) = ||\nabla^2(f)||^2 + \text{Ric}(\nabla f, \nabla f)$$

This final equation shows that you do not need physical measurements of a surface to calculate Ricci curvature or local geometry. By evaluating how functions (or vector fields) diffuse across a Markov transition matrix using the $\Gamma$ and $\Gamma_2$ operators, we are explicitly calculating the intrinsic curvature and Dirichlet energy of the data manifold.

### Background: The Weitzenböck Identity for Vector Fields
While the standard Bochner formula looks at scalar functions, the Weitzenböck identity analyses how a vector field $V$ diffuses across a curved manifold. It is written as:$$\frac{1}{2} \Delta (|V|^2) = \langle V, \Delta_{\text{Hodge}} V \rangle - |\nabla V|^2 - \text{Ric}(V, V)$$ 
* $\Delta_{\text{Hodge}} V$: The topological diffusion of the vector field.
* $\text{Ric}(V, V)$: The Ricci curvature (the physical shape of the manifold bending the vectors).
* $|\nabla V|^2$: The norm of the Jacobian or Frobenius norm (The Dirichlet Energy Density).

The Frobenius norm is the measurement of internal friction or shearing within a vector field. It measues how much the vectors stretch, warp or tear against the geometry of the space. Therefore: $$\Gamma(V, V) = |\nabla V|^2$$ 

Let's go back to the river. The data manifold is the physical, curved bedrock of a river. The vector field ($V$) is the water flowing over it.
* At every coordinate, the water has a velocity vector (a speed and a direction).
* The Jacobian  ($\nabla V$) compares the vector of one drop of water to the vectors of the drops immediately next to it.

**Low Dirichlet Energy ($|\nabla V|^2 \approx 0$):**
If the water is flowing smoothly down the river in perfect unison—a state physicists call laminar flow—the vectors of neighbouring drops are nearly identical. They do not rub against each other. There is no internal friction. The flow perfectly respects the shape of the bedrock.

**High Dirichlet Energy ($|\nabla V|^2 \gg 0$):**
Now imagine a whirlpool, or two currents crashing into each other. The drops of water right next to each other are suddenly trying to move in violently different directions or at vastly different speeds.
This creates massive shearing forces. The water molecules grind against each other, creating heat and friction.

$|\nabla V|^2$ is the exact mathematical measurement of that local friction. It measures how much the vector field is fighting itself and fighting the geometry of the space.

### Background: From Continuous to Discrete thorough Diffusion Geometry
In differential geometry, a 1-form is a mathematical object that measures how a function changes along a given direction. Imagine a mountain. The physical elevation at any point is a scalar function, $f$. A tangent vector is an arrow pointing in a specific direction you could walk, with a specific speed. A 1-form is the set of contour lines on the map. Much of the Riemannian geometry of a manifold can be described in terms of its Laplacian operator $\Delta$, via the carre du champ operator: $$\frac{1}{2}(\Delta(fh) - f\Delta(h) - g\Delta(f)) = G(\nabla f, \nabla h)$$ The main idea behind diffusion geometry is to replace the manifold Laplacian $\Delta$ with a Markov diffusion operator on a more general space so that can become a definition for Riemannian geometry on that space. 

**Markov Triple** ($E, \mu, \Gamma$): We can replace the continuous manifold with any measurable state space $E$ (such as a finite point cloud), the continuous volume with a density measure $\mu$, and the physical Laplacian with a generic infinitesimal generator $L$. When the state space is a finite set of data points, the diffusion process becomes a standard Markov chain (a stochastic transition matrix $P$). The continuous generator $\Delta$ is replaced by the discrete graph Laplacian, $P - I = L$ 


Now using a discrete matrix instead of continuous derivatives, the CDC operator becomes: $$\Gamma(f, f)(x_i) = \frac{1}{2} \sum_j P_{ij} (f(x_j) - f(x_i))^2$$ The gradient inner product is entirely replaced by the weighted sum of pairwise differences between a point and its neighbours. 

### Background: Ambient Coordinates as Scalar Functions to build the Metric Tensor
Data lives as a point cloud in a high-dimensional ambient space $\mathbb{R}^D$. Every data point $x_i$ is a vector of $D$ coordinates: $x_i = (c_1, c_2, \dots, c_D)$. For each of the $D$ coordinate axes $c: \mathbb{R} \rightarrow \mathbb{R}$. For example, $c_1(x_i)$ returns the first coordinate of point $x$. 

Using the discrete Markov formula established earlier, for any two coordinate functions $c_a$ and $c_b$ we can calculate the local Riemannian metric tensor ay point $x_i$: $$\Gamma(c_a, c_b)(x_i) = \frac{1}{2} \sum_j L(i, j) (c_a(x_j) - c_a(x_i))(c_b(x_j) - c_b(x_i))$$ When we combine these differences across all dimensions, the resulting vector $(x_j - x_i)$ is a chord. In the discrete setting, these chords act as the geometric substitute for the continuous continuous gradient ($\nabla$). The CDC operator calculates the local geometry by taking the weighted inner products of these local chords.

**Secants vs. Tangents:** In continuous calculus, a tangent vector is defined as the limit of a secant line (a chord) as the distance between two points approaches zero.  Due to operating on a discrete point cloud, that limit cannot reach zero. Therefore, the chord $(x_j - x_i)$ between a point and its nearest neighbour is the mathematically optimal discrete approximation of the tangent vector. As long as the data manifold is sampled densely enough, the linear span of the $K$-nearest neighbour chords will capture the local tangent space. 

### Calculating the Tangent Space: Bypassing the Metric Tensor
In high-dimensional ambient space ($\mathbb{R}^D$), explicitly constructing and performing an eigendecomposition on the full $D \times D$ Riemannian metric tensor at every integration step of a generative ODE is computationally prohibitive ($\mathcal{O}(D^3)$). To resolve this bottleneck, we employ a first-order finite-difference approximation.

Under the assumtion that the set of local chords pointing to the $K$-nearest neighbours directly spans a reliable approximation of the local tangent space, we can bypass the abstract tensor calculus entirely.  Because these $K$ neighbours are constrained to the underlying data manifold, their spatial chords $(x_j - x_i)$ point along the manifold's surface. Therefore, the linear span of these chords constructs a flat hyperplane that serves as a discrete substitute for the continuous tangent plane.

**1. Density-Normalized Weighting**
The $K$ chords are not treated equally. A standard nearest-neighbour projection is vulnerable to sampling bias; if one region of the manifold is sampled more densely, the tangent space approximation will artificially warp toward it. To eliminate this bias, we weight the chords using the Coifman-Lafon density-normalized Markov transition probabilities ($P_{ij}$). This normalization guarantees that our discrete graph generator ($L = P - I$) converges to the true, unweighted Laplace-Beltrami operator ($\Delta$) of the underlying geometry, independent of empirical sampling density.

**2. The CDC Projection** When our generative model produces a drift vector $V_i$ at point $x_i$, we project it directly onto these normalised, weighted chords. By doing so, we are computing the discrete directional derivative along the edges that define diffusion on the graph. Computing the dot product of the drift vector against these chords is the discrete Carré du Champ operator:$$\Gamma(V, V)(x_i) = \frac{1}{2} \sum_j P_{ij} |V(x_j) - V(x_i)|^2$$

### Mathematical Justification of the CDC Operator
To defend our methodology, we must clarify that our application of the CDC operator is not a heuristic, but a rigorous discrete formulation rooted in Bakry-Émery theory. 

* **The Discrete Generator:** In standard Riemannian geometry, the CDC operator $\Gamma(f,f)$ relies on the continuous Laplace-Beltrami operator. Because our data is a discrete point cloud, we model diffusion as a Markov chain. The exact equivalent of the continuous generator is the discrete transition matrix minus the identity matrix: $L = P - I$. By using Coifman-Lafon density normalisation, we ensure $P$ represents an unbiased geometric random walk.
* **The Algebraic Collapse:** When we apply this discrete generator to the abstract CDC equation $\Gamma(f, f) = \frac{1}{2} ( L(f^2) - 2fL(f) )$, the algebra perfectly collapses into a sum of squared differences: $\Gamma(f, f)(x_i) = \frac{1}{2} \sum_j P_{ij} ( f(x_j) - f(x_i) )^2$.
* **Vector Extension (Dirichlet Energy):** While the CDC is traditionally applied to scalar functions, extending it to coordinate vectors evaluates the Dirichlet energy of a vector field (justified by the Bochner-Weitzenböck formula). By taking the dot product of an incoming drift vector against the local chords ($\Delta = x_j - x_i$) weighted by $P_{ij}$, we are calculating the geometric gradient of the discrete sub-manifold.

---

## 📊 Results & Insights
*What were the results? What broke? What worked?*
- Insight 1: 
- Insight 2: 

*(Tip: You can use markdown to embed images like `![Result Plot](/path/to/plot.png)`)*

---

## ⏭️ Next Steps
*What needs to be done next week based on these results?*

---

## Citations

* Bakry, D., Gentil, I., & Ledoux, M. (2014). Analysis and Geometry of Markov Diffusion Operators (Vol. 348). Springer Science & Business Media.