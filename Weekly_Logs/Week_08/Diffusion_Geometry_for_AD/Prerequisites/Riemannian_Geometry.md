# 1. Foundations of Manifolds and Phase Space

**Smooth Manifold**
A smooth manifold $M$ is a topological space that is locally homeomorphic to Euclidean space, permitting the execution of differential calculus globally. At each point $p \in M$, there exists a tangent space $T_pM$ comprising all possible directional velocities through $p$.

**Phase Space and the Manifold Hypothesis in Time Series**
In dynamical systems, a univariate time series is typically embedded into a higher-dimensional phase space using a sliding window (Takens' Embedding Theorem). Under normal operating conditions, the deterministic dynamics of the system constrain these embedded vectors to a low-dimensional topological manifold.

**Tensor Field**
A tensor field assigns a tensor to the tangent space at every point $p$ on the manifold, varying continuously and differentiably as $p$ translates across the geometry.

**(0, k)-Tensor**
A $(0, k)$-tensor is a purely covariant multilinear map. At a specific point $p$, it accepts exactly $k$ tangent vectors as inputs and outputs a single real number:
$$ \omega: \underbrace{T_pM \times T_pM \times \dots \times T_pM}_{k \text{ times}} \to \mathbb{R} $$
The mapping is multilinear; scaling any single input vector by a scalar strictly scales the final real output by that exact scalar, and the operation distributes linearly over vector addition.

**Total Antisymmetry**
A multilinear map is totally antisymmetric if swapping the positions of any two input vectors precisely inverts the mathematical sign of the scalar output:
$$ \omega(v_1, \dots, v_i, \dots, v_j, \dots, v_k) = -\omega(v_1, \dots, v_j, \dots, v_i, \dots, v_k) $$
A strict algebraic consequence of this property is that if any two input tangent vectors are identical, linearly dependent, or collinear, the output evaluates to exactly zero.


**Wedge Product ($\wedge$)**
The wedge product is the fundamental algebraic operation used to combine lower-degree totally antisymmetric tensors into higher-degree ones. If you have a $k$-form $\alpha$ and an $l$-form $\beta$, their wedge product $\alpha \wedge \beta$ constructs a new $(k+l)$-form.
*   **Anticommutativity:** The wedge product strictly enforces the property of total antisymmetry: $\alpha \wedge \beta = (-1)^{kl} \beta \wedge \alpha$. A profound algebraic consequence of this rule is that the wedge product of any 1-form with itself—or with any scalar multiple of itself—is strictly zero ($\omega \wedge \omega = 0$).
*   **Geometric Intuition:** If a 1-form acts as a measurement tool for a 1-dimensional rate of change (an oriented length), the wedge product of two 1-forms ($\omega_1 \wedge \omega_2$) computes their joint 2-dimensional rate of change (an oriented area). If the two 1-forms are linearly dependent (e.g., they measure the exact same underlying physical phenomenon), they cannot span a 2D plane. The area collapses, and their wedge product evaluates to exactly zero.

---

# 2. Tangent and Cotangent Spaces

**Tangent Bundle ($TM$)**
The tangent space $T_pM$ is the real vector space containing all possible tangent vectors to smooth curves passing through a speciﬁc point $p$ on the manifold. The tangent bundle $TM$ is the disjoint union of all tangent spaces across the entire manifold $M$. Smooth sections of the tangent bundle are vector ﬁelds.
*   **Time Series Relevance:** $T_pM$ represents the set of all mathematically valid, "healthy" state transitions from the current timestamp $x_t$. If the system evolves along a vector residing strictly within the tangent space, it adheres to normal physical dynamics. Any component of a movement $\Delta X$ that cannot be represented in $T_pM$ lies in the extrinsic normal space, representing a structural break or anomaly.

**Cotangent Bundle ($T^*M$)**
The cotangent space $T_p^*M$ is the formal dual vector space to the tangent space. It comprises linear functionals (covectors or 1-forms) that map tangent vectors at $p$ to the real numbers ($\omega: T_pM \to \mathbb{R}$). The cotangent bundle $T^*M$ is the disjoint union of all cotangent spaces across $M$. Smooth sections of the cotangent bundle are diﬀerential 1-forms.

*   **Time Series Relevance:** Cotangent vectors represent measurements or physical observables of the dynamical system. If a scalar field $f(x)$ represents a specific sensor reading (e.g., temperature or pressure) over the phase space, its differential $df$ is a 1-form. Evaluating this 1-form on a physical state transition vector $v \in T_pM$ computes the exact instantaneous rate of change of that sensor. In architectures like Neural CDC, the network learns a basis of these 1-forms to map the expected coupled constraints of the system. An anomaly can be detected not by a predefined score, but when the geometric relationships (wedge products) between these expected rates of change suddenly break down.

> **Example: The Tangent vs. Cotangent Geometric Intuition**
> 
> Consider a 2-dimensional manifold $M$ representing the surface of a hill, embedded in a 3-dimensional ambient space $\mathbb{R}^3$. Let $p$ be a specific location on this hill.
> 
> **The Tangent Vector ($v \in T_pM$):** A tangent vector represents physical velocity. Visually, $T_pM$ is the 2D flat plane touching the hill at $p$. A vector $v$ represents the action of "walking North-East at 2 metres per second".
> 
> **The Normal Vector (Extrinsic):** The normal space consists of vectors in $\mathbb{R}^3$ that are strictly orthogonal to the 2D tangent plane at $p$ (pointing straight up into the sky). It describes how the manifold sits within the ambient space, not the intrinsic geometry.
> 
> **The Cotangent Vector or 1-form ($\omega \in T_p^*M$):** Suppose there is a scalar field defined on the hill, such as temperature $T$. The gradient of this temperature, $dT$, is a cotangent vector at $p$. It is visualised as a series of linearised contour lines (isotherms) spaced at specific intervals. 
> 
> **The Dual Pairing:** When the cotangent vector $dT$ evaluates the tangent vector $v$, it computes how many contour lines the vector $v$ pierces per unit of time, yielding a real number: $dT(v) = \text{rate of temperature change}$. If the velocity vector moves exactly parallel to the contour lines, it pierces zero lines, and $dT(v) = 0$.

---

# 3. Mechanics of 1-Forms and System Observables

**1-Forms as Hyperplanes**
A cotangent vector (1-form) $\omega \in T_p^*M$ is geometrically visualised as a family of parallel, equally spaced hyperplanes spanning the local tangent space. The mathematical evaluation $\omega(v)$ is the linear operation of counting the exact number of hyperplanes the vector $v$ pierces. A vector pierces a greater number of planes if it is longer, if it points perpendicularly to the planes, or if the planes are densely packed.

**The Null Space of a 1-Form**
The null space (or kernel) of the 1-form consists of all tangent vectors $v$ such that $\omega(v) = 0$. If $\omega(v) = 0$, the vector $v$ is perfectly parallel to the hyperplanes, piercing zero of them. In an $n$-dimensional tangent space, the null space of a non-zero 1-form constitutes an $(n-1)$-dimensional linear subspace. Any movement confined within this null space results in zero observed change relative to the 1-form.

**1-Forms as Physical Observables**
In a time-series phase space, a point $x$ represents the complete hidden state of the dynamical system. A scalar field on this manifold is simply a measurable property or sensor reading (e.g., $T(x)$ = Temperature, or $R(x)$ = RPM). 

The differential of this sensor, $dT$, is a 1-form. Geometrically, $dT$ is visualised as the linearised contour lines (isotherms) of constant temperature mapped across the tangent space. 
*   **The Null Space (Isoclines):** The null space of $dT$ defines all healthy state transitions (tangent vectors $v$) where the temperature does not change ($dT(v) = 0$). The system is shifting its internal state, but moving perfectly parallel to the temperature contours.
*   **Piercing the Hyperplanes:** If the system accelerates, the state transition vector $v$ points against the contours and pierces the hyperplanes. The algebraic evaluation $dT(v)$ yields the exact instantaneous rate of temperature change.



--- 


# 4. Riemannian Geometry

**Riemannian Metric Tensor ($g$)**
A Riemannian metric tensor $g$ is a smooth assignment of a symmetric, positive-definite bilinear form $g_p: T_pM \times T_pM \to \mathbb{R}$ to the tangent space at each point $p$. It establishes a local inner product structure, enabling the measurement of lengths, angles, and volumes. The metric tensor provides a canonical isomorphism (musical isomorphisms $\sharp$ and $\flat$) between the tangent and cotangent bundles.
*   **Time Series Relevance:** Standard Euclidean distance triggers false positives in highly volatile, healthy regimes. The metric tensor dynamically scales distance measurements based on the local geometry of the attractor. Neural CDC explicitly regresses a low-rank metric tensor $M(Y)^T M(Y)$, which dampens dimensions corresponding to normal volatility and amplifies dimensions corresponding to structural drift.

**Covariant Derivative ($\nabla$)**
A covariant derivative specifies the intrinsic rate of change of a vector field along the direction of another vector field. It projects the change of a vector field onto the local tangent space, discarding apparent changes caused by the curvature of the coordinate system itself.
*   **Time Series Relevance:** In a non-stationary time series, the "healthy" tangent space constantly rotates. The covariant derivative computes the intrinsic evolution of an anomaly vector by subtracting the natural rotation of the healthy dynamics.

**Levi-Civita Connection**
The Levi-Civita connection is the unique affine connection (parallel transport protocol) on a Riemannian manifold satisfying two strict criteria:
1.  **Metric Compatibility ($\nabla g = 0$):** The inner product of two vectors remains strictly constant during parallel transport. Lengths and angles are preserved.
2.  **Torsion-Free ($\nabla_X Y - \nabla_Y X = [X, Y]$):** The connection is symmetric. The manifold does not inherently twist the coordinate frame.


**Laplace-Beltrami Operator ($\Delta_g$)**
The Laplace-Beltrami operator generalises the Euclidean Laplacian to Riemannian manifolds. It measures the divergence of the gradient of a twice-differentiable scalar function: $\Delta_g f = \text{div}(\text{grad} f)$. In local coordinates:
$$ \Delta_g f = \frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} g^{ij} \partial_j f \right) $$
*   **Time Series Relevance:** $\Delta_g$ acts as the infinitesimal generator of Brownian motion on the manifold. Its eigenfunctions form an optimal, non-linear coordinate basis (Diffusion Maps) for the phase space. These coordinates are invariant to isometric deformations of the time series, providing a feature space where Euclidean distance equates directly to the probability of Markov transitions.

# 5. Exterior Calculus and Topology

**Differential Forms**
A differential $k$-form ($\Omega^k(M)$) is a totally antisymmetric $(0, k)$-tensor field. It geometrically computes the oriented (signed) $k$-dimensional volume of the parallelepiped spanned by $k$ input vectors within the tangent space. 0-forms are scalar functions; 1-forms compute oriented length; 2-forms compute oriented area. If the input vectors fail to span a full $k$-dimensional subspace, the $k$-form yields a strict zero.
*   **Time Series Relevance:** Differential forms provide a coordinate-free framework to measure fluxes and gradients across phase space.

**Wedge Product ($\wedge$)**
The wedge product combines lower-degree forms into higher-degree forms. For a $k$-form $\alpha$ and an $l$-form $\beta$, $\alpha \wedge \beta$ is a $(k+l)$-form. It is strictly anticommutative: $\alpha \wedge \beta = (-1)^{kl} \beta \wedge \alpha$. The wedge product of any 1-form with itself is strictly zero.
*   **Time Series Relevance:** It constructs multi-dimensional geometric feature descriptors. Taking the wedge product of independent 1-forms generates higher-order representations that capture joint correlations and multidimensional interactions within the dynamic system.

**Exterior Derivative ($d$)**
A linear operator $d: \Omega^k(M) \to \Omega^{k+1}(M)$ that generalises the gradient, curl, and divergence. It satisfies the graded Leibniz rule and the nilpotency condition $d^2 = 0$ (the boundary of a boundary is zero).
*   **Time Series Relevance:** The operator computes the structural gradient of a feature across the phase space, quantifying how a specific dynamic measurement evolves across the manifold.

**Hodge Star Operator ($\star$)**
A linear isomorphism $\star: \Omega^k(M) \to \Omega^{n-k}(M)$ that maps $k$-forms to their orthogonal dual $(n-k)$-forms. It relies strictly on the Riemannian metric tensor $g$ to define orthogonal complements.
*   **Time Series Relevance:** The Hodge star allows the explicit mapping of tangent-space dynamics (healthy behaviour) into the normal space (structural error).

**Codifferential ($\delta$)**
The formal adjoint of the exterior derivative with respect to the $L^2$ inner product on differential forms. It maps $\delta: \Omega^k(M) \to \Omega^{k-1}(M)$ and is constructed using the Hodge star operator: $\delta = (-1)^{nk + n + 1} \star d \star$. It generalises the divergence.
*   **Time Series Relevance:** It measures the geometric divergence or convergence of a feature field on the manifold, identifying local sources or sinks in the dynamical system's phase space.

**De Rham Cohomology ($H^k_{dR}(M)$)**
A $k$-form $\omega$ is *closed* if $d\omega = 0$ and *exact* if it is the derivative of a $(k-1)$-form ($\omega = d\eta$). Because $d^2 = 0$, all exact forms are closed. The $k$-th de Rham cohomology group is the quotient space of closed forms modulo exact forms: $H^k_{dR}(M) = \frac{\ker(d_k)}{\text{im}(d_{k-1})}$.
*   **Time Series Relevance:** The dimension of the $k$-th cohomology group (the Betti number) counts the number of $k$-dimensional macroscopic holes in the manifold. In time series, $H^1_{dR}(M)$ detects fundamental periodic cycles. Diffusion geometry computes this cohomology via the Hodge Laplacian ($\Delta_k = d_{k-1}\delta_k + \delta_{k+1}d_k$), providing a topological classification metric strictly invariant to noise and deformations.