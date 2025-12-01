# A log-linear model for short-video dwell-time classification

> Some of refinement suggestions are given by ChatGPT. It also helps me to compose the post.

## Problem statement

On a short-video platform, at time $t$ we pick the **next video** to push to a user. Given the recent interaction history (for example, past 5 video interactions) and current context (time, region, etc.), we want a model that assigns a probability to **how long the user will stay** on a candidate video. Formally, this is a conditional distribution

$$
p(y_t \mid x_{\text{past}}, x_t),
$$

where the label $y_t \in \{\text{0–20s},\ \text{20–40s},\ \text{>40s}\}$, the candidate video descriptor is $x_t$, and the history/context is $x_{\text{past}}$.

---

## Objects and notation

Let's formally (may not that easy, please help to refine) define the objects and notation for this problem.

* **Label**: $\mathcal{Y}=\{S,M,L\}$ corresponding to $S=\text{0–20s}$, $M=\text{20–40s}$, $L=\text{>40s}$.
* **Candidate video descriptor** $x_t$: video-side attributes at time $t$ (defined below).
* **History/context** $x_{\text{past}}$: user-side state, environment (time/region), and the last five interactions
  $(x_{t-1},y_{t-1}),\dots,(x_{t-5},y_{t-5})$. We should notice that all $y_t$ share the same label space $\mathcal{Y}$ and all $x_t$ share the same candidate video descriptor space $\mathcal{X}$.
* **Functions and symbols**:

  * $\mathbf{1}\{\cdot\}$: indicators; 
  * $\text{onehot}(c;\mathcal{S})$: one-hot vector over set $\mathcal{S}$; 
  * $\oplus$: vector concatenation; 
  * $\odot$: elementwise product.
  * For simplicity, we design categorical features for any video $x$:
    * $\text{type}(x)\in\{\text{Parenting},\text{Animals},\text{Politics}\}$,
    * $\text{len\_bucket}(x)\in\{[0,20),[20,40),[40,+)\}$,
    * $\text{lang}(x)\in \mathcal{L}$, language of the video used.
    * $\text{author}(x)$.
  * Binary / numeric video attributes: $\text{subtitle}(x)\in\{0,1\}$,
    * $\text{trending}(x)\in\{0,1\}$, 
    * global mean dwell $\overline{\text{dwell}}(x)\ge 0$.
  * User attributes at $t$: 
    * follow flag $\text{follow}_u(\text{author}(x_t))$, 
    * per-category interests $\text{interest}_u(c)$, 
    * today’s mean dwell $m_u(t)$.
  * Environment at $t$: hour $\in\{0,\dots,23\}$ encoded as just integers or $\big[\sin(2\pi\,\text{hour}/24),\cos(2\pi\,\text{hour}/24)\big]$, region $\in\mathcal{R}$.
  * Decay weights for sequence features: $\alpha_k=\exp(-\lambda k),\ \lambda>0$. It is for the sequence features. Sequencial history $x_{\text{past}}$ is weighted by $\alpha_k$.

---

## Feature design $f(x_t,x_{\text{past}},y)$

> In the online materials, $f(x,y)$ are always scaler functions, while we can also use vector functions. Then the $\theta$ (also as vectors) will map $f$ to the scalar by inner product. 

I design several  **class-dependent** features which is easy to interpret. Each $f(x_t,x_{\text{past}},y)$ should rely on video features, user-video interaction features, environment features and history features. Let

$$
f(x_t,x_{\text{past}},y)\;=\;
\Big(\ \underbrace{\phi_{\text{video}}(x_t)}_{\text{video}}\ \oplus\
\underbrace{\phi_{\text{user}}(x_{\text{past}},x_t)}_{\text{user}\times\text{video}}\ \oplus\
\underbrace{\phi_{\text{env}}(x_{\text{past}},x_t)}_{\text{environment}}\ \oplus\
\underbrace{\phi_{\text{seq}}(x_{\text{past}},x_t)}_{\text{sequence}}\ \oplus\
1\ \Big)\ \odot\ \text{onehot}(y;\mathcal{Y}).
$$

Recall that historical interactions includes previous $y$, ensuring the expression $f$ depends on $y$.

### A) Video-side block $\phi_{\text{video}}(x_t)$

$$
\begin{aligned}
\phi_{\text{video}}(x_t)=\;&
\text{onehot}(\text{type}(x_t))\ \oplus\
\text{onehot}(\text{len\_bucket}(x_t))\ \oplus\
\mathbf{1}\{\text{subtitle}(x_t)=1\}\ \oplus\
\mathbf{1}\{\text{trending}(x_t)=1\}\\
&\oplus\ \log\!\big(1+\overline{\text{dwell}}(x_t)\big)\ \oplus\
\log\!\big(1+\text{pop}(\text{author}(x_t))\big)\ \oplus\
\text{onehot}(\text{lang}(x_t)).
\end{aligned}
$$

The use of **log** transform (for example, `log(1 + dwell)`)  is to compress the possible heavy-tailed watch-time distribution, stabilize feature scale for learning, and encode diminishing returns so extreme values don’t dominate. (Thanks to ChatGPT's suggestion)

### B) User × Video block $\phi_{\text{user}}(x_{\text{past}},x_t)$

Let $\mathcal{C}=\{\text{Parenting},\text{Animals},\text{Politics}\}$.

$$
\phi_{\text{user}}=\;
\mathbf{1}\{\text{follow}_u(\text{author}(x_t))=1\}
\ \oplus\ 
\Big[\text{interest}_u(c)\Big]_{c\in\mathcal{C}}\ \odot\ \text{onehot}(\text{type}(x_t))
\ \oplus\
m_u(t)\cdot \text{onehot}(\text{type}(x_t)).
$$

### C) Environment block $\phi_{\text{env}}(x_{\text{past}},x_t)$

With region $r\in\mathcal{R}$ and a (predefined) set $\mathcal{P}$ of common region–language pairs (thanks to ChatGPT's suggestion of common region–language pairs)

$$
\phi_{\text{env}}=\;
\big[\sin(2\pi\,\text{hour}/24),\ \cos(2\pi\,\text{hour}/24)\big]
\ \oplus\ \text{onehot}(r)
\ \oplus\ \mathbf{1}\{(r,\text{lang}(x_t))\in \mathcal{P}\}.
$$

### D) Short-term sequence block $\phi_{\text{seq}}(x_{\text{past}},x_t)$

Use the last five interactions $(x_{t-k},y_{t-k})$ with exponential decay:

$$
\phi_{\text{seq}}=\sum_{k=1}^{5}\alpha_k\,
\Big[
\mathbf{1}\{\text{type}(x_t)=\text{type}(x_{t-k})\}\ \oplus\
\mathbf{1}\{\text{author}(x_t)=\text{author}(x_{t-k})\}\ \oplus\
\mathbf{1}\{\text{lang}(x_t)=\text{lang}(x_{t-k})\}
\Big]
$$

and optionally **performance-aware** refinements that remain causal (use only $y_{t-k}$, never $y_t$):

$$
\sum_{k=1}^{5}\alpha_k\,
\Big[
\mathbf{1}\{y_{t-k}=L\}\cdot \mathbf{1}\{\text{type}(x_t)=\text{type}(x_{t-k})\}
\ \oplus\
\mathbf{1}\{y_{t-k}=S\}\cdot \mathbf{1}\{\text{type}(x_t)=\text{type}(x_{t-k})\}
\Big].
$$


> Moreover, one may use more complex features of previous videos just like (A) does for the candidate video. It is quite a long term sequence features that I may leave here for simplicity.

---

## Log-linear model

In the online materials, $f(x,y)$ are always scaler functions, while we can also use vector functions. Then the $\theta$ (also as vectors) will map $f$ to the scalar by inner product. Let $\theta$ be the parameter vector (same dimension as $f$). Define the linear score

$$
s_\theta(x_t,x_{\text{past}},y)\;=\;\theta^\top f(x_t,x_{\text{past}},y).
$$

The conditional distribution over dwell-time classes is

$$
\boxed{
\quad
p_\theta(y_t \mid x_{\text{past}}, x_t)
\;=\;
\frac{\exp\!\big(\theta^\top f(x_t,x_{\text{past}},y_t)\big)}
{\displaystyle\sum_{y'\in\mathcal{Y}}\exp\!\big(\theta^\top f(x_t,x_{\text{past}},y')\big)}
\quad}
$$

which is a standard **conditional log-linear** model. The construction guarantees that features **depend on the class $y$** via any class-specific indicators included above.
