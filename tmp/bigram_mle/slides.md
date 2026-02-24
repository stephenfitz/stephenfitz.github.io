# Bigram MLE Derivation

---

## Outline

- Bigram model and likelihood
- Rewriting with bigram counts
- Log-likelihood
- Constraints
- Lagrangian formulation
- Partial derivatives and stationarity
- Solving for the multiplier
- Final MLE result
- Why it's the global maximum

---

## 1) Bigram Model + Likelihood

Let the observed corpus be a sequence of tokens:

$$(w_1, w_2, \ldots, w_T)$$

A bigram model assumes:

$$P(w_1, \ldots, w_T) = \prod_{t=2}^{T} P(w_t \mid w_{t-1})$$

We want the MLE:

$$\hat{\theta} = \arg\max_\theta \prod_{t=2}^{T} P_\theta(w_t \mid w_{t-1})$$

---

## 2) Rewrite Using Bigram Counts

Define the bigram count for a specific ordered pair $(a, b) \in V \times V$:

$$C(a, b) = \#\{t \in \{2, \ldots, T\} : w_{t-1} = a,\ w_t = b\}$$

Then the likelihood becomes:

$$\prod_{t=2}^{T} P(w_t \mid w_{t-1}) = \prod_{a \in V} \prod_{b \in V} P(b \mid a)^{C(a,b)}$$

Because each conditional probability $P(b \mid a)$ is multiplied once per time that transition occurs.

---

## 3) Take Logs

Maximizing likelihood is equivalent to maximizing log-likelihood:

$$\ell(\theta) = \log \prod_{a,b} P(b \mid a)^{C(a,b)} = \sum_{a,b} C(a, b) \log P(b \mid a)$$

So the optimization problem is:

$$\max_{\{P(b|a)\}} \sum_{a \in V} \sum_{b \in V} C(a, b) \log P(b \mid a)$$

---

## 4) Constraints

For each context token $a$, the conditional distribution over next tokens must sum to 1:

$$\sum_{b \in V} P(b \mid a) = 1 \qquad \forall a \in V$$

(And implicitly $P(b \mid a) \geq 0$.)

---

## 5) Build the Lagrangian

Introduce a Lagrange multiplier $\lambda_a$ for each constraint.

Define the Lagrangian:

$$\mathcal{L} = \sum_{a \in V} \sum_{b \in V} C(a, b) \log P(b \mid a) + \sum_{a \in V} \lambda_a \left(1 - \sum_{b \in V} P(b \mid a)\right)$$

This is the key step: one constraint per $a \to$ one $\lambda_a$.

---

## 6) Take Partial Derivatives

For each ordered pair $(a, b)$, differentiate with respect to the parameter $P(b \mid a)$.

Only two terms depend on $P(b \mid a)$:

- $C(a, b) \log P(b \mid a)$
- $-\lambda_a P(b \mid a)$

So:

$$\frac{\partial \mathcal{L}}{\partial P(b \mid a)} = \frac{C(a, b)}{P(b \mid a)} - \lambda_a$$

Set equal to 0 for stationarity:

$$\frac{C(a, b)}{P(b \mid a)} - \lambda_a = 0$$

Rearrange:

$$\frac{C(a, b)}{P(b \mid a)} = \lambda_a \quad \Rightarrow \quad P(b \mid a) = \frac{C(a, b)}{\lambda_a}$$

So for each fixed $a$, the distribution is proportional to counts.

---

## 7) Apply the Constraint

Now apply the constraint:

$$\sum_{b \in V} P(b \mid a) = 1$$

Substitute $P(b \mid a) = \frac{C(a,b)}{\lambda_a}$:

$$\sum_{b \in V} \frac{C(a, b)}{\lambda_a} = 1$$

Factor out $1/\lambda_a$:

$$\frac{1}{\lambda_a} \sum_{b \in V} C(a, b) = 1$$

Thus:

$$\lambda_a = \sum_{b \in V} C(a, b)$$

Define the unigram-as-context count:

$$C(a) := \sum_{b \in V} C(a, b)$$

So: $\lambda_a = C(a)$.

---

## 8) Final MLE

Plug back in:

$$\hat{P}(b \mid a) = \frac{C(a, b)}{C(a)}$$

This is exactly **"relative frequency."**

---

## 9) Why This Is the Global Maximum

For each fixed $a$, the function

$$\sum_b C(a, b) \log P(b \mid a)$$

is **concave** in the vector $(P(b \mid a))_{b \in V}$ over the simplex.

So the stationary point found by Lagrange multipliers is the **global maximum**.
