# Information Theory and Statistics

---

## Outline

- Random variables and probability distributions
- Expectation and variance
- Common distributions
- Sampling and empirical distributions
- Bayes' theorem and Bayesian modeling
- Maximum likelihood estimation
- Entropy
- Cross-entropy and KL divergence
- Perplexity
- Precision, recall, and F-measure
- Statistical significance testing

---

## Part I: Random Variables

---

## Random Variables

A **random variable** maps outcomes to numbers

- **Discrete**: countable values (words, integers)
- **Continuous**: uncountable values (real numbers)

Convention: capital $X$ for the variable, lowercase $x$ for realized values

---

## Probability Mass Function

For discrete $X$:

$$p(x) = P(X = x)$$

Requirements: $p(x) \geq 0$ and $\sum_x p(x) = 1$

---

## Probability Density Function

For continuous $X$:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

Requirements: $f(x) \geq 0$ and $\int_{-\infty}^{\infty} f(x) \, dx = 1$

---

## Joint, Marginal, Conditional

**Joint**: $p(x, y) = P(X = x, Y = y)$

**Marginal**: $p(x) = \sum_y p(x, y)$

**Conditional**: $p(y \mid x) = \frac{p(x, y)}{p(x)}$

**Independence**: $p(x, y) = p(x) \, p(y)$

---

## Part II: Expectation and Variance

---

## Expected Value

$$\E[X] = \sum_x x \, p(x) \quad \text{(discrete)}$$

$$\E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx \quad \text{(continuous)}$$

**Linearity**: $\E[aX + bY] = a\E[X] + b\E[Y]$

---

## Variance

$$\Var(X) = \E[(X - \E[X])^2] = \E[X^2] - (\E[X])^2$$

Standard deviation: $\sigma = \sqrt{\Var(X)}$

- $\Var(aX + b) = a^2 \Var(X)$
- Independent $X, Y$: $\Var(X + Y) = \Var(X) + \Var(Y)$

---

## Part III: Common Distributions

---

## Bernoulli and Binomial

**Bernoulli**: single binary trial

$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

$\E[X] = p$, $\Var(X) = p(1-p)$

**Binomial**: $n$ independent trials

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

---

## Categorical and Multinomial

**Categorical**: one of $K$ values with probabilities $p_1, \ldots, p_K$

**Multinomial**: counts of $K$ categories over $n$ trials

$$\Pr(X_1 = x_1, \ldots, X_K = x_K) = \frac{n!}{x_1! \cdots x_K!} \prod_{k=1}^K p_k^{x_k}$$

---

## Gaussian (Normal)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Written $X \sim \mathcal{N}(\mu, \sigma^2)$

Standard normal: $\mu = 0$, $\sigma^2 = 1$

---

## Part IV: Sampling

---

## The Sampling Process

**Sampling**: drawing observations from a distribution

i.i.d. samples: $x_i \stackrel{\text{iid}}{\sim} p$

**Empirical distribution**:

$$\hat{p}(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[x_i = x]$$

Approximates true distribution as $n \to \infty$

---

## Sample Statistics

**Sample mean**: $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$

**Sample variance**: $s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$

The $n - 1$ (Bessel's correction) makes $s^2$ unbiased for $\sigma^2$

---

## Part V: Bayesian Modeling

---

## Bayes' Theorem

$$p(\theta \mid x) = \frac{p(x \mid \theta) \, p(\theta)}{p(x)}$$

- $p(\theta)$: **prior** — belief before data
- $p(x \mid \theta)$: **likelihood** — probability of data given parameters
- $p(\theta \mid x)$: **posterior** — updated belief after data
- $p(x)$: **marginal likelihood** (evidence)

---

## Bayesian Inference

Treat parameters as random variables:

$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)$$

Posterior combines prior knowledge with evidence from data

---

## Point Estimates

**MAP** (maximum a posteriori):

$$\hat{\theta}_{\text{MAP}} = \argmax_\theta \, p(\mathcal{D} \mid \theta) \, p(\theta)$$

**Posterior mean**:

$$\hat{\theta}_{\text{mean}} = \E[\theta \mid \mathcal{D}]$$

---

## Part VI: Maximum Likelihood Estimation

---

## The Likelihood Function

Given i.i.d. data $\mathcal{D} = \{x_1, \ldots, x_n\}$:

$$\mathcal{L}(\theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

**Log-likelihood**:

$$\ell(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$$

---

## The MLE

$$\hat{\theta}_{\text{MLE}} = \argmax_\theta \, \ell(\theta)$$

Maximizing likelihood = maximizing log-likelihood ($\log$ is monotonic)

---

## Example: Bernoulli MLE

$n$ coin flips, $k$ heads:

$$\ell(p) = k \log p + (n-k) \log(1-p)$$

Set derivative to zero:

$$\frac{k}{p} - \frac{n-k}{1-p} = 0 \implies \hat{p}_{\text{MLE}} = \frac{k}{n}$$

The empirical frequency

---

## Properties of MLE

Under regularity conditions:

- **Consistent**: converges to true parameter as $n \to \infty$
- **Asymptotically normal**: sampling distribution approaches Gaussian
- **Asymptotically efficient**: lowest variance among consistent estimators

---

## Part VII: Entropy

---

## Definition of Entropy

$$H(X) = -\sum_x p(x) \log p(x) = \E[-\log p(X)]$$

Base 2 → **bits**; base $e$ → **nats**

Convention: $0 \log 0 = 0$

---

## Interpretations

**Uncertainty**: how unpredictable is $X$?

- Deterministic → $H = 0$
- Uniform over $K$ outcomes → $H = \log K$ (maximum)

**Surprisal**: $-\log p(x)$ = information content of outcome $x$

**Compression**: minimum average bits to encode samples from $p$

---

## Properties of Entropy

- $H(X) \geq 0$, with equality iff $X$ is deterministic
- $H(X) \leq \log |\mathcal{X}|$, with equality iff uniform
- **Chain rule**: $H(X, Y) = H(X) + H(Y \mid X)$

---

## Conditional Entropy

$$H(Y \mid X) = -\sum_{x,y} p(x,y) \log p(y \mid x)$$

Remaining uncertainty in $Y$ after observing $X$

---

## Part VIII: Cross-Entropy and KL Divergence

---

## Cross-Entropy

$$H(p, q) = -\sum_x p(x) \log q(x)$$

Average bits to encode samples from $p$ using code optimized for $q$

Always: $H(p, q) \geq H(p)$, with equality iff $p = q$

---

## Cross-Entropy as Loss

In ML: $p$ = true distribution, $q$ = model prediction

For single example with true label $y$:

$$\mathcal{L} = -\log q(y)$$

Negative log-likelihood of the correct class

Minimizing cross-entropy = training the model to match truth

---

## KL Divergence

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

- Non-negative: $D_{\text{KL}}(p \| q) \geq 0$
- Zero iff $p = q$
- **Not symmetric**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$

---

## Minimizing Cross-Entropy = Minimizing KL

Since $H(p)$ is constant w.r.t. model parameters:

$$\argmin_q H(p, q) = \argmin_q D_{\text{KL}}(p \| q)$$

This is why cross-entropy loss works

---

## Part IX: Perplexity

---

## Definition

$$\text{PPL} = p(w_1, \ldots, w_N)^{-1/N} = \exp(H)$$

where $H$ is cross-entropy (in nats) on the test data

---

## Interpretation

**Effective vocabulary size** the model is uncertain about at each step

- PPL = 1: perfect prediction
- PPL = 100: uncertain among ~100 words
- PPL = $|V|$: equivalent to random guessing

**Lower is better**

---

## Relation to Entropy

In bits: $\text{PPL} = 2^H$

In nats: $\text{PPL} = e^H$

---

## Part X: Precision, Recall, F-Measure

---

## The Confusion Matrix

|  | Predicted + | Predicted − |
|--|-------------|-------------|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

---

## Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

Of all predicted positives, how many are correct?

High precision → few false positives

---

## Recall

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

Of all actual positives, how many were found?

High recall → few false negatives

---

## The Precision-Recall Trade-off

Increasing threshold → higher precision, lower recall

Decreasing threshold → higher recall, lower precision

Often in tension — cannot maximize both simultaneously

---

## F-Measure

Harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Penalizes extreme imbalances: low precision or low recall → low $F_1$

---

## Generalized F-Score

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $F_2$: weights recall higher
- $F_{0.5}$: weights precision higher

**Multi-class**: micro-average (aggregate counts) or macro-average (average per-class)

---

## Part XI: Significance Testing

---

## Hypothesis Testing Framework

1. **Null hypothesis** $H_0$: default assumption (no effect)
2. **Alternative hypothesis** $H_1$: what we want to show
3. Compute **test statistic** from data
4. How likely is this statistic under $H_0$?
5. Reject $H_0$ if sufficiently unlikely

---

## P-Values

$$p\text{-value} = P(\text{data as extreme or more} \mid H_0)$$

Small $p$-value → evidence against $H_0$

**Not** the probability that $H_0$ is true!

---

## Significance Level and Errors

Reject $H_0$ if $p < \alpha$ (typically $\alpha = 0.05$)

|  | $H_0$ True | $H_0$ False |
|--|------------|-------------|
| **Reject** | Type I (false +) | Correct |
| **Fail to reject** | Correct | Type II (false −) |

**Power** = $1 - \beta$ = probability of correctly rejecting false $H_0$

---

## Common Tests

**t-test**: compare means (assumes approx. normal)

**Paired t-test**: paired observations (before/after)

**Chi-squared**: independence in contingency tables

**Permutation test**: non-parametric, empirical null distribution

---

## Multiple Comparisons

With $m$ tests at level $\alpha$:

$$P(\text{at least one false positive}) = 1 - (1-\alpha)^m$$

**Bonferroni correction**: use $\alpha/m$ per test

---

## Confidence Intervals

A 95% CI: if we repeated the experiment many times, 95% of intervals contain the true parameter

$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

For 95%: $z_{\alpha/2} = 1.96$

---

## Summary

- **Random variables**: discrete (PMF) and continuous (PDF)
- **Expectation**: linear; **variance**: measures spread
- **Bayes' theorem**: posterior $\propto$ likelihood $\times$ prior
- **MLE**: $\argmax_\theta \sum_i \log p(x_i \mid \theta)$
- **Entropy**: $H = -\sum p \log p$ — uncertainty measure
- **Cross-entropy**: $H(p, q) = -\sum p \log q$ — the loss function
- **KL divergence**: $D_{\text{KL}} = H(p, q) - H(p)$ — distributional gap
- **Perplexity**: $\exp(H)$ — effective vocabulary size
- **Precision/Recall/F1**: classification evaluation
- **Significance testing**: p-values quantify evidence against $H_0$
