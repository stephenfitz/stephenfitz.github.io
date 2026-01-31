# Information Theory and Statistics

This article introduces fundamental concepts from probability theory, information theory, and statistics that underpin modern machine learning and natural language processing. We cover random variables, estimation, entropy and its relatives, evaluation metrics, and statistical significance testing.

---

## Random Variables and Probability Distributions

### Random Variables

A **random variable** is a function that assigns a numerical value to each outcome in a sample space. We distinguish two types:

- **Discrete random variables** take values from a countable set (e.g., integers, words in a vocabulary)
- **Continuous random variables** take values from an uncountable set (e.g., real numbers)

We denote random variables with capital letters (e.g., $X$) and their realized values with lowercase letters (e.g., $x$).

### Probability Mass and Density Functions

For a discrete random variable $X$, the **probability mass function (PMF)** is
\[
p(x) = P(X = x)
\]
satisfying $p(x) \geq 0$ for all $x$ and $\sum_x p(x) = 1$.

For a continuous random variable, the **probability density function (PDF)** $f(x)$ satisfies
\[
P(a \leq X \leq b) = \int_a^b f(x) \, dx
\]
with $f(x) \geq 0$ and $\int_{-\infty}^{\infty} f(x) \, dx = 1$.

### Joint and Conditional Distributions

For two random variables $X$ and $Y$, the **joint distribution** is $p(x, y) = P(X = x, Y = y)$.

The **marginal distribution** of $X$ is obtained by summing (or integrating) over $Y$:
\[
p(x) = \sum_y p(x, y)
\]

The **conditional distribution** of $Y$ given $X = x$ is
\[
p(y \mid x) = \frac{p(x, y)}{p(x)}
\]
provided $p(x) > 0$.

Two random variables are **independent** if $p(x, y) = p(x) p(y)$ for all $x, y$.

---

## Expectation and Variance

### Expected Value

The **expected value** (or mean) of a random variable $X$ is
\[
\E[X] = \sum_x x \, p(x) \quad \text{(discrete)}
\]
or
\[
\E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx \quad \text{(continuous)}
\]

For a function $g(X)$:
\[
\E[g(X)] = \sum_x g(x) \, p(x)
\]

Key properties:

- **Linearity**: $\E[aX + bY] = a\E[X] + b\E[Y]$
- If $X$ and $Y$ are independent: $\E[XY] = \E[X]\E[Y]$

### Variance and Standard Deviation

The **variance** measures spread around the mean:
\[
\Var(X) = \E[(X - \E[X])^2] = \E[X^2] - (\E[X])^2
\]

The **standard deviation** is $\sigma = \sqrt{\Var(X)}$.

Properties:

- $\Var(aX + b) = a^2 \Var(X)$
- If $X$ and $Y$ are independent: $\Var(X + Y) = \Var(X) + \Var(Y)$

---

## Common Distributions

### Bernoulli and Binomial

A **Bernoulli** random variable represents a single binary trial:
\[
P(X = 1) = p, \quad P(X = 0) = 1 - p
\]
with $\E[X] = p$ and $\Var(X) = p(1-p)$.

The **Binomial** distribution counts successes in $n$ independent Bernoulli trials:
\[
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
\]

### Categorical and Multinomial

A **Categorical** random variable takes one of $K$ discrete values with probabilities $p_1, \ldots, p_K$ where $\sum_k p_k = 1$.

The **Multinomial** distribution generalizes the binomial to $K$ categories.

Consider \(n\) independent and identical trials, where each trial produces exactly one outcome from a finite set of \(K\) categories, with fixed probabilities \((p_1,\dots,p_K)\) that sum to one. Let \(X_k\) denote the number of times outcome \(k\) occurs across the \(n\) trials. The random vector \((X_1,\dots,X_K)\) is said to follow a multinomial distribution, with the defining constraint \(\sum_{k=1}^K X_k = n\), since each trial contributes to exactly one category count. Its probability mass function is given by 
\[
\Pr(X_1=x_1,\dots,X_K=x_K) = \frac{n!}{x_1!\cdots x_K!}\prod_{k=1}^K p_k^{x_k}
\]

Intuitively, the multinomial distribution describes the joint distribution of category counts obtained by aggregating \(n\) independent categorical outcomes.

### Gaussian (Normal)

The **Gaussian** distribution with mean $\mu$ and variance $\sigma^2$ has density
\[
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

We write $X \sim \mathcal{N}(\mu, \sigma^2)$. The standard normal has $\mu = 0$ and $\sigma^2 = 1$.

---

## Sampling

### The Sampling Process

**Sampling** is the process of drawing observations from a probability distribution. Given a distribution $p(x)$, a **sample** is a collection of values $x_1, x_2, \ldots, x_n$ drawn independently from $p$.

We write $x_i \stackrel{\text{iid}}{\sim} p$ to indicate that samples are **independent and identically distributed (i.i.d.)**.

### Empirical Distribution

Given samples $x_1, \ldots, x_n$, the **empirical distribution** assigns probability $1/n$ to each observed value:
\[
\hat{p}(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[x_i = x]
\]

The empirical distribution approximates the true distribution as $n \to \infty$ (law of large numbers).

### Sample Statistics

The **sample mean** estimates the population mean:
\[
\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i
\]

The **sample variance** estimates the population variance:
\[
s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
\]

The factor $n-1$ (Bessel's correction) makes $s^2$ an unbiased estimator of $\sigma^2$.

---

## Bayes' Theorem and Bayesian Modeling

### Bayes' Theorem

**Bayes' theorem** relates conditional probabilities:
\[
p(\theta \mid x) = \frac{p(x \mid \theta) \, p(\theta)}{p(x)}
\]

The components have specific names:

- $p(\theta)$: **prior** — our belief about $\theta$ before seeing data
- $p(x \mid \theta)$: **likelihood** — probability of data given parameters
- $p(\theta \mid x)$: **posterior** — updated belief after seeing data
- $p(x) = \sum_\theta p(x \mid \theta) p(\theta)$: **marginal likelihood** (or evidence)

### Bayesian Inference

In Bayesian modeling, we treat parameters as random variables with prior distributions. After observing data $\mathcal{D}$, we compute the posterior:
\[
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
\]

The posterior combines prior knowledge with evidence from data.

### Point Estimates from the Posterior

Two common point estimates:

**Maximum a posteriori (MAP)**:
\[
\hat{\theta}_{\text{MAP}} = \argmax_\theta \, p(\theta \mid \mathcal{D}) = \argmax_\theta \, p(\mathcal{D} \mid \theta) \, p(\theta)
\]

**Posterior mean**:
\[
\hat{\theta}_{\text{mean}} = \E[\theta \mid \mathcal{D}] = \int \theta \, p(\theta \mid \mathcal{D}) \, d\theta
\]

---

## Maximum Likelihood Estimation

### The Likelihood Function

Given observed data $\mathcal{D} = \{x_1, \ldots, x_n\}$ and a parametric model $p(x \mid \theta)$, the **likelihood function** is
\[
\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)
\]
assuming i.i.d. samples.

The **log-likelihood** is often more convenient:
\[
\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)
\]

### Maximum Likelihood Estimator

The **maximum likelihood estimator (MLE)** is the parameter value that maximizes the likelihood:
\[
\hat{\theta}_{\text{MLE}} = \argmax_\theta \, \mathcal{L}(\theta) = \argmax_\theta \, \ell(\theta)
\]

Since $\log$ is monotonic, maximizing likelihood and log-likelihood yield the same solution.

### Example: Bernoulli MLE

Given $n$ coin flips with $k$ heads, the log-likelihood is
\[
\ell(p) = k \log p + (n-k) \log(1-p)
\]

Taking the derivative and setting to zero:
\[
\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0
\]

Solving yields $\hat{p}_{\text{MLE}} = k/n$, the empirical frequency.

### Properties of MLE

Under regularity conditions, MLEs are:

- **Consistent**: $\hat{\theta}_{\text{MLE}} \to \theta_{\text{true}}$ as $n \to \infty$
- **Asymptotically normal**: the sampling distribution approaches Gaussian
- **Asymptotically efficient**: achieves the lowest possible variance among consistent estimators

---

## Entropy

### Definition

The **entropy** of a discrete random variable $X$ with PMF $p$ is
\[
H(X) = -\sum_x p(x) \log p(x) = \E[-\log p(X)]
\]

By convention, $0 \log 0 = 0$. The base of the logarithm determines units: base 2 gives **bits**, base $e$ gives **nats**.

### Interpretations of Entropy

**1. Uncertainty**: Entropy measures the uncertainty or unpredictability of a random variable. A deterministic variable (one outcome has probability 1) has entropy 0. A uniform distribution over $K$ outcomes has maximum entropy $\log K$.

**2. Information content**: The quantity $-\log p(x)$ is the **information content** (or surprisal) of outcome $x$. Rare events carry more information. Entropy is the expected information content.

**3. Compression**: Entropy is the theoretical minimum average number of bits needed to encode samples from $p$. No lossless compression scheme can do better on average.

### Properties of Entropy

- $H(X) \geq 0$, with equality iff $X$ is deterministic
- $H(X) \leq \log |\mathcal{X}|$, with equality iff $X$ is uniform over its support
- **Chain rule**: $H(X, Y) = H(X) + H(Y \mid X)$

### Conditional Entropy

The **conditional entropy** of $Y$ given $X$ is
\[
H(Y \mid X) = \E_X[H(Y \mid X = x)] = -\sum_{x,y} p(x,y) \log p(y \mid x)
\]

This measures the remaining uncertainty in $Y$ after observing $X$.

---

## Cross-Entropy and KL Divergence

### Cross-Entropy

The **cross-entropy** between a true distribution $p$ and a model distribution $q$ is
\[
H(p, q) = -\sum_x p(x) \log q(x) = \E_{x \sim p}[-\log q(x)]
\]

Cross-entropy measures the average number of bits needed to encode samples from $p$ using a code optimized for $q$.

Note: $H(p, q) \geq H(p)$, with equality iff $p = q$.

### Cross-Entropy as a Loss Function

In machine learning, we typically have:

- $p$: the true (empirical) distribution over labels
- $q$: the model's predicted distribution

Minimizing cross-entropy trains the model to match the true distribution. For a single example with true label $y$ and predicted probabilities $q$:
\[
\mathcal{L} = -\log q(y)
\]

This is the **negative log-likelihood** of the correct class.

### KL Divergence

The **Kullback-Leibler divergence** (or relative entropy) from $q$ to $p$ is
\[
D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
\]

KL divergence measures how much $q$ differs from $p$. It is:

- Non-negative: $D_{\text{KL}}(p \| q) \geq 0$
- Zero iff $p = q$
- **Not symmetric**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general

Since $H(p)$ is constant with respect to model parameters, minimizing cross-entropy is equivalent to minimizing KL divergence.

---

## Perplexity

### Definition

**Perplexity** is a measure of how well a probability model predicts a sample. For a distribution $p$ over sequences, the perplexity on a test sequence $w_1, \ldots, w_N$ is
\[
\text{PPL} = p(w_1, \ldots, w_N)^{-1/N} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \log p(w_i \mid w_1, \ldots, w_{i-1})\right)
\]

Equivalently:
\[
\text{PPL} = \exp(H)
\]
where $H$ is the cross-entropy (in nats) of the model on the test data.

### Interpretation

Perplexity can be interpreted as the **effective vocabulary size** the model is uncertain about at each position. A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 equally likely options.

Lower perplexity indicates a better model:

- Perplexity 1: perfect prediction (the model assigns probability 1 to each correct token)
- Perplexity $V$: equivalent to random guessing over vocabulary of size $V$

### Relation to Entropy and Cross-Entropy

If $H$ is the cross-entropy in bits:
\[
\text{PPL} = 2^H
\]

If $H$ is in nats (natural log):
\[
\text{PPL} = e^H
\]

---

## Evaluation Metrics: Precision, Recall, and F-Measure

### The Confusion Matrix

For binary classification, predictions fall into four categories:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

### Precision

**Precision** measures the fraction of predicted positives that are correct:
\[
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
\]

High precision means few false positives. A model that only predicts positive when very confident will have high precision but may miss many true positives.

### Recall

**Recall** (or sensitivity, or true positive rate) measures the fraction of actual positives that are correctly identified:
\[
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
\]

High recall means few false negatives. A model that predicts positive liberally will have high recall but may have many false positives.

### The Precision-Recall Trade-off

Precision and recall are often in tension. Increasing the threshold for predicting positive typically increases precision but decreases recall, and vice versa.

### F-Measure

The **F-measure** (or F-score, F1-score) is the harmonic mean of precision and recall:
\[
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \, \text{TP}}{2 \, \text{TP} + \text{FP} + \text{FN}}
\]

The harmonic mean penalizes extreme imbalances: if either precision or recall is low, $F_1$ will be low.

The general **$F_\beta$ score** weights recall $\beta$ times as much as precision:
\[
F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
\]

- $F_2$ weights recall higher than precision
- $F_{0.5}$ weights precision higher than recall

### Multi-class Extensions

For multi-class problems, precision, recall, and F-measure can be computed:

- **Micro-averaged**: aggregate TP, FP, FN across all classes, then compute metrics
- **Macro-averaged**: compute metrics for each class, then average

---

## Statistical Significance Testing

### Hypothesis Testing Framework

Statistical hypothesis testing asks: could the observed result have occurred by chance?

The framework:

1. **Null hypothesis** $H_0$: a default assumption (e.g., no effect, no difference)
2. **Alternative hypothesis** $H_1$: what we want to show (e.g., there is an effect)
3. Compute a **test statistic** from the data
4. Determine how likely such a statistic would be under $H_0$
5. Reject $H_0$ if this probability is sufficiently small

### P-Values

The **p-value** is the probability of observing a result at least as extreme as the one obtained, assuming the null hypothesis is true:
\[
p\text{-value} = P(\text{data as extreme or more} \mid H_0)
\]

A small p-value suggests the data is unlikely under $H_0$, providing evidence against it.

**Important**: The p-value is *not* the probability that $H_0$ is true. It is the probability of the observed data (or more extreme) given that $H_0$ is true.

### Significance Level

The **significance level** $\alpha$ is the threshold for rejecting $H_0$. Common choices are $\alpha = 0.05$ or $\alpha = 0.01$.

- If $p < \alpha$: reject $H_0$, the result is "statistically significant"
- If $p \geq \alpha$: fail to reject $H_0$

### Types of Errors

|  | $H_0$ True | $H_0$ False |
|---|---|---|
| **Reject $H_0$** | Type I Error (false positive) | Correct |
| **Fail to reject $H_0$** | Correct | Type II Error (false negative) |

- $\alpha$ = probability of Type I error (set by the researcher)
- $\beta$ = probability of Type II error
- **Power** = $1 - \beta$ = probability of correctly rejecting a false $H_0$

### Common Tests

**t-test**: Tests whether the mean of a sample differs from a hypothesized value, or whether two sample means differ. Assumes approximately normal distributions.

**Paired t-test**: For paired observations (e.g., before/after measurements on the same subjects).

**Chi-squared test**: Tests independence in contingency tables or goodness-of-fit to a distribution.

**Permutation test**: A non-parametric test that computes the test statistic for many random permutations of the data to build a null distribution empirically.

### Multiple Comparisons Problem

When performing many hypothesis tests, the probability of at least one false positive increases. With $m$ independent tests at level $\alpha$:
\[
P(\text{at least one false positive}) = 1 - (1-\alpha)^m
\]

**Bonferroni correction**: Use significance level $\alpha/m$ for each test to maintain family-wise error rate at $\alpha$.

### Confidence Intervals

A **confidence interval** provides a range of plausible values for a parameter. A 95% confidence interval means: if we repeated the experiment many times, 95% of the computed intervals would contain the true parameter.

For a sample mean with known variance:
\[
\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}
\]

where $z_{\alpha/2}$ is the critical value from the standard normal (e.g., 1.96 for 95% confidence).

---

## Summary

This article covered the statistical and information-theoretic foundations essential for machine learning:

- **Random variables** and probability distributions provide the language for modeling uncertainty
- **Expectation and variance** summarize distributions with single numbers
- **Sampling** connects theory to data through empirical distributions
- **Bayes' theorem** enables updating beliefs with evidence; Bayesian modeling treats parameters as random variables
- **Maximum likelihood estimation** finds parameters that make the observed data most probable
- **Entropy** measures uncertainty; **cross-entropy** measures the cost of using the wrong distribution; **KL divergence** measures distributional difference
- **Perplexity** exponentiates cross-entropy, giving an interpretable measure of model uncertainty
- **Precision, recall, and F-measure** evaluate classification performance with different trade-offs
- **Statistical significance testing** quantifies whether observed effects could be due to chance; p-values measure evidence against the null hypothesis

These concepts appear throughout machine learning: cross-entropy loss trains neural networks, perplexity evaluates language models, precision and recall assess information retrieval systems, and significance testing validates experimental results.
