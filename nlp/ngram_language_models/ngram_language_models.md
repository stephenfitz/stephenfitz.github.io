# N-gram Language Models

A **language model** assigns probabilities to sequences of words. Given a sequence $w_1, w_2, \ldots, w_n$, a language model estimates $P(w_1, w_2, \ldots, w_n)$—the probability that this particular sequence occurs in the language. Language models are foundational to NLP: they power speech recognition, machine translation, spelling correction, and text generation.

This article develops **n-gram language models**, the classical count-based approach to language modeling. We derive the mathematical framework from first principles, examine why naive estimation fails, and systematically develop smoothing and back-off techniques that address the fundamental sparsity problem.

---

## The Language Modeling Problem

### Joint Probability of Word Sequences

Consider a sentence like "the cat sat on the mat". A language model should assign this grammatical, semantically coherent sentence a higher probability than "green to jumped now then". Formally, we want to estimate:
$$
P(w_1, w_2, \ldots, w_n)
$$
for any sequence of words $w_1, w_2, \ldots, w_n$ from a vocabulary $V$.

### The Chain Rule of Probability

Any joint probability can be decomposed using the **chain rule**:
$$
P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_1, w_2) \cdots P(w_n \mid w_1, \ldots, w_{n-1})
$$

More compactly:
$$
P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})
$$

This decomposition is exact—no approximation has been made. It reframes the problem: instead of estimating the joint probability directly, we estimate a sequence of conditional probabilities, each predicting the next word given all previous words.

**Why this decomposition?** The chain rule converts an exponentially large joint distribution into a product of conditional distributions. If our vocabulary has $|V|$ words, the joint distribution over sequences of length $n$ has $|V|^n$ possible outcomes. The chain rule doesn't reduce this complexity directly, but it provides a natural structure for approximation.

### The Curse of History

The challenge is clear: to compute $P(w_n \mid w_1, \ldots, w_{n-1})$, we need to condition on the entire history $w_1, \ldots, w_{n-1}$. For a sentence of 20 words, we'd need to estimate probabilities conditioned on 19-word histories. With a vocabulary of 50,000 words, the number of possible 19-word histories is $50000^{19}$—far more than atoms in the observable universe.

We will never observe most histories in any training corpus, no matter how large. This motivates the key approximation in n-gram models.

---

## The Markov Assumption

### Finite History Approximation

The **Markov assumption** truncates the conditioning history to a fixed length $k$:
$$
P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-k}, \ldots, w_{i-1})
$$

This assumes that the probability of the next word depends only on the previous $k$ words, not the entire history. The assumption is linguistically imperfect—long-range dependencies exist in language—but it makes estimation tractable.

### N-gram Models

An **n-gram** is a contiguous sequence of $n$ words. An **n-gram model** conditions on the previous $n-1$ words:

| Model | Conditions on | Approximation |
|-------|---------------|---------------|
| Unigram ($n=1$) | Nothing | $P(w_i)$ |
| Bigram ($n=2$) | Previous word | $P(w_i \mid w_{i-1})$ |
| Trigram ($n=3$) | Previous 2 words | $P(w_i \mid w_{i-2}, w_{i-1})$ |
| 4-gram ($n=4$) | Previous 3 words | $P(w_i \mid w_{i-3}, w_{i-2}, w_{i-1})$ |

The terminology can be confusing: a **bigram model** predicts one word from one previous word (pairs of words are considered), a **trigram model** predicts one word from two previous words (triples are considered), and so on.

### Mathematical Justification

Why is the Markov assumption reasonable? Consider two perspectives:

**Information-theoretic**: If language has finite entropy rate $H$, then by the asymptotic equipartition property, most of the information about the next word is captured in recent context. Longer histories provide diminishing returns.

**Statistical**: With limited training data, we face a bias-variance tradeoff. Longer histories reduce bias (more accurate modeling of dependencies) but increase variance (fewer observations per context). The Markov assumption controls variance at the cost of some bias.

**Empirical**: In practice, trigram and 4-gram models capture most useful local structure. Perplexity improvements diminish rapidly beyond $n=5$ for typical corpora.

---

## Maximum Likelihood Estimation

### Counting N-grams

Given a training corpus, we estimate n-gram probabilities by counting. Let $C(w_1, \ldots, w_n)$ denote the count of the n-gram $w_1, \ldots, w_n$ in the corpus.

**Bigram MLE**:
$$
P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

The numerator counts how often the bigram $(w_{i-1}, w_i)$ appears; the denominator counts how often the context $w_{i-1}$ appears (which equals the sum of counts of all bigrams starting with $w_{i-1}$).

**Trigram MLE**:
$$
P_{\text{MLE}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i)}{C(w_{i-2}, w_{i-1})}
$$

**General n-gram MLE**:
$$
P_{\text{MLE}}(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) = \frac{C(w_{i-n+1}, \ldots, w_i)}{C(w_{i-n+1}, \ldots, w_{i-1})}
$$

### Derivation from Maximum Likelihood

Why this formula? MLE finds parameters that maximize the likelihood of the training data. For bigrams, we want:
$$
\hat{\theta} = \argmax_\theta \prod_{i} P_\theta(w_i \mid w_{i-1})
$$

Taking logs and treating each context independently, this becomes maximizing:
$$
\sum_{w, w'} C(w, w') \log P(w' \mid w)
$$
subject to $\sum_{w'} P(w' \mid w) = 1$ for each context $w$.

Using Lagrange multipliers, the optimal solution is:
$$
P(w' \mid w) = \frac{C(w, w')}{\sum_{w''} C(w, w'')} = \frac{C(w, w')}{C(w)}
$$

This confirms that relative frequency is the maximum likelihood estimator.

### Example Calculation

Consider a tiny corpus: "the cat sat. the cat ate. the dog sat."

Bigram counts:

| Bigram | Count |
|--------|-------|
| (the, cat) | 2 |
| (the, dog) | 1 |
| (cat, sat) | 1 |
| (cat, ate) | 1 |
| (dog, sat) | 1 |

Context counts:

| Context | Count |
|---------|-------|
| the | 3 |
| cat | 2 |
| dog | 1 |

Bigram probabilities:

$$
P(\text{cat} \mid \text{the}) = \frac{2}{3}, \quad P(\text{dog} \mid \text{the}) = \frac{1}{3}
$$
$$
P(\text{sat} \mid \text{cat}) = \frac{1}{2}, \quad P(\text{ate} \mid \text{cat}) = \frac{1}{2}
$$
$$
P(\text{sat} \mid \text{dog}) = \frac{1}{1} = 1
$$

---

## The Sparsity Problem

### Zero Counts

The fundamental problem with MLE is **sparsity**: most n-grams never appear in the training corpus, yielding zero probability estimates.

If "the elephant" never appears in training:
$$
P_{\text{MLE}}(\text{elephant} \mid \text{the}) = \frac{C(\text{the}, \text{elephant})}{C(\text{the})} = \frac{0}{C(\text{the})} = 0
$$

Any sentence containing "the elephant" receives probability zero, regardless of how reasonable the sentence is.

### Why Sparsity Is Catastrophic

The problem compounds multiplicatively. The probability of a sentence is a product:
$$
P(w_1, \ldots, w_n) = \prod_i P(w_i \mid w_{i-1})
$$

A single zero factor makes the entire product zero. One unseen bigram invalidates the entire sentence.

Moreover, zero probabilities make **perplexity** (the standard evaluation metric) undefined. Perplexity involves $\log P$, and $\log 0 = -\infty$.

### The Scale of Sparsity

Consider a vocabulary of $V = 50{,}000$ words:

| N-gram | Possible contexts | Typical corpus coverage |
|--------|-------------------|------------------------|
| Bigram | $5 \times 10^4$ | ~10-30% |
| Trigram | $2.5 \times 10^9$ | ~0.1-1% |
| 4-gram | $1.25 \times 10^{14}$ | ~0.001% |

Even with billions of words of training data, most trigrams and nearly all 4-grams will never be observed. Sparsity isn't a minor inconvenience—it's the dominant challenge in n-gram modeling.

### Mathematical Perspective on Zero Estimates

From a Bayesian viewpoint, observing zero occurrences of an n-gram doesn't mean its true probability is zero—it means we lack evidence. The MLE estimate conflates "unobserved" with "impossible."

Let $\theta$ be the true probability of an n-gram. If we observe $k$ occurrences in $N$ trials, the MLE is $\hat{\theta} = k/N$. When $k=0$, we get $\hat{\theta} = 0$, but the true $\theta$ could be any small positive value we haven't yet observed.

This motivates **smoothing**: adjusting probability estimates to reserve some mass for unseen events.

---

## Smoothing: The Core Idea

### Redistributing Probability Mass

**Smoothing** (or **discounting**) takes probability mass from observed n-grams and redistributes it to unobserved ones. The total probability must still sum to 1:
$$
\sum_{w} P(w \mid \text{context}) = 1
$$

If we increase $P(w \mid \text{context})$ for unseen words $w$, we must decrease it for seen words.

### Desiderata for Smoothing

A good smoothing method should:

1. **Assign nonzero probability to all n-grams** (avoid zeros)
2. **Preserve relative ordering** (likely n-grams should remain more likely than unlikely ones)
3. **Be consistent** (probabilities should still sum to 1)
4. **Perform well empirically** (improve perplexity on held-out data)

We now examine progressively more sophisticated smoothing methods.

---

## Laplace (Add-One) Smoothing

### The Method

**Laplace smoothing** adds 1 to every count:
$$
P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}
$$

The denominator adds $|V|$ (vocabulary size) to ensure normalization. If we add 1 to each of $|V|$ possible bigram continuations, the denominator must increase by $|V|$.

### Derivation and Justification

Laplace smoothing has a Bayesian interpretation. Suppose we place a symmetric Dirichlet prior on the distribution over next words:
$$
P(w \mid \text{context}) \sim \text{Dirichlet}(\alpha, \alpha, \ldots, \alpha)
$$

With $\alpha = 1$ (uniform prior), the posterior mean after observing counts $C$ is:
$$
\E[P(w \mid \text{context})] = \frac{C(\text{context}, w) + 1}{\sum_{w'} [C(\text{context}, w') + 1]} = \frac{C(\text{context}, w) + 1}{C(\text{context}) + |V|}
$$

This is exactly Laplace smoothing. The interpretation: before seeing any data, we act as if we've seen each n-gram once.

### Example

Continuing our tiny corpus with $|V| = 5$ words (the, cat, dog, sat, ate):

Without smoothing:
$$
P(\text{cat} \mid \text{the}) = \frac{2}{3} \approx 0.667
$$

With Laplace smoothing:
$$
P_{\text{Laplace}}(\text{cat} \mid \text{the}) = \frac{2 + 1}{3 + 5} = \frac{3}{8} = 0.375
$$

For an unseen bigram like (the, the):
$$
P_{\text{Laplace}}(\text{the} \mid \text{the}) = \frac{0 + 1}{3 + 5} = \frac{1}{8} = 0.125
$$

### Problems with Laplace Smoothing

Laplace smoothing is simple but deeply flawed for language modeling:

**Excessive discounting**: It takes too much probability from seen n-grams. In the example above, $P(\text{cat} \mid \text{the})$ dropped from 0.667 to 0.375—nearly halved.

**Uniform treatment of unseen events**: All unseen n-grams receive equal probability, but some unseen n-grams are more plausible than others. "The elephant" should be more likely than "the the."

**Scale dependence**: The discount ratio $C/(C + |V|)$ depends heavily on vocabulary size. With $|V| = 50{,}000$, even high-count n-grams are severely discounted.

**Poor perplexity**: Laplace smoothing typically performs worse than more sophisticated methods in practice.

---

## Add-k Smoothing

### Generalizing Laplace

**Add-k smoothing** generalizes Laplace by adding a fractional count $k < 1$:
$$
P_{\text{add-}k}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k|V|}
$$

The parameter $k$ controls how much probability mass shifts to unseen events. When $k = 1$, we recover Laplace smoothing. When $k \to 0$, we approach MLE.

### Choosing k

The optimal $k$ is chosen to minimize perplexity on held-out data (validation set). Typical values for bigrams range from 0.01 to 0.5.

**Grid search**: Try values like $k \in \{0.001, 0.01, 0.1, 0.5, 1.0\}$ and select the one with lowest validation perplexity.

### Bayesian Interpretation

Add-k smoothing corresponds to a Dirichlet prior with parameter $\alpha = k$:

- $k > 1$: Prior believes the distribution is more uniform than data suggests
- $k = 1$: Uniform prior (Laplace)
- $0 < k < 1$: Prior believes the distribution is more peaked than uniform
- $k \to 0$: Maximum likelihood (no prior influence)

### Limitations

Add-k smoothing improves on Laplace but still has fundamental limitations:

1. **Single parameter for all contexts**: The same $k$ is used regardless of how much data we have for a context
2. **Uniform smoothing for unseen events**: Still treats all unseen n-grams equally
3. **No use of lower-order information**: Ignores potentially useful unigram or lower-order n-gram statistics

---

## Good-Turing Smoothing

### The Intuition

**Good-Turing smoothing** (Good, 1953) addresses a key question: how much probability mass should be reserved for unseen events?

The insight: **n-grams seen once** provide information about n-grams seen zero times. If many n-grams appear exactly once, we should expect many unseen n-grams to appear once in new data.

### Notation

Let $N_r$ be the number of n-grams that appear exactly $r$ times in the training corpus:

- $N_0$ = number of unseen n-grams
- $N_1$ = number of n-grams appearing once (hapax legomena)
- $N_2$ = number of n-grams appearing twice
- etc.

The total number of n-gram tokens is:
$$
N = \sum_{r=1}^{\infty} r \cdot N_r
$$

### The Good-Turing Estimator

Good-Turing replaces each count $r$ with an adjusted count $r^*$:
$$
r^* = (r + 1) \frac{N_{r+1}}{N_r}
$$

The probability estimate for an n-gram with count $r$ is then:
$$
P_{\text{GT}}(\text{n-gram with count } r) = \frac{r^*}{N}
$$

For unseen n-grams ($r = 0$):
$$
0^* = 1 \cdot \frac{N_1}{N_0}
$$

The total probability mass for unseen events is:
$$
P(\text{unseen}) = N_0 \cdot \frac{0^*}{N} = N_0 \cdot \frac{N_1}{N_0 \cdot N} = \frac{N_1}{N}
$$

This elegant result says: **the probability mass for unseen events equals the fraction of tokens that are singletons** (n-grams occurring exactly once).

### Derivation

The derivation relies on a key assumption: n-gram frequencies follow a statistical distribution where $N_r$ estimates the expected number of n-grams with true frequency yielding $r$ observations.

Consider the probability mass from n-grams occurring $r$ times:
$$
\text{Mass}_r = r \cdot N_r / N
$$

Good-Turing reallocates this mass based on the empirical relationship between $N_r$ and $N_{r+1}$. The adjusted count $r^*$ satisfies:
$$
r^* \cdot N_r = (r+1) \cdot N_{r+1}
$$

This ensures the total mass is preserved while shifting estimates based on frequency-of-frequency statistics.

### Example

Suppose we count bigrams in a corpus and find:

| Count $r$ | $N_r$ (n-grams with this count) |
|-----------|-------------------------------|
| 0 | 1,000,000 (unseen) |
| 1 | 5,000 |
| 2 | 2,000 |
| 3 | 1,000 |
| 4 | 600 |
| 5 | 400 |

Adjusted counts:
$$
0^* = 1 \cdot \frac{5000}{1000000} = 0.005
$$
$$
1^* = 2 \cdot \frac{2000}{5000} = 0.8
$$
$$
2^* = 3 \cdot \frac{1000}{2000} = 1.5
$$

A bigram seen once gets adjusted count 0.8 instead of 1. The "missing" 0.2 goes toward unseen bigrams.

### Practical Issues

Good-Turing has theoretical elegance but practical challenges:

**Sparse $N_r$ for large $r$**: For high counts, $N_{r+1}$ may be zero or very noisy. Solution: smooth the $N_r$ values (e.g., fit a power-law regression) before computing $r^*$.

**Not directly applicable**: Good-Turing gives a total probability for unseen events but doesn't specify how to distribute it among specific unseen n-grams.

**Combination with back-off**: In practice, Good-Turing is combined with back-off (covered below) to allocate the reserved probability mass.

---

## Back-off Models

### The Intuition

**Back-off** models address a key limitation of fixed n-gram models: when we lack evidence for a high-order n-gram, we should consult lower-order statistics.

If we've never seen "colorless green ideas," we can still estimate its probability using:

- The bigram probability $P(\text{ideas} \mid \text{green})$
- Or the unigram probability $P(\text{ideas})$

### Stupid Back-off

**Stupid back-off** (Brants et al., 2007) is the simplest approach:

$$
S(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) =
\begin{cases}
\dfrac{C(w_{i-n+1}, \ldots, w_i)}{C(w_{i-n+1}, \ldots, w_{i-1})} & \text{if } C(w_{i-n+1}, \ldots, w_i) > 0 \\[10pt]
\lambda \cdot S(w_i \mid w_{i-n+2}, \ldots, w_{i-1}) & \text{otherwise}
\end{cases}
$$

where $\lambda$ is a fixed back-off weight (typically 0.4).

**Key properties**:

- Uses MLE when counts are available
- Backs off to shorter context when count is zero
- The factor $\lambda$ penalizes backing off
- Scores don't sum to 1 (not true probabilities—hence "stupid")

Stupid back-off is fast and works well at scale (used by Google for web-scale language models), but the non-normalized scores limit some applications.

### Katz Back-off

**Katz back-off** (Katz, 1987) combines Good-Turing discounting with back-off:

$$
P_{\text{Katz}}(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) =
\begin{cases}
\dfrac{C^*(w_{i-n+1}, \ldots, w_i)}{C(w_{i-n+1}, \ldots, w_{i-1})} & \text{if } C(w_{i-n+1}, \ldots, w_i) > 0 \\[10pt]
\alpha(w_{i-n+1}, \ldots, w_{i-1}) \cdot P_{\text{Katz}}(w_i \mid w_{i-n+2}, \ldots, w_{i-1}) & \text{otherwise}
\end{cases}
$$

where:

- $C^*$ is the Good-Turing adjusted count
- $\alpha$ is a normalization factor to preserve probability mass

### Computing the Back-off Weight

The back-off weight $\alpha$ is determined by the constraint that probabilities sum to 1:
$$
\sum_w P_{\text{Katz}}(w \mid \text{context}) = 1
$$

Let $\mathcal{S}$ be the set of words $w$ where $C(\text{context}, w) > 0$ (seen continuations) and $\mathcal{U}$ be unseen continuations. Then:

$$
\sum_{w \in \mathcal{S}} \frac{C^*(\text{context}, w)}{C(\text{context})} + \alpha \sum_{w \in \mathcal{U}} P_{\text{Katz}}(w \mid \text{shorter context}) = 1
$$

Solving for $\alpha$:
$$
\alpha(\text{context}) = \frac{1 - \sum_{w \in \mathcal{S}} \frac{C^*(\text{context}, w)}{C(\text{context})}}{\sum_{w \in \mathcal{U}} P_{\text{Katz}}(w \mid \text{shorter context})}
$$

The numerator is the probability mass reserved for unseen events (from Good-Turing discounting). The denominator normalizes the back-off distribution.

---

## Interpolation

### The Idea

**Interpolation** combines estimates from multiple n-gram orders simultaneously, rather than backing off only when counts are zero:
$$
P_{\text{interp}}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 P_3(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P_2(w_i \mid w_{i-1}) + \lambda_1 P_1(w_i)
$$

where $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and each $\lambda_j \geq 0$.

### Motivation

Why interpolate rather than back off?

1. **Lower-order models provide useful signal even when higher-order counts exist**: Even if we've seen "New York" 100 times, the unigram probability of "York" is still informative.

2. **Robustness**: When higher-order counts are small, they're noisy. Interpolation provides regularization.

3. **Smooth degradation**: Rather than abrupt transitions at count thresholds, interpolation smoothly blends information.

### Choosing Interpolation Weights

The weights $\lambda$ are typically learned by maximizing likelihood on held-out data using the **EM algorithm** (expectation-maximization).

**E-step**: Given current $\lambda$, compute the expected contribution of each component:
$$
\gamma_j^{(i)} = \frac{\lambda_j P_j(w_i \mid \text{context})}{\sum_k \lambda_k P_k(w_i \mid \text{context})}
$$

**M-step**: Update $\lambda$ based on expected contributions:
$$
\lambda_j = \frac{1}{N} \sum_i \gamma_j^{(i)}
$$

Iterate until convergence.

### Context-Dependent Weights

**Jelinek-Mercer smoothing** uses interpolation with context-dependent weights:
$$
P(w \mid u, v) = \lambda(u, v) P_{\text{MLE}}(w \mid u, v) + (1 - \lambda(u, v)) P(w \mid v)
$$

where $\lambda(u, v)$ depends on how reliable the context $(u, v)$ is (e.g., higher $\lambda$ when $C(u, v)$ is large).

A simple approach: bucket contexts by count and learn a separate $\lambda$ for each bucket.

---

## Absolute Discounting

### The Observation

Empirical studies (Church and Gale, 1991) revealed a striking pattern: the optimal discount for n-gram counts is approximately **constant** across different count values.

If an n-gram appears $r$ times, its expected count in held-out data is approximately $r - d$, where $d \approx 0.75$ is nearly independent of $r$ (for $r \geq 1$).

### The Method

**Absolute discounting** subtracts a fixed discount $d$ from each non-zero count:
$$
P_{\text{abs}}(w \mid v) = \frac{\max(C(v, w) - d, 0)}{C(v)} + \lambda(v) P(w)
$$

where $\lambda(v)$ is chosen to ensure normalization:
$$
\lambda(v) = \frac{d \cdot |\{w : C(v, w) > 0\}|}{C(v)}
$$

The term $|\{w : C(v, w) > 0\}|$ counts the number of distinct word types following context $v$.

### Interpretation

The discount $d$ represents probability mass taken from each observed n-gram. This mass is collected into $\lambda(v)$ and redistributed via the lower-order model $P(w)$.

The factor $|\{w : C(v, w) > 0\}|$ in $\lambda$ means: contexts with more diverse continuations redistribute more mass to the back-off distribution. This makes linguistic sense—a context that precedes many different words is less predictive.

### Estimating the Discount

The optimal discount can be estimated from the training data:
$$
d = \frac{N_1}{N_1 + 2N_2}
$$

where $N_1$ and $N_2$ are the number of n-grams occurring exactly once and twice, respectively. This formula emerges from Good-Turing analysis and typically yields $d \approx 0.75$.

---

## Kneser-Ney Smoothing

### The Problem with Standard Back-off

Standard interpolation and back-off have a subtle flaw. Consider the word "Francisco." In the phrase "San Francisco," "Francisco" appears frequently. But "Francisco" almost never appears in other contexts.

Standard unigram probability: $P(\text{Francisco})$ is relatively high (frequent word).

But when backing off from an unseen context like "I visited ___," we shouldn't favor "Francisco"—it's not a word that appears in diverse contexts.

### Continuation Probability

**Kneser-Ney smoothing** (Kneser and Ney, 1995) addresses this by replacing the lower-order probability with **continuation probability**—how likely a word is to appear as a novel continuation.

Instead of $P(w) \propto C(w)$, use:
$$
P_{\text{continuation}}(w) \propto |\{v : C(v, w) > 0\}|
$$

This counts the number of distinct contexts that $w$ follows. "Francisco" follows few contexts (mainly "San"), so its continuation probability is low. A word like "the" follows many contexts, so its continuation probability is high.

### The Full Kneser-Ney Formula

**Bigram Kneser-Ney**:
$$
P_{\text{KN}}(w \mid v) = \frac{\max(C(v, w) - d, 0)}{C(v)} + \lambda(v) P_{\text{continuation}}(w)
$$

where:
$$
\lambda(v) = \frac{d \cdot |\{w : C(v, w) > 0\}|}{C(v)}
$$
$$
P_{\text{continuation}}(w) = \frac{|\{v' : C(v', w) > 0\}|}{\sum_{w'} |\{v : C(v, w') > 0\}|}
$$

### Recursive Definition for Higher Orders

For n-grams with $n > 2$, Kneser-Ney is defined recursively. The highest-order model uses regular counts; lower-order models use continuation counts:

$$
P_{\text{KN}}(w \mid u, v) = \frac{\max(C(u, v, w) - d, 0)}{C(u, v)} + \lambda(u, v) P_{\text{KN}}(w \mid v)
$$

where $P_{\text{KN}}(w \mid v)$ uses continuation counts rather than raw counts.

### Modified Kneser-Ney

**Modified Kneser-Ney** (Chen and Goodman, 1999) uses different discounts for different count levels:

$$
d(c) = \begin{cases}
0 & \text{if } c = 0 \\
d_1 & \text{if } c = 1 \\
d_2 & \text{if } c = 2 \\
d_{3+} & \text{if } c \geq 3
\end{cases}
$$

The discounts are estimated from training data:
$$
d_1 = 1 - 2 \frac{N_2}{N_1} \cdot \frac{N_1}{N_1 + N_2}
$$
$$
d_2 = 2 - 3 \frac{N_3}{N_2} \cdot \frac{N_1}{N_1 + N_2}
$$
$$
d_{3+} = 3 - 4 \frac{N_4}{N_3} \cdot \frac{N_1}{N_1 + N_2}
$$

Modified Kneser-Ney consistently achieves the best perplexity among classical smoothing methods and remained state-of-the-art for count-based language models.

### Why Kneser-Ney Works

The effectiveness of Kneser-Ney stems from two insights:

1. **Absolute discounting**: Constant subtraction matches empirical behavior better than proportional discounting.

2. **Continuation probability**: The lower-order distribution should reflect how words behave as novel continuations, not their raw frequency. This captures the difference between words that are frequent because they follow specific contexts vs. words that are genuinely versatile.

---

## Evaluation: Perplexity

### Definition

**Perplexity** measures how well a language model predicts a test set. For a test sequence $w_1, \ldots, w_N$:
$$
\text{PP}(W) = P(w_1, \ldots, w_N)^{-1/N} = \sqrt[N]{\frac{1}{P(w_1, \ldots, w_N)}}
$$

Equivalently, using cross-entropy $H$:
$$
\text{PP}(W) = 2^{H(W)}
$$
where:
$$
H(W) = -\frac{1}{N} \log_2 P(w_1, \ldots, w_N) = -\frac{1}{N} \sum_i \log_2 P(w_i \mid w_1, \ldots, w_{i-1})
$$

### Interpretation

Perplexity can be interpreted as the **effective vocabulary size** the model is uncertain among at each step. A perplexity of 100 means the model is, on average, as uncertain as if it were choosing uniformly among 100 words.

**Lower perplexity is better**: The model assigns higher probability to the test data.

**Relationship to entropy**: Perplexity is the exponential of cross-entropy. Minimizing perplexity is equivalent to minimizing cross-entropy, which is equivalent to maximizing likelihood.

### Typical Values

| Model | Typical Perplexity (Wall Street Journal) |
|-------|------------------------------------------|
| Unigram | 950 |
| Bigram | 170 |
| Trigram | 100 |
| 4-gram + Kneser-Ney | 80 |

Modern neural language models achieve perplexity below 20 on similar benchmarks.

### Caveats

Perplexity is only comparable across models with the same vocabulary and test set. Different preprocessing (tokenization, handling of unknown words) can significantly affect perplexity.

Perplexity also doesn't capture all aspects of language model quality. A model might have good perplexity but generate incoherent text, or vice versa.

---

## Practical Considerations

### Handling Unknown Words

Real text contains words not in the training vocabulary. Common strategies:

**Closed vocabulary**: Treat out-of-vocabulary (OOV) words as a special `<UNK>` token. Replace rare training words (count < threshold) with `<UNK>` to create training examples.

**Open vocabulary**: Use subword units (BPE, etc.) that can represent any word. This moves the OOV problem to a lower level.

### Start Tokens

The beginning-of-sentence token `<s>` (or `<BOS>`) provides context for the first word(s). Without it, we'd need a separate distribution for "sentence-initial words" versus the conditional distributions used elsewhere.

For an n-gram model, pad the beginning with $n-1$ start tokens:

```
Sentence: "the cat sat"
Bigram contexts: (<s>, the), (the, cat), (cat, sat), ...
Trigram contexts: (<s>, <s>, the), (<s>, the, cat), (the, cat, sat), ...
```

This is a notational convenience—it lets us use the same conditional probability machinery throughout.

---

## The Necessity of the End-of-Sentence Token

The end-of-sentence token `</s>` (or `<EOS>`) is not merely a notational convenience—it is **mathematically essential** for the language model to define a valid probability distribution over sentences. Without it, the model is fundamentally broken as a probability distribution.

### The Problem: What Are We Modeling?

A language model should define a probability distribution over **sentences** (finite sequences of words). For any probability distribution, the total probability over all possible outcomes must equal 1:
$$
\sum_{\text{all sentences } S} P(S) = 1
$$

The question is: does our n-gram model satisfy this requirement?

### Without EOS: The Probabilities Sum to Infinity

Consider a model without an end token. For a bigram model, we compute:
$$
P(w_1, w_2, \ldots, w_n) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_2) \cdots P(w_n \mid w_{n-1})
$$

At each position, the conditional probabilities sum to 1 over the vocabulary:
$$
\sum_{w \in V} P(w \mid \text{context}) = 1
$$

Now let's sum over all sentences. First, all one-word sentences:
$$
\sum_{w_1 \in V} P(w_1) = 1
$$

All two-word sentences:
$$
\sum_{w_1 \in V} \sum_{w_2 \in V} P(w_1) P(w_2 \mid w_1) = \sum_{w_1 \in V} P(w_1) \underbrace{\sum_{w_2 \in V} P(w_2 \mid w_1)}_{= 1} = \sum_{w_1 \in V} P(w_1) = 1
$$

All three-word sentences:
$$
\sum_{w_1, w_2, w_3 \in V} P(w_1) P(w_2 \mid w_1) P(w_3 \mid w_2) = 1
$$

By induction, for any length $n$:
$$
\sum_{\text{all } n\text{-word sentences}} P(w_1, \ldots, w_n) = 1
$$

Therefore, the total probability over **all** sentences is:
$$
\sum_{\text{all sentences}} P(S) = \sum_{n=1}^{\infty} \sum_{\text{all } n\text{-word sentences}} P(w_1, \ldots, w_n) = \sum_{n=1}^{\infty} 1 = \infty
$$

**The probabilities sum to infinity, not 1.** This is not a valid probability distribution.

### The Interpretation Problem

What does the model without EOS actually represent? It defines a probability distribution over the **next word given any prefix**, but not over complete sentences. The quantity $P(w_1, \ldots, w_n)$ is not "the probability of the sentence $w_1 \ldots w_n$"—it's something else entirely.

More precisely, without EOS the model defines a **stochastic process** that generates infinite sequences. At each step, it samples a word from $P(w \mid \text{context})$. But since there's no stopping condition, every generated sequence is infinite. Finite sentences have probability zero under this interpretation (they're measure-zero events in the space of infinite sequences).

### With EOS: A Proper Distribution

Now include `</s>` as a special token in the vocabulary. At each position, the model can either:

- Generate a regular word $w \in V$ and continue
- Generate `</s>` and stop

The conditional distributions now include `</s>`:
$$
P(\text{EOS} \mid \text{context}) + \sum_{w \in V} P(w \mid \text{context}) = 1
$$

The probability of a complete sentence $w_1, \ldots, w_n$ followed by termination is:
$$
P(w_1, \ldots, w_n, \text{EOS}) = P(w_1) \cdot P(w_2 \mid w_1) \cdots P(w_n \mid w_{n-1}) \cdot P(\text{EOS} \mid w_n)
$$

**Claim**: The sum over all finite sentences equals 1.

**Proof**: Let $p_{\text{stop}}(w) = P(\text{EOS} \mid w)$ and $p_{\text{cont}}(w) = 1 - p_{\text{stop}}(w) = \sum_{w' \in V} P(w' \mid w)$.

The probability of all 1-word sentences:
$$
\sum_{w_1} P(w_1) \cdot p_{\text{stop}}(w_1)
$$

The probability of all 2-word sentences:
$$
\sum_{w_1, w_2} P(w_1) \cdot P(w_2 \mid w_1) \cdot p_{\text{stop}}(w_2)
$$

And so on. The total is:
$$
\sum_{n=1}^{\infty} \sum_{w_1, \ldots, w_n} P(w_1) \prod_{i=2}^{n} P(w_i \mid w_{i-1}) \cdot p_{\text{stop}}(w_n)
$$

This equals 1 by a standard result: the probability of eventually stopping in a Markov chain where each state has positive stopping probability sums to 1, provided $p_{\text{stop}}(w) > 0$ for all $w$ (which holds if `</s>` has positive probability in every context).

### A Concrete Example

Consider a tiny vocabulary $V = \{a, b\}$ plus `</s>`, with a unigram model:

**Without EOS**: $P(a) = 0.6$, $P(b) = 0.4$

Sum over 1-word sentences: $0.6 + 0.4 = 1$
Sum over 2-word sentences: $(0.6 + 0.4)^2 = 1$
Sum over all sentences: $1 + 1 + 1 + \cdots = \infty$

**With EOS**: $P(a) = 0.4$, $P(b) = 0.3$, $P(\text{EOS}) = 0.3$

Sum over 1-word sentences: $0.3$ (just `</s>`)

Sum over 2-word sentences: $(0.4 + 0.3) \times 0.3 = 0.21$

Sum over 3-word sentences: $(0.4 + 0.3)^2 \times 0.3 = 0.147$

Sum over $n$-word sentences: $0.7^{n-1} \times 0.3$

Total: $\sum_{n=1}^{\infty} 0.3 \times 0.7^{n-1} = 0.3 \times \frac{1}{1 - 0.7} = 0.3 \times \frac{10}{3} = 1$ ✓

### Implications for Perplexity

Perplexity requires a valid probability, so the EOS token must be included:
$$
\text{PP}(w_1, \ldots, w_n) = P(w_1, \ldots, w_n, \text{EOS})^{-1/(n+1)}
$$

Note the denominator is $n + 1$, counting the EOS prediction. This is important: it means the model is evaluated on its ability to predict both words and sentence boundaries.

If we excluded EOS from perplexity:
$$
\text{PP}'(w_1, \ldots, w_n) = P(w_1, \ldots, w_n)^{-1/n}
$$

This would **favor longer sentences** because $P(w_1, \ldots, w_n)$ doesn't include the "penalty" for continuing (the model never has to predict when to stop). Two models could have identical per-word predictions but different perplexities under this flawed metric if they differ only in EOS probabilities.

### Deficient Models

A model without EOS is called **deficient**: it assigns total probability less than 1 to the event space it's supposed to model (finite sentences), with the "missing" mass going to infinite sequences.

More generally, a model can be deficient even with EOS if the EOS probability is too low. If $P(\text{EOS} \mid w) = 0$ for some word $w$, and $w$ can be reached with positive probability, then there's positive probability of generating an infinite sequence, and the model is deficient.

In practice, smoothing ensures $P(\text{EOS} \mid w) > 0$ for all $w$, avoiding deficiency.

### The BOS Token Is Different

Note that the beginning token BOS serves a different purpose than EOS. The start token is a **conditioning context**: it is never predicted, only conditioned on. It provides the initial context for predicting $w_1$.

Without BOS, we would need a separate "initial distribution" $P_0(w_1)$ distinct from the conditional distributions $P(w \mid \text{context})$. The start token lets us unify these: $P(w_1 \mid \text{BOS})$ is just another conditional probability.

The asymmetry is clear:

- BOS appears on the **right side** of the conditioning bar: $P(w_1 \mid \text{BOS})$
- EOS appears on the **left side**: $P(\text{EOS} \mid w_n)$

---

## Efficiency

**Storage**: N-gram models can be compressed using tries, perfect hashing, or quantization. Tools like KenLM provide highly optimized implementations.

**Computation**: Probability lookups are O(1) with hash tables. The main cost is loading the model into memory.

**Pruning**: Remove n-grams below a count threshold or that contribute little to perplexity. Entropy pruning selectively removes n-grams where backing off loses minimal information.

---

## Summary

N-gram language models estimate the probability of word sequences by decomposing via the chain rule and applying the Markov assumption to limit context. Maximum likelihood estimation from corpus counts provides the foundation, but **sparsity**—the fact that most n-grams never appear in training—necessitates smoothing.

**Smoothing methods** redistribute probability mass from observed to unobserved events:

- **Laplace (add-one)**: Simple but over-smooths; Bayesian interpretation as uniform prior
- **Add-k**: Tunable smoothing; Bayesian interpretation with Dirichlet prior
- **Good-Turing**: Uses frequency-of-frequencies to estimate mass for unseen events
- **Absolute discounting**: Subtracts constant from counts; matches empirical discount patterns

**Back-off** consults lower-order models when higher-order counts are unavailable:

- **Stupid back-off**: Simple, unnormalized; effective at scale
- **Katz back-off**: Combines Good-Turing with normalized back-off

**Interpolation** combines multiple orders simultaneously, with weights learned from held-out data.

**Kneser-Ney smoothing** achieves the best performance by combining absolute discounting with **continuation probability**—using how many contexts a word appears in, rather than raw frequency, for back-off distributions.

While neural language models have largely supplanted n-gram models for generation tasks, understanding n-gram models illuminates fundamental concepts in language modeling: the tradeoff between model expressiveness and data requirements, the relationship between counts and probabilities, and the critical importance of handling sparsity. These concepts carry forward directly into neural approaches, where similar challenges manifest in different forms.
