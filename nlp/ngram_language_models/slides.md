# N-gram Language Models

---

## Outline

- The language modeling problem
- The Markov assumption
- Maximum likelihood estimation
- The sparsity problem
- Smoothing methods (Laplace, add-k, Good-Turing)
- Back-off and interpolation
- Kneser-Ney smoothing
- Perplexity
- The necessity of EOS

---

## Part I: The Problem

---

## What Is a Language Model?

Assigns probabilities to word sequences:

$$P(w_1, w_2, \ldots, w_n)$$

"the cat sat on the mat" → high probability

"mat the on sat cat the" → low probability

---

## Applications

- Speech recognition
- Machine translation
- Spelling correction
- Text generation

---

## The Chain Rule

Any joint probability decomposes exactly:

$$P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \ldots, w_{i-1})$$

Predict each word given all previous words

---

## The Curse of History

To compute $P(w_n \mid w_1, \ldots, w_{n-1})$:

- 20-word sentence → condition on 19 words
- Vocabulary of 50,000 → $50000^{19}$ possible histories
- More than atoms in the universe

We'll never observe most histories

---

## Part II: The Markov Assumption

---

## Truncating History

Assume next word depends only on previous $k$ words:

$$P(w_i \mid w_1, \ldots, w_{i-1}) \approx P(w_i \mid w_{i-k}, \ldots, w_{i-1})$$

Linguistically imperfect, but tractable

---

## N-gram Models

| Model | Conditions on |
|-------|---------------|
| Unigram | Nothing: $P(w_i)$ |
| Bigram | Previous word: $P(w_i \mid w_{i-1})$ |
| Trigram | Previous 2: $P(w_i \mid w_{i-2}, w_{i-1})$ |
| 4-gram | Previous 3 |

---

## Why Markov Works

**Information-theoretic**: Most information in recent context

**Statistical**: Bias-variance tradeoff — longer history = more variance

**Empirical**: Diminishing returns beyond $n=5$

---

## Part III: Maximum Likelihood Estimation

---

## Counting N-grams

Estimate probabilities by counting:

$$P_{\text{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

Bigram count over context count

---

## General MLE Formula

$$P_{\text{MLE}}(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) = \frac{C(w_{i-n+1}, \ldots, w_i)}{C(w_{i-n+1}, \ldots, w_{i-1})}$$

Relative frequency is the MLE

---

## Example

Corpus: "the cat sat. the cat ate. the dog sat."

| Bigram | Count |
|--------|-------|
| (the, cat) | 2 |
| (the, dog) | 1 |
| (cat, sat) | 1 |
| (cat, ate) | 1 |

---

## Example Probabilities

$$P(\text{cat} \mid \text{the}) = \frac{2}{3}$$

$$P(\text{dog} \mid \text{the}) = \frac{1}{3}$$

$$P(\text{sat} \mid \text{cat}) = \frac{1}{2}$$

---

## Part IV: The Sparsity Problem

---

## Zero Counts

If "the elephant" never appears:

$$P_{\text{MLE}}(\text{elephant} \mid \text{the}) = \frac{0}{C(\text{the})} = 0$$

Any sentence with "the elephant" gets probability **zero**

---

## Why Sparsity Is Catastrophic

Sentence probability is a product:

$$P(w_1, \ldots, w_n) = \prod_i P(w_i \mid w_{i-1})$$

One zero → entire product is zero

Also: $\log 0 = -\infty$ breaks perplexity

---

## Scale of Sparsity

With vocabulary $V = 50{,}000$:

| N-gram | Possible | Typical coverage |
|--------|----------|------------------|
| Bigram | $5 \times 10^4$ | 10-30% |
| Trigram | $2.5 \times 10^9$ | 0.1-1% |
| 4-gram | $1.25 \times 10^{14}$ | 0.001% |

Most n-grams **never observed**

---

## The Bayesian View

Zero count ≠ impossible

It means: **lack of evidence**

MLE conflates "unobserved" with "impossible"

---

## Part V: Smoothing

---

## The Core Idea

**Smoothing**: Take mass from observed n-grams, give to unobserved

Must still sum to 1:

$$\sum_{w} P(w \mid \text{context}) = 1$$

---

## Desiderata

1. Assign nonzero probability to all n-grams
2. Preserve relative ordering
3. Probabilities sum to 1
4. Improve perplexity on held-out data

---

## Laplace (Add-One) Smoothing

Add 1 to every count:

$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + |V|}$$

Bayesian interpretation: uniform Dirichlet prior

---

## Laplace Example

Without smoothing: $P(\text{cat} \mid \text{the}) = \frac{2}{3} = 0.667$

With Laplace ($|V|=5$): $P = \frac{2+1}{3+5} = \frac{3}{8} = 0.375$

For unseen (the, the): $P = \frac{0+1}{3+5} = 0.125$

---

## Problems with Laplace

- **Excessive discounting**: 0.667 → 0.375 (nearly halved!)
- **Uniform for unseen**: "the elephant" = "the the"?
- **Scale dependent**: Large $|V|$ → severe discount
- **Poor perplexity** in practice

---

## Add-k Smoothing

Generalize with fractional $k < 1$:

$$P_{\text{add-}k}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + k|V|}$$

Choose $k$ to minimize validation perplexity

---

## Part VI: Good-Turing Smoothing

---

## The Key Insight

N-grams seen **once** tell us about n-grams seen **zero** times

If many appear once, expect many unseen to appear once in new data

---

## Notation

$N_r$ = number of n-grams appearing exactly $r$ times

- $N_0$: unseen n-grams
- $N_1$: appearing once (hapax legomena)
- $N_2$: appearing twice

---

## The Good-Turing Estimator

Replace count $r$ with adjusted count:

$$r^* = (r + 1) \frac{N_{r+1}}{N_r}$$

For unseen ($r=0$): $0^* = \frac{N_1}{N_0}$

---

## Elegant Result

Total mass for unseen events:

$$P(\text{unseen}) = \frac{N_1}{N}$$

**Mass for unseen = fraction of singletons**

---

## Part VII: Back-off Models

---

## The Intuition

When high-order n-gram unseen, consult lower-order

Never seen "colorless green ideas"?

Use $P(\text{ideas} \mid \text{green})$ or $P(\text{ideas})$

---

## Stupid Back-off

$$S(w_i \mid \text{context}) = \begin{cases} \text{MLE} & \text{if count} > 0 \\ \lambda \cdot S(w_i \mid \text{shorter}) & \text{otherwise} \end{cases}$$

$\lambda \approx 0.4$, scores don't sum to 1

Fast, used at Google scale

---

## Katz Back-off

Combine Good-Turing with normalized back-off:

$$P_{\text{Katz}} = \begin{cases} \frac{C^*}{C(\text{context})} & \text{if count} > 0 \\ \alpha \cdot P_{\text{Katz}}(\text{shorter}) & \text{otherwise} \end{cases}$$

$\alpha$ ensures probabilities sum to 1

---

## Part VIII: Interpolation

---

## Combining Multiple Orders

$$P_{\text{interp}}(w \mid u, v) = \lambda_3 P_3(w \mid u, v) + \lambda_2 P_2(w \mid v) + \lambda_1 P_1(w)$$

where $\lambda_1 + \lambda_2 + \lambda_3 = 1$

---

## Why Interpolate?

- Lower-order useful even when higher-order exists
- Robustness when counts are small
- Smooth degradation (no abrupt transitions)

Learn $\lambda$ with EM on held-out data

---

## Part IX: Absolute Discounting

---

## The Observation

Empirical finding (Church & Gale, 1991):

Optimal discount is **constant** across count values

Expected count ≈ $r - d$ where $d \approx 0.75$

---

## The Method

$$P_{\text{abs}}(w \mid v) = \frac{\max(C(v, w) - d, 0)}{C(v)} + \lambda(v) P(w)$$

Subtract fixed $d$ from each count

Redistribute via lower-order model

---

## Estimating the Discount

$$d = \frac{N_1}{N_1 + 2N_2}$$

Typically yields $d \approx 0.75$

---

## Part X: Kneser-Ney Smoothing

---

## The Problem

"Francisco" appears often (in "San Francisco")

Standard back-off: high $P(\text{Francisco})$

But "Francisco" rarely follows other contexts!

---

## Continuation Probability

Instead of raw frequency, count distinct contexts:

$$P_{\text{continuation}}(w) \propto |\{v : C(v, w) > 0\}|$$

"Francisco": few contexts → low continuation prob

"the": many contexts → high continuation prob

---

## Kneser-Ney Formula

$$P_{\text{KN}}(w \mid v) = \frac{\max(C(v, w) - d, 0)}{C(v)} + \lambda(v) P_{\text{continuation}}(w)$$

Absolute discounting + continuation probability

---

## Why It Works

1. **Absolute discounting**: matches empirical patterns
2. **Continuation probability**: captures word versatility

Modified Kneser-Ney: state-of-the-art for count-based LMs

---

## Part XI: Perplexity

---

## Definition

$$\text{PP}(W) = P(w_1, \ldots, w_N)^{-1/N}$$

Equivalently:

$$\text{PP} = 2^{H(W)}$$

where $H$ is cross-entropy

---

## Interpretation

**Effective vocabulary size** the model is uncertain among

Perplexity 100 → uncertain among ~100 words per step

**Lower is better**

---

## Typical Values

| Model | Perplexity |
|-------|------------|
| Unigram | 950 |
| Bigram | 170 |
| Trigram | 100 |
| 4-gram + KN | 80 |
| Neural LMs | < 20 |

---

## Part XII: The Necessity of EOS

---

## The Problem Without EOS

Sum over all 1-word sentences: 1

Sum over all 2-word sentences: 1

Sum over all sentences: $1 + 1 + 1 + \cdots = \infty$

**Not a valid probability distribution!**

---

## With EOS: Proper Distribution

At each step, model can:
- Generate word and continue
- Generate EOS and stop

$$P(\text{EOS} \mid \text{context}) + \sum_{w} P(w \mid \text{context}) = 1$$

---

## Example

Without EOS: $P(a) = 0.6$, $P(b) = 0.4$

- 1-word sentences sum to 1
- 2-word sentences sum to 1
- Total: $\infty$

With EOS: $P(a) = 0.4$, $P(b) = 0.3$, $P(\text{EOS}) = 0.3$

- Total: $\sum_{n=1}^{\infty} 0.3 \times 0.7^{n-1} = 1$ ✓

---

## EOS vs. BOS

**BOS**: conditioning context (right side of $\mid$)

$$P(w_1 \mid \text{BOS})$$

**EOS**: predicted token (left side of $\mid$)

$$P(\text{EOS} \mid w_n)$$

Asymmetric roles!

---

## Deficient Models

Without EOS: model is **deficient**

- Total probability < 1 over finite sentences
- "Missing" mass on infinite sequences
- Perplexity undefined

---

## Summary

- **Chain rule** decomposes joint into conditionals
- **Markov assumption** truncates history
- **MLE** = relative frequency (but zero counts!)
- **Sparsity** is the dominant challenge
- **Smoothing**: Laplace → add-k → Good-Turing → absolute
- **Back-off**: consult lower-order when needed
- **Kneser-Ney**: absolute discounting + continuation probability
- **EOS** is mathematically essential
