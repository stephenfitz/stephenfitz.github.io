# GloVe: Global Vectors for Word Representation

In the preceding article on spectral word embeddings, we constructed word vectors by measuring distributional context overlap, building a similarity graph, and computing the eigenvectors of the graph Laplacian. That approach works—it produces embeddings where similar words are nearby—but it discards information at every step. The Dice coefficient reduces rich co-occurrence counts to a single similarity score. The Laplacian eigendecomposition finds directions of minimal variation, but it does not directly model the statistical structure of the co-occurrence data itself.

**GloVe** (Global Vectors for Word Representation), introduced by Pennington, Socher, and Manning (2014), takes a different approach. Rather than converting co-occurrence counts into similarities and then into a graph, GloVe works directly with the co-occurrence matrix. It asks: can we find word vectors whose dot products approximate the logarithm of how often two words co-occur? The answer leads to a weighted least-squares objective that factorizes the log-count matrix into low-rank word vectors. Despite sometimes being grouped with neural methods like Word2Vec, GloVe is fundamentally a **matrix factorization** method—it belongs squarely in the count-based tradition alongside LSA and PMI-SVD.

The key innovation is not the matrix factorization itself, but the *motivation* for it. GloVe derives its objective function from a simple observation about co-occurrence probability ratios: the ratio $P(k \mid \text{ice}) / P(k \mid \text{steam})$ tells us far more about the relationship between "ice" and "steam" than either probability alone. This ratio-based reasoning leads, through a sequence of algebraic steps, to the specific form of the GloVe objective—making it one of the few embedding methods whose loss function has a principled derivation rather than being chosen by intuition.

---

## The Co-occurrence Matrix

### Counting Co-occurrences

The starting point for GloVe is a **word-word co-occurrence matrix** $X$, where $X_{ij}$ counts the number of times word $j$ appears in the context of word $i$. Context is defined by a symmetric window of fixed size around each occurrence of word $i$ in the corpus. If the window size is 10, then for each occurrence of word $i$, we count all words within 10 positions to the left and 10 positions to the right.

A common refinement is **distance weighting**: words closer to the target word contribute more to the count than words further away. With a window of size $L$, a context word at distance $d$ contributes $1/d$ to the count rather than 1. This reflects the intuition that immediately adjacent words are more informative about a word's meaning than words several positions away.

Formally, let $x_1, x_2, \ldots, x_T$ be the sequence of words in the corpus. For a window of size $L$ with distance weighting:

$$
X_{ij} = \sum_{t=1}^{T} \mathbf{1}[x_t = i] \sum_{\substack{l = -L \\ l \neq 0}}^{L} \frac{\mathbf{1}[x_{t+l} = j]}{|l|}
$$

The matrix $X$ is symmetric: $X_{ij} = X_{ji}$. This follows from the symmetry of the context window—if word $j$ appears in the context of word $i$, then word $i$ appears in the context of word $j$, with the same distance weight.

### From Counts to Probabilities

From the co-occurrence matrix, we define two derived quantities. The **row sum** $X_i = \sum_k X_{ik}$ is the total number of context words observed for word $i$. The **co-occurrence probability** is:

$$
P_{ij} = P(j \mid i) = \frac{X_{ij}}{X_i}
$$

This is the probability that word $j$ appears in the context of word $i$, estimated by relative frequency. It is the same distributional signal that spectral embedding captures through context set overlap, but here we retain the full probability rather than reducing it to a binary similarity score.

### Connection to Spectral Embedding

Both spectral embedding and GloVe start from co-occurrence statistics. Spectral embedding converts these into a similarity matrix (via the Dice coefficient), constructs a graph Laplacian, and finds its eigenvectors. GloVe skips the similarity computation entirely and works directly with the co-occurrence matrix—or more precisely, with its logarithm. This directness is an advantage: by preserving the full count information, GloVe can model fine-grained distinctions that a binary similarity measure would flatten.

---

## From Counts to Ratios

### Why Ratios Matter

The core insight of GloVe is that raw co-occurrence probabilities are hard to interpret, but their **ratios** are highly informative. Consider two words, "ice" and "steam," and ask: which probe words $k$ discriminate between them?

The probability $P(k \mid \text{ice})$ tells us how often $k$ appears near "ice," but this conflates information about $k$'s relationship to "ice" with $k$'s overall frequency. A common word like "the" will have high $P(\text{the} \mid \text{ice})$ and high $P(\text{the} \mid \text{steam})$, but this tells us nothing about the difference between ice and steam.

The ratio $P(k \mid \text{ice}) / P(k \mid \text{steam})$ cancels out this noise. If $k$ is related to ice but not steam, the ratio is large. If $k$ is related to steam but not ice, the ratio is small. If $k$ is related to both or neither, the ratio is close to 1.

### The Ice and Steam Example

Pennington et al. (2014) illustrate this with concrete co-occurrence statistics from a large corpus:

| Probe word $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | Ratio |
|---|---|---|---|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | 8.9 |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | 0.085 |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | 1.36 |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | 0.96 |

Four cases emerge:

**Ratio $\gg 1$ (solid).** The probe word is strongly associated with "ice" but not "steam." The ratio captures this asymmetry cleanly.

**Ratio $\ll 1$ (gas).** The probe word is strongly associated with "steam" but not "ice." The ratio is small, indicating the opposite asymmetry.

**Ratio $\approx 1$, both large (water).** The probe word is related to both "ice" and "steam." The individual probabilities are both high, but the ratio is near 1 because the association is symmetric.

**Ratio $\approx 1$, both small (fashion).** The probe word is unrelated to both. Again the ratio is near 1, but for a different reason—neither probability is meaningful.

The ratio provides a clear signal for specific associations (Cases 1 & 2) and a neutral signal (~1) for non-discriminative ones (Cases 3 & 4). This allows the model to ignore global word frequency and focus on the relative associations that define word meaning. This is the key observation: **probability ratios encode word relationships more cleanly than probabilities themselves**.

### Formalizing the Insight

We want a model that captures these ratios. Specifically, we want a function $F$ of word vectors that satisfies:

$$
F(w_i, w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}
$$

where $w_i$ and $w_j$ are the word vectors for the two target words, and $\tilde{w}_k$ is the context vector for the probe word. The tilde notation distinguishes **context vectors** from **word vectors**—GloVe maintains two separate sets of vectors, one for words appearing as targets and one for words appearing as contexts.

---

## Deriving the Objective Function

### Step 1: Encode Ratios in Vector Differences

Since the ratio $P_{ik}/P_{jk}$ depends on the *difference* between words $i$ and $j$ (how one relates to probe $k$ compared to the other), we encode this via vector subtraction:

$$
F(w_i - w_j, \tilde{w}_k) = \frac{P_{ik}}{P_{jk}}
$$

The function $F$ now takes a vector difference and a context vector as arguments.

### Step 2: Dot Product to Get a Scalar

The left side of our equation should produce a scalar (the right side is a scalar ratio). The most natural way to combine two vectors into a scalar is the dot product:

$$
F\!\left((w_i - w_j)^\top \tilde{w}_k\right) = \frac{P_{ik}}{P_{jk}}
$$

Now $F$ is a function from $\R$ to $\R_{>0}$.

### Step 3: The Homomorphism Requirement

The co-occurrence matrix is symmetric ($X_{ij} = X_{ji}$), so the roles of word and context should be interchangeable. This imposes a structural constraint: the function $F$ must be a **homomorphism** from $(\R, +)$ to $(\R_{>0}, \times)$. That is, $F$ must satisfy:

$$
F(a + b) = F(a) \cdot F(b)
$$

The unique continuous solution is the exponential function: $F = \exp$.

**Why does symmetry require this?** Expanding the left side:

$$
F\!\left((w_i - w_j)^\top \tilde{w}_k\right) = F\!\left(w_i^\top \tilde{w}_k - w_j^\top \tilde{w}_k\right)
$$

And the right side is a ratio:

$$
\frac{P_{ik}}{P_{jk}}
$$

If $F$ is a homomorphism, then $F(a - b) = F(a)/F(b)$, which gives us $F(w_i^\top \tilde{w}_k) / F(w_j^\top \tilde{w}_k) = P_{ik}/P_{jk}$. This means we can identify $F(w_i^\top \tilde{w}_k) = P_{ik}$ for each $(i,k)$ pair independently—and the word and context roles become interchangeable.

To enforce symmetry between words and contexts, we must be able to express the individual probability $P_{ik}$ purely as a function of their dot product, say $P_{ik} = G(w_i^\top \tilde{w}_k)$. The simplest and most parsimonious choice that preserves this structural symmetry is to assume $G$ is the same function as our original function $F$, yielding $F(w_i^\top \tilde{w}_k - w_j^\top \tilde{w}_k) = \frac{F(w_i^\top \tilde{w}_k)}{F(w_j^\top \tilde{w}_k)}$. For this equivalence to hold, $F$ must systematically transform a subtraction in its input into a division in its output. This strictly requires $F$ to be a group homomorphism between the additive group $(\mathbb{R}, +)$ and the multiplicative group $(\mathbb{R}_{>0}, \times)$. This homomorphism is the *only* mathematical mechanism that can cleanly separate the terms for word $i$ from word $j$; if we used any other function (like a polynomial), the terms would remain inextricably tangled (e.g., $(A-B)^2 \neq A^2 / B^2$). The unique continuous solution that satisfies this strict homomorphism requirement is the exponential function, $F(x) = \exp(x)$.

### Step 4: Apply the Exponential

Setting $F = \exp$:

$$
\exp(w_i^\top \tilde{w}_k) = P_{ik} = \frac{X_{ik}}{X_i}
$$

Taking logarithms of both sides:

$$
w_i^\top \tilde{w}_k = \log P_{ik} = \log X_{ik} - \log X_i
$$

### Step 5: Absorb the Normalization

The term $\log X_i$ on the right depends only on word $i$, not on the context word $k$. We absorb it into a **bias term** $b_i$ associated with word $i$. For symmetry, we also add a bias $\tilde{b}_k$ for the context word:

$$
w_i^\top \tilde{w}_k + b_i + \tilde{b}_k = \log X_{ik}
$$

This is the **GloVe model equation**. It says that the dot product of a word vector and a context vector, plus bias terms, should approximate the logarithm of their co-occurrence count.

### Step 6: The Weighted Least-Squares Objective

The model equation will not hold exactly—it is an approximation. GloVe minimizes the squared error, summed over all observed co-occurrences, with a weighting function $f(X_{ik})$:

$$
J = \sum_{i,j=1}^{|V|} f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

The weighting function $f$ serves two purposes. First, it ensures that pairs with $X_{ij} = 0$ do not contribute to the loss (since $\log 0$ is undefined). Second, it prevents extremely frequent co-occurrences from dominating the objective—without weighting, function words like "the" and "of" would overwhelm the signal from content words.

---

## The Weighting Function

### Design Requirements

The weighting function $f: [0, \infty) \to [0, \infty)$ must satisfy three conditions:

1. **$f(0) = 0$.** If two words never co-occur, the pair should not contribute to the loss. This also avoids evaluating $\log 0$.

2. **Non-decreasing.** More frequent co-occurrences should receive at least as much weight as less frequent ones—they carry more statistical evidence.

3. **Bounded for large $x$.** Extremely frequent co-occurrences (mostly involving function words) should not dominate. The weight should saturate beyond some threshold.

### The GloVe Weighting Function

Pennington et al. propose the following piecewise function:

$$
f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}
$$

For co-occurrence counts below the threshold $x_{\max}$, the weight grows as a power law. Above the threshold, the weight is capped at 1.

### Parameter Choices

The GloVe paper finds that the results are relatively insensitive to the exact values, but recommends:

- **$x_{\max} = 100$**: co-occurrence counts above 100 are considered "frequent enough" and receive full weight.
- **$\alpha = 3/4$**: the sublinear exponent. This was found to outperform the linear case $\alpha = 1$ in experiments.

**Why $\alpha = 3/4$?** A linear weighting ($\alpha = 1$) would give weight proportional to the count, which still allows high-frequency pairs to dominate. The sublinear exponent $3/4$ compresses the range: a pair with count 100 gets weight 1, but a pair with count 50 gets weight $(50/100)^{3/4} \approx 0.59$ rather than $0.50$. This gives moderate-frequency co-occurrences relatively more influence, which turns out to produce better embeddings.

### Why Weighting Matters

Without weighting ($f \equiv 1$ for all nonzero entries), the loss function is dominated by the most frequent co-occurrences. In English, these are pairs involving function words: "the–of," "the–and," "in–the," and so on. These pairs carry little semantic information—they tell us that function words are common, which we already know. The weighting function downweights these uninformative pairs relative to rarer but more semantically meaningful co-occurrences like "ice–solid" or "king–crown."

---

## Optimization

### Parameters

The GloVe model has four sets of parameters:

- **Word vectors** $w_i \in \R^d$ for $i = 1, \ldots, |V|$
- **Context vectors** $\tilde{w}_j \in \R^d$ for $j = 1, \ldots, |V|$
- **Word biases** $b_i \in \R$ for $i = 1, \ldots, |V|$
- **Context biases** $\tilde{b}_j \in \R$ for $j = 1, \ldots, |V|$

The total number of parameters is $2|V|d + 2|V| = 2|V|(d + 1)$. For a vocabulary of 400,000 words and embedding dimension $d = 300$, this is about 240 million parameters.

### Gradient Computation

The loss for a single co-occurrence pair $(i, j)$ is:

$$
J_{ij} = f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2
$$

Let $\Delta_{ij} = w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$ denote the residual. The gradients are:

$$
\frac{\partial J_{ij}}{\partial w_i} = 2\, f(X_{ij})\, \Delta_{ij}\, \tilde{w}_j
$$

$$
\frac{\partial J_{ij}}{\partial \tilde{w}_j} = 2\, f(X_{ij})\, \Delta_{ij}\, w_i
$$

$$
\frac{\partial J_{ij}}{\partial b_i} = 2\, f(X_{ij})\, \Delta_{ij}
$$

$$
\frac{\partial J_{ij}}{\partial \tilde{b}_j} = 2\, f(X_{ij})\, \Delta_{ij}
$$

The gradient for a word vector $w_i$ is the context vector $\tilde{w}_j$ scaled by the weighted residual, and symmetrically for the context vector.

### AdaGrad

GloVe uses **AdaGrad** (Duchi et al., 2011) for optimization, which adapts the learning rate for each parameter based on the history of its gradients. For a parameter $\theta$ with gradient $g_t$ at iteration $t$:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}}\, g_t
$$

where $G_t = \sum_{\tau=1}^{t} g_\tau^2$ is the accumulated squared gradient and $\eta$ is the initial learning rate. Parameters that receive large gradients frequently (common words) get smaller effective learning rates; parameters that receive small gradients infrequently (rare words) get larger effective learning rates. This is well-suited to GloVe because the gradient magnitudes vary enormously across the vocabulary.

### A Batch Method

An important distinction between GloVe and neural methods like Word2Vec is that GloVe is a **batch** method. The full co-occurrence matrix $X$ is computed once from the corpus before any optimization begins. Training then iterates over the nonzero entries of $X$, computing the loss and gradients for each pair. There is no concept of "seeing a word in context" during training—all the contextual information has already been distilled into counts.

This has practical implications. The co-occurrence matrix must fit in memory (or be stored on disk and streamed), and its construction requires a full pass over the corpus. But once built, training is fast: each iteration sweeps over the sparse matrix $X$, and convergence typically occurs within 50–100 passes. The separation of counting and optimization reinforces GloVe's identity as a count-based method.

### Combining Word and Context Vectors

After training, GloVe has two sets of vectors for each word: the word vector $w_i$ and the context vector $\tilde{w}_i$. Since the co-occurrence matrix is symmetric ($X_{ij} = X_{ji}$), these two sets of vectors are theoretically interchangeable—they differ only due to random initialization.

The final embedding is their sum:

$$
w_i^{\text{final}} = w_i + \tilde{w}_i
$$

Pennington et al. report that this averaging consistently improves performance over using either set alone. The intuition is that while $w_i$ and $\tilde{w}_i$ capture the same distributional information, they learn slightly different aspects due to the asymmetry of the optimization procedure. Summing them produces a more robust representation.

---

## Relationship to Matrix Factorization

### The Core Factorization

Strip away the biases and the GloVe model equation reads:

$$
w_i^\top \tilde{w}_j \approx \log X_{ij}
$$

In matrix form, let $W \in \R^{|V| \times d}$ be the matrix of word vectors and $\tilde{W} \in \R^{|V| \times d}$ be the matrix of context vectors. Then:

$$
W\tilde{W}^\top \approx \log X
$$

This is a **low-rank matrix factorization** of the log-count matrix. The embedding dimension $d$ determines the rank of the approximation. GloVe seeks the best rank-$d$ factorization of $\log X$ under the weighted least-squares criterion defined by $f(X_{ij})$.

### Comparison to SVD and LSA

Latent Semantic Analysis (LSA) also factorizes a co-occurrence matrix, but the details differ in three important ways:

**Log transform.** LSA typically operates on raw counts (possibly weighted by TF-IDF). GloVe factorizes the *logarithm* of the counts. The log transform compresses the dynamic range—a co-occurrence count of 10,000 becomes $\log 10{,}000 \approx 9.2$, while a count of 1 becomes $\log 1 = 0$. This prevents extremely frequent co-occurrences from dominating the factorization.

**Weighting.** LSA treats all entries of the matrix equally (or uses a fixed TF-IDF weighting). GloVe applies the learned weighting function $f(X_{ij})$, which gives zero weight to unobserved co-occurrences and bounded weight to extremely frequent ones.

**Optimization method.** LSA uses the singular value decomposition (SVD), which finds the globally optimal low-rank approximation under the unweighted Frobenius norm. GloVe uses stochastic gradient descent (AdaGrad), which finds a local minimum of the weighted objective. The SVD solution is unique; the GloVe solution depends on initialization. However, GloVe's weighted objective is arguably more appropriate for the data, since not all entries of the co-occurrence matrix are equally reliable.

### Connection to PMI

The log-count that GloVe targets can be decomposed in terms of **pointwise mutual information** (PMI):

$$
\text{PMI}(i, j) = \log \frac{P(i, j)}{P(i)\,P(j)} = \log \frac{X_{ij} \cdot |\text{corpus}|}{X_i \cdot X_j}
$$

Rearranging:

$$
\log X_{ij} = \text{PMI}(i, j) + \log X_i + \log X_j - \log |\text{corpus}|
$$

In the GloVe objective, the bias terms $b_i$ and $\tilde{b}_j$ can absorb the word-specific terms $\log X_i$ and $\log X_j$ (and the corpus-size constant). This means that the word vectors $w_i^\top \tilde{w}_j$ effectively approximate the PMI, up to terms that are absorbed by the biases.

This connection was highlighted by Levy and Goldberg (2014), who showed that Word2Vec's skip-gram model with negative sampling implicitly factorizes a shifted PMI matrix. GloVe and skip-gram Word2Vec are thus approximating the same underlying quantity—the PMI—through different computational means. The count-based and prediction-based paradigms converge on the same target.

---

## Worked Example

### A Small Corpus

We use the same toy corpus from the spectral embedding article, to illustrate how GloVe processes the same data differently:

1. "the cat sat on the mat"
2. "the dog sat on the rug"
3. "a cat chased a dog"
4. "the bird sat on the branch"
5. "a dog chased a bird"

We use a symmetric context window of size 1 (immediate neighbors only), with no distance weighting.

### Building the Co-occurrence Matrix

For each word in the corpus, we count how many times each other word appears immediately to its left or right. Scanning through the corpus:

In sentence 1 ("the cat sat on the mat"): "the" and "cat" co-occur, "cat" and "sat" co-occur, "sat" and "on" co-occur, "on" and "the" co-occur, "the" and "mat" co-occur.

Continuing through all five sentences and accumulating counts, we obtain the co-occurrence matrix $X$. Here is a portion for the content words:

|  | cat | dog | bird | sat | chased |
|--|-----|-----|------|-----|--------|
| cat | 0 | 1 | 0 | 1 | 1 |
| dog | 1 | 0 | 1 | 1 | 1 |
| bird | 0 | 1 | 0 | 1 | 1 |
| sat | 1 | 1 | 1 | 0 | 0 |
| chased | 1 | 1 | 1 | 0 | 0 |

Notice the symmetry: $X_{\text{cat, dog}} = X_{\text{dog, cat}} = 1$.

### Computing Probabilities and Ratios

The row sums are: $X_{\text{cat}} = 3$, $X_{\text{dog}} = 4$, $X_{\text{bird}} = 3$. The co-occurrence probabilities for "cat" and "bird" are:

$$
P(\text{sat} \mid \text{cat}) = \frac{1}{3}, \quad P(\text{sat} \mid \text{bird}) = \frac{1}{3}
$$

$$
P(\text{dog} \mid \text{cat}) = \frac{1}{3}, \quad P(\text{dog} \mid \text{bird}) = \frac{1}{3}
$$

The ratios are:

$$
\frac{P(\text{sat} \mid \text{cat})}{P(\text{sat} \mid \text{bird})} = \frac{1/3}{1/3} = 1.0
$$

$$
\frac{P(\text{dog} \mid \text{cat})}{P(\text{dog} \mid \text{bird})} = \frac{1/3}{1/3} = 1.0
$$

The ratios are 1.0 across all probes, confirming that "cat" and "bird" have identical distributional profiles in this corpus. GloVe would learn very similar vectors for them.

### GloVe Loss Terms

For the pair (cat, sat) with $X_{\text{cat, sat}} = 1$:

$$
f(1) = \left(\frac{1}{100}\right)^{3/4} = 0.01^{0.75} \approx 0.032
$$

$$
J_{\text{cat, sat}} = 0.032 \cdot \left(w_{\text{cat}}^\top \tilde{w}_{\text{sat}} + b_{\text{cat}} + \tilde{b}_{\text{sat}} - \log 1\right)^2 = 0.032 \cdot \left(w_{\text{cat}}^\top \tilde{w}_{\text{sat}} + b_{\text{cat}} + \tilde{b}_{\text{sat}}\right)^2
$$

Since $\log 1 = 0$, the model is driven to make $w_{\text{cat}}^\top \tilde{w}_{\text{sat}} + b_{\text{cat}} + \tilde{b}_{\text{sat}} \approx 0$.

For a pair with count 0, like (cat, bird), $f(0) = 0$, so the pair contributes nothing to the loss. GloVe only trains on observed co-occurrences.

---

## Connection to Other Methods

### The Count-Based Spectrum

GloVe occupies a specific position in the lineage of count-based embedding methods:

**LSA** (Deerwester et al., 1990). Factorizes the raw (or TF-IDF weighted) term-document matrix via SVD. Does not use word-word co-occurrence or log transforms.

**PMI-SVD** (Levy and Goldberg, 2014; Church and Hanks, 1990). Constructs a word-word PMI matrix and factorizes it via SVD. The PMI transform replaces raw counts with a measure of association strength.

**Spectral embedding.** Converts co-occurrence counts into a similarity graph and computes the Laplacian eigenvectors. Works through the graph rather than directly with counts.

**GloVe** (Pennington et al., 2014). Factorizes the log-count co-occurrence matrix via weighted least squares. The log transform and learned weighting function are the distinguishing features.

Each method in this progression refines the treatment of co-occurrence statistics: from raw counts (LSA), to PMI-transformed counts (PMI-SVD), to similarity-based spectral methods, to the log-weighted factorization of GloVe.

### Forward: Prediction-Based Methods

**Word2Vec** (Mikolov et al., 2013), covered in a later article, takes a fundamentally different approach. Rather than constructing a co-occurrence matrix and factorizing it, Word2Vec trains a neural network to *predict* context words from target words (or vice versa). Yet as Levy and Goldberg (2014) showed, Word2Vec's skip-gram with negative sampling implicitly factorizes a shifted PMI matrix—the same quantity that GloVe targets explicitly. The two methods are computational duals: GloVe counts first and then optimizes, while Word2Vec optimizes directly on the corpus.

### The Count-Based / Prediction-Based Bridge

GloVe was introduced in part to bridge the gap between count-based and prediction-based methods. The authors argued that both families are fundamentally doing the same thing—capturing co-occurrence statistics—and that the distinction lies in the computational approach, not the underlying signal. GloVe demonstrates this by deriving a principled objective function from co-occurrence ratios (a count-based perspective) and then optimizing it with stochastic gradient descent (a technique borrowed from neural network training). It is count-based in its data, but prediction-like in its optimization.

---

## Summary

GloVe derives word embeddings by factorizing the log of the word-word co-occurrence matrix through a weighted least-squares objective. The method is motivated by a single key observation: **co-occurrence probability ratios** encode word relationships more cleanly than raw probabilities.

- The **co-occurrence matrix** $X_{ij}$ counts how often word $j$ appears in the context of word $i$, using a symmetric window over the corpus
- **Probability ratios** $P(k \mid i) / P(k \mid j)$ discriminate word relationships: large for $k$ related to $i$ only, small for $k$ related to $j$ only, near 1 otherwise
- The requirement that word vectors encode these ratios, combined with a symmetry constraint, leads uniquely to the **model equation**: $w_i^\top \tilde{w}_k + b_i + \tilde{b}_k = \log X_{ik}$
- The **weighting function** $f(x) = \min\!\left((x/x_{\max})^\alpha, 1\right)$ prevents function words from dominating and excludes zero counts
- **Optimization** uses AdaGrad over the nonzero entries of $X$; the final embedding is $w_i + \tilde{w}_i$
- In matrix form, GloVe computes a **low-rank factorization**: $W\tilde{W}^\top \approx \log X$
- The biases absorb word-frequency terms, so the vectors approximate **pointwise mutual information**

GloVe sits at the boundary between count-based and prediction-based methods. It uses the full co-occurrence matrix—a count-based object—but optimizes with gradient descent, a technique characteristic of neural methods. A later article develops Word2Vec, which approaches the same underlying statistical target from the prediction side: instead of counting co-occurrences first, it learns embeddings by predicting context words directly from the corpus.
