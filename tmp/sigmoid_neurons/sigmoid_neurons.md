# Sigmoid Neurons and Logistic Regression

The perceptron article introduced the simplest neural network: a single neuron that computes $y = \sgn(w^\top x)$ and learns by flipping its decision boundary whenever it makes a mistake. The perceptron is elegant, its convergence theorem is beautiful, and it works—but it has two fundamental limitations. First, the sign function is discontinuous: the output jumps from $-1$ to $+1$ at $w^\top x = 0$, producing no gradient almost everywhere and an undefined gradient at the boundary. Second, the perceptron produces a hard binary decision, not a probability. It cannot express that it is 90% confident an email is spam—only that it is or isn't.

The **sigmoid neuron** resolves both limitations by replacing the sign activation with the **sigmoid function** $\sigma(z) = 1/(1 + e^{-z})$, which smoothly maps any real number to the interval $(0, 1)$. This seemingly small change has profound consequences: the output can now be interpreted as a probability, the loss function can be derived from first principles via maximum likelihood estimation, and the resulting gradient is everywhere well-defined and smooth. The model that results—a single sigmoid neuron trained on labeled data—is known as **logistic regression**, one of the most widely used classification algorithms in statistics and machine learning.

This article develops the sigmoid neuron as a **probabilistic model**. We begin with the sigmoid function and its properties, then frame the neuron's output as a Bernoulli probability. The cross-entropy loss function emerges naturally as the negative log-likelihood—that is, as the **surprisal** of the training data under the model. We derive the gradient of this loss in full detail, arriving at a remarkably simple update rule that can be expressed in a single line of linear algebra.

---

## The Sigmoid Function

### Definition

The **sigmoid function** (also called the **logistic function**) is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

An equivalent form, obtained by multiplying numerator and denominator by $e^z$, is:

$$
\sigma(z) = \frac{e^z}{e^z + 1}
$$

![The sigmoid function maps any real number to the interval $(0, 1)$, with $\sigma(0) = 1/2$. The function saturates toward 0 for large negative inputs and toward 1 for large positive inputs.](figures/sigmoid_function.png)

### Key Properties

The sigmoid has several properties that make it ideal as an activation function for probabilistic classification:

1. **Range**: $\sigma(z) \in (0, 1)$ for all $z \in \R$. The output can be interpreted as a probability.

2. **Monotonicity**: $\sigma$ is strictly increasing. Larger inputs produce larger outputs.

3. **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$. This follows directly from the definition:

$$
\sigma(-z) = \frac{1}{1 + e^{z}} = \frac{e^{-z}}{e^{-z} + 1} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z)
$$

4. **Boundary behavior**:

$$
\lim_{z \to -\infty} \sigma(z) = 0, \qquad \lim_{z \to +\infty} \sigma(z) = 1, \qquad \sigma(0) = \frac{1}{2}
$$

5. **Differentiability**: $\sigma$ is infinitely differentiable everywhere, unlike the sign function which is discontinuous at zero.

### The Derivative of the Sigmoid

The derivative of the sigmoid has a particularly elegant form:

$$
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)
$$

To derive this, write $\sigma(z) = (1 + e^{-z})^{-1}$ and apply the chain rule:

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}}
$$

Since $1 - \sigma(z) = 1 - \frac{1}{1 + e^{-z}} = \frac{e^{-z}}{1 + e^{-z}}$, we have $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

![The sigmoid $\sigma(z)$ (blue) and its derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ (red). The derivative reaches its maximum of $1/4$ at $z = 0$ and decays symmetrically toward zero for large $|z|$.](figures/sigmoid_derivative.png)

The maximum of $\sigma'(z)$ occurs at $z = 0$, where $\sigma(0) = 1/2$ and $\sigma'(0) = 1/4$. This means the sigmoid is steepest at the decision boundary and progressively flatter in the "saturated" regions where the output is close to 0 or 1. This property will be important when we discuss the gradient of the loss function.

---

## From Perceptron to Sigmoid Neuron

### The Perceptron's Limitation

Recall the perceptron: given input $x \in \R^d$, it computes the weighted sum $z = w^\top x$ and applies the sign function to produce a binary output $y = \sgn(z) \in \{-1, +1\}$. The sign function is a "hard" activation—it makes an all-or-nothing decision with no notion of confidence.

### The Sigmoid Neuron

The sigmoid neuron replaces the sign activation with the sigmoid function. Given input $x \in \R^d$ (with bias absorbed as before), the neuron computes:

$$
\hat{y} = \sigma(w^\top x) = \frac{1}{1 + e^{-w^\top x}}
$$

The output $\hat{y} \in (0, 1)$ is no longer a discrete class label. Instead, we interpret it as the **probability** that the input belongs to class 1:

$$
\hat{y} = P(y = 1 \mid x)
$$

![Comparison of the perceptron's sign activation (left) and the sigmoid neuron's smooth activation (right). The perceptron makes a hard binary decision; the sigmoid neuron outputs a continuous probability.](figures/activation_comparison.png)

This probabilistic interpretation transforms the classification problem from a geometric one (find a separating hyperplane) into a statistical one (find the parameters of a probability model that best explain the observed data).

---

## The Probabilistic Model

### Labels and the Bernoulli Distribution

We adopt the convention that labels are $y^{(i)} \in \{0, 1\}$ rather than $\{-1, +1\}$. This is the natural choice for a probabilistic model: $y = 1$ means the example belongs to the positive class, $y = 0$ means it does not.

Given a dataset $D = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(n)}, y^{(n)})\}$ where each $x^{(i)} \in \R^d$ is a $d$-dimensional input and each $y^{(i)} \in \{0, 1\}$ is a binary label, the sigmoid neuron defines a probabilistic model:

$$
y^{(i)} \sim \text{Bernoulli}\bigl(\sigma(w^\top x^{(i)})\bigr)
$$

This means we model each label as a coin flip, where the probability of heads (class 1) depends on the input through the sigmoid:

$$
P(y^{(i)} = 1) = \sigma(w^\top x^{(i)}) = \frac{1}{1 + e^{-w^\top x^{(i)}}}
$$

$$
P(y^{(i)} = 0) = 1 - \sigma(w^\top x^{(i)})
$$

### Maximum Likelihood Estimation

The question is: how should we choose $w$? The probabilistic framework gives a principled answer. We want to find the weight vector $w$ that makes the observed training data **as probable as possible** under the model. That is, we want to maximize the probability that the sigmoid neuron assigns to its experience:

$$
\argmax_w \; P(D \mid w)
$$

This is **maximum likelihood estimation** (MLE): find the parameters that maximize the likelihood of the data.

### The Likelihood Function

Assuming the training examples are independent given $w$, the likelihood of the entire dataset factorizes as a product:

$$
P(D \mid w) = \prod_{i=1}^{n} p_i
$$

where $p_i$ is the probability that the model assigns to the **correct label** for example $i$. Since $y^{(i)} \in \{0, 1\}$, we can write this compactly using the Bernoulli probability mass function:

$$
p_i = \sigma(w^\top x^{(i)})^{y^{(i)}} \cdot \bigl(1 - \sigma(w^\top x^{(i)})\bigr)^{1 - y^{(i)}}
$$

To see why this works, check the two cases:

- When $y^{(i)} = 1$: $p_i = \sigma(w^\top x^{(i)})^1 \cdot (1 - \sigma(w^\top x^{(i)}))^0 = \sigma(w^\top x^{(i)})$. That is the probability of class 1.
- When $y^{(i)} = 0$: $p_i = \sigma(w^\top x^{(i)})^0 \cdot (1 - \sigma(w^\top x^{(i)}))^1 = 1 - \sigma(w^\top x^{(i)})$. That is the probability of class 0.

---

## Deriving the Cross-Entropy Loss

### From Likelihood to Surprisal

Maximizing the likelihood $P(D \mid w) = \prod_i p_i$ is equivalent to minimizing the **negative log-likelihood**:

$$
\mathcal{L}(w) = -\log P(D \mid w)
$$

The logarithm turns the product into a sum, and the negation turns the maximization into a minimization:

$$
\mathcal{L}(w) = -\log \prod_{i=1}^{n} p_i = -\sum_{i=1}^{n} \log p_i
$$

This quantity has a natural interpretation from information theory: $-\log p_i$ is the **surprisal** (or self-information) of observing the correct label for example $i$. If the model assigns high probability to the correct label ($p_i \approx 1$), the surprisal is low ($-\log p_i \approx 0$). If the model assigns low probability to the correct label ($p_i \approx 0$), the surprisal is high ($-\log p_i \to \infty$). Minimizing the total surprisal means finding the model that is **least surprised** by the training data.

### Expanding the Loss

Substituting the Bernoulli expression for $p_i$:

$$
\mathcal{L}(w) = -\sum_{i=1}^{n} \log \left[\sigma(w^\top x^{(i)})^{y^{(i)}} \cdot \bigl(1 - \sigma(w^\top x^{(i)})\bigr)^{1 - y^{(i)}}\right]
$$

Using $\log(a^b \cdot c^d) = b \log a + d \log c$:

$$
\boxed{\mathcal{L}(w) = -\sum_{i=1}^{n} \left[y^{(i)} \log \sigma(w^\top x^{(i)}) + (1 - y^{(i)}) \log \bigl(1 - \sigma(w^\top x^{(i)})\bigr)\right]}
$$

This is the **binary cross-entropy loss** (also called the **log loss**). It did not come from nowhere—it is the unique loss function that arises from treating the sigmoid neuron as a Bernoulli probability model and performing maximum likelihood estimation. The cross-entropy loss is not a design choice; it is a consequence of the probabilistic interpretation.

### Intuition: The Two Cases

Examining the loss for a single example, with $\hat{y} = \sigma(w^\top x)$:

$$
L(\hat{y}, y) = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]
$$

**When $y = 1$**: The loss reduces to $L = -\log \hat{y}$. We want $\hat{y}$ close to 1 (high predicted probability for the positive class). As $\hat{y} \to 1$, $L \to 0$. As $\hat{y} \to 0$, $L \to \infty$.

**When $y = 0$**: The loss reduces to $L = -\log(1 - \hat{y})$. We want $\hat{y}$ close to 0 (low predicted probability for the positive class). As $\hat{y} \to 0$, $L \to 0$. As $\hat{y} \to 1$, $L \to \infty$.

![The cross-entropy loss for the two cases. Left: when $y = 1$, the loss is $-\log \hat{y}$, which penalizes low predicted probabilities. Right: when $y = 0$, the loss is $-\log(1 - \hat{y})$, which penalizes high predicted probabilities.](figures/cross_entropy_loss.png)

In both cases, the loss is zero when the prediction is perfect and grows without bound as the prediction approaches the wrong extreme. The logarithmic growth means that **confident wrong predictions are punished severely**—much more than mildly wrong predictions. This is a direct consequence of the surprisal interpretation: a model that assigns probability 0.01 to an event that actually occurs is very surprised indeed.

---

## Why Not Quadratic Loss?

A natural first instinct might be to use the quadratic (mean squared error) loss:

$$
L_{\text{quad}}(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2
$$

This doesn't work well with the sigmoid activation. The problem is that composing the quadratic loss with the sigmoid produces a **non-convex** function of the weights $w$. The loss landscape has local minima and flat regions where the gradient nearly vanishes, making optimization by gradient descent unreliable.

![local optima](figures/local_optima.png)

![energy landscape](figures/energy_landscape.png)

The cross-entropy loss, by contrast, is **convex** in $w$ when composed with the sigmoid. This means gradient descent is guaranteed to find the global minimum—there are no local minima to get trapped in. This convexity is not a coincidence: it is a general property of maximum likelihood estimation for exponential family distributions, of which the Bernoulli-sigmoid model is a special case.

---

## Deriving the Gradient

We now derive the gradient of the cross-entropy loss with respect to the weight vector $w$. This derivation follows the full calculation step by step, introducing a useful substitution that simplifies the algebra.

### Simplifying the Log Terms

Define the shorthand $a_i = \sigma(w^\top x^{(i)})$ for the activation (predicted probability) of example $i$, and define:

$$
\alpha_i = \log \sigma(w^\top x^{(i)}) = \log \frac{1}{1 + e^{-w^\top x^{(i)}}} = -\log(1 + e^{-w^\top x^{(i)}})
$$

This is the log-probability of the positive class. Now compute the log-probability of the negative class:

$$
\log(1 - a_i) = \log\left(1 - \frac{1}{1 + e^{-w^\top x^{(i)}}}\right) = \log\left(\frac{e^{-w^\top x^{(i)}}}{1 + e^{-w^\top x^{(i)}}}\right)
$$

$$
= -w^\top x^{(i)} - \log(1 + e^{-w^\top x^{(i)}}) = \alpha_i - w^\top x^{(i)}
$$

This is a key identity: $\log(1 - a_i) = \alpha_i - w^\top x^{(i)}$.

### The Loss in Terms of $\alpha$

Substituting into the cross-entropy loss:

$$
\mathcal{L}(w) = -\sum_{i=1}^{n} \left[y^{(i)} \alpha_i + (1 - y^{(i)})(\alpha_i - w^\top x^{(i)})\right]
$$

### Computing $\partial \alpha_i / \partial w$

We need the partial derivative of $\alpha_i = -\log(1 + e^{-w^\top x^{(i)}})$ with respect to $w$:

$$
\frac{\partial \alpha_i}{\partial w} = -\frac{1}{1 + e^{-w^\top x^{(i)}}} \cdot e^{-w^\top x^{(i)}} \cdot (-x^{(i)}) = \frac{x^{(i)} e^{-w^\top x^{(i)}}}{1 + e^{-w^\top x^{(i)}}}
$$

This simplifies beautifully. The fraction $\frac{e^{-w^\top x^{(i)}}}{1 + e^{-w^\top x^{(i)}}}$ is exactly $1 - \sigma(w^\top x^{(i)}) = 1 - a_i$, so:

$$
\frac{\partial \alpha_i}{\partial w} = x^{(i)}(1 - a_i)
$$

### Assembling the Gradient

Now differentiate the loss:

$$
\frac{\partial \mathcal{L}}{\partial w} = -\sum_{i=1}^{n} \left[y^{(i)} \frac{\partial \alpha_i}{\partial w} + (1 - y^{(i)})\left(\frac{\partial \alpha_i}{\partial w} - x^{(i)}\right)\right]
$$

Substituting $\frac{\partial \alpha_i}{\partial w} = x^{(i)}(1 - a_i)$:

$$
= -\sum_{i=1}^{n} \left[y^{(i)} x^{(i)}(1 - a_i) + (1 - y^{(i)})\bigl(x^{(i)}(1 - a_i) - x^{(i)}\bigr)\right]
$$

Expanding the second term: $x^{(i)}(1 - a_i) - x^{(i)} = -x^{(i)} a_i$. So:

$$
= -\sum_{i=1}^{n} \left[y^{(i)} x^{(i)}(1 - a_i) - (1 - y^{(i)}) x^{(i)} a_i\right]
$$

Distributing:

$$
= -\sum_{i=1}^{n} \left[y^{(i)} x^{(i)} - y^{(i)} x^{(i)} a_i - x^{(i)} a_i + y^{(i)} x^{(i)} a_i\right]
$$

The $y^{(i)} x^{(i)} a_i$ terms cancel:

$$
= -\sum_{i=1}^{n} x^{(i)} \left(y^{(i)} - a_i\right) = \sum_{i=1}^{n} x^{(i)} \left(a_i - y^{(i)}\right)
$$

Recalling that $a_i = \sigma(w^\top x^{(i)})$:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial w} = \sum_{i=1}^{n} x^{(i)} \left(\sigma(w^\top x^{(i)}) - y^{(i)}\right)}
$$

This is a striking result. The gradient is simply the sum, over all training examples, of the input vector $x^{(i)}$ scaled by the **prediction error** $(\hat{y}^{(i)} - y^{(i)})$. When the prediction is correct ($\hat{y}^{(i)} \approx y^{(i)}$), the contribution is near zero. When the prediction is wrong, the input vector pushes the weights in the direction that would correct the mistake. This has the same intuitive structure as the perceptron update rule—but it emerged from a probabilistic interpretation.

---

## The Matrix Form

### Batch Notation

Stacking the $n$ input vectors as rows of a matrix and the labels as a column vector:

$$
X = \begin{bmatrix} x^{(1)\top} \\ x^{(2)\top} \\ \vdots \\ x^{(n)\top} \end{bmatrix} \in \R^{n \times d}, \qquad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(n)} \end{bmatrix} \in \R^{n}
$$

The vector of activations (predicted probabilities) is $\sigma(Xw) \in \R^n$, where $\sigma$ is applied elementwise.

### The Gradient in Matrix Form

The gradient derived above can be written as a single matrix expression:

$$
\boxed{\frac{\partial \mathcal{L}}{\partial w} = X^\top \bigl(\sigma(Xw) - y\bigr)}
$$

This is the transpose of the design matrix times the vector of prediction errors. Each column of $X^\top$ is a training input; the vector $\sigma(Xw) - y$ contains the signed errors. The matrix-vector product sums the error-weighted inputs, exactly as the elementwise formula does.

---

## Gradient Descent

### The Update Rule

With the gradient in hand, we minimize the loss by **gradient descent**: repeatedly adjust $w$ in the direction opposite to the gradient.

$$
w \leftarrow w - \eta \cdot X^\top \bigl(\sigma(Xw) - y\bigr)
$$

where $\eta > 0$ is the **learning rate**, a scalar that controls the step size. This update is repeated until convergence.

![Gradient descent on a convex loss surface. Starting from an initial point $w^{(0)}$, each step moves in the direction of steepest descent (negative gradient), with steps becoming smaller as the minimum is approached.](figures/gradient_descent.png)

The intuition is geometric: the gradient $\partial \mathcal{L}/\partial w$ points in the direction of steepest increase of the loss. Moving in the opposite direction decreases the loss. The learning rate $\eta$ determines how far we step—too large and we overshoot; too small and convergence is slow.

### Connection to the Perceptron

Compare the sigmoid neuron's update to the perceptron's update rule $w \leftarrow w + t^{(i)} x^{(i)}$ (applied on misclassified examples). Both have the same structure: the weight vector is adjusted by adding a scaled version of the input. The difference is in the scaling:

- **Perceptron**: updates only on mistakes, with a fixed scale of $\pm 1$
- **Sigmoid neuron**: updates on all examples, with a scale proportional to the prediction error $(\hat{y}^{(i)} - y^{(i)})$

The sigmoid neuron's update is "softer"—examples that are almost correctly classified contribute small updates, while examples that are badly misclassified contribute large updates. This smooth scaling is a direct consequence of using a differentiable activation function and a principled loss function.

### Implementation

In code, the entire training loop is concise:

```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train(X, y, lr=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    for epoch in range(epochs):
        y_hat = sigmoid(X @ w)              # predictions
        gradient = X.T @ (y_hat - y)        # gradient
        w = w - lr * gradient               # update
    return w
```

The three lines inside the loop correspond exactly to the three mathematical operations: compute activations, compute gradient, take a gradient step.

---

## Odds, Log-Odds, and the Logit

### The Logit Function

The sigmoid function maps a real number (the weighted sum $z = w^\top x$) to a probability $p \in (0, 1)$. Its inverse, the **logit function**, maps a probability back to a real number:

$$
\logit(p) = \log \frac{p}{1 - p} = \sigma^{-1}(p)
$$

The quantity $p/(1 - p)$ is the **odds** of the event: the ratio of the probability that it occurs to the probability that it does not. The logit is the logarithm of the odds, or **log-odds**.

![The logit function maps probabilities in $(0, 1)$ to the entire real line. At $p = 1/2$, the log-odds are zero. Probabilities above $1/2$ correspond to positive log-odds; below $1/2$, to negative log-odds.](figures/log_odds.png)

### Log-Odds Are Linear

A key property of logistic regression is that the log-odds are a linear function of the input:

$$
\log \frac{P(y = 1 \mid x)}{P(y = 0 \mid x)} = \log \frac{\sigma(w^\top x)}{1 - \sigma(w^\top x)} = w^\top x
$$

This follows because $\sigma(z)/(1 - \sigma(z)) = e^z$, so $\log(\sigma(z)/(1 - \sigma(z))) = z = w^\top x$. The weights have a direct interpretation: each weight $w_j$ is the partial derivative of the log-odds with respect to feature $x_j$:

$$
w_j = \frac{\partial}{\partial x_j} \log \frac{P(y = 1 \mid x)}{P(y = 0 \mid x)}
$$

Equivalently, a unit increase in feature $x_j$ (holding all other features constant) **multiplies the odds** by $e^{w_j}$. If $w_j = 0.5$, each unit increase in $x_j$ multiplies the odds by $e^{0.5} \approx 1.65$—a 65% increase. If $w_j = -1$, each unit increase in $x_j$ multiplies the odds by $e^{-1} \approx 0.37$—a 63% decrease.

This interpretability is one of the reasons logistic regression remains widely used in fields like medicine, economics, and social science, even when more complex models might achieve higher accuracy.

---

## Convexity of the Cross-Entropy Loss

We claimed that the cross-entropy loss is convex in $w$. This is worth verifying. Using the logit-form identity derived earlier, the loss for a single example with label $y$ and logit $z = w^\top x$ can be written as:

$$
L(z, y) = -\left[y \log \sigma(z) + (1 - y) \log(1 - \sigma(z))\right] = \log(1 + e^z) - yz
$$

The second form follows from our earlier substitutions: $\log \sigma(z) = -\log(1 + e^{-z})$ and $\log(1 - \sigma(z)) = -z - \log(1 + e^{-z})$, combined and simplified.

To check convexity, compute the second derivative with respect to $z$:

$$
\frac{\partial L}{\partial z} = \sigma(z) - y
$$

$$
\frac{\partial^2 L}{\partial z^2} = \sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

Since $\sigma(z) \in (0, 1)$, the second derivative is strictly positive for all $z$. The loss is convex in $z$, and since $z = w^\top x$ is a linear function of $w$, the loss is convex in $w$ as well. The sum of convex functions is convex, so the total loss $\mathcal{L}(w) = \sum_i L(w^\top x^{(i)}, y^{(i)})$ is convex in $w$.

This guarantees that gradient descent converges to the global minimum—a guarantee the perceptron provides only for linearly separable data, and the quadratic loss with sigmoid activation does not provide at all.

---

## Connection to Information Theory

The cross-entropy loss has a deep connection to information theory that goes beyond the surprisal interpretation. Given the true distribution of labels $q$ and the model's predicted distribution $p$, the **cross-entropy** is:

$$
H(q, p) = -\sum_c q(c) \log p(c)
$$

For binary classification with $q = (y, 1-y)$ and $p = (\hat{y}, 1-\hat{y})$:

$$
H(q, p) = -\left[y \log \hat{y} + (1 - y) \log(1 - \hat{y})\right]
$$

which is exactly our loss function. The cross-entropy is always at least as large as the **entropy** of the true distribution:

$$
H(q, p) = H(q) + D_{\text{KL}}(q \| p) \geq H(q)
$$

where $D_{\text{KL}}(q \| p) \geq 0$ is the **Kullback-Leibler divergence**. Minimizing the cross-entropy is equivalent to minimizing the KL divergence between the true label distribution and the model's predictions—that is, finding the model whose predicted probabilities are closest to the true probabilities in the information-theoretic sense.

This perspective generalizes: when we later encounter softmax regression (multi-class logistic regression), language models, and the training objectives of large language models, they are all minimizing cross-entropy—fitting a parameterized distribution to observed data by minimizing the KL divergence.

---

## Linear Probes

The logistic regression framework extends naturally to analyzing learned representations. In the perceptron sentiment article, we trained a linear classifier on GloVe word embeddings to discover that sentiment is linearly encoded in embedding space. That was a **linear probe**—and logistic regression provides the principled probabilistic version.

Given a representation $h(x)$ produced by any model (a word embedding, a hidden layer of a neural network, or an LLM's internal activations), we fit:

$$
P(y = 1 \mid x) = \sigma(w^\top h(x) + b)
$$

The learned weights $w$ reveal which directions in representation space encode the property of interest. Since each $w_j$ is the partial derivative of the log-odds with respect to representation dimension $j$, the weight vector provides a complete linear explanation of how the representation encodes the label.

Linear probes trained with cross-entropy loss—rather than the perceptron's heuristic update—provide calibrated probabilities and a well-defined measure of how linearly separable a property is. This technique is now a standard tool for mechanistic interpretability in deep learning.

---

## Summary

- The **sigmoid function** $\sigma(z) = 1/(1 + e^{-z})$ maps real numbers smoothly to probabilities in $(0, 1)$, resolving the perceptron's discontinuous activation and enabling a probabilistic interpretation
- The **sigmoid neuron** models each label as a Bernoulli random variable: $y^{(i)} \sim \text{Bernoulli}(\sigma(w^\top x^{(i)}))$, turning classification into probabilistic inference
- The **cross-entropy loss** $\mathcal{L}(w) = -\sum_i [y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)})]$ is not a design choice—it is the **negative log-likelihood** (surprisal) that arises from maximum likelihood estimation on the Bernoulli model
- The **gradient** simplifies to $\frac{\partial \mathcal{L}}{\partial w} = X^\top(\sigma(Xw) - y)$: the design matrix transposed times the vector of prediction errors
- The **gradient descent** update $w \leftarrow w - \eta \cdot X^\top(\sigma(Xw) - y)$ has the same structure as the perceptron update but with smooth, proportional corrections instead of binary flips
- The **log-odds** $\log(P(y=1)/P(y=0)) = w^\top x$ are linear in the features, making the weights directly interpretable as log-odds-ratio multipliers
- The cross-entropy loss is **convex** in $w$, guaranteeing that gradient descent finds the global minimum—unlike the quadratic loss composed with the sigmoid
- Cross-entropy minimization is equivalent to minimizing the **KL divergence** between the true labels and the model's predictions, connecting logistic regression to information theory
- Logistic regression serves as the foundation for **linear probes**, a standard technique for analyzing the representations learned by deep neural networks
