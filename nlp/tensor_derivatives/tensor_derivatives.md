# Tensor Derivatives

Modern deep learning is built on a small number of mathematical primitives repeated at enormous scale. Among these, tensor calculus (and in particular, derivatives of vectors, matrices, and higher-order tensors) plays a central role.

In this post, I want to walk through the basic ideas behind tensor derivatives as they appear in supervised learning with neural networks. The goal is not formal rigor for its own sake, but practical clarity: how to reason about shapes, indices, and derivatives in a way that directly maps to working code.

---

## Why Tensor Derivatives Matter

Modern AI is predominantly based on supervised learning in deep neural networks. At its core, deep learning is nothing more than **gradient-based optimization of parameters**.

Every training step involves repeatedly computing derivatives of:

- Vectors with respect to vectors  
- Vectors with respect to matrices  
- Matrices with respect to matrices  
- And, in batch settings, higher-order tensors  

If you are comfortable only with scalar calculus, this is where things often become confusing.

The key strategy we will use throughout is simple:

> To compute derivatives of tensor expressions, reduce the problem to derivatives of *individual scalar components*.

Once you understand the scalar structure, the tensor result becomes almost automatic.

---

## A First Example: Matrix–Vector Product

Consider the basic linear transformation

\[
\vec{y} = W \vec{x}
\]

where:

- \( \vec{x} \in \mathbb{R}^D \) is a column vector  
- \( W \in \mathbb{R}^{C \times D} \) is a matrix  
- \( \vec{y} \in \mathbb{R}^C \) is a column vector  

What does the derivative \( \frac{\partial \vec{y}}{\partial \vec{x}} \) mean?

By definition, it is the matrix of all partial derivatives

\[
\frac{\partial y_i}{\partial x_j}
\]

so the result must be a \( C \times D \) matrix: the **Jacobian**.

---

## Computing a Single Component

To understand the full Jacobian, we start with a single entry.

The \( i \)-th component of \( \vec{y} \) is

\[
y_i = \sum_{j=1}^D W_{i,j} x_j
\]

Now fix indices and compute

\[
\frac{\partial y_3}{\partial x_7}
\]

Only one term in the sum depends on \( x_7 \):

\[
\frac{\partial}{\partial x_7}
\left( W_{3,1}x_1 + \cdots + W_{3,7}x_7 + \cdots + W_{3,D}x_D \right)
= W_{3,7}
\]

So in general,

\[
\frac{\partial y_i}{\partial x_j} = W_{i,j}
\]

---

## The Full Jacobian

Collecting all components, the Jacobian is

\[
\frac{\partial \vec{y}}{\partial \vec{x}} =
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_D} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_C}{\partial x_1} & \cdots & \frac{\partial y_C}{\partial x_D}
\end{bmatrix}
=
\begin{bmatrix}
W_{1,1} & \cdots & W_{1,D} \\
\vdots & \ddots & \vdots \\
W_{C,1} & \cdots & W_{C,D}
\end{bmatrix}
= W
\]

Hence,

\[
\frac{\partial}{\partial \vec{x}} (W \vec{x}) = W
\]

This result is intuitive and appears constantly in backpropagation.

---

## Row Vectors

Now consider the row-vector version

\[
\vec{y} = \vec{x} W
\]

where:

- \( \vec{x} \in \mathbb{R}^{1 \times D} \)  
- \( W \in \mathbb{R}^{D \times C} \)  
- \( \vec{y} \in \mathbb{R}^{1 \times C} \)  

A single component is

\[
y_3 = \sum_{j=1}^D x_j W_{j,3}
\]

Then

\[
\frac{\partial y_3}{\partial x_7} = W_{7,3}
\]

and assembling all components gives

\[
\frac{\partial \vec{y}}{\partial \vec{x}} = W^T
\]

The orientation of vectors determines whether \( W \) or \( W^T \) appears.

---

## Derivative with Respect to a Matrix

Now consider differentiating with respect to the matrix itself.

Let

\[
\vec{y} = \vec{x} W
\]

and ask for

\[
\frac{\partial \vec{y}}{\partial W}
\]

This is no longer a matrix, but a **three-dimensional tensor**, since:

- One axis indexes components of \( \vec{y} \)  
- Two axes index components of \( W \)  

Start with a single component:

\[
\frac{\partial y_3}{\partial W_{7,8}}
\]

Since

\[
y_3 = \sum_{j=1}^D x_j W_{j,3}
\]

we see that \( y_3 \) does not depend on \( W_{7,8} \), so

\[
\frac{\partial y_3}{\partial W_{7,8}} = 0
\]

But for elements in the third column,

\[
\frac{\partial y_3}{\partial W_{2,3}} = x_2
\]

In general,

\[
\frac{\partial y_j}{\partial W_{i,j}} = x_i, \quad \text{and all other entries are zero.}
\]

---

## Sparsity and Reduced Representations

Let

\[
F_{i,j,k} = \frac{\partial y_i}{\partial W_{j,k}}
\]

Then the only nonzero entries satisfy \( i = k \), and

\[
F_{i,j,i} = x_j
\]

All other components vanish.

This derivative tensor is highly **sparse**.

We can summarize the nontrivial part as a matrix

\[
G_{i,j} = F_{i,j,i}
\]

Such reductions are essential for efficient implementations of backpropagation.

---

## Batch Computation

In practice, we never process a single example at a time.

Let \( X \in \mathbb{R}^{N \times D} \) be a batch of \( N \) inputs, and

\[
Y = X W, \quad Y \in \mathbb{R}^{N \times C}
\]

with

\[
Y_{i,j} = \sum_{k=1}^D X_{i,k} W_{k,j}
\]

Now consider

\[
\frac{\partial Y_{a,b}}{\partial X_{c,d}}
\]

This is nonzero only when \( a = c \), since rows correspond to independent examples.

In fact,

\[
\frac{\partial Y_{i,j}}{\partial X_{i,k}} = W_{k,j}
\]

and for each row,

\[
\frac{\partial Y_{i,:}}{\partial X_{i,:}} = W^T
\]

The same parameters are applied independently to every example in the batch.

---

## Multiple Layers and the Chain Rule

Now consider two layers in sequence:

\[
\vec{y} = V W \vec{x}
\]

Let

\[
\vec{m} = W \vec{x}, \quad \vec{y} = V \vec{m}
\]

Then by the chain rule,

\[
\frac{\partial \vec{y}}{\partial \vec{x}}
= \frac{\partial \vec{y}}{\partial \vec{m}} \frac{\partial \vec{m}}{\partial \vec{x}}
\]

Component-wise,

\[
\frac{\partial y_i}{\partial x_j}
= \sum_{k=1}^M \frac{\partial y_i}{\partial m_k} \frac{\partial m_k}{\partial x_j}
\]

Since

\[
\frac{\partial y_i}{\partial m_k} = V_{i,k}, \quad
\frac{\partial m_k}{\partial x_j} = W_{k,j}
\]

we obtain

\[
\frac{\partial y_i}{\partial x_j} = \sum_{k=1}^M V_{i,k} W_{k,j}
\]

which is exactly the matrix product \( (VW)_{i,j} \).

Thus,

\[
\frac{\partial \vec{y}}{\partial \vec{x}} = V W
\]

Backpropagation involves repeated application of this idea.

---

## Summary: How to Think About Tensor Derivatives

When working with tensor derivatives in neural networks, three principles suffice:

1. **Introduce intermediate variables**  
   Break long expressions into simple steps.

2. **Work component-wise**  
   Always start from a single scalar derivative.

3. **Reduce along the right axes**  
   Apply summation to collapse intermediate tensors into usable forms.

If you can consistently reduce tensor calculus to scalar reasoning plus careful bookkeeping of indices and shapes, it will become easier to reason about backpropagation in complex neural architectures.
