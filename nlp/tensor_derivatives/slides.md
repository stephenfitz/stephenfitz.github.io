# Tensor Derivatives

Derivatives of vectors, matrices, and higher-order tensors in deep learning

---

## Overview

- Why tensor derivatives matter for deep learning
- The key strategy: reduce to scalar components
- Matrix-vector product and the Jacobian
- Derivatives with respect to matrices
- Sparsity and reduced representations
- Batch computation
- The chain rule for multiple layers

---

## Why Tensor Derivatives Matter

- Deep learning is **gradient-based optimization of parameters**
- Every training step involves derivatives of:
    - Vectors with respect to vectors
    - Vectors with respect to matrices
    - Matrices with respect to matrices
    - Higher-order tensors in batch settings
- Scalar calculus alone is not enough

---

## The Key Strategy

> To compute derivatives of tensor expressions, reduce the problem to derivatives of *individual scalar components*.

- Understand the scalar structure first
- The tensor result becomes almost automatic
- Three principles: intermediate variables, component-wise, reduce along axes

---

## First Example: Matrix-Vector Product

$$\vec{y} = W\vec{x}$$

- $\vec{x} \in \R^D$ (column vector), $W \in \R^{C \times D}$ (matrix), $\vec{y} \in \R^C$ (column vector)
- What does $\frac{\partial \vec{y}}{\partial \vec{x}}$ mean?
- It is the matrix of all partial derivatives $\frac{\partial y_i}{\partial x_j}$
- Result: a $C \times D$ matrix --- the **Jacobian**

---

## Computing a Single Component

- The $i$-th component: $y_i = \sum_{j=1}^D W_{i,j} x_j$
- Fix indices, e.g. compute $\frac{\partial y_3}{\partial x_7}$:

$$\frac{\partial}{\partial x_7}\left(W_{3,1}x_1 + \cdots + W_{3,7}x_7 + \cdots\right) = W_{3,7}$$

- In general:

$$\frac{\partial y_i}{\partial x_j} = W_{i,j}$$

---

## The Full Jacobian

$$\frac{\partial \vec{y}}{\partial \vec{x}} = \begin{bmatrix} W_{1,1} & \cdots & W_{1,D} \\ \vdots & \ddots & \vdots \\ W_{C,1} & \cdots & W_{C,D} \end{bmatrix} = W$$

- Hence:

$$\frac{\partial}{\partial \vec{x}}(W\vec{x}) = W$$

- This result appears constantly in **backpropagation**

---

## Row-Vector Variant

$$\vec{y} = \vec{x}W$$

- $\vec{x} \in \R^{1 \times D}$, $W \in \R^{D \times C}$, $\vec{y} \in \R^{1 \times C}$
- Single component: $y_3 = \sum_{j=1}^D x_j W_{j,3}$
- Then: $\frac{\partial y_3}{\partial x_7} = W_{7,3}$
- Assembling all components:

$$\frac{\partial \vec{y}}{\partial \vec{x}} = W^\top$$

- The orientation of vectors determines whether $W$ or $W^\top$ appears

---

## Derivative with Respect to a Matrix

$$\vec{y} = \vec{x}W \quad \Rightarrow \quad \frac{\partial \vec{y}}{\partial W} = \text{?}$$

- This is a **3-dimensional tensor**:
    - One axis indexes components of $\vec{y}$
    - Two axes index components of $W$
- Since $y_3 = \sum_j x_j W_{j,3}$:
    - $\frac{\partial y_3}{\partial W_{7,8}} = 0$ (wrong column)
    - $\frac{\partial y_3}{\partial W_{2,3}} = x_2$ (correct column)

---

## General Rule for Matrix Derivatives

$$\frac{\partial y_j}{\partial W_{i,j}} = x_i \quad \text{(all other entries are zero)}$$

- The derivative tensor $F_{i,j,k} = \frac{\partial y_i}{\partial W_{j,k}}$ is nonzero only when $i = k$
- And $F_{i,j,i} = x_j$
- This tensor is highly **sparse**

---

## Sparsity and Reduced Representations

- The full derivative tensor has shape $(C, D, C)$ --- mostly zeros
- We can summarize the nontrivial part as a matrix: $G_{i,j} = F_{i,j,i}$
- Such reductions are essential for **efficient backpropagation**
- In practice, we never materialize the full tensor

---

## Batch Computation

- In practice: batch of $N$ inputs, $X \in \R^{N \times D}$

$$Y = XW, \quad Y \in \R^{N \times C}$$

- Element: $Y_{i,j} = \sum_{k=1}^D X_{i,k} W_{k,j}$
- $\frac{\partial Y_{a,b}}{\partial X_{c,d}}$ is nonzero only when $a = c$ (rows are independent examples)
- For each row:

$$\frac{\partial Y_{i,:}}{\partial X_{i,:}} = W^\top$$

---

## Multiple Layers and the Chain Rule

$$\vec{y} = VW\vec{x}$$

- Let $\vec{m} = W\vec{x}$, then $\vec{y} = V\vec{m}$
- Chain rule:

$$\frac{\partial \vec{y}}{\partial \vec{x}} = \frac{\partial \vec{y}}{\partial \vec{m}} \cdot \frac{\partial \vec{m}}{\partial \vec{x}}$$

---

## Chain Rule: Component Form

$$\frac{\partial y_i}{\partial x_j} = \sum_{k=1}^M \frac{\partial y_i}{\partial m_k} \cdot \frac{\partial m_k}{\partial x_j} = \sum_{k=1}^M V_{i,k} \, W_{k,j}$$

- This is exactly the matrix product $(VW)_{i,j}$
- Therefore:

$$\frac{\partial \vec{y}}{\partial \vec{x}} = VW$$

- **Backpropagation** = repeated application of this idea

---

## Summary: Three Principles

1. **Introduce intermediate variables** --- break long expressions into simple steps
2. **Work component-wise** --- always start from a single scalar derivative
3. **Reduce along the right axes** --- apply summation to collapse intermediate tensors

- Reduce tensor calculus to scalar reasoning + careful bookkeeping of indices and shapes
- This directly maps to reasoning about backpropagation in complex architectures
