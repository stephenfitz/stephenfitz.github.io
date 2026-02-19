# Matrix Computations

From definition to implementation

---

## Overview

- The matrix product: definition and structure
- Selecting columns with standard basis vectors
- Geometric interpretation as linear maps
- Matrix-vector product as linear combination of columns
- Matrix product as composition of maps
- Implementing matrix multiplication in Python
- Performance from naive loops to BLAS

---

## Notation

- $A$ is an $n \times m$ matrix, $B$ is an $m \times l$ matrix
- The product $AB$ is an $n \times l$ matrix
- Critical constraint: **columns of $A$** must equal **rows of $B$** (both $m$)

$$A = (a_{i,j}), \quad B = (b_{i,j})$$

---

## The Matrix Product: Definition

$$(AB)_{i,j} = \sum_{k=1}^{m} a_{i,k} \, b_{k,j}$$

- Each entry is a sum of $m$ terms
- Walk along the $i$-th row of $A$ and the $j$-th column of $B$
- Multiply corresponding entries and accumulate

---

## Elements Written Out

![Elements of the matrix product](figures/mm3.jpg)

---

## A Single Element as an Inner Product

$$(AB)_{2,1} = a_{2,1} b_{1,1} + a_{2,2} b_{2,1} + a_{2,3} b_{3,1} = \sum_{k=1}^{m} a_{2,k} \, b_{k,1}$$

- Each element of the product = **inner product** of a row of $A$ with a column of $B$
- Row index $\to$ which row from $A$; column index $\to$ which column from $B$

---

## A Single Element as an Inner Product

![Single element as inner product](figures/mm4.jpg)

---

## Selecting a Column with $\vec{e}_k$

$$\vec{e}_k = \begin{bmatrix} 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix} \leftarrow k\text{-th position}$$

- Multiplying $A$ by $\vec{e}_k$ yields the $k$-th column of $A$:

$$A\vec{e}_k = \vec{a}_k$$

---

## Selecting a Column with $\vec{e}_k$

![Extracting a column](figures/mm6.jpg)

---

## Geometric Interpretation

- A matrix $A \in \R^{n \times m}$ defines a **linear map** from $\R^m$ to $\R^n$
- $A\vec{e}_k = \vec{a}_k$: each standard basis vector maps to the corresponding column
- The **columns of a matrix are the images of the standard basis**

---

## Geometric Interpretation

![Geometric interpretation](figures/mm8.jpg)

---

## Example: 2x2 Transformation

$$A = \begin{bmatrix} -2 & -1 \\ 1 & 2 \end{bmatrix}$$

- $\vec{e}_1 = (1,0) \mapsto (-2, 1)$ and $\vec{e}_2 = (0,1) \mapsto (-1, 2)$
- The unit square is transformed into a **parallelogram** spanned by the column vectors
- A matrix encodes a linear transformation; its columns tell you where the basis vectors go

---

## Matrix-Vector Product as Column Combination

$$A\vec{x} = x_1 \vec{a}_1 + x_2 \vec{a}_2 + \cdots + x_m \vec{a}_m$$

> The matrix-vector product $A\vec{x}$ is a **linear combination of the columns of $A$**, with the components of $\vec{x}$ as coefficients.

- The range of $A$ is the **span of its columns**

---

## Matrix-Vector Product as Column Combination

![Linear combination of columns](figures/mm10.jpg)

---

## Matrix Product as Composition

- $AB$ represents: **first apply $B$, then apply $A$**
- $C\vec{e}_k = A(B\vec{e}_k)$: the composition sends each basis vector to the same place
- The "rows by columns" rule is precisely the algebraic expression of **function composition** for linear maps

---

## Matrix Product as Composition

![Composition of linear maps](figures/mm12.jpg)

---

## Columns of the Product

$$(AB)\vec{e}_j = A(B\vec{e}_j) = A\vec{b}_j = b_{1,j}\vec{a}_1 + b_{2,j}\vec{a}_2 + \cdots + b_{m,j}\vec{a}_m$$

> Each column of $AB$ is a linear combination of the columns of $A$, with coefficients given by the corresponding column of $B$.

---

## Implementing in Python: Setup

```python
import numpy as np
from timeit import timeit

a = np.random.randn(50, 200)
b = np.random.randn(200, 20)
```

- Product will be $50 \times 20$; each entry requires a dot product of length 200
- Goal: implement at progressively higher abstraction, measure performance

---

## Version 1: Triple Nested Loops

```python
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i,j] += a[i,k] * b[k,j]
    return c
```

- Faithful to the definition but very slow
- $50 \times 20 \times 200 = 200{,}000$ Python-level scalar multiplications
- **0.540 seconds**

---

## Version 2: Elementwise Operations

```python
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        for j in range(bc):
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
```

- Inner loop replaced by vectorized NumPy operation
- Heavy arithmetic now runs in compiled C code
- **0.027 seconds** (~20x speedup)

---

## Version 3: Broadcasting

```python
def matmul(a, b):
    ar, ac = a.shape
    br, bc = b.shape
    c = np.zeros(shape=(ar, bc))
    for i in range(ar):
        c[i] = (a[i, :, None] * b).sum(axis=0)
    return c
```

- Eliminates the second loop using NumPy broadcasting
- Only one Python loop remains (over 50 rows)
- **0.007 seconds** (~4x speedup)

---

## Version 4: Einstein Summation

```python
matmul = lambda a, b: np.einsum('ik,kj->ij', a, b)
```

- `'ik,kj->ij'`: for each $(i,j)$, sum over shared index $k$
- The definition of matrix multiplication in index notation
- NumPy optimizes contraction order and memory access
- **0.002 seconds**

---

## Version 5: BLAS

```python
matmul = np.matmul
```

- Calls through to BLAS (OpenBLAS or Intel MKL)
- Exploits cache hierarchies, SIMD instructions, CPU-specific tuning
- **0.002 seconds**

---

## Results: Three Orders of Magnitude

| Implementation | Time (s) | vs. BLAS |
|:---|---:|---:|
| Triple nested loops | 0.540 | 282x |
| Elementwise | 0.027 | 14x |
| Broadcasting | 0.007 | 3.8x |
| Einstein summation | 0.002 | 1.2x |
| BLAS (`np.matmul`) | 0.002 | 1.0x |

---

## Results: Three Orders of Magnitude

![Performance results](figures/results.png)

---

## The Lesson

- At each step: replaced Python-level looping with operations exposing more structure to compiled code
- The **mathematics is identical**; the implementation makes all the difference
- Every `nn.Linear`, every attention head, every gradient computation calls BLAS or GPU equivalents
- Understanding the math is essential; **efficiency means letting the hardware do the work**

---

## Summary

- The matrix product: inner products of rows and columns, or composition of linear maps
- Columns of $A$ are images of the standard basis under the linear map
- $A\vec{x}$ is a linear combination of columns; $AB$ is function composition
- Implementation: from 0.54s (naive loops) to 0.002s (BLAS) --- **282x speedup**
- Expose structure to optimized libraries for maximum performance
