---
title: "Math For Machine Learning Notes"
date: 2023-01-03
mathjax: true
toc: true
categories:
  - blog
tags:
  - study
  - cs229
---

# Linear Algebra Review

## 1. Vectors and Matrices
Vectors and matrices are fundamental objects in linear algebra. A vector can be thought of as a list of numbers, and a matrix as a grid of numbers arranged in rows and columns. For example:

```text
Vector: v = [1, 2, 3]
Matrix: M = [1 2
             3 4
             5 6]
```

## 2. Matrix Operations

### 2.1 Addition
We add two matrices by adding their corresponding entries:

```text
[1 2]     [4 5]     [1+4 2+5]     [5 7]
[3 4]  +  [6 7]  =  [3+6 4+7]  =  [9 11]
```

### 2.2 Subtraction
Similar to addition, we subtract two matrices by subtracting their corresponding entries:

```text
[5 7]     [4 5]     [5-4 7-5]     [1 2]
[9 11]  - [6 7]  =  [9-6 11-7] =  [3 4]
```

### 2.3 Multiplication
Multiplication is a bit more complex. You multiply the elements of the rows of the first matrix by the elements of the columns of the second, and then add them up.

```text
[1 2]     [5 6]     [(1*5+2*7) (1*6+2*8)]     [19 22]
[3 4]  x  [7 8]  =  [(3*5+4*7) (3*6+4*8)]  =  [43 50]
```

## 3. Vector Spaces

A vector space is a set of vectors that can be added together and multiplied by scalars (real or complex numbers) in such a way that these operations obey certain axioms. For example, the set of all 2D vectors forms a vector space.

## 4. Linear Transformations and Matrices

Linear transformations are a way of 'transforming' one vector into another while preserving the operations of addition and scalar multiplication. For example, scaling and rotating vectors are both linear transformations. Every linear transformation can be represented by a matrix. If `v` is a vector and `A` is the matrix representing a linear transformation, the transformed vector is obtained by the matrix-vector product `Av`.

## 5. Eigenvalues and Eigenvectors

Given a square matrix `A`, if there is a non-zero vector `v` such that multiplying `A` by `v` gives a vector that's a scaled version of `v`, then `v` is an eigenvector of `A` and the scaling factor is the corresponding eigenvalue.

If `Av = λv`, then `v` is an eigenvector of `A` and `λ` is the corresponding eigenvalue. For instance:

```text
Let A = [4 1]
         [2 3]

Then, for v = [1]
               [2]

We have Av = [4*1 + 1*2]
             [2*1 + 3*2]

          = [6]
            [8]

which is 2*v, so λ = 2.
```

## 6. Diagonalization

A matrix `A` is diagonalizable if we can

 write it as `A = PDP^(-1)` where `D` is a diagonal matrix, and `P` is a matrix whose columns are the eigenvectors of `A`.

## 7. Orthogonal and Unitary Matrices

An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (i.e., orthonormal vectors). Orthogonal matrices have a lovely property: the inverse of an orthogonal matrix is its transpose.

## 8. Dot Product and Norms

The dot product of two vectors `a = [a1, a2, ..., an]` and `b = [b1, b2, ..., bn]` is defined as `a1*b1 + a2*b2 + ... + an*bn`.

The norm of a vector `v` (often interpreted as the length of the vector) is the square root of the dot product of the vector with itself, and is denoted as ||v||.

## 9. Projections

The projection of a vector `y` onto another vector `x` is the vector in the direction of `x` that best approximates `y`. If `y` is projected onto `x` to get the projection vector `p`, then `y - p` is orthogonal to `x`.

## 10. Singular Value Decomposition (SVD)

Every matrix `A` can be decomposed into the product of three matrices `U`, `Σ`, and `V^T`, where `U` and `V` are orthogonal matrices and `Σ` is a diagonal matrix. The diagonal entries of `Σ` are the singular values of `A`.

## 11. Linear Systems of Equations

Many problems in linear algebra boil down to solving a system of linear equations, which can often be represented in matrix form as `Ax = b`. Gaussian elimination and similar methods can be used to solve these systems.


# Probability and Statistics Review

## 1. Probability Basics

### 1.1 Sample Space and Events
A sample space is the set of all possible outcomes of an experiment. An event is a subset of the sample space.

### 1.2 Probability of an Event
The probability of an event A, denoted by P(A), is a number between 0 and 1 inclusive that measures the likelihood of the occurrence of the event.

### 1.3 Probability Axioms
- For any event A, 0 <= P(A) <= 1.
- P(S) = 1, where S is the sample space.
- If A1, A2, A3, ... are disjoint events, then P(A1 ∪ A2 ∪ A3 ∪ ...) = P(A1) + P(A2) + P(A3) + ...

## 2. Conditional Probability

Conditional probability is the probability of an event given that another event has occurred. If we denote A and B as two events, the conditional probability of A given that B has occurred is written as P(A | B).

## 3. Random Variables

A random variable is a function that assigns a real number to each outcome in a sample space.

## 4. Probability Distributions

A probability distribution assigns a probability to each possible value of the random variable. A probability distribution is described by a probability density function (pdf) for continuous variables or a probability mass function (pmf) for discrete variables.

### 4.1 Common Distributions
- Uniform Distribution
- Normal (Gaussian) Distribution
- Binomial Distribution
- Poisson Distribution
- Exponential Distribution

## 5. Expected Value and Variance

The expected value (mean) of a random variable is the long-run average value of the outcomes. The variance measures how spread out the values are around the expected value.

## 6. Covariance and Correlation

Covariance and correlation are measures of the relationship between two random variables. Correlation is a normalized form of covariance and provides a measure between -1 and 1 of how related two variables are.

## 7. Hypothesis Testing

Hypothesis testing is a method for testing a claim or hypothesis about a parameter in a population, using data measured in a sample.

### 7.1 Null and Alternative Hypotheses
The null hypothesis (H0) is a statement about the population parameter that implies no effect or difference, while the alternative hypothesis (Ha) is the statement being tested.

### 7.2 p-value
The p-value is the probability of obtaining the observed data (or data more extreme) if the null hypothesis is true. A smaller p-value provides stronger evidence against the null hypothesis.

### 7.3 Significance Level
The significance level (often denoted by α) is the probability of rejecting the null hypothesis when it is true. It’s a threshold used to determine when the null hypothesis can be rejected.

## 8. Confidence Intervals

A confidence interval provides an estimated range of values which is likely to include an unknown population parameter. The confidence level describes the uncertainty of this range.





