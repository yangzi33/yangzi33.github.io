---
layout: post
title:  "Preliminaries to Neural Networks"
summary: "Linear Regression, Gradient Descent, Vectorization"
---

# Setting Up the Problem

Our goal is to let the machine to "learn" in some sort of way, but how?

To mathematically formulate a *learning problem*, there are two essential things for us to define: a **model** and a **loss function**.

## The Model

The model is an architecture that defines the set of allowable hypotheses, usually functions that predict from the inputs based on computations. Consider the simple case of linear regression, the model consists of linear functions. Usually, the linear function of $$D$$ inputs is paramertrized as $$D$$ coefficients,

* In the fashion of undergraduate statistics, we often interpret them as slope terms, denoted $$\beta_1,\dots,\beta_D$$.

We will call these coefficients using an alternative name: **weights**, usually denoted by $w$'s; we will also have an intercept term, called **bias**. Hence we can write the model mathematically as

$$ y = \sum_j w_j x_j+b.$$

## Loss Function

It is common to encounter cases where we have multiple linear fits, but how do decide which linear fits are "better" than the others?

In order to quantify how *good* a fit is, we will define a **loss function** $$\mathcal{L}(y,t)$$ which assesses how far the predictions $$y$$ is from the target $$t$$. In the case of linear regression, we often use the **squared error** defined as

$$\mathcal{L}(y,t)=\frac{1}{2}(y-t)^2,$$

which is small when $y$ and $t$ are close, and large when they are far apart. The value $$y-t$$ is known as the **residual**, and we want to minimize which (to be close to zero).

## Cost Function

When combining the model and loss function, we obtain an **optimization problem**, in which we try to minimize a **cost function** with respect to the model parameters (i.e. the weights and bias). The cost function is simply the loss averaged over all training examples:

$$
\begin{align}
\mathcal{J}(w_1,w_2,\dots,w_D,b)&=\frac{1}{N}\sum^N_{i=1}\mathcal{L}(y^{(i)},t^{(i)})\\
&=\frac{1}{2N}\sum^N_{i=1}(y^{(i)}-t^{(i)})^2\\
&=\frac{1}{2N}\sum^N_{i=1}\Bigg(\sum_j w_jx_j^{(i)}+b-t^{(i)}\Bigg)^2.
\end{align}
$$

Our goal is to choose $$w_1,w_2,\dots,w_D$$ and $b$ to minimize $\mathcal{J}$.

# The Optimization Problem

To solve the optimization problem, we will bring in the concept of **partial derivatives** from multivariable calculus.

A good place to start is to compute the partial derivatives of the cost function.
Consider the case of linear regression, applying the chain rule yields

$$
\frac{\partial\mathcal{J}}{\partial w_j}=\frac{1}{N}\sum^N_{i=1}x_j^{(i)}\Bigg(\sum_{j'}w_{j'}x_{j'}^{(i)}+b-t^{(i)}\Bigg)=\frac{1}{N}\sum^N_{i=1}x_j^{(i)}(y^{(i)}-t^{(i)})
$$

$$
\frac{\partial\mathcal{J}}{\partial b}=\frac{1}{N}\sum^N_{i=1}\Bigg(\sum_{j'}w_{j'}x_{j'}^{(i)}+b-t^{(i)}\Bigg)=\frac{1}{N}\sum^N_{i=1}(y^{(i)}-t^{(i)})
$$

To use these partial derivatives, we will have two methods to compute the minimum. That is, **direction solution** and **gradient descent**.

## Gradient Descent

The gradient descent is an example of an **iterative algorithm**, in which we apply a certain update rule over and over again. If we are *lucky enough*, our **iterates** will improve gradually according to our objective function.

To start the gradient descent procedure, we will initialze the weights to some value (e.g. all zeros),
If we visualize the cost function as a surface, the lower is better,
so we repeatedly adjust them in the direction which **decreases the cost function the most**-the direction of **steepest descent**.
We will repeat this procedure until the iterates **converge** (in practice, we usually run it until we're tired of waiting).
The final iterate will be close to the actual optimum *if we are lucky enough*.

Now consider the gradient: the direction of steepest ascent of a function

$$
\nabla_{\mathbf{w}}\mathcal{J}=\begin{bmatrix}
\frac{\partial\mathcal{J}}{\partial w_1}\\
\vdots\\
\frac{\partial\mathcal{J}}{\partial w_D}
\end{bmatrix}
$$

Our aim is to decrease the function as quickly as possible, hence we should update the parameters in the direction that is **opposite to the gradient**.

We can formalize this using the **gradient descent** update rule:

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha\nabla_{\mathbf{w}}\mathcal{J}.
$$

The constant $$\alpha$$ here is known as the **learning rate**.
* The larger it is, the larger a "step" we take.
* Generally, it's good to choose small values like $$0.01$$ or $$0.001$$

Now a question may arise: why do we need gradient descent if we can set the partial derivatives to zero and compute the "actual" solution? The gradient descent only gives solutions that are close to the optimum, but doesn't give us the real optimum!

Well, there are two reasons

1. Solving the system of equations **explicitly** only applies to a handful of models, but the gradient descent works on all models for which we can compute the gradient (without solving it analytically by setting to $$0$$). This is often easy to do, and it can be easily *automated*.
2. Solving a large system of linear equations can be expensive, possibly much much much much ... more expensive than a single update in the gradient descent. Hence gradient descent can sometimes find a reasonable solution **much faster** than solving the linear systems.

Therefore, gradient descent can often more practical than computing the exact solutions, even for models which we are able to derive.

# Vectorization

For sake of getting more compact, readable forms of formulas, we will rewrite the model (linear regression as example again) in terms of operations on matrices and vectors. This process is called **vectorization**.

Suppose we have $$N$$ training examples, where each of the examples is $$D$$-dimensional. We will represent the inputs as an $$N\times D$$ matrix $$\mathbf{X}$$, where each row of $$\mathbf{X}$$ corresponds to a training example, and each column corresponds to one input dimention.

The weights are represented as a $$D$$-dimensional vector $$w$$, and the targets are represented as a $$N$$-dimensional vector $$\mathbf{t}$$.

Therefore the predictions can be computed using a matrix-vector product

$$
\mathbf{y}=\mathbf{Xw}+b\mathbf{1}.
$$

As well as the cost function:

$$
\mathcal{J}=\frac{1}{2N}\|\mathbf{y}-\mathbf{t}\|^2=\frac{1}{2N}\| \mathbf{Xw}+b\mathbf{1}-\mathbf{t} \|^2
$$

Additionally, we can vectorize the gradient descent update (for linear regression) as

$$
\mathbf{w}\leftarrow\mathbf{w}-\alpha\frac{1}{N}\mathbf{X}^T(\mathbf{y-t}).
$$