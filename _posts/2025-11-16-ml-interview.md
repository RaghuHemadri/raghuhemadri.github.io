---
layout: post
title: "ML Interview: Fundamentals and Implementation"
date: 2025-11-16 09:00:00
description: 
tags: ml
categories: ml
thumbnail: assets/img/ml_cheatsheet.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

## 1. Foundations of Machine Learning

### 1.1 Types of Learning

At a high level, you can think of each learning paradigm in terms of:

* What data you have (labels? structure? feedback?)
* What objective you optimize
* How you deploy it in practice

#### 1.1.1 Supervised Learning

You are given a dataset of input–output pairs  
$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n,
$$
where $x_i \in \mathcal{X}$ (features) and $y_i \in \mathcal{Y}$ (labels).

You choose a parametric model $f_\theta : \mathcal{X} \to \mathcal{Y}$ and minimize expected loss:
$$
\theta^* = \arg\min_\theta \mathbb{E}_{(x,y)\sim p_{\text{data}}}[\ell(f_\theta(x), y)].
$$

In practice, you approximate this with empirical risk:
$$
\hat{\theta} = \arg\min_\theta \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i).
$$

Typical tasks: classification, regression, ranking (when labels are relevance scores, clicks, etc).

Interpretation: You’re directly learning a mapping “features → label”, under the assumption that train and test come from the same distribution.

---

#### 1.1.2 Unsupervised Learning

Here you only observe inputs:
$$
\mathcal{D} = \{x_i\}_{i=1}^n
$$
with **no labels**. The goal is to uncover structure: clusters, low-dimensional manifolds, density, etc.

Common formulations:

* **Density modeling**: learn $p_\theta(x)$ that approximates the data distribution.
* **Representation learning**: learn $z = g_\theta(x)$ such that $z$ captures “useful” structure (e.g., through reconstruction loss in autoencoders, or contrastive objectives).
* **Clustering**: assign each $x_i$ to a latent group $c_i$, often via maximizing likelihood under a mixture model.

This is about understanding the geometry/statistics of the data rather than predicting a given label.

---

#### 1.1.3 Semi-Supervised Learning

You have a small labeled set and a large unlabeled set:
$$
\mathcal{D}_L = \{(x_i, y_i)\}_{i=1}^{n_L}, \quad
\mathcal{D}_U = \{x_j\}_{j=1}^{n_U}, \quad n_U \gg n_L.
$$

The idea is to exploit the unlabeled data to regularize or guide the supervised model. Typical assumptions:

* **Cluster assumption**: points in the same cluster likely share labels.
* **Manifold assumption**: data lies on a low-dimensional manifold; the decision boundary should be smooth along it.

Many methods optimize an objective like
$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{sup}}(\theta; \mathcal{D}_L) + \lambda\,\mathcal{L}_{\text{unsup}}(\theta; \mathcal{D}_U),
$$
where $\mathcal{L}_{\text{unsup}}$ enforces consistency or smoothness (e.g., consistency regularization, pseudo-labeling).

---

#### 1.1.4 Self-Supervised Learning

A special case (conceptually between unsupervised and supervised) where you create **artificial labels** from the data itself via a “pretext task”.

Examples:

* Masked prediction (e.g., mask a word/token and predict it).
* Contrastive learning: given two views of the same input $x$, push their representations together and others apart.
* Next-step prediction in sequences or time series.

Formally, you define a transformation $T$ that maps raw data to a supervised-like task:
$$
(x, y_{\text{pseudo}}) = T(x).
$$

Then you train as if it were supervised. The goal is not the pretext task itself, but to learn **representations** that transfer well to downstream tasks with few labels.

---

#### 1.1.5 Weak Supervision

Here labels are noisy, indirect, or programmatically generated:

* Heuristic labeling functions  
* Distant supervision (e.g., using knowledge bases)  
* Crowdsourced labels with varying quality  

You might model label noise explicitly or treat multiple noisy signals as evidence, e.g., learning a latent “true” label. The key idea is trading off label quality for quantity, then correcting for noise via modeling.

---

#### 1.1.6 Reinforcement Learning (Light View)

Data arrives as **trajectories** of interaction with an environment:
$$
\{(s_t, a_t, r_t, s_{t+1})\}_{t=0}^{T}
$$

A policy $\pi_\theta(a \mid s)$ is learned to maximize expected return:
$$
J(\theta) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right].
$$

Unlike supervised learning, the signal is **delayed and sparse**; you don’t get a direct “correct action” at each step, just scalar rewards. This makes exploration, credit assignment, and variance reduction central.

---

> **Quick Check:** can you briefly restate how you’d distinguish: supervised,unsupervised, and self-supervised learning
in one or two sentences each, focusing on **what data you have** and **what you’re trying to learn**?

---

## 1.2 Problem Types

At interview level, you want to be able to map a product question (e.g., “rank apps for a query”) to a **formal ML problem type**.

### 1.2.1 Classification

* Inputs: $(x \in \mathcal{X})$, labels: $(y \in {1, \dots, K})$ (or $({0,1})$ for binary).
* Goal: learn $(p_\theta(y \mid x))$ or directly a decision rule $(\hat{y}(x))$.

Typical formulation:

$$
\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \ell\big(f_\theta(x_i), y_i\big),
$$
where $(f_\theta(x))$ outputs logits or probabilities and $(\ell)$ is usually cross-entropy.

Examples:

* Spam vs not spam.
* Fraud vs non-fraud.
* Multi-class intent classification for queries.

Key intuition: decision boundary in feature space splitting classes.

---

### 1.2.2 Regression

* Inputs: $(x \in \mathcal{X})$, labels: $(y \in \mathbb{R})$ (or $(\mathbb{R}^d)$).
* Goal: estimate conditional expectation $(f^*(x) = \mathbb{E}[y \mid x])$ (under squared loss).

Typical objective:

$$
\theta^* = \arg\min_\theta \frac{1}{n}\sum_{i=1}^n (f_\theta(x_i) - y_i)^2.
$$

Examples:

* Predicting CTR (as a probability between 0 and 1, but treated as a real-valued regression target).
* Predicting price, rating, time-to-event.

In practice, many “probability prediction” tasks (like CTR) are a hybrid of regression and classification perspectives.

---

### 1.2.3 Ranking

Ranking is about **ordering** items rather than predicting an absolute value.

Setup: For a query $(q)$, you have candidate items $({d_1, \dots, d_m})$ and want a permutation $(\pi)$ such that:

$$
\text{relevance}(q, d_{\pi(1)}) \ge \text{relevance}(q, d_{\pi(2)}) \ge \dots
$$

Three common perspectives:

* **Pointwise**: treat each (query, item) pair as regression/classification on a relevance label (e.g., 0/1 click, 1–5 rating).
* **Pairwise**: learn $(P(d_i \succ d_j \mid q))$; optimize pairwise losses (e.g., hinge loss on score differences).
* **Listwise**: optimize a loss that operates on the whole ranked list (e.g., approximate nDCG).

Even if you train with pointwise loss, evaluation and product impact are list-based (nDCG, MRR, etc.), so you should be comfortable switching mental models.

---

### 1.2.4 Clustering

Unsupervised grouping: assign each point $(x_i)$ a cluster label $(c_i \in {1, \dots, K})$ without ground-truth labels.

Typical objective (k-means):

$$
\min_{c_i, {\mu_k}} \sum_{i=1}^n \left\lVert x_i - \mu_{c_i} \right\rVert^2.
$$

Here, $(\mu_k)$ are cluster centroids; the algorithm alternates between:

1. Assigning each point to its nearest centroid.
2. Recomputing centroids as means of assigned points.

Use cases:

* User segmentation.
* Query/topic clustering.
* Pre-bucketing items before ranking.

---

### 1.2.5 Reinforcement Learning as a Problem Type

From a product lens, RL is used when **actions influence future states and rewards**:

* State $(s_t)$, action $(a_t)$, reward $(r_t)$, next state $(s_{t+1})$.
* Policy $(\pi_\theta(a \mid s))$ chooses actions.
* Objective: maximize expected return $(J(\theta))$.

Difference from supervised learning: you do not get $(a_t^*)$ labels; you only get scalar feedback (reward), often delayed and noisy.

Applied ML framing:

* Ad selection with sequential user behavior and budgets.
* Ranking with long-term engagement, not just immediate click.
* A/B test-like exploration via bandits (light RL).

---

## 1.3 Bayes Rule, Conditional Independence, Graphical Model Basics

These concepts are the backbone of many classical models (Naive Bayes, HMMs, CRFs) and show up in interview questions around probability reasoning.

### 1.3.1 Bayes Rule

Given joint distribution $(p(x, y))$, Bayes rule relates posterior, likelihood, and prior:

$$
p(y \mid x) = \frac{p(x \mid y), p(y)}{p(x)},
$$
where
$$
p(x) = \sum_y p(x \mid y), p(y)
\quad \text{(discrete case)}
$$
or
$$
p(x) = \int p(x \mid y), p(y), dy
\quad \text{(continuous case)}.
$$

Interpretation:

* $(p(y))$: prior belief about label or parameter $(y)$.
* $(p(x \mid y))$: how likely we are to see this data if $(y)$ were true (likelihood).
* $(p(y \mid x))$: updated belief after observing data (posterior).

In classification, many generative models estimate $(p(x \mid y))$ and $(p(y))$, then classify via:
$$
\hat{y}(x) = \arg\max_y p(y \mid x) = \arg\max_y p(x \mid y)p(y).
$$

You often drop $(p(x))$ since it is common to all classes.

---

### 1.3.2 Conditional Independence

Two random variables $(X)$ and $(Y)$ are conditionally independent given $(Z)$ if:

$$
p(x, y \mid z) = p(x \mid z), p(y \mid z)
$$
for all values.

Equivalently, knowing $(Y)$ gives no extra information about $(X)$ once you know $(Z)$.

Naive Bayes uses a strong conditional independence assumption. For features $(X = (X_1, \dots, X_d))$ and class $(Y)$:

$$
p(x \mid y) = \prod_{j=1}^d p(x_j \mid y).
$$

This is rarely literally true, but it often works surprisingly well in practice because:

* Estimation is simpler and lower-variance.
* The model can still capture useful directional signals from correlated features.

In interviews, mention conditional independence when asked why Naive Bayes is “naive” or why graphical models factorize nicely.

---

### 1.3.3 Graphical Model Basics (DAGs and Factorization)

Probabilistic graphical models represent dependencies with graphs.

For **Bayesian networks** (directed acyclic graphs, DAGs):

* Each node is a random variable.
* An edge $(X \to Y)$ means $(X)$ is a direct parent (influences) $(Y)$.
* The joint distribution factorizes as

$$
p(x_1, \dots, x_n) = \prod_{i=1}^n p\big(x_i \mid \text{parents}(x_i)\big).
$$

Example: Simple Naive Bayes graph:

* One root node $(Y)$.
* Directed edges from $(Y)$ to each feature $(X_j)$.
* Factorization:

$$
p(y, x_1, \dots, x_d) = p(y)\prod_{j=1}^d p(x_j \mid y).
$$

Key interview-relevant points:

* Graph structure encodes conditional independence assumptions.
* Inference = computing marginals or posteriors (e.g., $(p(y \mid x))$).
* Learning = estimating conditional distributions consistent with the graph.

You rarely derive full message-passing in interviews, but being able to say “given this DAG, the joint factorizes into a product of local conditionals” is very useful.

---

> **quick check:**
Suppose you build a model that outputs a **real-valued “relevance score”** for each (query, document) pair, and at serving time you **only care about sorting documents by this score**.
From a modeling perspective, is this closer to: a regression problem, a ranking problem, or both?
Answer in one or two sentences, and explain your reasoning.

---

## 1.4 Curse of Dimensionality

The “curse” is the collection of unintuitive phenomena that appear when the feature dimension ($d$) is large.

### 1.4.1 Volume and sparsity

Consider a unit interval ($[0,1]$) in 1D: its “volume” is $1$.
In $d$ dimensions, the unit hypercube ($[0,1]^d$) also has volume $1$, but points become extremely sparse.

Suppose you need a fraction $\alpha$ of the side length around each point to cover $\rho$ fraction of the total volume. In 1D:
$$
\alpha = \rho
$$
because length scales linearly with volume.

In $d$ dimensions:
$$
\alpha^d = \rho \quad \Rightarrow \quad \alpha = \rho^{1/d}.
$$

Example: To cover $10\%$ of the volume ($\rho = 0.1$):

* In 1D: $\alpha = 0.1$.
* In 10D: $\alpha = 0.1^{1/10} \approx 0.79$.

In 10D, you must cover almost 80% of each dimension’s range to see just 10% of the volume. Intuition: **local neighborhoods explode in size** as dimension grows.

Consequences:

* Any notion of “locality” (k-NN, kernel methods) becomes fragile without huge data.
* Data requirements grow roughly exponentially in $d$ for fixed density.

---

### 1.4.2 Distances concentrate

In high dimensions, distances between random points tend to concentrate around a common value. For many distributions:

* $(\text{max distance} - \text{min distance})$ shrinks relative to the mean distance.

So the “nearest neighbor” is not much closer than a random point. This harms:

* k-NN classifiers and regressors.
* Any method relying on Euclidean distance in raw feature space.

Practical impact: you typically need:

* Strong feature engineering or representation learning.
* Dimensionality reduction (PCA, autoencoders).
* Regularization to keep effective complexity smaller than ambient dimension.

---

## 1.5 No Free Lunch Theorem (NFL)

The NFL theorem formalizes the idea that **no learning algorithm is universally best** over all possible problems.

One common version: consider all possible labelings $f$ of a finite input space $\mathcal{X}$. Let a learning algorithm $A$ observe a training set and produce predictions on unseen points. If you average its generalization performance over all possible functions $f : \mathcal{X} \to \mathcal{Y}$, then:

* Every algorithm has the same average performance.

Intuition:

* If you place the same prior weight on every possible labeling of the data, then any structure exploited by one algorithm is “canceled” by adversarial structure elsewhere.
* To do better than chance in real life, you must **bias** yourself toward some subset of functions (smoothness, linear separability, low-rank structure, etc.).

Interview takeaway:

* Generalization requires **inductive bias** (model class, regularization, features).
* Comparing algorithms only makes sense relative to a **distribution of problems** (e.g., natural images, text, logs), not “all possible worlds.”

---

## 1.6 Bias–Variance Trade-off

Consider a regression setting with true relationship:
$$
Y = f^*(X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid X] = 0, \quad \text{Var}(\varepsilon \mid X) = \sigma^2.
$$

Let $\hat{f}(x)$ be the predictor learned from a training set (itself random). We want the expected squared prediction error at a fixed point $x$:
$$
\mathbb{E}\big[(Y - \hat{f}(x))^2 \mid X = x\big].
$$

We can decompose this expectation into:

$$
\mathbb{E}\big[(Y - \hat{f}(x))^2 \mid X = x\big]
= \underbrace{\big(\mathbb{E}[\hat{f}(x)] - f^*(x)\big)^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}\big[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2\big]}_{\text{Variance}}
+ \underbrace{\sigma^2}_{\text{Irreducible\ noise}}.
$$

Where expectations are taken over the randomness of the training data (and possibly algorithm).

* **Bias**: systematic error from using a too-simple or misspecified model class (e.g., linear model for a nonlinear relationship).
* **Variance**: sensitivity to training data fluctuations; complex models can fit noise and vary a lot between samples.
* **Irreducible noise**: inherent randomness in $Y$ given $X$; no model can get rid of it.

The **trade-off**: increasing model complexity typically decreases bias but increases variance. Regularization, model selection, and early stopping are ways to navigate this trade-off.

---

## 1.7 Overfitting vs Underfitting

Bias–variance manifests practically as over- vs underfitting, visible in training vs validation performance.

### 1.7.1 Underfitting

* The model is too simple or too constrained to capture the underlying structure.
* Both training and validation errors are high.
* Bias is dominant.

Examples:

* Fitting a linear model to data generated by a highly nonlinear function.
* Using too strong regularization.

Fixes:

* Increase model capacity (more features, more flexible model).
* Reduce regularization.
* Improve feature engineering.

---

### 1.7.2 Overfitting

* The model has enough capacity to fit not only the true signal but also the noise specific to the training set.
* Training error is very low (often near zero), but validation/test error is high.
* Variance is dominant.

Overfitting mechanisms:

* High-dimensional hypothesis class relative to data size.
* Poor regularization.
* Extensive hyperparameter search on the same validation set (implicit overfitting to the validation data).

Typical symptoms:

* Large gap between training and validation metrics.
* Very complex decision boundaries.

Mitigations:

* Regularization (L1/L2, early stopping, dropout in DL, etc.).
* Cross-validation and careful model selection.
* Feature selection or dimensionality reduction.
* More data (if available) or more informative features.

---

> a quick check (one conceptual question):

Suppose you train a very flexible model on a small dataset and observe:

* Training accuracy: 99%
* Validation accuracy: 70%

Give a concise explanation of what this indicates in terms of **bias**, **variance**, and **overfitting**, using those words explicitly.

---

## 2. Classical ML Algorithms

### 2.1 Linear Models

Linear models are the workhorses of classical ML: cheap, interpretable, and often surprisingly strong baselines.

We assume features $x\in\mathbb{R}^d$ and model a linear score:
$$
f_\theta(x)=w^\top x+b,\quad \theta=(w,b).
$$

Depending on the task (regression vs classification), we put a different likelihood and loss on top.

---

### 2.1.1 Linear Regression (OLS)

We observe pairs $(x_i,y_i)$ with $y_i\in\mathbb{R}$ and assume:
$$
y_i=w^\top x_i+b+\varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,\sigma^2).
$$

In matrix form:

* Let $X\in\mathbb{R}^{n\times d}$ be the design matrix (rows are $x_i^\top$).
* Let $y\in\mathbb{R}^n$ be the label vector.

The ordinary least squares (OLS) problem is:
$$
\min_{w}\;\frac{1}{2n}\lVert Xw-y\rVert_2^2.
$$

#### Normal equation (closed form)

If $X^\top X$ is invertible, the minimizer is:
$$
\hat{w}=(X^\top X)^{-1}X^\top y.
$$

This comes from setting the gradient to zero:
$$
\nabla_w\frac{1}{2n}\lVert Xw-y\rVert_2^2=\frac{1}{n}X^\top(Xw-y)=0.
$$

Interpretation:

* This is also the **maximum likelihood estimator** (MLE) under the Gaussian noise assumption.

#### Gradient descent perspective

For large $n$ and $d$, you typically use gradient-based methods:

$$
w^{(t+1)}=w^{(t)}-\eta\,\frac{1}{n}X^\top(Xw^{(t)}-y),
$$

possibly with mini-batches (SGD).

---

### 2.1.2 Ridge, Lasso, Elastic-Net

Linear regression is high-variance when features are many or highly correlated. Regularization shrinks coefficients to control complexity.

#### Ridge Regression (L2)

Objective:
$$
\min_{w}\;\frac{1}{2n}\lVert Xw-y\rVert_2^2+\frac{\lambda}{2}\lVert w\rVert_2^2.
$$

Closed-form solution:
$$
\hat{w}_{\text{ridge}}=(X^\top X+\lambda I)^{-1}X^\top y.
$$

Effect:

* Shrinks coefficients toward zero but rarely exactly zero.
* Stabilizes solution when $X^\top X$ is ill-conditioned.
* Reduces variance at the cost of some bias.

Bayesian view:

* Equivalent to MAP estimation with Gaussian prior $w\sim\mathcal{N}(0,\tau^2 I)$ for suitable $\tau$.

#### Lasso (L1)

Objective:
$$
\min_{w}\;\frac{1}{2n}\lVert Xw-y\rVert_2^2+\lambda\lVert w\rVert_1.
$$

Here $\lVert w\rVert_1=\sum_j|w_j|$. No closed form; solved via coordinate descent, proximal gradient, etc.

Effect:

* Encourages **sparsity**: many coefficients exactly zero.
* Performs variable selection and regularization simultaneously.

Bayesian view:

* Corresponds to a Laplace prior on coefficients.

#### Elastic-Net

Combines L1 and L2:

$$
\min_{w}\;\frac{1}{2n}\lVert Xw-y\rVert_2^2
+\lambda_1\lVert w\rVert_1
+\frac{\lambda_2}{2}\lVert w\rVert_2^2.
$$

Often used when:

* You expect many irrelevant features (need sparsity), but
* Features are correlated (pure Lasso can behave erratically by picking one arbitrarily).

Elastic-net tends to **group** correlated features.

---

### 2.1.3 Logistic Regression (Binary Classification)

For binary labels $y_i\in\{0,1\}$, we model:
$$
p_\theta(y=1\mid x)=\sigma(w^\top x+b),\quad \sigma(z)=\frac{1}{1+e^{-z}}.
$$

The likelihood of the dataset:
$$
p(y\mid X,\theta)=\prod_{i=1}^n\sigma(z_i)^{y_i}(1-\sigma(z_i))^{1-y_i},\quad z_i=w^\top x_i+b.
$$

Negative log-likelihood (cross-entropy loss):

$$
\mathcal{L}(\theta)
=-\sum_{i=1}^n\Big[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\Big].
$$

We minimize $\mathcal{L}(\theta)$ via gradient-based methods (no closed form).

Decision rule:

$$
\hat{y}(x)=
\begin{cases}
1 & \text{if }\sigma(w^\top x + b)\ge 0.5,\\
0 & \text{otherwise}.
\end{cases}
$$

Geometric picture:

* Decision boundary $w^\top x+b=0$ is a **hyperplane**.
* Logistic regression is a **linear classifier in feature space**, but with probabilistic outputs.

---

### 2.1.4 Multinomial Logistic Regression (Softmax)

For $K$ classes $y\in\{1,\dots,K\}$, we define scores:
$$
s_k(x)=w_k^\top x+b_k.
$$

Softmax probabilities:
$$
p(y=k\mid x)=\frac{\exp(s_k(x))}{\sum_{j=1}^K\exp(s_j(x))}.
$$

Loss (cross-entropy):

$$
\mathcal{L}(\theta)=-\sum_{i=1}^n\log p(y_i\mid x_i).
$$

Again, we optimize via gradient descent. This is the classical “softmax classifier” and is extensively used as the final layer even in deep nets (but here we stay in the linear setting).

---

### 2.1.5 Regularization in Linear and Logistic Models: When L1 vs L2?

You can regularize logistic regression just like linear regression:

* L2-regularized logistic:
  $$
  \mathcal{L}(\theta)
  =-\sum_{i=1}^n\log p(y_i\mid x_i)+\frac{\lambda}{2}\lVert w\rVert_2^2.
  $$
* L1-regularized logistic:
  $$
  \mathcal{L}(\theta)
  =-\sum_{i=1}^n\log p(y_i\mid x_i)+\lambda\lVert w\rVert_1.
  $$

Heuristics:

* Prefer **L2** when:

  * You believe most features are at least mildly useful.
  * You care less about sparsity and more about smooth shrinkage.
  * Features are dense and not extremely high-dimensional.

* Prefer **L1** when:

  * You expect many irrelevant features.
  * Feature dimension is large compared to number of samples.
  * Model interpretability / feature selection is important.

* Prefer **Elastic-Net** when:

  * You want sparsity but also robustness with correlated features.
  * Pure Lasso tends to pick one of several correlated features arbitrarily.

---

> Quick check:

Suppose you have a dataset with **10,000 features**, many of which you suspect are irrelevant, and you care about learning which features matter. Would you lean toward:

* Ridge,
* Lasso,
* or Elastic-Net,

and why, in one or two sentences?

---

## 2.2 Tree-Based Models

Tree-based methods are extremely common in industry (tabular data, ranking, ads, risk models). You should know:

* How a decision tree is built (splitting criteria, recursion, stopping).
* Why ensembles like Random Forests and Gradient Boosting work.
* What makes XGBoost/LightGBM/CatBoost fast and strong.

---

### 2.2.1 Decision Trees (CART)

A decision tree recursively partitions the feature space into regions and assigns a prediction (constant) to each region.

Consider a binary tree where each internal node corresponds to a test of the form:
$$
x_j \le t \quad \text{(go left) or (go right)}.
$$

Leaves correspond to regions $R_m$, each with a prediction:

* Regression: $\hat{y}(x) = \text{mean of } y_i \text{ in region } R_m$.
* Classification: $\hat{y}(x) = \arg\max_k \hat{p}_k$ in region $R_m$.

#### Splitting criterion

At each node, we choose the split $(j,t)$ that maximizes decrease in impurity.

For classification, common impurity measures for a node with class proportions $p_k$:

* Gini:
  $$
  G = \sum_{k} p_k (1 - p_k) = 1 - \sum_k p_k^2.
  $$
* Entropy:
  $$
  H = - \sum_k p_k \log p_k.
  $$

For regression, impurity is often the variance (MSE) within the node:
$$
\text{MSE} = \frac{1}{N} \sum_{i \in \text{node}} (y_i - \bar{y})^2.
$$

Given a candidate split that partitions data at node into left and right subsets $L$ and $R$, with sizes $N_L$ and $N_R$, impurity after split is:

$$
I_{\text{split}} = \frac{N_L}{N} I(L) + \frac{N_R}{N} I(R),
$$

and we choose the split minimizing $I_{\text{split}}$ (or equivalently maximizing impurity reduction).

#### Recursion and stopping

Algorithm:

1. Start at root node with all data.
2. Find best split (feature, threshold).
3. Partition data into children.
4. Recurse on each child until stopping criteria:

   * Max depth reached.
   * Node has too few samples.
   * No split reduces impurity “enough.”
   * Node is pure (all labels same).

Decision trees are:

* Low bias (can approximate complex functions).
* High variance (small changes in data can drastically change the tree).

This motivates ensembles.

---

### 2.2.2 Random Forests (Bagging)

Random Forests reduce variance of trees via averaging.

Idea:

* Train many decision trees on different bootstrapped samples of the data.
* At each split, consider a random subset of features.

Formally:

1. For $b = 1,\dots,B$:

   * Sample a bootstrap dataset $D^{(b)}$ by drawing $n$ examples with replacement from the original training set.
   * Train a decision tree on $D^{(b)}$, but at each node:

     * Randomly select $m \ll d$ features, and search for the best split only among those $m$.
2. For prediction:

   * Regression: average predictions:
     $$
     \hat{f}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B f^{(b)}(x).
     $$
   * Classification: majority vote or average class probabilities.

Why it works:

* Each tree is a high-variance estimator.
* Averaging reduces variance as long as individual trees are not perfectly correlated.
* Feature subsampling at each split decorrelates trees further.

Important concept: **Out-of-bag (OOB) error**:

* For each example, consider only trees where that example was not in the bootstrap sample.
* Use them to get an OOB prediction and estimate generalization error without separate validation set.

---

### 2.2.3 Gradient Boosting (GBM Concepts)

Boosting builds an ensemble sequentially, where each new model tries to correct the residuals/errors of the current ensemble.

General gradient boosting framework:

We want to approximate a function $F^*(x)$ by an additive model:
$$
F_M(x) = \sum_{m=1}^M \gamma_m h_m(x),
$$
where $h_m$ are base learners (typically shallow trees, “decision stumps” or depth 3–8 trees).

We minimize an empirical loss:
$$
\mathcal{L}(F) = \sum_{i=1}^n \ell(y_i, F(x_i)).
$$

At iteration $m$, we have current model $F_{m-1}$. We add a new tree $h_m$ to reduce the loss. Using functional gradient descent:

1. Compute pseudo-residuals:
   $$
   r_{im} = - \left.\frac{\partial \ell(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}}.
   $$
   These are the negative gradients of the loss wrt the model predictions.

2. Fit a tree $h_m$ to predict $r_{im}$ from $x_i$:
   $$
   h_m = \arg\min_h \sum_{i=1}^n (r_{im} - h(x_i))^2.
   $$

3. Optionally, find an optimal step size $\gamma_m$ via line search:
   $$
   \gamma_m = \arg\min_\gamma \sum_{i=1}^n \ell\left(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)\right).
   $$

4. Update:
   $$
   F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x),
   $$
   where $\nu \in (0,1]$ is the learning rate (shrinkage).

For squared loss $\ell(y, F(x)) = \frac{1}{2}(y - F(x))^2$:

* Pseudo-residuals $r_{im} = y_i - F_{m-1}(x_i)$ (actual residuals).
* So boosting is directly fitting residuals.

Properties:

* Low bias (additive ensemble of trees).
* Can achieve strong performance but prone to overfitting if not regularized (learning rate, tree depth, number of trees).

---

### 2.2.4 XGBoost, LightGBM, CatBoost Fundamentals

All three are gradient boosting frameworks with engineering and algorithmic optimizations.

#### XGBoost

Key ideas:

* Uses second-order Taylor approximation of loss for each leaf split decision:

  * For each example $i$ in a node:

    * Gradient $g_i = \partial \ell / \partial F(x_i)$
    * Hessian $h_i = \partial^2 \ell / \partial F(x_i)^2$
* For a candidate leaf with examples $I$, it defines a regularized objective contribution:
  $$
  \tilde{\mathcal{L}} = -\frac{1}{2} \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} + \gamma,
  $$
  where $\lambda$ and $\gamma$ are regularization parameters.
* Fast approximate split finding via histogram-based methods:

  * Continuous features are bucketed into bins.
  * Split search is done over bins instead of raw values.

Regularization:

* L2 regularization on leaf weights.
* Tree complexity penalties (e.g., number of leaves).

Engineering:

* Sparse-aware (handles missing values efficiently).
* Column block structure, cache-aware.

#### LightGBM

Optimized for speed and memory on large datasets.

Key differences:

* **Leaf-wise growth** with depth constraints:

  * It grows the tree by always splitting the leaf that yields the largest loss reduction (best-first).
  * This can produce deeper trees with fewer leaves than level-wise growth for the same loss reduction.
* **Histogram-based splitting** with efficient binning.
* **Gradient-based one-side sampling (GOSS)**:

  * Keeps all instances with large gradients (hard examples).
  * Randomly samples from small gradient instances.
  * Maintains approximation of full-data gradient while reducing computation.

This makes LightGBM fast on large-scale tabular data and often very performant.

#### CatBoost

Designed to handle categorical features and reduce prediction shift:

* Uses **ordered boosting**:

  * Avoids using target statistics that leak information from the future.
  * Constructs permutations of data and computes target encodings using only “past” data in that permutation.
* Handles categorical variables natively with target statistics:

  * E.g., encodes a category as a smoothed average of the target for that category.

The main selling point: strong performance on datasets with many categorical features, with less need for manual preprocessing.

---

If we summarize tree-based models:

* Decision trees: expressive but high variance and prone to overfitting.
* Random Forests: reduce variance via bagging and feature randomness.
* Gradient Boosting (XGBoost/LightGBM/CatBoost): sequentially fit residuals/gradients, often achieve state-of-the-art on tabular data, at risk of overfitting without careful regularization.

---

> one quick reflection question for you to keep this active in your head:

Why do Random Forests primarily help with **variance** rather than **bias**? Answer in one or two sentences, using the language of “complex trees” and “averaging.”

---

## 2.3 Support Vector Machines (SVMs)

SVMs are margin-based classifiers. They try to find a hyperplane that separates classes with **maximum margin**, which empirically improves generalization.

### 2.3.1 Hard-Margin SVM (Linearly Separable Case)

Binary labels $y_i \in \{-1, +1\}$, features $x_i \in \mathbb{R}^d$.

We seek a separating hyperplane $w^\top x + b = 0$ such that:
$$
y_i (w^\top x_i + b) \ge 1 \quad \forall i,
$$
and we maximize the geometric margin $1/\lVert w \rVert$, equivalently minimize $\frac{1}{2}\lVert w \rVert^2$.

Primal problem:
$$
\begin{aligned}
\min_{w,b} \quad & \frac{1}{2}\lVert w \rVert^2 \\
\text{s.t.} \quad & y_i (w^\top x_i + b) \ge 1, \; i = 1,\dots,n.
\end{aligned}
$$

Only points on the margin boundaries (the **support vectors**) influence the solution.

---

### 2.3.2 Soft-Margin SVM (Non-Separable Case)

We introduce slack variables $\xi_i \ge 0$ to allow margin violations:

$$
\begin{aligned}
\min_{w,b,{\xi_i}} \quad & \frac{1}{2}\lVert w \rVert^2 + C \sum_{i=1}^n \xi_i \\
\text{s.t.} \quad & y_i (w^\top x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0.
\end{aligned}
$$

Here:

* $\frac{1}{2}\lVert w \rVert^2$ controls margin (complexity).
* $\sum \xi_i$ penalizes violations/misclassifications.
* $C > 0$ trades off margin size vs training error.

Large $C$: prioritize fitting training data (lower bias, potentially higher variance).  
Small $C$: allow more violations, larger margin (higher bias, lower variance).

---

### 2.3.3 Dual Formulation and Kernel Trick

The dual of the soft-margin problem gives:

$$
\begin{aligned}
\max_{\alpha} \quad & \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \, K(x_i, x_j) \\
\text{s.t.} \quad & 0 \le \alpha_i \le C, \\
& \sum_{i=1}^n \alpha_i y_i = 0.
\end{aligned}
$$

Here $K(x_i, x_j) = x_i^\top x_j$ is the inner product in feature space. In the kernelized SVM, we replace the inner product by a **kernel function**:

$$
K(x, x') = \phi(x)^\top \phi(x'),
$$

for some implicit feature map $\phi$. We never need to compute $\phi(x)$ explicitly.

Decision function:

$$
f(x) = \text{sign}\Big(\sum_{i=1}^n \alpha_i y_i K(x_i, x) + b\Big).
$$

Only points with $\alpha_i > 0$ (support vectors) appear in the sum.

---

### 2.3.4 Common Kernels and Geometric Intuition

Typical kernels:

* Linear: $K(x, x') = x^\top x'$.
* Polynomial: $K(x, x') = (\gamma x^\top x' + r)^d$.
* RBF (Gaussian): $K(x, x') = \exp(-\gamma \lVert x - x'\rVert^2)$.

Geometric view:

* Linear SVM: finds maximum-margin hyperplane in original feature space.
* Kernel SVM: implicitly maps data into a high (possibly infinite) dimensional feature space and finds a **linear** separator there, which corresponds to a nonlinear decision boundary in input space.

Interview-level message: SVMs maximize the margin and use kernels to handle nonlinearity while keeping optimization in terms of pairwise similarities.

---

## 2.4 Nearest-Neighbor Methods

Nearest-neighbor methods are conceptually simple, non-parametric baselines that can be surprisingly strong when you have good representations and low-to-moderate dimensions.

### 2.4.1 k-Nearest Neighbors (k-NN)

Given a query point $x$:

1. Find its $k$ nearest neighbors among training points under some distance metric $d(\cdot,\cdot)$.
2. For classification:
   * Predict the majority label among neighbors (possibly weighted by distance).
3. For regression:
   * Predict the average of neighbor labels.

Formally, let $N_k(x)$ be the index set of the $k$ nearest training points. Then:

* Classification prediction:
  $$
  \hat{y}(x) = \arg\max_{c} \sum_{i \in N_k(x)} \mathbf{1}{y_i = c}.
  $$

* Regression prediction:
  $$
  \hat{y}(x) = \frac{1}{k} \sum_{i \in N_k(x)} y_i.
  $$

Properties:

* **Lazy learning**: no explicit training; all computation at query time.
* As $n \to \infty$ and $k$ grows appropriately, k-NN can be consistent (risk approaches Bayes optimal).
* Suffers badly from curse of dimensionality in raw feature space.

---

### 2.4.2 Distance Metrics and Their Effects

Common distances:

* Euclidean: $d(x, x') = \lVert x - x' \rVert_2$.
* Manhattan: $d(x, x') = \lVert x - x' \rVert_1$.
* Cosine distance: $1 - \frac{x^\top x'}{\lVert x \rVert \lVert x' \rVert}$.
* Problem-specific metrics (e.g., learned embeddings).

The choice of distance + feature representation **defines the inductive bias**: what it means for two points to be “similar.”

In high dimensions, raw Euclidean distance is often not meaningful; in practice, k-NN is typically used on top of learned embeddings (e.g., in retrieval systems, metric learning).

Data structures:

* KD-trees, Ball trees, and ANN structures (like HNSW, IVF-PQ) are used to speed up neighbor search.

---

## 2.5 Clustering

Clustering partitions data into groups such that points in the same group are more similar to each other than to those in other groups. This is unsupervised; no ground-truth cluster labels.

### 2.5.1 k-Means Clustering (Lloyd’s Algorithm)

Goal: Given $K$ clusters, find centroids $\{\mu_1, \dots, \mu_K\}$ minimizing within-cluster squared distances:

$$
\min_{{c_i},{\mu_k}} \sum_{i=1}^n \lVert x_i - \mu_{c_i} \rVert^2.
$$

where $c_i \in \{1, \dots, K\}$ is the cluster assignment for $x_i$.

Lloyd’s algorithm:

1. Initialize centroids $\mu_k$ (e.g., randomly or via k-means++).
2. Assignment step:
   $$
   c_i \leftarrow \arg\min_k \lVert x_i - \mu_k \rVert^2.
   $$
3. Update step:
   $$
   \mu_k \leftarrow \frac{1}{|{i: c_i = k}|} \sum_{i: c_i = k} x_i.
   $$
4. Repeat steps 2–3 until convergence (assignments stop changing or objective change is small).

k-means++ initialization picks initial centroids with probability proportional to squared distance from existing centroids, which greatly improves robustness.

Limitations:

* Assumes spherical clusters with similar variance.
* Sensitive to choice of $K$ and scaling of features.
* Finds local minima of the objective.

---

### 2.5.2 Gaussian Mixture Models (GMMs) and EM

GMMs generalize k-means by modeling clusters as Gaussians with full covariance matrices.

Model:

$$
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k),
$$

with mixture weights $\pi_k \ge 0$ and $\sum_k \pi_k = 1$.

We want to estimate parameters $\{\pi_k, \mu_k, \Sigma_k\}$ via maximum likelihood. This is often done with the EM algorithm.

EM algorithm:

* Introduce latent cluster assignments $z_i \in \{1,\dots,K\}$ indicating which component generated $x_i$.
* E-step: compute responsibilities
  $$
  \gamma_{ik} = p(z_i = k \mid x_i) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}.
  $$
* M-step: update parameters using responsibilities:
  $$
  N_k = \sum_{i=1}^n \gamma_{ik},
  $$
  $$
  \mu_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} x_i,
  $$
  $$
  \Sigma_k = \frac{1}{N_k} \sum_{i=1}^n \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top,
  $$
  $$
  \pi_k = \frac{N_k}{n}.
  $$

k-means can be seen as a special case of GMM with spherical covariances and hard assignments.

---

### 2.5.3 DBSCAN and Hierarchical Clustering (Intuition)

DBSCAN (density-based clustering):

* Defines clusters as dense regions separated by low-density regions.
* Parameters: neighborhood radius $\varepsilon$ and minimum points $\text{minPts}$.
* A point is a “core” point if it has at least $\text{minPts}$ neighbors within $\varepsilon$.
* Clusters are formed by connecting core points that are density-reachable.
* Can find arbitrarily shaped clusters and detect noise/outliers.

Hierarchical clustering:

* Agglomerative (bottom-up):
  * Start with each point as its own cluster.
  * Iteratively merge the two closest clusters according to a linkage criterion (single, complete, average).
* Produces a dendrogram (cluster tree); you can cut at different levels for different numbers of clusters.

Both are often used in exploratory analysis and when cluster shapes are non-spherical or unknown.

---

## 2.6 Probabilistic Models (Naive Bayes, HMMs, CRFs – High Level)

### 2.6.1 Naive Bayes

For classification with features $x = (x_1, \dots, x_d)$ and label $y$, Naive Bayes assumes:

$$
p(x \mid y) = \prod_{j=1}^d p(x_j \mid y).
$$

Using Bayes rule:

$$
p(y \mid x) \propto p(y) \prod_{j=1}^d p(x_j \mid y).
$$

Different variants:

* Multinomial NB: common for text (features are word counts).
* Gaussian NB: continuous features modeled as Gaussians.

Pros:

* Very fast to train.
* Low sample complexity due to strong independence assumptions.

Cons:

* Independence assumption rarely holds; can limit accuracy.

---

### 2.6.2 Hidden Markov Models (HMMs) – Basics

HMMs model sequences with hidden states:

* Hidden state sequence $(z_1,\dots,z_T)$.
* Observations $(x_1,\dots,x_T)$.

Assumptions:

* Markov on hidden states:
  $$
  p(z_t \mid z_{1:t-1}) = p(z_t \mid z_{t-1}).
  $$
* Emissions independent given state:
  $$
  p(x_t \mid z_{1:t}, x_{1:t-1}) = p(x_t \mid z_t).
  $$

Parameters:

* Initial distribution $\pi_k = p(z_1 = k)$.
* Transition matrix $A_{kl} = p(z_{t+1} = l \mid z_t = k)$.
* Emission distributions $p(x_t \mid z_t = k)$.

Tasks:

* Likelihood: $p(x_{1:T})$ (forward algorithm).
* Decoding: most likely hidden state sequence (Viterbi).
* Learning: EM (Baum–Welch).

---

### 2.6.3 CRFs (Very Light)

Conditional Random Fields (CRFs) are discriminative models for structured outputs, e.g., sequence labeling:

$$
p(y \mid x) \propto \exp\left(\sum_k \lambda_k f_k(y, x)\right),
$$

where $f_k$ are feature functions over input–output configurations (like label transitions, label–token pairs).

Compared to HMMs:

* HMM: generative over $(x,y)$, with specific independence assumptions.
* CRF: directly models $p(y \mid x)$, allowing rich, overlapping features of the input without modeling $p(x)$.

In interviews, high-level recognition of “CRF = discriminative, HMM = generative sequence model” is usually enough unless you are in structured prediction roles.

---

## 2.7 Dimensionality Reduction

Dimensionality reduction seeks a low-dimensional representation of data that preserves important structure.

### 2.7.1 PCA (Principal Component Analysis)

Given centered data $X \in \mathbb{R}^{n \times d}$ (columns mean zero), PCA finds directions of maximal variance.

Compute the covariance matrix:
$$
\Sigma = \frac{1}{n} X^\top X.
$$

Eigen-decomposition:
$$
\Sigma = V \Lambda V^\top,
$$
where columns of $V$ are eigenvectors (principal components) and $\Lambda$ diagonal with eigenvalues.

To reduce to $k$ dimensions:

* Take top $k$ eigenvectors $V_k$.
* Project data: $Z = X V_k$.

Equivalent view: SVD of $X$:
$$
X = U S V^\top,
$$
and $V_k$ are top right singular vectors.

Explained variance ratio:

$$
\text{Explained variance by first } k \text{ PCs} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}.
$$

---

### 2.7.2 Kernel PCA (High-Level)

Kernel PCA extends PCA to nonlinear embeddings via a kernel function $K(x_i, x_j)$.

Key idea: perform PCA in feature space $\phi(x)$ implicitly:

* Construct kernel matrix $K$ with entries $K_{ij} = K(x_i, x_j)$.
* Center and eigen-decompose $K$.
* The top eigenvectors correspond to principal components in feature space.

This yields nonlinear projections without ever computing $\phi(x)$ explicitly.

---

### 2.7.3 t-SNE and UMAP (Intuition)

t-SNE:

* Converts high-dimensional pairwise distances into probabilities $p_{ij}$ (similarity of points $i$ and $j$).
* Finds low-dimensional embeddings $y_i$ whose pairwise similarities $q_{ij}$ match $p_{ij}$ as closely as possible.
* Objective: minimize KL divergence $\text{KL}(P \parallel Q)$.
* Good for visualization (2D/3D), but not a general-purpose embedding for downstream supervised tasks (non-parametric, non-linear, non-metric).

UMAP:

* Also builds a graph of local neighborhoods in high-dimensional space.
* Optimizes a low-dimensional embedding preserving this graph structure.
* Often faster than t-SNE and can preserve more of the global structure.

Both are mainly used for **visualizing** high-dimensional data (e.g., embedding spaces) rather than as features for production models.

---

## 3. Model Evaluation & Metrics

Evaluation is about turning model behavior into numbers that reflect product objectives. You should always think: “What quantity is this metric actually estimating?”

---

### 3.1 Classification Metrics

#### 3.1.1 Confusion Matrix

For binary classification with positive class $1$ and negative class $0$:

* True Positives (TP): predicted $1$, true $1$.
* False Positives (FP): predicted $1$, true $0$.
* False Negatives (FN): predicted $0$, true $1$.
* True Negatives (TN): predicted $0$, true $0$.

Most metrics are functions of TP, FP, FN, TN.

Formally, for a set of predictions ${\hat{y}_i}$ and labels ${y_i}$:

* $\text{TP}=\sum_i \mathbf{1}{y_i = 1, \hat{y}_i = 1}$, etc.

---

#### 3.1.2 Accuracy vs Precision/Recall

**Accuracy**

$$
\text{Accuracy}=\frac{\text{TP}+\text{TN}}{\text{TP}+\text{FP}+\text{FN}+\text{TN}}.
$$

Good when:

* Classes are balanced.
* FP and FN have similar cost.

Misleading under class imbalance (e.g., 99% negatives → trivial classifier gets 99% accuracy).

**Precision and Recall**

Precision (Positive Predictive Value):

$$
\text{Precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}.
$$

Interpretation: Among all predicted positives, how many are truly positive?

Recall (Sensitivity, True Positive Rate):

$$
\text{Recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}.
$$

Interpretation: Among all actual positives, how many did we catch?

In many applications (fraud, medical alerts, abuse detection, search relevance), you choose a threshold on model scores to trade off precision vs recall depending on costs.

---

#### 3.1.3 F1, Macro/Micro Averaging

**F1 Score**

The F1 score is the harmonic mean of precision and recall:

$$
\text{F1}=2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}.
$$

Why harmonic mean? It penalizes extreme imbalance (e.g., very high precision and very low recall).

Multi-class and multi-label:

* **Macro-average**: compute metric per class, then average:

  $$
  \text{F1}_{\text{macro}}=\frac{1}{K}\sum_{k=1}^K \text{F1}_k.
  $$

  Treats all classes equally, even rare ones.

* **Micro-average**: aggregate TP, FP, FN across classes first, then compute a single precision/recall/F1:

  $$
  \text{Precision}_{\text{micro}}=\frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k+\text{FP}_k)},\quad
  \text{Recall}_{\text{micro}}=\frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k+\text{FN}_k)}.
  $$

  Dominated by large/majority classes.

Choice depends on whether you care more about overall volume (micro) or per-class fairness/coverage (macro).

---

#### 3.1.4 ROC Curves and AUC

Most modern classifiers output scores $s(x)$ interpreted as confidence. By varying a threshold $t$, you get different predictions.

For each threshold $t$, define:

* True Positive Rate (TPR, same as recall):

  $$
  \text{TPR}(t)=\frac{\text{TP}(t)}{\text{TP}(t)+\text{FN}(t)}.
  $$

* False Positive Rate (FPR):

  $$
  \text{FPR}(t)=\frac{\text{FP}(t)}{\text{FP}(t)+\text{TN}(t)}.
  $$

An ROC curve plots $\text{TPR}(t)$ versus $\text{FPR}(t)$ as $t$ sweeps.

**AUC (Area Under ROC Curve)**:

* Interpreted as the probability that the classifier ranks a random positive higher than a random negative:

  $$
  \text{AUC}=\mathbb{P}(s(x^+)>s(x^-)).
  $$

* Threshold-independent; focuses on ranking quality between positive and negative examples.

ROC/AUC is less informative under extreme class imbalance because FPR can be tiny even if FP count is large in absolute terms.

---

#### 3.1.5 Precision-Recall Curves and PR-AUC

In highly imbalanced settings, precision–recall curves are often more informative.

* For each threshold $t$, compute $\text{precision}(t)$ and $\text{recall}(t)$.
* Plot precision vs recall.

**PR-AUC** is simply the area under the precision–recall curve.

Interpretation:

* Focuses on performance on the positive class.
* More sensitive to performance on rare classes than ROC-AUC.

Heuristic:
Use ROC-AUC when classes are balanced and costs are symmetric; use PR-AUC when the positive class is rare and you care about “how many of the flagged items are truly positive”.

---

#### 3.1.6 Calibration and Reliability

A classifier is **calibrated** if its predicted probabilities match empirical frequencies. For all scores $p$,

$$
\mathbb{P}(Y=1\mid \hat{p}(X)=p)\approx p.
$$

For example, among all examples where the model outputs $0.7$, about $70\%$ should be positive.

**Reliability diagram**:

* Bucket predictions into bins (e.g., $[0,0.1),[0.1,0.2),\dots$).
* For each bin, plot:

  * x-axis: mean predicted probability.
  * y-axis: empirical fraction of positives.

Perfect calibration lies on the diagonal.

**Brier score** (for binary):

$$
\text{Brier}=\frac{1}{n}\sum_{i=1}^n(\hat{p}_i-y_i)^2.
$$

Smaller is better. It mixes calibration and sharpness (how extreme probabilities are).

Calibration matters a lot when:

* You feed probabilities into downstream decision systems (pricing, risk, bids).
* You combine multiple models or simulate outcomes.

---

### 3.2 Regression Metrics

Regression metrics quantify how close predictions $\hat{y}_i$ are to real-valued targets $y_i$.

Let $n$ examples with labels $y_i$ and predictions $\hat{y}_i$.

#### 3.2.1 MSE, RMSE, MAE, MAPE

* Mean Squared Error (MSE):

  $$
  \text{MSE}=\frac{1}{n}\sum_{i=1}^n (y_i-\hat{y}_i)^2.
  $$

* Root Mean Squared Error (RMSE):

  $$
  \text{RMSE}=\sqrt{\text{MSE}}.
  $$

MSE/RMSE heavily penalize large errors; sensitive to outliers.

* Mean Absolute Error (MAE):

  $$
  \text{MAE}=\frac{1}{n}\sum_{i=1}^n |y_i-\hat{y}_i|.
  $$

More robust to outliers; corresponds to median estimator under Laplace noise.

* Mean Absolute Percentage Error (MAPE):

  $$
  \text{MAPE}=\frac{100}{n}\sum_{i=1}^n\left|\frac{y_i-\hat{y}_i}{y_i}\right|.
  $$

Interpretable as average percentage error, but undefined when $y_i=0$ and unstable for small $|y_i|$.

Choice in practice:

* Use MSE/RMSE when large errors are particularly bad.
* Use MAE when you care about median-like behavior and robustness.
* Use MAPE when scale-free percentage interpretation is important and $y_i$ are positive and not near zero.

---

#### 3.2.2 $R^2$ and Adjusted $R^2$

Let $\bar{y}=\frac{1}{n}\sum_i y_i$.

* Total Sum of Squares (TSS):

  $$
  \text{TSS}=\sum_{i=1}^n (y_i-\bar{y})^2.
  $$

* Residual Sum of Squares (RSS):

  $$
  \text{RSS}=\sum_{i=1}^n (y_i-\hat{y}_i)^2.
  $$

Coefficient of determination:

$$
R^2=1-\frac{\text{RSS}}{\text{TSS}}.
$$

Interpretation:

* Fraction of variance in $y$ explained by the model compared to a constant baseline $\hat{y}=\bar{y}$.
* $R^2=0$ means no better than predicting the mean; negative means worse.

Adjusted $R^2$ penalizes adding more predictors:

$$
R^2_{\text{adj}}=1-\frac{\text{RSS}/(n-p-1)}{\text{TSS}/(n-1)},
$$

where $p$ is the number of predictors. It discourages overfitting by not rewarding useless features.

---

### 3.3 Ranking / Search Metrics

In ranking/search, evaluation is list-based. The model outputs an ordering of items for each query; metrics measure the quality of that ranked list.

Let $q$ be a query, with associated items $d_1,\dots,d_n$ and relevance labels $rel_i$ (e.g., $0,1,2,\dots$).

#### 3.3.1 DCG and nDCG

Discounted Cumulative Gain (DCG) at cut-off $k$:

$$
\text{DCG@}k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}.
$$

* Higher relevance at higher ranks contributes more.
* Log discount penalizes items that appear lower in the list.

Ideal DCG (IDCG@k) is DCG of the ideal ranking (sorted by true relevance).

Normalized DCG (nDCG@k):

$$
\text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}.
$$

Thus $\text{nDCG@}k\in[0,1]$ and accounts for both relevance and ranking position. It is the standard metric for search/recommendation ranking.

---

#### 3.3.2 MRR (Mean Reciprocal Rank)

For tasks where you care about the position of the first relevant item (e.g., question answering):

* For query $q$, let $r_q$ be rank of the first relevant item (smallest index where $rel_i>0$).

* Reciprocal Rank:

  $$
  \text{RR}_q=\frac{1}{r_q}.
  $$

* Mean Reciprocal Rank:

  $$
  \text{MRR}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} \text{RR}_q.
  $$

High MRR means relevant items appear very early.

---

#### 3.3.3 MAP (Mean Average Precision) and Hit/Recall@k

Average Precision (AP) for one query:

* For each position $i$ where $d_i$ is relevant, compute precision up to $i$,
  $$
  P(i)=\frac{\#\text{relevant in top } i}{i}.
  $$
* AP is the average of $P(i)$ over all relevant items.

Then Mean Average Precision:

$$
\text{MAP}=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}} \text{AP}_q.
$$

HitRate@k:

* Indicator of whether any relevant item appears in top $k$:
  $$
  \text{Hit@}k(q)=
  \begin{cases}
  1,&\text{if any relevant item is in top }k,\\
  0,&\text{otherwise}.
  \end{cases}
  $$

* Average over queries.

Recall@k:

$$
\text{Recall@}k(q)=\frac{\#\text{relevant items in top }k}{\#\text{relevant items overall}}.
$$

Hit@k is a special case of Recall@k when there is at most one relevant item.

---

### 3.4 Probabilistic Predictions and Proper Scoring Rules

When a model outputs probabilities $\hat{p}$ instead of hard labels, you want metrics that reward both sharpness and calibration.

#### 3.4.1 Log-Loss (Cross-Entropy)

Binary case:

$$
\text{LogLoss}=-\frac{1}{n}\sum_{i=1}^n \big[y_i\log \hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\big].
$$

Multi-class case (with one-hot labels):

$$
\text{LogLoss}=-\frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K y_{ik}\log \hat{p}_{ik}.
$$

Log-loss is a **strictly proper scoring rule**:

* The expected log-loss is minimized when $\hat{p}$ equals the true conditional probability.
* Encourages honest probabilities and penalizes overconfident errors heavily.

#### 3.4.2 Brier Score

Already mentioned for calibration, but more general:

Binary:

$$
\text{Brier}=\frac{1}{n}\sum_{i=1}^n (\hat{p}_i-y_i)^2.
$$

Multi-class:

$$
\text{Brier}=\frac{1}{n}\sum_{i=1}^n\sum_{k=1}^K (\hat{p}_{ik}-y_{ik})^2.
$$

Also a proper scoring rule, but less harsh on very wrong, overconfident predictions than log-loss.

---

### 3.5 Cross-Validation and Data Leakage

#### 3.5.1 k-Fold and Stratified CV

k-fold cross-validation:

1. Split dataset into $k$ folds.
2. For each fold $j$:

   * Train on $k-1$ folds.
   * Evaluate on fold $j$.
3. Average metrics over folds.

Stratified k-fold:

* Ensures each fold has approximately the same label distribution (important for classification with imbalance).

Cross-validation estimates generalization performance more robustly than a single train/validation split, especially for small datasets.

---

#### 3.5.2 Time-Series CV

For temporal data, random shuffling breaks causality. Use time-aware splits:

* Train on $[t_0,t_1]$, validate on $(t_1,t_2]$.
* Rolling or expanding windows: simulate how the model would operate in production.

Key principle: never train on future data relative to the evaluation period.

---

#### 3.5.3 Data Leakage Pitfalls

Data leakage happens when information from the test/validation set leaks into training, causing overly optimistic metrics.

Common sources:

* Preprocessing (scaling, imputation, feature selection) fit on full data instead of train only.
* Cross-validation where target-derived features (e.g., mean target encoding) are computed using all folds.
* Time-series tasks where you accidentally use future aggregates.

In an interview, if asked about poor production performance vs offline metrics, you should always consider leakage as a top hypothesis.

---

> quick check for yourself:

If your binary classifier has very high ROC-AUC but poor PR-AUC on a highly imbalanced dataset, what does that tell you about how it ranks positives vs negatives, and why might PR-AUC be low despite good ROC-AUC? Try to phrase it in 2–3 sentences.

---

## 4. Optimization & Training Dynamics

In interviews, this section is about showing you understand **how** models are actually trained, not just the final closed-form solutions.

Assume a generic empirical risk minimization problem:

$$
\min_\theta J(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i) + \lambda R(\theta).
$$

---

### 4.1 Gradient Descent, SGD, and Momentum

#### 4.1.1 Batch Gradient Descent

At each iteration, you compute the gradient on the **full dataset**:

$$
\nabla_\theta J(\theta) = \frac{1}{n} \sum_{i=1}^n \nabla_\theta \ell(f_\theta(x_i), y_i) + \lambda \nabla_\theta R(\theta),
$$

and update:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t),
$$

where $\eta > 0$ is the learning rate.

Pros:

* Stable, monotonic decrease for convex losses with small enough $\eta$.
* Easy to analyze theoretically.

Cons:

* Very expensive per step for large $n$.
* Not used directly in large-scale settings; replaced by mini-batch methods.

---

#### 4.1.2 Stochastic and Mini-Batch Gradient Descent

Instead of using all data each step, use a single example or a mini-batch $B_t$.

SGD (single sample):

$$
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \ell(f_\theta(x_{i_t}), y_{i_t}),
$$

where $i_t$ is a random index.

Mini-batch SGD:

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{1}{|B_t|} \sum_{i \in B_t} \nabla_\theta \ell(f_\theta(x_i), y_i).
$$

Key ideas:

* Cheap per iteration; each step is noisy but trends toward the minimizer.
* With appropriate learning rate schedule $\{\eta_t\}$, SGD converges to a neighborhood of a local minimum (and to the global minimum in strongly convex settings).

In practice, mini-batches give a good trade-off between noise (helps escape shallow local minima) and compute efficiency.

---

#### 4.1.3 Momentum

Plain SGD can be slow when the loss landscape has narrow valleys: gradients oscillate across steep directions.

Momentum introduces an exponentially decaying moving average of gradients:

$$
v_{t+1} = \beta v_t + (1 - \beta)\nabla_\theta J(\theta_t),
$$
$$
\theta_{t+1} = \theta_t - \eta v_{t+1},
$$

with $\beta \in [0,1)$ (e.g., 0.9). Intuition:

* $v_t$ is like a “velocity” that accumulates gradients over time.
* In consistent descent directions, momentum accelerates.
* In oscillating directions, positive and negative gradients cancel.

Nesterov momentum is a slight variant where you look ahead before computing the gradient, improving theoretical convergence in convex cases. Conceptually similar in practice.

---

### 4.2 Newton’s Method and Quasi-Newton (BFGS, L-BFGS)

Gradient descent uses **first-order** information only. Newton’s method uses curvature (second derivatives) for faster local convergence.

#### 4.2.1 Newton Step

Assume $J(\theta)$ is twice differentiable. Around $\theta_t$, use a second-order Taylor expansion:

$$
J(\theta) \approx J(\theta_t) + \nabla J(\theta_t)^\top (\theta - \theta_t) + \frac{1}{2} (\theta - \theta_t)^\top H_t (\theta - \theta_t),
$$

where $H_t = \nabla^2 J(\theta_t)$ is the Hessian.

Minimizing this quadratic approximation yields:

$$
\theta_{t+1} = \theta_t - H_t^{-1} \nabla J(\theta_t).
$$

If you are near a local minimum and the Hessian is positive definite, Newton’s method converges **quadratically** (very fast).

Problems:

* Computing and inverting $H_t \in \mathbb{R}^{d \times d}$ is expensive for large $d$.
* Requires careful damping/line search in non-convex problems to avoid divergence.

---

#### 4.2.2 Quasi-Newton (BFGS, L-BFGS)

Quasi-Newton methods approximate the inverse Hessian using gradient history, avoiding explicit second derivatives.

BFGS maintains an estimate $B_t \approx H_t^{-1}$ and updates it each iteration using $\Delta \theta_t = \theta_{t+1} - \theta_t$ and $\Delta g_t = \nabla J(\theta_{t+1}) - \nabla J(\theta_t)$.

Update:

$$
\theta_{t+1} = \theta_t - \eta B_t \nabla J(\theta_t),
$$

with a rank-two update rule for $B_t$.

L-BFGS is a memory-efficient variant that stores only a few recent $(\Delta \theta, \Delta g)$ pairs.

These methods are widely used for **medium-scale** problems (e.g., classical ML with up to millions of parameters, not billions).

---

### 4.3 Coordinate Descent (Important for Lasso)

Coordinate descent optimizes one parameter at a time while keeping others fixed.

For an objective $J(\theta_1, \dots, \theta_d)$:

1. Pick coordinate $j$.
2. Solve:
   $$
   \theta_j^{\text{new}} = \arg\min_{\theta_j} J(\theta_1, \dots, \theta_j, \dots, \theta_d),
   $$
   holding all other coordinates fixed.
3. Cycle through coordinates or use a selection rule.

For the Lasso:

$$
J(w) = \frac{1}{2n}\lVert Xw - y \rVert_2^2 + \lambda \lVert w \rVert_1,
$$

the update for each $w_j$ has a **closed-form soft-thresholding** solution:

$$
w_j \leftarrow S\left(\frac{1}{n} \sum_{i=1}^n x_{ij} (y_i - \hat{y}_{-j,i}), \frac{\lambda}{2}\right),
$$

where $S(\cdot,\lambda)$ is the soft-thresholding operator:

$$
S(z,\lambda) = \text{sign}(z)\max(|z| - \lambda, 0),
$$

and $\hat{y}_{-j,i}$ is the prediction excluding feature $j$.

This makes coordinate descent very efficient for high-dimensional sparsity problems.

---

### 4.4 Convex vs Non-Convex Optimization

A function $J(\theta)$ is convex if:

$$
J(\alpha \theta_1 + (1 - \alpha)\theta_2) \le \alpha J(\theta_1) + (1 - \alpha) J(\theta_2)
\quad \forall \theta_1,\theta_2, \alpha \in [0,1].
$$

Consequences:

* Any local minimum is a global minimum.
* Gradient descent with suitable step size converges to the global optimum (under mild conditions).
* Many classical models (linear regression, logistic regression with convex regularizers, SVMs) are convex.

Non-convex:

* Multiple local minima, saddle points, flat regions.
* No guarantee that gradient methods find global optimum.
* Deep nets and many complex models are non-convex.

Empirically, for high-dimensional non-convex problems, SGD-like methods often still find good minima, possibly due to the structure of the loss landscape (many “good” minima with similar value).

---

### 4.5 Loss Landscape Intuition (Sharp vs Flat Minima)

Even in non-convex settings, not all minima are equal.

* **Sharp minimum**: small change in parameters leads to large increase in loss.
* **Flat minimum**: loss stays low over a relatively large neighborhood in parameter space.

Flat minima are often associated with better generalization:

* Intuition: if performance is stable under small parameter perturbations, the model might be more robust to sampling noise and distributional shifts.

Some heuristics that implicitly bias toward flatter minima:

* Smaller learning rates or decaying schedules.
* SGD noise (mini-batching) tends to escape very sharp, narrow minima.
* Strong regularization (weight decay) and early stopping.

In interviews, you rarely need formal theory; it is enough to explain that optimization interacts with generalization through the geometry of the loss landscape.

---

### 4.6 Hyperparameter Tuning: Grid, Random, Bayesian

Hyperparameters (learning rate, regularization strength, number of trees, depth, etc.) control the bias–variance trade-off and training dynamics.

#### 4.6.1 Grid Search

* Define a discrete grid of values for each hyperparameter.
* Evaluate all combinations via cross-validation or a validation set.
* Choose the combination with best metric.

Pros: simple and parallelizable.  
Cons: exponential in the number of hyperparameters; wastes evaluations on unimportant dimensions.

---

#### 4.6.2 Random Search

Instead of a fixed grid, sample hyperparameters at random from distributions (often log-uniform for scales like $\lambda$, $\eta$).

Key insight (Bergstra & Bengio): in many problems, only a few hyperparameters matter a lot:

* Random search explores more unique values along each dimension.
* More efficient than grid search under a fixed evaluation budget.

---

#### 4.6.3 Bayesian Optimization (BO)

Bayesian optimization treats the validation metric as an unknown function $f(\lambda)$ of hyperparameters $\lambda$ and tries to **model** it to guide exploration.

Typical setup:

1. Put a prior on $f$ (e.g., Gaussian process or Tree-structured Parzen Estimator).
2. After observing some evaluated points $(\lambda_i, f(\lambda_i))$, compute posterior over $f$.
3. Choose the next hyperparameter $\lambda_{\text{next}}$ by maximizing an acquisition function (e.g., Expected Improvement, Upper Confidence Bound).
4. Evaluate $f(\lambda_{\text{next}})$ (train model, measure metric), update posterior, repeat.

Benefits:

* Efficient use of expensive evaluations (training models).
* Good for low- to medium-dimensional hyperparameter spaces.

Variants: bandit-based resource allocation (Hyperband, ASHA) that combine early stopping with multi-armed bandit ideas.

---

## 5. Feature Engineering & Data Handling

In many real MLE roles, especially with non-deep models, **feature work matters more than model choice**. Interviewers want to see that you understand how preprocessing interacts with algorithms.

---

### 5.1 Normalization vs Standardization

Many models are sensitive to feature scales, especially:

* Distance-based methods (k-NN, k-means).
* Gradient-based optimization (conditioning).
* Regularized linear models (L1/L2).

**Standardization (z-score scaling)**

For feature $x$:

$$
\tilde{x} = \frac{x - \mu}{\sigma},
$$

where $\mu$ is mean and $\sigma$ is standard deviation estimated on the training set.

Effects:

* Features have zero mean and unit variance (on train).
* Important for models assuming roughly isotropic Gaussian structure (e.g., PCA, L2-regularized linear/logistic regression).

**Normalization**

Several meanings in practice; two common ones:

* Min–max scaling to $[0,1]$:

  $$
  \tilde{x} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}.
  $$

* Vector (row-wise) normalization to unit norm:

  $$
  \tilde{x} = \frac{x}{\lVert x \rVert_p},
  $$

  common with cosine similarity (e.g., text embeddings).

Key practice:

* Fit scalers on training data only (to avoid leakage).
* Apply same transformation to validation/test.

---

### 5.2 Encoding Categorical Variables

Most classical models require numeric features; how you encode categories can strongly affect performance.

**One-hot encoding**

For a categorical variable with $K$ distinct values, create $K$ binary indicators:

* For sample with category $c$, set feature $x_c = 1$, all others 0.

Pros:

* Simple, works well when $K$ is modest.
* Interpretable.

Cons:

* High-dimensional and sparse for large-cardinality features (e.g., user IDs, query terms).
* No notion of similarity between categories.

**Target / mean encoding**

Replace a category with a smoothed estimate of target statistics (e.g., mean label):

For category $c$:

$$
\text{Enc}(c) = \frac{\sum_{i: x_i = c} y_i + \alpha \mu}{N_c + \alpha},
$$

where:

* $N_c$ is count for category $c$,
* $\mu$ is global mean of $y$,
* $\alpha$ is a smoothing parameter.

This is powerful but dangerous:

* Very prone to leakage if computed on full data.
* Best practice: compute encodings in a cross-validated, out-of-fold manner so each point’s encoding is derived from other points, not itself.

**Embeddings**

Learn dense vectors for categories, typically with deep models or specialized factorization methods. In a pure classical setup, you might use matrix factorization or learned embeddings in a hybrid model.

---

### 5.3 Handling Missing Values

Missingness types:

* MCAR: Missing Completely At Random.
* MAR: Missing At Random (depends on observed features).
* MNAR: Missing Not At Random (depends on unobserved values).

Strategies:

**Simple imputations**

* Mean/median imputation for continuous features.
* Mode (most frequent) for categorical features.

Cheap and works surprisingly well for many models, especially trees which can also model imputed-value artifacts.

**Indicator for missingness**

Augment with a binary feature:

* For feature $x$, create $m = \mathbf{1}{x \text{ is missing}}$.
* Impute $x$ with some constant (e.g., 0 or mean).
* The model can then learn patterns like “missingness correlates with label”.

**More advanced methods**

* k-NN imputation: impute based on nearest neighbors in feature space.
* MICE (Multiple Imputation by Chained Equations): iterative modeling of each feature as a function of others.

In practice:

* For tree-based models (XGBoost, LightGBM), you can often let the model handle missing values natively (they learn default directions for missing).
* For linear models, simple imputation + missing indicators is a strong baseline.

---

### 5.4 Outliers and Robustness

Outliers can heavily affect:

* MSE-based models (linear regression).
* PCA (covariance sensitive to extreme values).

Identification:

* Using Interquartile Range (IQR):

  * Compute $Q_1$ and $Q_3$.
  * IQR = $Q_3 - Q_1$.
  * Flag as outlier if $x < Q_1 - k \cdot \text{IQR}$ or $x > Q_3 + k \cdot \text{IQR}$ (commonly $k = 1.5$ or $3$).

* Z-score: values with $|z| > k$ (e.g., 3) may be considered outliers.

Handling:

* Winsorization: cap values at some quantile (e.g., 1st and 99th percentiles).
* Transformations: log/Box–Cox to compress large values.
* Robust models: use MAE or Huber loss instead of MSE.

Huber loss (for residual $r = y - \hat{y}$):

$$
\ell_\delta(r) =
\begin{cases}
\frac{1}{2}r^2 & \text{if } |r| \le \delta, \\
\delta(|r| - \frac{1}{2}\delta) & \text{if } |r| > \delta.
\end{cases}
$$

Quadratic near zero (like MSE), linear in tails (like MAE).

---

### 5.5 Feature Selection

Goals:

* Reduce variance and overfitting.
* Improve interpretability.
* Reduce computation and latency.

Approaches:

**Filter methods**

* Score each feature individually using some criterion:

  * Correlation with target.
  * Mutual information.
  * ANOVA F-statistics.

* Select top-$k$ features or threshold by score.

Fast and model-agnostic, but ignore feature interactions.

**Wrapper methods**

* Evaluate subsets of features using a predictive model:

  * Recursive Feature Elimination (RFE): fit a model, rank features by importance, remove least important, repeat.
  * Forward selection: start with none, add features greedily.

More accurate but expensive, particularly with many features.

**Embedded methods**

* Feature selection happens during model training:

  * L1-regularized models (Lasso, L1 logistic regression) drive coefficients exactly to zero.
  * Tree-based models with feature importance metrics (split gain, permutation importance).

Permutation importance:

* Measure performance on validation set.
* For each feature, randomly permute its values across samples and re-evaluate.
* Drop in performance indicates how important that feature is.

---

### 5.6 Interaction and Polynomial Features

Linear models can be made more expressive via hand-crafted features.

Given original features $(x_1, \dots, x_d)$:

**Polynomial features**

* Add squared terms $x_j^2$, cubic terms $x_j^3$, etc.
* Add interactions $x_j x_k$.

The model stays linear in the expanded feature space:

$$
f(x) = w_0 + \sum_j w_j x_j + \sum_{j \le k} w_{jk} x_j x_k + \dots
$$

But the function becomes nonlinear in original $x$.

Risk:

* Dimensionality explodes (e.g., all pairwise interactions is $O(d^2)$).
* Overfitting unless you have sufficient data + regularization.

In practice:

* Add only domain-informed interactions (e.g., price × category, user\_age × device\_type).
* For generic expansion, combine with strong regularization (L1/L2) or use models that handle high dimension (like linear models with L1).

---

### 5.7 Time-Series Feature Engineering

For time-dependent data (forecasts, temporal behavior), you often transform raw time into features.

Given a time series ${y_t}$ and covariates $x_t$:

**Lags**

* Include past values as features:

  * $y_{t-1}, y_{t-2}, \dots, y_{t-L}$.
  * Past covariates $x_{t-1}, \dots$.

This allows static models (like linear regression, tree ensembles) to capture autoregressive structure.

**Rolling windows**

* Statistics over a window:

  * Moving average, min, max, std over last $W$ steps:

    $$
    \text{MA}\_W(t) = \frac{1}{W} \sum_{j=0}^{W-1} y_{t-j}.
    $$

* Rolling counts or sums for events.

**Seasonality and calendar features**

* Extract time-of-day, day-of-week, month-of-year as categorical features.
* Add indicators for holidays, weekends, special events.
* Include sine/cosine encodings for cyclical features:

  For hour-of-day $h \in \{0, \dots, 23\}$:

  $$
  \text{sin\_hour} = \sin\left(2\pi \frac{h}{24}\right), \quad
  \text{cos\_hour} = \cos\left(2\pi \frac{h}{24}\right).
  $$

**Causality and leakage**

Critical rule: when constructing features at time $t$, only use information available up to time $t$ (no peeking into the future).

* Rolling windows must be defined using past data only.
* If you aggregate over a time range, ensure it ends before the prediction time.

This is a common interview trap: they will explicitly ask how you avoid temporal leakage when building lag or aggregate features.

---

## 6. Probability & Statistics for ML

### 6.1 Random Variables, Expectation, Variance

A random variable $X$ is a mapping from outcomes to real numbers.

* Discrete: $P(X = x)$.
* Continuous: density $p_X(x)$, with $\int p_X(x),dx = 1$.

**Expectation**

Discrete:
$$
\mathbb{E}[X] = \sum_x x , P(X = x).
$$

Continuous:
$$
\mathbb{E}[X] = \int x , p_X(x) , dx.
$$

Linearity:
$$
\mathbb{E}[aX + bY] = a \mathbb{E}[X] + b \mathbb{E}[Y].
$$

**Variance**

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
= \mathbb{E}[X^2] - (\mathbb{E}[X])^2.
$$

Covariance:
$$
\text{Cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])].
$$

These are the primitives behind bias–variance, CLT, and most probabilistic reasoning in ML.

---

### 6.2 Common Distributions (ML-Relevant)

You mostly need to know the form, typical use, and key moments.

* Bernoulli($p$): $X \in {0,1}$, $P(X=1)=p$.

  * $\mathbb{E}[X]=p$, $\text{Var}(X)=p(1-p)$.
  * Used for binary labels and logistic regression likelihood.

* Binomial($n,p$): sum of $n$ i.i.d. Bernoullis.

  * Models count of successes in $n$ trials.

* Poisson($\lambda$): count of events in interval.

  * $P(X=k) = e^{-\lambda} \lambda^k / k!$.
  * $\mathbb{E}[X] = \lambda$, $\text{Var}(X)=\lambda$.
  * Used in count models, e.g., events per time window.

* Gaussian $\mathcal{N}(\mu,\sigma^2)$:

  * Density $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-(x-\mu)^2/(2\sigma^2))$.
  * Central in linear regression, PCA, CLT.

* Laplace($\mu,b$): double exponential.

  * Heavier tails than Gaussian.
  * Corresponds to L1-type optimization (MAP with Laplace prior).

* Student-$t(\nu)$: heavier tails than Gaussian; robust to outliers.

You rarely derive these in interviews; you use them as modeling assumptions and to interpret likelihoods and priors.

---

### 6.3 Law of Large Numbers (LLN) and Central Limit Theorem (CLT)

Let $X_1,\dots,X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$.

**LLN**

Sample mean:
$$
\bar{X}*n = \frac{1}{n}\sum*{i=1}^n X_i
$$

converges (with high probability) to $\mu$ as $n \to \infty$.

Interpretation: empirical averages approximate expectations with enough data. This justifies empirical risk minimization (replacing expectations with sample averages).

**CLT**

The normalized sum:
$$
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0,1)
$$

as $n \to \infty$.
So for large $n$, $\bar{X}_n$ is approximately Gaussian even if $X_i$ are not.

This underpins confidence intervals, approximate normality of many estimators, and approximate test statistics.

---

### 6.4 Maximum Likelihood Estimation (MLE)

Given data $\mathcal{D} = {x_i}*{i=1}^n$ and parametric density $p*\theta(x)$, the likelihood is:

$$
L(\theta) = \prod_{i=1}^n p_\theta(x_i).
$$

Log-likelihood:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p_\theta(x_i).
$$

MLE:

$$
\hat{\theta}*{\text{MLE}} = \arg\max*\theta \ell(\theta).
$$

In supervised learning, $p_\theta(x,y)$ or $p_\theta(y \mid x)$ leads to maximizing log-likelihood equivalent to minimizing familiar losses:

* Linear regression with Gaussian noise $\Rightarrow$ minimizing MSE.
* Logistic regression $\Rightarrow$ minimizing cross-entropy.

Asymptotically, under regularity conditions:

* $\hat{\theta}_{\text{MLE}}$ is consistent and asymptotically normal.
* It is efficient (achieves Cramér–Rao lower bound).

---

### 6.5 Maximum A Posteriori (MAP) and Bayesian Inference

Bayesian view:

* Prior $p(\theta)$ encodes beliefs about parameters before data.
* Likelihood $p(\mathcal{D} \mid \theta)$.
* Posterior:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta).
$$

**MAP estimator**:

$$
\hat{\theta}*{\text{MAP}} = \arg\max*\theta p(\theta \mid \mathcal{D})
= \arg\max_\theta \big[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\big].
$$

So MAP is like MLE plus a regularization term arising from $\log p(\theta)$.

Examples:

* Gaussian prior $p(\theta) \propto \exp(-\lambda \lVert \theta \rVert_2^2)$
  $\Rightarrow$ L2-regularized MLE (ridge).
* Laplace prior $p(\theta) \propto \exp(-\lambda \lVert \theta \rVert_1)$
  $\Rightarrow$ L1-regularized MLE (lasso).

Bayesian inference goes beyond MAP:

* Keep full posterior $p(\theta \mid \mathcal{D})$ instead of a point estimate.
* Predictive distribution:

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta), p(\theta \mid \mathcal{D}) , d\theta.
$$

In practice, exact integration is often intractable; we approximate with MCMC or variational inference.

---

### 6.6 Conjugate Priors (High-Level)

A prior $p(\theta)$ is conjugate to the likelihood family if the posterior $p(\theta \mid \mathcal{D})$ has the same functional form as the prior.

Classics:

* Bernoulli/Binomial likelihood with Beta prior $\Rightarrow$ Beta posterior.
* Poisson likelihood with Gamma prior $\Rightarrow$ Gamma posterior.
* Gaussian mean with Gaussian prior $\Rightarrow$ Gaussian posterior.

Conjugacy gives closed-form posteriors and easy updates, useful for online/streaming Bayesian updates and for intuition about how priors and data combine.

---

### 6.7 Sampling Methods: Rejection Sampling and Gibbs Sampling

When you cannot compute expectations analytically, you approximate by sampling from a target distribution $\pi(x)$.

**Rejection sampling**

* Proposal distribution $q(x)$ that is easy to sample from.
* Constant $M$ such that $\pi(x) \le M q(x)$ for all $x$.
* Algorithm:

  1. Sample $x \sim q(x)$.
  2. Accept $x$ with probability $\pi(x)/(M q(x))$; otherwise reject and resample.

Yields i.i.d samples from $\pi$, but can be very inefficient in high dimensions or if $M$ is large.

**Markov Chain Monte Carlo (MCMC)** methods (e.g., Gibbs, Metropolis–Hastings) construct a Markov chain whose stationary distribution is $\pi$.

Gibbs sampling:

* Suppose $\pi(x_1,\dots,x_d)$ factorizes so that conditionals are easy: $\pi(x_j \mid x_{-j})$.

* Iterate:

  * For $j=1,\dots,d$, sample:

    $$
    x_j^{(t+1)} \sim \pi(x_j \mid x_1^{(t+1)},\dots,x_{j-1}^{(t+1)}, x_{j+1}^{(t)},\dots,x_d^{(t)}).
    $$

* After burn-in, samples approximate $\pi$.

These tools underlie Bayesian inference and some probabilistic models but are usually high-level topics in interviews unless the role is explicitly Bayesian/ML research.

---

### 6.8 A/B Testing and Hypothesis Testing

A/B testing compares two variants (A and B) on some metric (conversion rate, CTR, etc.).

Let $p_A$ and $p_B$ be true conversion rates, with empirical estimates $\hat{p}_A$ and $\hat{p}_B$ from sample sizes $n_A$ and $n_B$.

**Null hypothesis** $H_0$: $p_A = p_B$.
**Alternative** $H_1$: $p_A \neq p_B$ (two-sided) or $p_A < p_B$ / $>$ (one-sided).

Approximate standard error of difference:

$$
\text{SE}(\hat{p}_A - \hat{p}_B) \approx \sqrt{\frac{\hat{p}_A (1 - \hat{p}_A)}{n_A} + \frac{\hat{p}_B (1 - \hat{p}_B)}{n_B}}.
$$

z-statistic:

$$
z = \frac{\hat{p}_B - \hat{p}_A}{\text{SE}}.
$$

Under $H_0$, for large samples, $z$ is approximately standard normal. You compute a $p$-value and compare to significance level $\alpha$ (e.g., $0.05$).

**Confidence intervals**

An approximate $(1-\alpha)$ CI for $\hat{p}_B - \hat{p}_A$:

$$
(\hat{p}_B - \hat{p}*A) \pm z*{1-\alpha/2} \cdot \text{SE}.
$$

Hypothesis tests and CIs are two sides of the same coin.

Data pitfalls:

* Peeking/stopping early inflates false positive rates.
* Multiple comparisons require corrections or more careful design.
* Non-stationarity and interference (users influencing each other) invalidate independent Bernoulli assumptions.

---

## 7. ML Theory Essentials

This section is usually tested at a high level unless it is a theoretical role. The goal is to show you understand **capacity**, **overfitting**, and **regularization** beyond hand-wavy statements.

---

### 7.1 Capacity, VC Dimension, and Rademacher Complexity (High-Level)

Model capacity is about how rich the hypothesis class $\mathcal{H}$ is.

VC dimension (for binary classifiers):

* The VC dimension of $\mathcal{H}$ is the largest $n$ such that there exists a set of $n$ points that can be labeled in all $2^n$ ways and for each labeling there exists an $h \in \mathcal{H}$ that fits it perfectly.
* Example (informally): linear separators in $\mathbb{R}^d$ have VC dimension $d+1$.

Generalization bounds (very roughly) say that with high probability:

$$
R(h) \le \hat{R}(h) + \text{complexity term}(\mathcal{H}, n),
$$

where $R$ is true risk, $\hat{R}$ is empirical risk, and the complexity term shrinks as $n$ grows and grows with capacity (VC or Rademacher complexity).

Rademacher complexity:

* Data-dependent measure of how well $\mathcal{H}$ can fit random $\pm 1$ labels on the given sample.
* Tighter than VC in many cases.

Key message: To generalize, you want good trade-offs between empirical fit and complexity.

---

### 7.2 Regularization as Complexity Control

Regularization constrains or penalizes the hypothesis space, effectively shrinking the set of functions the model can represent or prefers.

* L2 regularization shrinks parameters; in linear models, it controls the L2 norm of $w$ and hence the size of the function.
* L1 regularization encourages sparsity; it implicitly restricts the effective number of active features.

From a theory perspective:

* Regularization reduces capacity (or effective capacity), making generalization bounds better.
* Many regularizers correspond to priors in Bayesian view, making predictions Bayesian posteriors or MAP.

---

### 7.3 Generalization Bounds (Intuition, Not Formulas)

A typical form (very simplified) is:

$$
R(h) \le \hat{R}(h) + O\left(\sqrt{\frac{\text{capacity}(\mathcal{H}) + \log(1/\delta)}{n}}\right),
$$

with probability at least $1-\delta$ over the sample.

Interpretation:

* For fixed model class, as $n$ grows, true risk approaches empirical risk.
* For fixed $n$, larger capacity increases the gap between train and test risk.

In deep learning, formal bounds are often vacuous, but the intuition still guides practice: some mechanism must control effective capacity (architecture, regularization, implicit bias of optimization, early stopping).

---

### 7.4 Kernel Methods Fundamentals

In kernel methods, instead of working with explicit features $\phi(x)$, you work with a kernel function:

$$
K(x,x') = \langle \phi(x), \phi(x') \rangle.
$$

Many algorithms (SVM, kernel ridge regression) can be written in dual form with only inner products between training examples. Replacing inner product with $K$ implicitly maps into a high-dimensional feature space and learns a linear model there.

Kernel ridge regression example:

* Objective:

  $$
  \min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \lVert f \rVert_\mathcal{H}^2,
  $$

  where $\mathcal{H}$ is a reproducing kernel Hilbert space (RKHS) associated with $K$.

* Representer theorem: solution has the form

  $$
  f(x) = \sum_{i=1}^n \alpha_i K(x_i, x).
  $$

Kernel methods used to be SOTA before deep learning; now they’re more niche but still conceptually important (e.g., GP regression).

---

### 7.5 EM Algorithm (Expectation-Maximization)

EM is used for maximum likelihood when there are latent variables $Z$ and the complete-data log-likelihood is easier than the marginal over $Z$.

We want to maximize:

$$
\ell(\theta) = \log p(X \mid \theta) = \log \sum_Z p(X,Z \mid \theta).
$$

EM alternates:

* E-step: compute posterior over latent variables under current parameters:

  $$
  q(Z) = p(Z \mid X, \theta^{(t)}).
  $$

* M-step: maximize expected complete-data log-likelihood:

  $$
  \theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{Z \sim q}[\log p(X,Z \mid \theta)].
  $$

EM guarantee: each iteration does not decrease the likelihood $p(X \mid \theta)$ (monotonic ascent), though it may converge to local maxima.

Example: Gaussian mixture models, HMMs (Baum–Welch).

---

### 7.6 Information Theory Basics (Entropy, KL, Mutual Information)

Entropy of a discrete random variable $X$:

$$
H(X) = -\sum_x p(x)\log p(x).
$$

Measures average uncertainty (in bits if log base 2).

Kullback–Leibler divergence between $P$ and $Q$:

$$
\text{KL}(P \Vert Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}.
$$

Not symmetric; measures how inefficient it is to encode samples from $P$ using a code optimized for $Q$.

Mutual information between $X$ and $Y$:

$$
I(X;Y) = H(X) - H(X \mid Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}.
$$

Measures reduction in uncertainty of $X$ given $Y$.

These quantities appear implicitly in:

* Cross-entropy loss (which is $H(P) + \text{KL}(P\Vert Q)$).
* Variational inference objectives (ELBO has explicit KL).
* Regularization via info bottleneck perspectives.

---

## 8. Model Explainability & Interpretability

Modern interviews often probe whether you can **explain models** and reason about fairness and accountability, especially in regulated domains.

---

### 8.1 Global vs Local Interpretability

* Global: understanding overall model behavior (e.g., which features are generally important, monotonic relationships).
* Local: understanding why the model made a specific prediction for a specific input.

Linear models and small trees are inherently more interpretable globally. Complex ensembles and deep models need post-hoc methods.

---

### 8.2 Feature Importance

**Model-specific importance**

* Tree ensembles: importance based on total split gain, split counts, etc.
* Coefficients in linear models: magnitude of weights (after proper scaling).

These are easy but can be misleading under collinearity or interactions.

**Permutation importance (model-agnostic)**

1. Measure baseline performance on validation set.
2. For a feature $j$, permute its values across samples (breaking association with target).
3. Recompute performance.
4. The drop in performance is the importance of feature $j$.

Permutation importance captures how performance depends on that feature, accounting for interactions, but can be confounded by correlated features.

---

### 8.3 Partial Dependence Plots (PDPs)

PDPs show the average effect of a feature on the prediction, marginalizing over other features.

For feature $x_j$ and model $f$:

$$
f_{\text{PDP}}(x_j) = \mathbb{E}*{X*{-j}}[f(x_j, X_{-j})].
$$

Empirically, approximate by:

$$
f_{\text{PDP}}(x_j) \approx \frac{1}{n} \sum_{i=1}^n f(x_j, x_{i,-j}).
$$

Plot $f_{\text{PDP}}(x_j)$ versus $x_j$ to see: “on average, how does changing this feature affect predictions?”

Limitations:

* Assumes (or implicitly treats) $x_j$ as independent of $X_{-j}$; can be misleading under strong correlations.

---

### 8.4 LIME and SHAP

Both are **local explanation** methods.

**LIME (Local Interpretable Model-agnostic Explanations)**

* For a given instance $x_0$:

  * Sample perturbed points around $x_0$.
  * Get model predictions $f(x)$ for those points.
  * Fit a simple interpretable surrogate model $g$ (e.g., sparse linear model) weighted by a kernel favoring points near $x_0$.
* The coefficients of $g$ are the local explanation.

Essentially: approximate the complex model locally by a simple one.

**SHAP (SHapley Additive exPlanations)**

* Based on Shapley values from cooperative game theory.
* Each feature is a “player”; the model output is the “payout.”
* Shapley value for feature $j$ is the average marginal contribution of including feature $j$ across all subsets of other features.

Mathematically, for prediction $f(x)$, SHAP values $\phi_j$ satisfy:

$$
f(x) - \mathbb{E}[f(X)] = \sum_{j} \phi_j.
$$

Properties:

* Additivity: attributions sum to prediction difference.
* Symmetry and other axiomatic guarantees.

In practice, exact Shapley values are expensive; approximations (TreeSHAP, KernelSHAP) are used.

---

### 8.5 Counterfactual Explanations

Instead of asking “why this prediction?”, ask “what minimal change in input would flip the decision?”

For an instance $x$ with prediction $f(x)$, find $x'$ such that:

* $f(x')$ has desired label (e.g., from reject to approve),
* $x'$ is close to $x$ under some distance, and
* $x'$ respects constraints (e.g., age cannot decrease arbitrarily, causal constraints).

This is an optimization problem:

$$
\min_{x'} d(x, x')
\quad \text{s.t.}\quad f(x') = y_{\text{desired}},, x' \in \mathcal{C},
$$

where $\mathcal{C}$ imposes “realistic” constraints.

Useful in credit decisions, hiring models, etc., to provide actionable feedback.

---

### 8.6 Fairness Metrics (Brief)

Given sensitive attribute $A$ (e.g., group membership), outcome $\hat{Y}$, and true label $Y$, common group fairness criteria:

* Demographic parity:

  $$
  P(\hat{Y}=1 \mid A=a) \text{ same across groups}.
  $$

* Equalized odds:

  $$
  P(\hat{Y}=1 \mid Y=y, A=a) \text{ same across groups for each } y.
  $$

* Equal opportunity: equal TPR across groups:

  $$
  P(\hat{Y}=1 \mid Y=1, A=a) \text{ same across groups}.
  $$

These are often mutually incompatible with each other and with calibration under realistic label imbalance, so trade-offs are inevitable.

---

## 9. ML Systems & Practical Engineering (Non-DL Focus)

This is critical for MLE roles. You need to show you can move from “train a model in a notebook” to “operate a model in production”.

---

### 9.1 Data Pipelines: Batch vs Streaming, Versioning, Feature Stores

**Batch pipelines**

* Periodically process large chunks of data (e.g., daily).
* ETL: Extract → Transform → Load into data warehouse or feature store.
* Train models offline on accumulated data.

**Streaming pipelines**

* Process events as they arrive (e.g., clicks, impressions).
* Maintain near real-time aggregates (sliding windows, counts).
* Feed online features to models for low-latency decisions.

Data versioning:

* Track which raw data, transformations, and feature definitions produced a training dataset.
* Tools: data lakes with partitioning, table snapshots, or specialized tools (Delta, Lakehouse patterns).

Feature stores:

* Central repository of validated, reusable features.
* Key goals:

  * Consistent definitions between training and serving.
  * Point-in-time correctness (no leakage).
  * Low-latency serving for online features.

---

### 9.2 Scaling Classical ML

**Distributed training**

* For tree-based models like XGBoost:

  * Data-parallel: partition data across workers; aggregate gradient statistics.
  * Feature-parallel in some setups.
  * Use parameter servers or allreduce to sync state.

**Online learning**

* Model parameters updated continuously as new data arrives.
* Algorithms like FTRL (Follow-The-Regularized-Leader), online SGD:

  * Good for CTR models, recommendation systems, ads, where data is non-stationary.
* Often combined with warm-start from offline-trained models.

**Approximate Nearest Neighbors (ANN)**

* For large-scale retrieval (search, recsys), exact k-NN in embedding space is too slow.
* ANN indexes (e.g., HNSW graphs, IVF-PQ) support sub-millisecond approximate k-NN in millions–billions of vectors.
* Trade small recall loss for large speed gains.

---

### 9.3 Serving: Latency, Serialization, Testing, Monitoring

**Latency/throughput trade-offs**

* P99 or P95 latency budgets (e.g., must respond within 100 ms).
* Decompose latency into:

  * Feature lookup time.
  * Model compute time.
  * Network round trips.
* Techniques:

  * Precomputation and caching.
  * Simpler models for online scoring, with more complex models offline.

**Model formats**

* Serialized formats like pickled Python objects are common but fragile across environments.
* More robust: ONNX, PMML, or model servers (e.g., treating models as services).
* Ensure reproducibility and compatibility with serving stack (e.g., C++ services).

**Shadow testing and canary releases**

* Shadow: new model runs in parallel to current one but does not affect user experience; collect logs to compare.
* Canary: release new model to small traffic slice, monitor metrics and errors, then gradually ramp up.

**Monitoring**

* Prediction distributions and feature distributions (drift detection).
* Performance metrics over time (accuracy, AUC, business KPIs).
* Outlier detection on inputs and outputs.

Drift detection:

* Compare current feature or score distributions to training reference via statistical tests or divergence measures.
* Alert when divergence exceeds threshold.

---

### 9.4 Search, Recsys, Ads (Non-DL Components)

**Query understanding basics**

* Tokenization, stemming/lemmatization.
* Spelling correction, synonyms, query rewriting.
* Query classification (intent, topic).

**Lexical retrieval (BM25)**

BM25 scores a document $d$ for query $q$ (bag of terms) as:

$$
\text{BM25}(q,d) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})},
$$

where:

* $f(t,d)$ = term frequency.
* $|d|$ = document length, $\text{avgdl}$ = average document length.
* $k_1$ and $b$ are hyperparameters.
* IDF is inverse document frequency.

Used for efficient first-stage retrieval before ML ranking.

**Collaborative filtering (CF)**

* Matrix factorization for recommendations:

  * User–item interaction matrix $R$.
  * Factorize: $R \approx U V^\top$ with low-rank factors $U$, $V$.
  * Predict rating or implicit score as $\hat{r}_{ui} = u_u^\top v_i$.
* ALS (Alternating Least Squares) is a common optimization.

**Exploration vs exploitation (bandits)**

* In recommendations and ads, you must explore new items or strategies while exploiting known good ones.
* Multi-armed bandits (UCB, Thompson sampling) model this explicitly.
* Contextual bandits condition on user/context features.

---

## 10. Applied ML Domains (Breadth Topics)

These are broad areas where classical ML still plays a role. You rarely derive everything; you just need to map concepts to problem types and typical methods.

---

### 10.1 Search and Ranking

* First-stage retrieval:

  * Inverted indexes, BM25, lexical match.
  * Possibly ANN over embeddings (semantic retrieval).
* Re-ranking:

  * Learning-to-rank models (pointwise, pairwise, listwise).
  * Features: query–document features, behavioral signals (clicks, dwell time), popularity, freshness.
* Click models:

  * Model user behavior in ranked lists (position bias, examination).
  * DBN/CTR models as probabilistic frameworks to de-bias clicks.

---

### 10.2 Recommender Systems

* Collaborative filtering:

  * Explicit feedback (ratings) with matrix factorization.
  * Implicit feedback (clicks, views) with weighting and negative sampling.
* Neighborhood methods:

  * User–user or item–item similarity.
* Cold-start:

  * Use content-based features (metadata, text).
  * Use popularity or heuristic priors for new items/users.
* Ranking metrics: nDCG, MAP, Recall@k, coverage, diversity.

---

### 10.3 Time-Series Forecasting

Classical models:

* AR($p$): autoregressive:
  $$
  y_t = \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t.
  $$
* MA($q$): moving average:
  $$
  y_t = \sum_{j=1}^q \theta_j \varepsilon_{t-j} + \varepsilon_t.
  $$
* ARMA($p,q$): combination of both.
* ARIMA($p,d,q$): includes differencing $d$ times for non-stationary series.

Prophet (Facebook):

* Additive model combining trend, seasonality, and holidays:
  $$
  y(t) = g(t) + s(t) + h(t) + \varepsilon_t.
  $$

The practical questions are about handling seasonality, holidays, trend shifts, and evaluating with proper time splits.

---

### 10.4 Causal Inference

Causal inference asks “what if we intervene?” instead of just correlational prediction.

Average Treatment Effect (ATE):

* Let $Y(1)$, $Y(0)$ be potential outcomes under treatment and control.
* ATE is:

  $$
  \text{ATE} = \mathbb{E}[Y(1) - Y(0)].
  $$

Problem: you only observe one of $Y(1)$ or $Y(0)$ per unit.

Propensity scores:

* Probability of receiving treatment given covariates:

  $$
  e(x) = P(T=1 \mid X=x).
  $$

Inverse propensity weighting (IPW):

* Weight treated units by $1/e(x)$ and control units by $1/(1-e(x))$ to balance covariates and estimate causal effects under ignorability assumptions.

In industry, causal thinking matters for:

* Interpreting A/B tests.
* Policy changes (e.g., pricing, ranking changes).
* Avoiding confounding when evaluating ML-driven interventions.

---

### 10.5 Anomaly Detection

Goal: identify rare, unusual points.

Approaches:

* Simple statistical thresholds:

  * Z-score, distributional assumptions.
* Density-based methods:

  * Estimate $p(x)$ and flag low-density points.
* Isolation Forest:

  * Randomly split data; anomalies are isolated in fewer splits.
* Robust statistics:

  * Use robust estimators (median, MAD) to reduce influence of outliers.

Metrics:

* Often heavily imbalanced; use PR curves, ROC-AUC, recall at fixed false-positive budget.

---

### 10.6 Optimization / Operations Research

Interfaces with ML when you:

* Use ML predictions as inputs to optimization (demand forecasts → inventory optimization).
* Or use ML to approximate difficult cost functions.

Classical OR problems:

* Linear programming (LP):

  $$
  \min_x c^\top x \quad \text{s.t. } Ax \le b, x \ge 0.
  $$

* Integer programming (IP): some variables constrained to integers (assignment, routing).

Assignment problems:

* Hungarian algorithm for bipartite matching.
* Used in resource assignment, ranking with constraints, etc.

The key is being able to articulate the pipeline: ML predicts parameters; OR optimizes decisions subject to constraints.

---