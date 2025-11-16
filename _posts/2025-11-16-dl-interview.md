---
layout: post
title: "DL Interview: Fundamentals and Implementation"
date: 2025-11-16 09:00:00
description: 
tags: ml
categories: ml
thumbnail: assets/img/dl_cheatsheet.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

## Contents
- [Contents](#contents)
- [1. Core Foundations](#1-core-foundations)
  - [1.1 Perceptron and Linear Separability](#11-perceptron-and-linear-separability)
  - [1.2 Multi-Layer Perceptron (MLP)](#12-multi-layer-perceptron-mlp)
  - [1.3 Activation Functions](#13-activation-functions)
  - [1.4 Initialization](#14-initialization)
  - [1.5 Forward vs Backward Pass and Computational Graph](#15-forward-vs-backward-pass-and-computational-graph)
  - [1.6 Loss Functions](#16-loss-functions)
  - [1.7 Backpropagation and Gradient Pathologies](#17-backpropagation-and-gradient-pathologies)
- [2. Optimization \& Training Dynamics](#2-optimization--training-dynamics)
  - [2.1 Gradient Descent, SGD, and Minibatches](#21-gradient-descent-sgd-and-minibatches)
  - [2.2 Momentum and Nesterov](#22-momentum-and-nesterov)
  - [2.3 Adaptive Methods](#23-adaptive-methods)
  - [2.4 Learning Rate Scheduling](#24-learning-rate-scheduling)
  - [2.5 Gradient Clipping](#25-gradient-clipping)
  - [2.6 Batch Size Effects (Sharp vs Flat Minima)](#26-batch-size-effects-sharp-vs-flat-minima)
  - [2.7 Loss Landscape Intuition](#27-loss-landscape-intuition)
  - [2.8 Training Tricks](#28-training-tricks)
  - [2.9 Polyak Averaging and EMA Weights](#29-polyak-averaging-and-ema-weights)
  - [2.10 Lottery Ticket Hypothesis (LTH)](#210-lottery-ticket-hypothesis-lth)
- [3. Regularization Techniques](#3-regularization-techniques)
  - [3.1 Weight Decay and AdamW Decoupling](#31-weight-decay-and-adamw-decoupling)
  - [3.2 Dropout and DropConnect](#32-dropout-and-dropconnect)
  - [3.3 Normalization Layers: BatchNorm, LayerNorm, GroupNorm](#33-normalization-layers-batchnorm-layernorm-groupnorm)
  - [3.4 Stochastic Depth](#34-stochastic-depth)
  - [3.5 Data Augmentation](#35-data-augmentation)
  - [3.6 Curriculum Learning](#36-curriculum-learning)
- [4. Convolutional Neural Networks (CNNs)](#4-convolutional-neural-networks-cnns)
  - [4.1 CNN Fundamentals](#41-cnn-fundamentals)
  - [4.2 CNN Model Families (Historical Arc)](#42-cnn-model-families-historical-arc)
  - [4.3 Advanced CNN Ideas](#43-advanced-cnn-ideas)
- [5. Recurrent Neural Networks (RNNs)](#5-recurrent-neural-networks-rnns)
  - [5.1 Core RNN Formulation](#51-core-rnn-formulation)
  - [5.2 Vanishing/Exploding Gradients and Truncated BPTT](#52-vanishingexploding-gradients-and-truncated-bptt)
  - [5.3 LSTM](#53-lstm)
  - [5.4 GRU](#54-gru)
  - [5.5 Bi-directional and Deep RNNs](#55-bi-directional-and-deep-rnns)
  - [5.6 Attention on RNNs: Seq2Seq + Attention](#56-attention-on-rnns-seq2seq--attention)
  - [5.7 RNN Applications (Conceptual)](#57-rnn-applications-conceptual)
- [6. Attention \& Transformers](#6-attention--transformers)
  - [6.1 Scaled Dot-Product Attention](#61-scaled-dot-product-attention)
  - [6.2 Masked (Causal) Attention](#62-masked-causal-attention)
  - [6.3 Multi-Head Attention](#63-multi-head-attention)
  - [6.4 Positional Encodings](#64-positional-encodings)
  - [6.5 Transformer Block](#65-transformer-block)
  - [6.6 Transformer Model Families (High-Level)](#66-transformer-model-families-high-level)
  - [6.7 Scaling Laws (Qualitative)](#67-scaling-laws-qualitative)
- [7. Large Language Models (LLMs)](#7-large-language-models-llms)
  - [7.1 Pretraining Objectives](#71-pretraining-objectives)
  - [7.2 Distributed Training (Very High Level)](#72-distributed-training-very-high-level)
  - [7.3 Fine-Tuning and Alignment](#73-fine-tuning-and-alignment)
  - [7.4 Inference Acceleration](#74-inference-acceleration)
  - [7.5 Quantization and Distillation](#75-quantization-and-distillation)
- [8. Generative Models](#8-generative-models)
  - [8.1 Autoencoders (AEs, DAEs, Sparse AEs, VAEs)](#81-autoencoders-aes-daes-sparse-aes-vaes)
  - [8.2 GANs (Generative Adversarial Networks)](#82-gans-generative-adversarial-networks)
  - [8.3 Diffusion Models](#83-diffusion-models)
  - [8.4 Flow-Based Models](#84-flow-based-models)
  - [8.5 Token-Based Diffusion (for LLMs)](#85-token-based-diffusion-for-llms)
- [9. Vision Architectures (Advanced)](#9-vision-architectures-advanced)
  - [9.1 Vision Transformer (ViT)](#91-vision-transformer-vit)
  - [9.2 CLIP (Contrastive Pretraining)](#92-clip-contrastive-pretraining)
  - [9.3 MAE (Masked Autoencoder for Vision)](#93-mae-masked-autoencoder-for-vision)
  - [9.4 SAM (Segment Anything Model)](#94-sam-segment-anything-model)
  - [9.5 DETR and YOLO Family](#95-detr-and-yolo-family)
  - [9.6 NeRF and Diffusion for Vision](#96-nerf-and-diffusion-for-vision)
- [10. Multimodal Deep Learning](#10-multimodal-deep-learning)
  - [10.1 Vision–Language Models](#101-visionlanguage-models)
  - [10.2 Large Multimodal Models (LMMs)](#102-large-multimodal-models-lmms)
  - [10.3 Image Captioning and VQA](#103-image-captioning-and-vqa)
  - [10.4 Video Transformers and Speech](#104-video-transformers-and-speech)
- [11. Reinforcement Learning (DL-heavy Parts)](#11-reinforcement-learning-dl-heavy-parts)
  - [11.1 Policy Gradient and Actor–Critic](#111-policy-gradient-and-actorcritic)
  - [11.2 PPO (Proximal Policy Optimization)](#112-ppo-proximal-policy-optimization)
  - [11.3 Q-Learning and DQN](#113-q-learning-and-dqn)
  - [11.4 AlphaZero and MuZero Architectures (High Level)](#114-alphazero-and-muzero-architectures-high-level)
  - [11.5 World Models and Offline RL](#115-world-models-and-offline-rl)
- [12. Self-Supervised \& Contrastive Learning](#12-self-supervised--contrastive-learning)
  - [12.1 Contrastive Learning and InfoNCE](#121-contrastive-learning-and-infonce)
  - [12.2 SimCLR and MoCo](#122-simclr-and-moco)
  - [12.3 BYOL, SwAV, DINO](#123-byol-swav-dino)
  - [12.4 Self-Distillation and Masked Autoencoding](#124-self-distillation-and-masked-autoencoding)
- [13. Deep Learning Theory (High-Level Intuitions)](#13-deep-learning-theory-high-level-intuitions)
  - [13.1 Universal Approximation and Overparameterization](#131-universal-approximation-and-overparameterization)
  - [13.2 Double Descent and Sharp vs Flat Minima](#132-double-descent-and-sharp-vs-flat-minima)
  - [13.3 Lottery Ticket Hypothesis (revisited) and NTK](#133-lottery-ticket-hypothesis-revisited-and-ntk)
  - [13.4 Depth vs Width and Inductive Biases](#134-depth-vs-width-and-inductive-biases)
- [14. Deep Learning Systems](#14-deep-learning-systems)
  - [14.1 Distributed Training](#141-distributed-training)
  - [14.2 Serving and Inference](#142-serving-and-inference)
  - [14.3 Quantization: Static, Dynamic, QAT](#143-quantization-static-dynamic-qat)
  - [14.4 Compilers and Kernels](#144-compilers-and-kernels)
- [15. Safety, Alignment, and Responsible AI (LLM-Focused)](#15-safety-alignment-and-responsible-ai-llm-focused)
  - [15.1 RLHF and Constitutional AI](#151-rlhf-and-constitutional-ai)
  - [15.2 Jailbreak Robustness and Toxicity Filters](#152-jailbreak-robustness-and-toxicity-filters)
  - [15.3 Preference Modeling and Evaluation](#153-preference-modeling-and-evaluation)
- [16. Evaluation Metrics (DL-Specific)](#16-evaluation-metrics-dl-specific)
  - [16.1 Sequence and Text Metrics](#161-sequence-and-text-metrics)
  - [16.2 Vision and Generative Metrics](#162-vision-and-generative-metrics)
  - [16.3 Ranking Metrics](#163-ranking-metrics)
- [17. Modern Research Trends (High-Level Map)](#17-modern-research-trends-high-level-map)
  - [17.1 Retrieval-Augmented Models](#171-retrieval-augmented-models)
  - [17.2 Long-Sequence and Memory Models](#172-long-sequence-and-memory-models)
  - [17.3 Recurrent GPT / RWKV-Style and World Models for Reasoning](#173-recurrent-gpt--rwkv-style-and-world-models-for-reasoning)
  - [17.4 Diffusion Transformers, MoE, Tool Use, Agents](#174-diffusion-transformers-moe-tool-use-agents)


## 1. Core Foundations

### 1.1 Perceptron and Linear Separability

1. **Model**

A binary perceptron takes input $x\in\mathbb{R}^d$, weights $w\in\mathbb{R}^d$, bias $b\in\mathbb{R}$ and computes  
$$
z = w^\top x + b,\quad \hat{y} = \text{sign}(z)
$$
Typical label space: $\hat{y}\in{-1,+1}$ (or sometimes ${0,1}$).

2. **Decision boundary**

The decision boundary is the hyperplane  
$$
w^\top x + b = 0.
$$
Points with $w^\top x + b > 0$ go to one class, $<0$ to the other. Geometrically, $w$ is normal to the hyperplane.

3. **Linear separability**

A dataset ${(x_i, y_i)}_{i=1}^n$ with $y_i\in{-1,+1}$ is **linearly separable** if there exists $(w,b)$ such that  
$$
y_i(w^\top x_i + b) > 0\quad \forall i.
$$

4. **Perceptron learning rule (online)**

Given a misclassified example $(x_i, y_i)$, update:  
$$
w \leftarrow w + \eta y_i x_i,\quad b \leftarrow b + \eta y_i,
$$
where $\eta$ is the learning rate.

5. **Convergence**

If the data are linearly separable, the perceptron algorithm converges in finite steps to some separating hyperplane. If not separable, it does not converge and typically keeps cycling.

6. **Limitations**

Perceptron can only express linearly separable decision boundaries. Classic failure case: XOR in 2D cannot be separated by a single hyperplane; needs a multi-layer network.

---

### 1.2 Multi-Layer Perceptron (MLP)

1. **Architecture**

An MLP composes multiple affine layers with nonlinear activations. For a single hidden layer:  
$$
h = \sigma(W^{(1)} x + b^{(1)}),\quad
\hat{y} = f(W^{(2)} h + b^{(2)}),
$$
where:

* $W^{(1)}\in\mathbb{R}^{m\times d}$, $b^{(1)}\in\mathbb{R}^m$,  
* $W^{(2)}\in\mathbb{R}^{k\times m}$, $b^{(2)}\in\mathbb{R}^k$,  
* $\sigma$ is a pointwise nonlinearity (e.g., ReLU, tanh),  
* $f$ is typically identity (regression) or softmax (classification).

2. **Depth vs width**

* **Depth** = number of layers of nonlinear transformations.  
* **Width** = number of units per layer.

3. **Universal approximation**

A feedforward network with at least one hidden layer and a non-polynomial activation (like ReLU, sigmoid, tanh) can approximate any continuous function on a compact set to arbitrary precision, given enough hidden units. This is a capacity/existence result; it does not guarantee that gradient descent will find that approximation.

4. **Viewpoint**

MLPs are learned feature extractors:

* Earlier layers learn low-level combinations of input features.  
* Deeper layers form more abstract representations.

---

### 1.3 Activation Functions

We use nonlinear activations to break linearity; otherwise, stacked linear layers collapse to a single linear map.

#### 1.3.1 Sigmoid and Tanh

1. **Sigmoid**
$$
\sigma(x) = \frac{1}{1 + e^{-x}} \in (0,1).
$$

* Pros: probabilistic interpretation; historically used in output layers.  
* Cons: saturates for large $|x|$; gradients $\sigma(x)(1-\sigma(x))$ become tiny near 0 or 1 → vanishing gradients in deep nets.

2. **Tanh**
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1,1).
$$

* Zero-centered, which helps optimization compared to sigmoid.  
* Still suffers from saturation and vanishing gradients for large $|x|$.

Both are more common now in RNNs (e.g., LSTM gates) and some older architectures, less so in modern deep CNNs/Transformers.

#### 1.3.2 ReLU Family

1. **ReLU**
$$
\text{ReLU}(x) = \max(0, x).
$$

* Pros: cheap; does not saturate in positive region; mitigates vanishing gradients in practice.  
* Cons: zero gradient for $x<0$ → “dead ReLUs” that never activate if pushed too negative.

2. **LeakyReLU**
$$
\text{LeakyReLU}(x) =
\begin{cases}
x, & x \ge 0,\\
\alpha x, & x < 0,
\end{cases}
$$
with small $\alpha$ (e.g., $0.01$), to keep small gradient in negative region and reduce dead units.

3. **PReLU**

Same as LeakyReLU, but $\alpha$ is learned:  
$$
\text{PReLU}(x) =
\begin{cases}
x, & x \ge 0,\\
a x, & x < 0,
\end{cases}
$$
with trainable parameter $a$.

4. **GELU**

Gaussian Error Linear Unit:  
$$
\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right)\right).
$$

Interpretation: gates $x$ by its probability under a Gaussian. Smooth, non-monotonic, behaves roughly like $x\cdot \Phi(x)$ where $\Phi$ is the normal CDF. Used in many modern Transformers.

#### 1.3.3 Softmax

Softmax turns logits into a probability distribution:  
$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}.
$$

Key properties:

* $p_i > 0$ and $\sum_i p_i = 1$.  
* Invariant to adding a constant to all logits: $p(z) = p(z + c\mathbf{1})$.  
* Often combined with cross-entropy loss for multiclass classification.

---

### 1.4 Initialization

Goal: choose initial weights so that activations and gradients have reasonable variance across layers, avoiding early saturation or explosion.

Let $x$ be input to a layer with $n_\text{in}$ inputs and $n_\text{out}$ outputs, weights $w_{ij}$ with zero mean, variance $\text{Var}(w_{ij})$, and assume inputs have zero mean and unit variance.

1. **Xavier / Glorot initialization**

Designed for symmetric activations (e.g., tanh) to keep variance similar across layers. Roughly:  
$$
\text{Var}(w_{ij}) = \frac{2}{n_\text{in} + n_\text{out}}.
$$

Typically implemented as:

* Uniform: $w_{ij}\sim U\left[-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}}, \sqrt{\frac{6}{n_\text{in}+n_\text{out}}}\right]$  
* Normal: $w_{ij}\sim\mathcal{N}\left(0, \frac{2}{n_\text{in} + n_\text{out}}\right)$

2. **He / Kaiming initialization**

Tailored for ReLU-like activations, focusing more on preserving variance in the forward pass:  
$$
\text{Var}(w_{ij}) = \frac{2}{n_\text{in}}.
$$

Often:

* Normal: $w_{ij}\sim\mathcal{N}\left(0, \frac{2}{n_\text{in}}\right)$.

3. **LSUV (Layer-Sequential Unit-Variance)**

Layer-Sequential Unit-Variance initialization:

* Start from a standard initialization.  
* Pass a batch through the network.  
* For each layer (in order), rescale weights so that the output activations have unit variance (and optionally zero mean).  
* Repeat a few iterations.

This empirically stabilizes deep nets by normalizing activations without explicit normalization layers.

---

### 1.5 Forward vs Backward Pass and Computational Graph

1. **Forward pass**

* Compute outputs layer by layer from inputs to loss.  
* For each layer $y=f_\theta(x)$, store intermediate quantities needed for gradients (e.g., $x$, $y$, sometimes masks).

2. **Computational graph**

* Represents the sequence of operations as a directed acyclic graph (DAG).  
* Nodes: tensors (variables, activations, parameters).  
* Edges: operations (matmul, add, ReLU, etc.).  
* Modern frameworks build this graph dynamically (PyTorch) or statically (old TF1-style).

3. **Backward pass (backprop)**

Using the chain rule, gradients are propagated from loss back to parameters. For a scalar loss $L$ and intermediate $z$:  
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z}\cdot \frac{\partial z}{\partial x}.
$$

The framework traverses the computational graph in reverse topological order, applying local gradient formulas and reusing stored intermediates.

---

### 1.6 Loss Functions

1. **Mean Squared Error (MSE)**

For regression:  
$$
L = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|^2.
$$

For Gaussian noise assumptions, MSE corresponds to maximum likelihood.

2. **Cross-entropy (multiclass)**

Given true label $y_i$ and predicted probabilities $p_i$:  
$$
L = - \log p_y.
$$

Usually $p$ is softmax of logits. Equivalent to negative log-likelihood under a categorical distribution.

3. **Binary cross-entropy (logistic)**

For binary label $y\in{0,1}$ and predicted probability $p$:  
$$
L = -\big(y\log p + (1-y)\log(1-p)\big).
$$

4. **Hinge loss / max-margin**

For label $y\in{-1,+1}$ and score $f(x)$:  
$$
L = \max(0, 1 - y f(x)).
$$

Encourages a margin of at least 1. Used in SVM-style models; less common in deep nets now, but conceptually important.

5. **Focal loss**

Used for imbalanced detection tasks. For binary case:  
$$
L = - \alpha (1 - p_t)^\gamma \log p_t,
$$
where $p_t$ is the predicted probability for the true class (if $y=1$, $p_t=p$, else $1-p$). The factor $(1-p_t)^\gamma$ downweights easy examples and focuses on hard ones.

6. **KL divergence**

For distributions $P$ and $Q$ over same support:  
$$
D_{\text{KL}}(P\mid Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)}.
$$

In deep learning:

* Used in distillation (teacher $P$, student $Q$).  
* Many losses (cross-entropy) are KL up to a constant.

7. **Negative log-likelihood (NLL)**

If model outputs a parametric distribution $p_\theta(y\mid x)$:  
$$
L = - \log p_\theta(y\mid x).
$$

* Cross-entropy is a special case of NLL for categorical distributions.  
* Often implemented as `nn.NLLLoss` operating on log-probabilities.

---

### 1.7 Backpropagation and Gradient Pathologies

1. **Backprop as repeated chain rule**

For a composition $L = L(z_L)$, $z_l = f_l(z_{l-1})$:  
$$
\frac{\partial L}{\partial z_{l-1}} = \frac{\partial L}{\partial z_l}\cdot \frac{\partial z_l}{\partial z_{l-1}}.
$$

Over $L$ layers, gradients are products of Jacobians. Their norms can shrink to zero or blow up.

2. **Vanishing gradients**

If typical singular values of $\frac{\partial z_l}{\partial z_{l-1}}$ are $<1$, products across many layers $\prod_l J_l$ go to zero:

* Lower layers learn very slowly or not at all.  
* Sigmoid/tanh saturating activations and naive initialization exacerbate this.

3. **Exploding gradients**

If typical singular values are $>1$, products blow up:

* Gradients become huge, causing unstable updates.  
* Especially common in RNNs with long sequences.

4. **Mitigations**

* Proper initialization (Xavier, He).  
* Normalization layers (BatchNorm, LayerNorm) to keep activations in reasonable ranges.  
* Residual/skip connections: $z_{l+1} = z_l + f_l(z_l)$, which keep gradient paths closer to identity, preserving gradient flow.  
* Gradient clipping (more in Optimization section).  
* Using ReLU/GELU instead of saturating sigmoids in deep feedforward nets.

---

> **quick self-check:**  
Can you, in your own words, describe **why residual connections help with vanishing gradients** in deep networks? A short 2–3 sentence explanation is enough.

---

## 2. Optimization & Training Dynamics

### 2.1 Gradient Descent, SGD, and Minibatches

1. **Full-batch Gradient Descent**

For parameters $\theta$ and loss over dataset:
$$
L(\theta) = \frac{1}{N}\sum_{i=1}^N \ell(f_\theta(x_i), y_i),
$$
full-batch gradient descent does:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t).
$$

* Uses exact gradient, but each step is $O(N)$ — slow and not used in large-scale DL.

2. **Stochastic Gradient Descent (SGD)**

Uses a single sample $(x_i, y_i)$:
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \ell(f_{\theta_t}(x_i), y_i).
$$

* Very noisy gradient estimate, but cheap and can escape shallow local minima and some saddle points.

3. **Mini-batch SGD**

Compromise: pick batch $B_t$ of size $|B|$:
$$
g_t = \frac{1}{|B|} \sum_{(x_i,y_i)\in B_t} \nabla_\theta \ell(f_{\theta_t}(x_i), y_i),\quad
\theta_{t+1} = \theta_t - \eta g_t.
$$

* $|B|$ trades off noise vs computation.

---

### 2.2 Momentum and Nesterov

1. **Momentum SGD**

Maintain velocity $v_t$:
$$
v_{t+1} = \mu v_t + g_t,\quad
\theta_{t+1} = \theta_t - \eta v_{t+1},
$$
where $\mu \in [0,1)$ is the momentum coefficient, $g_t$ is (mini-batch) gradient.

Interpretation:

* Acts like an exponentially weighted moving average of past gradients.
* Smooths noisy gradients, accelerates in consistent directions, dampens oscillations.

2. **Nesterov Accelerated Gradient (NAG)**

Look ahead by momentum before computing gradient:
$$
g_t = \nabla_\theta L(\theta_t - \mu v_t),\ 
v_{t+1} = \mu v_t + g_t,\ 
\theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

* Intuition: gradient is evaluated at a “lookahead” point, giving a more responsive correction when heading in a bad direction.
* In practice, difference vs standard momentum is modest but often slightly better.

---

### 2.3 Adaptive Methods

All adaptive methods rescale gradients per-parameter based on historical statistics.

Let $g_t$ be gradient at step $t$.

#### 2.3.1 Adagrad

Accumulate squared gradients:
$$
G_t = G_{t-1} + g_t^2 \quad (\text{elementwise}),
$$
update:
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t.
$$

* Effective step size decays over time, especially for frequently updated parameters.
* Good for sparse features, but learning rate can become too small.

#### 2.3.2 RMSProp

Exponential moving average of squared gradients:
$$
v_t = \beta v_{t-1} + (1 - \beta) g_t^2,
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t.
$$

* Fixes Adagrad’s ever-shrinking LR via EMA.

#### 2.3.3 Adam and AdamW

Adam keeps EMA of gradients and squared gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,\ 
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2.
$$

Bias-corrected:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}.
$$

Update:
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

* Works well out-of-the-box, dominant for NLP/Transformers.

**AdamW** decouples weight decay from the Adam update:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right),
$$

rather than absorbing $\lambda\theta$ into $g_t$. This behaves more like true $L_2$ regularization for adaptive methods and is standard for modern LLMs.

#### 2.3.4 Other optimizers (Lion, NovoGrad, AdaFactor)

You don’t need to memorize their formulas, but:

* **NovoGrad**: uses normalized gradients and second-moment estimates, designed for large-batch training.
* **AdaFactor**: approximates $v_t$ with factored matrices (for large embedding matrices), reducing memory (used in T5).
* **Lion**: momentum-based optimizer using sign of momentum for updates; lower memory overhead and distinct dynamics, but conceptually still: “adaptive-like” behavior with different regularization.

For interviews: knowing Adam and AdamW well is usually enough; others are bonus topics.

---

### 2.4 Learning Rate Scheduling

The learning rate $\eta$ has huge impact. Schedules trade fast initial learning for stable convergence.

1. **Warmup**

Start with small $\eta$, linearly or gradually ramp up to target LR over first $T_\text{warmup}$ steps/epochs.

Motivation:

* Large models with many layers and normalization (Transformers) can be unstable at the beginning; warmup avoids blowing up early.

2. **Step decay**

Reduce LR by a factor $\gamma$ at predefined epochs:
$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T_\text{step}\rfloor}.
$$

3. **Exponential decay**

$$
\eta_t = \eta_0 \cdot e^{-kt}.
$$

4. **Cosine annealing**

$$
\eta_t = \eta_{\min} + \frac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)\left(1 + \cos\left(\frac{\pi t}{T}\right)\right).
$$

* Starts high, slowly decays to $\eta_{\min}$, smooth non-monotonic slope.

5. **Polynomial decay**

$$
\eta_t = (\eta_0 - \eta_{\min}) \left(1 - \frac{t}{T}\right)^p + \eta_{\min}.
$$

6. **One-cycle policy**

* Increase LR from low to high, then decay back to very low over a single cycle (often one epoch or training run).
* Often also cycles momentum inversely.
* Intuition: aggressive exploration early, then anneal for convergence.

---

### 2.5 Gradient Clipping

Used to prevent exploding gradients, especially in RNNs and large models.

1. **Norm-based clipping**

Compute global gradient norm:
$$
|g| = \sqrt{\sum_i g_i^2}.
$$

If $|g| > c$ (clip threshold), rescale:
$$
g \leftarrow g \cdot \frac{c}{|g|}.
$$

2. **Value-based clipping**

Clip each component:
$$
g_i \leftarrow \text{clip}(g_i, -c, c).
$$

Norm clipping is usually preferred; it preserves direction and just caps magnitude.

---

### 2.6 Batch Size Effects (Sharp vs Flat Minima)

Empirically:

* **Small batches** introduce gradient noise → optimization behaves like noisy gradient descent.

  * Can help escape sharp minima and saddle points.
  * Often associated with better generalization but slower wall-clock time.

* **Large batches** approximate full gradient:

  * Faster per-epoch convergence.
  * Can converge to sharper minima with worse generalization (though this can be mitigated by tuning LR, schedules, and regularization).

“Flat minima” loosely: regions where loss doesn’t increase sharply when perturbing $\theta$. They correlate with better generalization. Noise from small batches acts like a temperature, encouraging convergence to flatter minima.

---

### 2.7 Loss Landscape Intuition

Key mental pictures:

* **Non-convexity**: DL loss surfaces have many saddle points and flat regions, not just isolated local minima.
* **Saddle points**: points where gradient is zero but Hessian has both positive and negative eigenvalues; high-dimensional optimization tends to spend more time at saddles than at poor local minima.
* **Mode connectivity**: many minima are connected by low-loss paths in parameter space, suggesting “wide basins” rather than isolated deep wells.

You don’t need exact Hessian details for interviews, but remember:

* Stochasticity and momentum help navigate the landscape.
* Overparameterization tends to create many good minima (easier optimization than small, underparameterized nets).

---

### 2.8 Training Tricks

#### 2.8.1 Label Smoothing

Instead of one-hot targets, with smoothing factor $\alpha$:

* For correct class $y$: $q_y = 1 - \alpha$,
* For others: $q_j = \alpha / (K - 1)$, where $K$ is #classes.

Then minimize cross-entropy between $q$ and predicted distribution $p$.

Effects:

* Prevents overconfident predictions.
* Acts as regularization and improves calibration.
* Reduces vulnerability to label noise.

#### 2.8.2 Mixup and CutMix

* **Mixup**:

  * Create virtual samples $(\tilde{x}, \tilde{y})$ as convex combinations:
    $$
    \tilde{x} = \lambda x_i + (1-\lambda) x_j,\quad
    \tilde{y} = \lambda y_i + (1-\lambda) y_j,
    $$
    with $\lambda \sim \text{Beta}(\alpha,\alpha)$.
  * Encourages linear behavior between training examples; acts as strong regularizer.

* **CutMix**:

  * Cut a patch from one image and paste it into another, mixing labels proportionally to area.
  * Better preserves local structure than global mixup for vision.

#### 2.8.3 Early Stopping

Monitor validation loss; stop training when it stops improving for some patience window.

* Acts as implicit regularization by preventing overfitting.
* Cheap and widely used.

#### 2.8.4 Weight Decay vs $L_2$ Regularization

In plain SGD, adding $L_2$ penalty $\lambda |\theta|^2$ is equivalent to weight decay (multiplying $\theta$ by $(1 - \eta\lambda)$ at each step). For adaptive optimizers, this equivalence breaks, hence the need for **AdamW** (true decoupled weight decay).

---

### 2.9 Polyak Averaging and EMA Weights

1. **Polyak averaging**

Average parameters over training steps:
$$
\bar{\theta}^T = \frac{1}{T} \sum_{t=1}^T \theta_t.
$$

* The averaged model often generalizes better than the final iterate.
* Related to stochastic approximation theory.

2. **Exponential Moving Average (EMA)**

Maintain a smoothed version of weights:
$$
\theta^\text{EMA}_t = \alpha \theta^\text{EMA}_{t-1} + (1 - \alpha) \theta_t.
$$

* Evaluate on validation/test with $\theta^\text{EMA}$ rather than raw $\theta$.
* Helps stabilize training and improve final performance; widely used in diffusion models and vision.

---

### 2.10 Lottery Ticket Hypothesis (LTH)

Statement (informal):

* A randomly initialized dense network contains a **subnetwork** (“winning ticket”) that, when trained in isolation with the same training procedure, can match or exceed the performance of the full network.

Typical process (iterative pruning):

1. Train full network for some epochs.
2. Prune a fraction of smallest-magnitude weights.
3. Reset remaining weights to their **original initialization**.
4. Retrain only the subnetwork.
5. Repeat.

Implications:

* Overparameterized nets may primarily help optimization by containing many good subnetworks.
* Structured pruning guided by LTH can lead to efficient subnetworks.

---

> **Quick check, just to lock this in:**
**Why AdamW is preferred over classic Adam for modern LLM training**, what are the 1–2 key points you’d highlight? (Short bullet or 2–3 sentence answer is fine.)

---

## 3. Regularization Techniques

High-level: regularization controls **effective capacity** so the model fits signal rather than noise. You can think in three broad buckets:

* Penalties on parameters (weight decay, etc.).
* Stochastic structure on the network (dropout, stochastic depth).
* Data-side regularization (augmentations, curriculum).

### 3.1 Weight Decay and AdamW Decoupling

In its simplest form, weight decay adds an $L_2$ penalty to the loss:
$$
L'(\theta)=L(\theta)+\frac{\lambda}{2}|\theta|^2.
$$

For vanilla SGD:
$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L'(\theta_t)
=\theta_t-\eta(\nabla_\theta L(\theta_t)+\lambda\theta_t).
$$

Rewriting:
$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\nabla_\theta L(\theta_t),
$$
so parameters shrink multiplicatively each step (“decay”), plus the usual gradient step.

For **Adam**, if you naively add $\lambda\theta$ to the gradient, that $\lambda\theta$ also flows through the adaptive rescaling, which changes the regularization behavior. **AdamW** fixes this by **decoupling** weight decay from the adaptive step:

* Compute adaptive step as usual on loss gradient.
* Then apply decay:
  $$
  \theta_{t+1}=\theta_t-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}-\eta\lambda\theta_t.
  $$

This behaves much closer to true $L_2$ regularization and is standard for modern deep nets (especially Transformers).

---

### 3.2 Dropout and DropConnect

Both inject **multiplicative noise** during training to prevent co-adaptation.

#### 3.2.1 Dropout

Given hidden activations $h\in\mathbb{R}^d$:

* Sample mask $m\in{0,1}^d$ with $P(m_i=1)=p$.
* Training-time:
  $$
  \tilde{h}=\frac{m\odot h}{p}.
  $$

Scaling by $1/p$ keeps the expected activation constant:
$\mathbb{E}[\tilde{h}_i]=h_i$.

Interpretations:

* Ensemble: training is like averaging over many thinned networks.
* Regularizer: forces units not to rely on specific others, improving robustness.

At test time: use full network with no dropout (or equivalently, use $p=1$ and no scaling).

#### 3.2.2 DropConnect

Instead of dropping activations, DropConnect randomly zeros **weights**:

* For weight matrix $W$, sample mask $M$ with Bernoulli($p$) entries.
* Use $\tilde{W}=M\odot W$ during training.

Less common in practice than dropout, but conceptually similar: random sparse subnetwork each iteration.

---

### 3.3 Normalization Layers: BatchNorm, LayerNorm, GroupNorm

All normalize some set of activations to have roughly zero mean and unit variance, then learn scale/shift.

Let $x$ be activations for a given layer.

#### 3.3.1 Batch Normalization (BatchNorm)

For a given feature channel $c$, over a mini-batch (and sometimes spatial dims):

$$
\mu_c=\frac{1}{m}\sum_{i=1}^m x_{i,c},\quad
\sigma_c^2=\frac{1}{m}\sum_{i=1}^m(x_{i,c}-\mu_c)^2.
$$

Normalize, then scale/shift:
$$
\hat{x}_{i,c}=\frac{x_{i,c}-\mu_c}{\sqrt{\sigma_c^2+\epsilon}},\quad
y_{i,c}=\gamma_c\hat{x}_{i,c}+\beta_c.
$$

Effects:

* Stabilizes training by controlling activation statistics.
* Allows higher learning rates.
* Implicit regularization via batch noise.

Limitations:

* Depends on batch statistics; small batch sizes or variable-length sequences can hurt stability.
* Awkward for some RNN or autoregressive settings.

#### 3.3.2 Layer Normalization (LayerNorm)

Normalize **across features** for each sample independently:

For a given sample $x\in\mathbb{R}^d$:
$$
\mu=\frac{1}{d}\sum_{j=1}^d x_j,\quad
\sigma^2=\frac{1}{d}\sum_{j=1}^d(x_j-\mu)^2.
$$
$$
\hat{x}_j=\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}},\quad
y_j=\gamma_j\hat{x}_j+\beta_j.
$$

No dependency on batch dimension → works well for Transformers and sequence models.

#### 3.3.3 GroupNorm

Partition channels into groups, normalize within each group for each sample. Interpolates between LayerNorm (one group = all channels) and InstanceNorm. Designed to be stable across a wide range of batch sizes (e.g., detection/segmentation models).

---

### 3.4 Stochastic Depth

Used mainly in very deep residual networks.

Idea: randomly **drop entire residual blocks** during training:

For a residual block with input $x$ and transformation $F(x)$:
$$
\text{Standard: }y=x+F(x).
$$

With stochastic depth:

* With probability $p$, use $y=x$ (skip block).
* With probability $1-p$, use $y=x+F(x)$, possibly scaled to keep expectations aligned.

Interpretation:

* Trains a family of shallower networks (like dropout at the block level).
* Reduces effective depth during training, helping gradient flow and acting as regularizer.
* At test time, use all blocks deterministically.

---

### 3.5 Data Augmentation

Instead of restricting the hypothesis class directly, augment the data to enforce invariances and robustness.

#### 3.5.1 Image

* Classical: random crops, flips, rotations, color jitter, Gaussian noise, Cutout.
* Strong augmentations: RandAugment, AutoAugment, Mixup, CutMix.
* Correspond to assumptions like “object identity is invariant to small translations / color shifts / occlusions.”

#### 3.5.2 Text

More subtle; naive operations often break semantics.

* Back-translation (translate to another language and back).
* Token-level perturbations: synonym replacement, random deletion (used sparingly).
* Span masking (BERT-style), which is more pretraining than augmentation but plays a similar role.

#### 3.5.3 Audio

* Time-stretching, pitch-shift.
* SpecAugment: masking frequency/time bands on spectrograms.

---

### 3.6 Curriculum Learning

Train on data in a structured order, from “easy” to “hard”:

* Define a difficulty measure (heuristic, model-based, or label-based).
* Start training on simpler examples, gradually introduce harder ones.

Intuition:

* Optimization landscape is easier to navigate when starting with simple patterns.
* Analogous to human education.

Related concept: **self-paced learning**, where the model’s own confidence is used to select the next batch of examples.

---

## 4. Convolutional Neural Networks (CNNs)

CNNs exploit **spatial locality** and **translation invariance** for grid-like data (images, audio spectrograms).

### 4.1 CNN Fundamentals

#### 4.1.1 Convolution Operation

For a 2D input $X$ and kernel $K$, a single-channel “valid” convolution:

$$
Y(i,j)=\sum_{u,v}K(u,v),X(i+u,j+v).
$$

In practice:

* Multiple input channels and output channels.
* Convolution can be seen as a sliding dot product between local input patches and kernels.

Compared to fully connected layers:

* Parameter sharing: same kernel applied across spatial positions.
* Sparse connectivity: each output depends only on a local receptive field.

#### 4.1.2 Padding, Stride, Dilation

* **Padding**: extend input with zeros around borders to control output size.

  * “Same” padding ≈ preserves spatial size.
  * “Valid” padding = no padding → output shrinks.

* **Stride**: step size when sliding the kernel.

  * Stride $s>1$ down-samples feature maps.

* **Dilation**: “holes” in the kernel; spaced out kernel positions.

  * Dilation factor $d$ means sampling every $d$-th input position.
  * Increases receptive field without increasing kernel size.

#### 4.1.3 Transposed Convolution

Used for upsampling / “deconvolution” (segmentation, generators):

* Learnable upsampling that is (loosely) the inverse of convolution.
* You can think of inserting zeros between input positions and then applying a standard convolution.

Common pitfalls: checkerboard artifacts if stride and kernel size are not chosen carefully; often mitigated by using resize (nearest/bilinear) + standard conv.

#### 4.1.4 Parameter Sharing and Receptive Field

* **Parameter sharing**: each filter’s weights are reused across all spatial positions, yielding translation equivariance.
* **Receptive field**: region of input that affects a given output activation.

  * Grows with depth, kernel size, stride, and dilation.
  * Large receptive fields are crucial for capturing global context.

#### 4.1.5 Pooling

* **Max pooling**: take max over local window.
* **Average pooling**: take mean over window.
* **Global average pooling (GAP)**: average over entire spatial dimensions → one value per channel.

Benefits:

* Invariance to small translations.
* Down-sampling for computational savings.
* GAP in particular removes need for fully connected layers at the end of CNNs and acts as a form of regularization (used heavily in ResNet-like models).

---

### 4.2 CNN Model Families (Historical Arc)

You mainly need the **ideas** each family contributed.

#### 4.2.1 LeNet

Early CNN for digit recognition:

* Few conv + pooling layers → small fully connected head.
* Showed CNNs can work well on real tasks.

#### 4.2.2 AlexNet and VGG

* **AlexNet**: won ImageNet 2012.

  * Used ReLU, dropout, data augmentation, GPU training.
  * Deep for its time (8 layers), but large fully connected head.

* **VGG**:

  * Very deep stacks of $3\times 3$ convs.
  * Showed depth alone (with small filters) yields good performance.
  * Very heavy in parameters and computation.

#### 4.2.3 ResNet and Variants

* Residual blocks: $y=x+F(x)$.
* Enables training very deep networks (50, 101, 152 layers, etc.).
* Core idea: easier to learn residual mapping than direct mapping; gradients flow more directly via identity paths.

Variants:

* **ResNeXt**: cardinality (groups of transformations) as a new dimension; group convolutions in residual blocks.
* **WideResNet**: fewer layers but wider channels; showed that width can substitute for depth to some extent.

#### 4.2.4 DenseNet

* Dense connectivity: each layer receives concatenation of all previous feature maps.
  $$
  x_l=H_l([x_0,x_1,\dots,x_{l-1}]).
  $$
* Encourages feature reuse, mitigates vanishing gradients.
* Parameter-efficient for its depth, though memory-heavy due to concatenation.

#### 4.2.5 MobileNet, ShuffleNet, EfficientNet

Target: efficient inference on mobile / edge devices.

* **MobileNet**:

  * Heavy use of **depthwise separable convolutions** (see below).
  * Significantly reduces FLOPs.

* **ShuffleNet**:

  * Group convolutions + channel shuffle to maintain cross-channel information flow.

* **EfficientNet**:

  * Compound scaling: systematically scale **depth, width, and resolution** according to a simple formula.
  * Derived via architecture search + scaling laws.

#### 4.2.6 ConvNeXt and ViTs

* **ConvNeXt**:

  * A “modernized” CNN incorporating Transformer-era design patterns (large kernels, LayerNorm-like normalization, inverted bottlenecks).
  * Achieves ViT-level accuracy while retaining convolutional inductive biases.

* **Vision Transformers (ViT, DeiT)**:

  * Treat image patches as tokens, use Transformer blocks rather than convs.
  * Initially needed large datasets (JFT, etc.), later with DeiT and strong augmentation work well on ImageNet.
  * Conceptually: CNNs vs ViTs is mostly about **hard-coded locality vs learned attention**.

---

### 4.3 Advanced CNN Ideas

#### 4.3.1 Atrous/Dilated Convolution

Replace standard conv with dilation factor $d$:

$$
Y(i,j)=\sum_{u,v}K(u,v),X(i+du,j+dv).
$$

* Enlarges receptive field without increasing parameters or reducing resolution.
* Widely used in semantic segmentation (e.g., DeepLab).

#### 4.3.2 Depthwise Separable Convolution

Factor a standard conv into:

1. **Depthwise convolution**: per-channel spatial conv with kernel $k\times k$.
2. **Pointwise convolution**: $1\times 1$ conv across channels.

Standard conv with $C_\text{in}$ input channels, $C_\text{out}$ output channels, kernel $k$ has $k^2C_\text{in}C_\text{out}$ parameters.

Depthwise-separable:

* Depthwise: $k^2C_\text{in}$.
* Pointwise: $C_\text{in}C_\text{out}$.
  Total: $k^2C_\text{in}+C_\text{in}C_\text{out}$, much cheaper when $k$ is large.

Used heavily in MobileNet and related families.

#### 4.3.3 Squeeze-and-Excitation (SE) Blocks

Channel-wise attention:

1. **Squeeze**: global average pooling over spatial dimensions → vector $s\in\mathbb{R}^C$.
2. **Excitation**: small MLP to produce per-channel weights $a\in\mathbb{R}^C$ (often with sigmoid output).
3. **Scale**: multiply feature maps by $a$ channel-wise.

Intuition: let the network learn which channels are important for each input, akin to attention along the channel dimension.

#### 4.3.4 Feature Pyramid Networks (FPN)

For detection/segmentation, objects appear at multiple scales. FPN:

* Builds a **top-down** feature pyramid from deep, low-resolution, semantically rich feature maps combined with shallow, high-resolution ones via lateral connections.
* Produces multi-scale feature maps with strong semantics at each scale.
* Used extensively in modern detectors (e.g., RetinaNet).

#### 4.3.5 Anchor-based vs Anchor-free Detection

* **Anchor-based** (e.g., Faster R-CNN, RetinaNet, YOLOv3/v4):

  * Predefine a set of boxes (“anchors”) at each location, with various scales/aspect ratios.
  * Network predicts offsets + classification for each anchor.
  * Needs careful anchor design and tuning.

* **Anchor-free** (e.g., FCOS, CenterNet, some YOLO variants like YOLOX/RT-DETR-like ideas):

  * Predict bounding boxes directly from keypoints or direct regression from each location.
  * Simplifies architecture and hyperparameters; often easier to train.

---

## 5. Recurrent Neural Networks (RNNs)

### 5.1 Core RNN Formulation

A (vanilla) RNN processes a sequence $(x_1,\dots,x_T)$ with hidden state $h_t$:

* Hidden update:
$$
h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)
$$
* Output (optional):
$$
y_t=W_{hy}h_t+b_y
$$

Here $\phi$ is typically $\tanh$ or $\text{ReLU}$; $h_0$ is usually zeros (or learned).

You can view this as:

* A **time-unrolled MLP** with shared parameters at each time step.
* Depth in time is $T$; backprop must pass gradients through all these steps.

### 5.2 Vanishing/Exploding Gradients and Truncated BPTT

Backpropagation through time (BPTT) applies the chain rule across all time steps:

* Gradients involve products of Jacobians:
$$
\frac{\partial L}{\partial h_{t-k}}=\frac{\partial L}{\partial h_t}
\prod_{j=t-k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}
$$

If the typical singular values of $\frac{\partial h_j}{\partial h_{j-1}}$ are:

* $<1$: products shrink → **vanishing gradients** (early timesteps hardly update).
* $>1$: products blow up → **exploding gradients**.

Mitigations:

* Gradient clipping (norm-based).
* Gated RNNs (LSTM/GRU) with better gradient flow.
* **Truncated BPTT**: only backprop through a window of $k$ time steps (e.g., 128), not the whole sequence, to reduce both compute and instability.

---

### 5.3 LSTM

Long Short-Term Memory (LSTM) introduces a **cell state** $c_t$ and gates that control information flow.

Given input $x_t$, previous hidden $h_{t-1}$ and cell $c_{t-1}$:

Gates:
$$
\begin{aligned}
i_t&=\sigma(W_ix_t+U_ih_{t-1}+b_i)&&\text{(input gate)}\\
f_t&=\sigma(W_fx_t+U_fh_{t-1}+b_f)&&\text{(forget gate)}\\
o_t&=\sigma(W_ox_t+U_oh_{t-1}+b_o)&&\text{(output gate)}\\
\tilde{c}*t&=\tanh(W_cx_t+U_ch*{t-1}+b_c)&&\text{(candidate cell)}
\end{aligned}
$$

Cell and hidden updates:
$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t
$$
$$
h_t=o_t\odot\tanh(c_t)
$$

Key idea: the cell state $c_t$ has an **additive** update (rather than purely multiplicative), so gradients can flow along the “memory” path $c_{t-1}\to c_t$ controlled by $f_t$ and $i_t$.

Intuition:

* $f_t$ decides what to forget.
* $i_t$ decides what new information to store.
* $o_t$ decides what part of the cell to expose.

---

### 5.4 GRU

Gated Recurrent Unit (GRU) simplifies LSTM by merging cell and hidden state.

Given $h_{t-1}$ and $x_t$:

$$
\begin{aligned}
z_t&=\sigma(W_zx_t+U_zh_{t-1}+b_z)&&\text{(update gate)}\\
r_t&=\sigma(W_rx_t+U_rh_{t-1}+b_r)&&\text{(reset gate)}\\[4pt]
\tilde{h}*t&=\tanh(W_hx_t+U_h(r_t\odot h*{t-1})+b_h)\\
h_t&=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t
\end{aligned}
$$

Interpretation:

* $z_t$ balances between keeping old state $h_{t-1}$ and adopting new candidate $\tilde{h}_t$.
* Fewer parameters than LSTM; similar purpose (mitigating vanishing gradients).

---

### 5.5 Bi-directional and Deep RNNs

* **Bi-directional RNNs (BiRNNs)**:

  * Process sequence forwards and backwards:
    $$
    \overrightarrow{h}_t=\text{RNN}_f(x_1,\dots,x_t),\quad
    \overleftarrow{h}_t=\text{RNN}_b(x_T,\dots,x_t)
    $$
  * Concatenate: $h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]$.
  * Useful when you can access full context (e.g., tagging, classification), not for strict online/causal tasks.

* **Deep RNNs**:

  * Stack multiple RNN layers: each layer’s outputs serve as inputs to the next.
  * Increases representational capacity but exacerbates training difficulty.

---

### 5.6 Attention on RNNs: Seq2Seq + Attention

Classic **sequence-to-sequence** with attention:

1. **Encoder**: RNN (LSTM/GRU) encodes source sequence into hidden states $(h_1,\dots,h_T)$.
2. **Decoder**: another RNN generates target sequence; at each step $t$ it:

   * Computes attention weights over encoder states:
     $$
     \alpha_{t,i}=\text{softmax}*i(e*{t,i}),\quad e_{t,i}=\text{score}(s_{t-1},h_i)
     $$
     where $s_{t-1}$ is decoder state.
   * Context vector:
     $$
     c_t=\sum_i\alpha_{t,i}h_i
     $$
   * Uses $[c_t,y_{t-1}]$ as input to decoder RNN.

This solved the “information bottleneck” of encoding an entire sequence into a single vector; attention lets the decoder access all encoder states directly.

---

### 5.7 RNN Applications (Conceptual)

* **Language modeling**: predict next token given previous.
* **Machine translation**: seq2seq + attention (pre-Transformer).
* **Speech**: acoustic modeling, language modeling, end-to-end ASR (before Transformers).

In practice, Transformers have largely replaced RNNs in new architectures, but RNNs are still interview-relevant for understanding sequence modeling and attention’s motivation.

---

## 6. Attention & Transformers

### 6.1 Scaled Dot-Product Attention

Given queries $Q\in\mathbb{R}^{T_q\times d_k}$, keys $K\in\mathbb{R}^{T_k\times d_k}$, and values $V\in\mathbb{R}^{T_k\times d_v}$:

1. Compute similarity scores:
$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$
2. Apply softmax row-wise to get attention weights:
$$
A=\text{softmax}(S)
$$
3. Weight values:
$$
\text{Attn}(Q,K,V)=AV
$$

* Scaling by $\sqrt{d_k}$ avoids excessively large dot products when $d_k$ is large (stabilizes softmax gradients).

**Self-attention**: $Q=K=V$ (tokens attend to each other in the same sequence).  
**Cross-attention**: $Q$ from one sequence (e.g., decoder), $K,V$ from another (e.g., encoder).

---

### 6.2 Masked (Causal) Attention

For autoregressive language modeling, token $t$ must not attend to future tokens $>t$.

Implement with a mask $M$ where:

* $M_{ij}=0$ if $j\le i$,
* $M_{ij}=-\infty$ (or large negative) if $j>i$.

Attention logits become:
$$
S'=\frac{QK^\top}{\sqrt{d_k}}+M
$$

Softmax then zeroes out future positions.

---

### 6.3 Multi-Head Attention

Instead of a single attention, use $H$ heads:

For each head $h$:
$$
Q_h=XW_h^Q,\quad K_h=XW_h^K,\quad V_h=XW_h^V
$$
$$
\text{head}_h=\text{Attn}(Q_h,K_h,V_h)
$$

Concatenate and project:
$$
\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_H)W^O
$$

Intuition:

* Each head captures different relations (e.g., syntactic dependencies, semantic similarity).
* Multi-head helps the model represent multiple subspaces and attention patterns.

---

### 6.4 Positional Encodings

Attention is permutation-invariant; we must inject order information.

#### 6.4.1 Sinusoidal Absolute Encoding

For position $t$ and dimension $2i$ / $2i+1$:

$$
\text{PE}(t,2i)=\sin\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right),\quad
\text{PE}(t,2i+1)=\cos\left(\frac{t}{10000^{2i/d_{\text{model}}}}\right)
$$

Then:
$$
\tilde{x}_t=x_t+\text{PE}(t)
$$

Properties:

* Encodes relative distances via linear combinations.
* No extra parameters; extrapolates to longer sequences (in principle).

#### 6.4.2 Learned Positional Embeddings

Treat position index as another token ID; learn an embedding:
$$
\text{PE}(t)=P_t,\quad P\in\mathbb{R}^{L_{\max}\times d_{\text{model}}}
$$

Simple and flexible, but limited to $L_{\max}$ unless extended.

#### 6.4.3 Rotary Positional Embeddings (RoPE)

Apply a **rotation** in 2D subspaces of the embedding, parameterized by position. Roughly:

* Split each head dimension into 2D pairs; multiply each pair by a rotation matrix $\mathbf{R}(t)$ depending on $t$.
* Rotation is frequency-based, similar to sinusoidal encodings but applied at the level of $(Q,K)$.

Benefits:

* Encodes relative positions naturally.
* Works well for extrapolating context length when combined with interpolation/rescaling tricks.

#### 6.4.4 ALiBi

Attention with Linear Biases:

* No explicit positional embeddings.
* Add a learned linear bias to attention logits proportional to distance:
  $$
  S'*{ij}=S*{ij}+m_h\cdot(i-j)
  $$
  where $m_h$ is a head-specific slope.

Effect:

* Encourages attention to nearby tokens more than distant ones in a simple, extrapolation-friendly way.

---

### 6.5 Transformer Block

A standard Transformer **encoder block**:

1. Input $X\in\mathbb{R}^{T\times d}$.

2. Multi-head self-attention:
   $$
   H=\text{MHA}(X)
   $$

3. Residual + normalization:

   * Post-norm (original):
     $$
     X'=\text{LayerNorm}(X+H)
     $$
   * Pre-norm (modern):
     $$
     H=\text{MHA}(\text{LayerNorm}(X)),\quad
     X'=X+H
     $$

4. Feed-forward network (FFN):
   $$
   F=\text{FFN}(X')=\sigma(X'W_1+b_1)W_2+b_2
   $$
   often with expansion: $d\to4d\to d$.

5. Another residual + norm.

Decoder blocks additionally:

* Use **masked self-attention** over decoder inputs.
* Use **cross-attention** to encoder outputs.

**Pre-norm vs post-norm**:

* Pre-norm: LayerNorm applied before sublayers, improves training stability for very deep Transformers.
* Post-norm: LayerNorm after residual; original Transformer; more prone to instability at large depths.

---

### 6.6 Transformer Model Families (High-Level)

You mainly need **objective + architecture** for each.

1. **BERT (encoder-only, masked LM)**

* Bidirectional encoder stack.
* Pretraining objective: **masked language modeling (MLM)** + sometimes next-sentence prediction.
* MLM: randomly mask some tokens and predict them using full-context attention.
* Used as contextual encoder for classification, tagging, QA (with task-specific heads).

2. **GPT (decoder-only, causal LM)**

* Stack of masked self-attention decoder blocks.
* Pretraining: **next-token prediction**:
  $$
  \max_\theta\sum_t\log p_\theta(x_t\mid x_{<t})
  $$
* Used as generative LM; fine-tuned or instructed for downstream tasks.

3. **T5 (encoder–decoder, text-to-text)**

* Encoder–decoder Transformer.
* Everything cast as text-to-text: input and output are text sequences.
* Pretraining: span corruption (mask spans, predict them).
* Very flexible; good for sequence-to-sequence tasks.

4. **Encoder-only variants**: RoBERTa, DeBERTa, ALBERT

* **RoBERTa**: better training recipe (more data, no NSP, dynamic masking).
* **DeBERTa**: disentangled attention (content vs position), relative positional encodings.
* **ALBERT**: parameter sharing + factorized embeddings for efficiency.

5. **Decoder-only variants**: GPT-2/3/NeoX/Mistral, LLaMA

* Differences mostly in scale, training data, tokenizer, architecture tweaks (e.g., RMSNorm, SwiGLU, RoPE).
* Objective is still next-token prediction.

6. **Vision Transformers (ViT, DeiT)**

* Treat image patches as tokens, apply standard Transformer encoder.
* ViT: needs large pretraining data.
* DeiT: data-efficient training with distillation and strong augmentation.

7. **Swin Transformer**

* Hierarchical vision Transformer with local windows and window shifting.
* Adds locality and multi-scale structure akin to CNNs.

---

### 6.7 Scaling Laws (Qualitative)

Empirical scaling laws (Kaplan, Chinchilla) show:

* Loss decreases roughly as a power-law in model size, dataset size, and compute, when trained near optimally.
* **Chinchilla**: for a given compute budget, best performance comes from **smaller models trained on more data** compared to the earlier “GPT-3 style” very large models on relatively less data.
* Practical implication: tune **parameters, data, and compute** jointly; do not just scale parameters without enough data.

---

## 7. Large Language Models (LLMs)

### 7.1 Pretraining Objectives

At core, LLMs are just very large sequence models trained with **maximum likelihood**.

#### 7.1.1 Next-Token Prediction (Causal LM)

Given sequence $x_{1:T}$, maximize:
$$
\log p_\theta(x_{1:T}) = \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}).
$$

Training loss (per-token cross-entropy):
$$
L(\theta) = - \mathbb{E}_{x \sim \mathcal{D}} \sum_{t} \log p_\theta(x_t \mid x_{<t}).
$$

* Architecturally: **decoder-only** Transformer with causal mask.
* This is what GPT-style models do.

#### 7.1.2 Masked LM (MLM)

Given input sequence, randomly mask a subset of tokens $M$ and predict them using full context:

$$
L(\theta) = - \mathbb{E}_{x,M} \sum_{t \in M} \log p_\theta(x_t \mid x_{\setminus M}).
$$

* **Encoder-only** (BERT-style) with bidirectional attention.
* Great for representations, less natural for left-to-right generation.

#### 7.1.3 Denoising / Span Corruption

Generalization of MLM: you corrupt the input (e.g., noise tokens, remove spans) and train to reconstruct original.

* T5-style: mask spans and replace them with special sentinel tokens; model generates missing spans.
* Sequence-to-sequence formulation allows flexible input–output mappings.

#### 7.1.4 Mixture-of-Experts (MoE) Training

Instead of one dense FFN per layer, you have $E$ experts and a learned router:

* For token embedding $h$:

  * Router outputs scores $r \in \mathbb{R}^E$,
  * Select top-$k$ experts per token (e.g., $k=2$),
  * Weighted sum of their outputs:
    $$ 
    \text{MoE}(h) = \sum_{e \in \text{Top-}k} \alpha_e \cdot \text{FFN}_e(h).
    $$

Benefits:

* Parameter count can be huge, but **compute per token** similar to dense model.
* Training difficulty: load balancing, routing collapse, etc.

---

### 7.2 Distributed Training (Very High Level)

LLMs don’t fit into a single GPU, so you combine multiple forms of parallelism:

1. **Data parallel**: replicate model across GPUs, each processes different mini-batches; gradients averaged.
2. **Tensor model parallel**: split individual layers (e.g., matmul) across devices.
3. **Pipeline parallel**: split layers across stages, feed micro-batches through pipeline.
4. **ZeRO/FSDP**:

   * ZeRO stages: partition optimizer states, gradients, and parameters across devices.
   * FSDP: fully shard model parameters across devices and all-gather as needed.

The goal is to minimize **peak memory** and **communication overhead**, while keeping GPUs busy.

---

### 7.3 Fine-Tuning and Alignment

LLM pipelines often look like:

1. Pretrain with next-token prediction.
2. Supervised fine-tuning on **instruction / chat** data.
3. Preference-based alignment (RLHF, DPO, etc.).
4. Optional self-improvement loops.

#### 7.3.1 Supervised Fine-Tuning (SFT)

Given instruction–response pairs $(x, y)$, train generatively:

$$
L(\theta) = - \sum_{t=1}^{T_y} \log p_\theta(y_t \mid x, y_{<t}).
$$

* Encourages the model to produce desired answers given prompts.
* Still just cross-entropy under teacher forcing.

#### 7.3.2 RLHF (with PPO)

Pipeline:

1. **SFT model**: baseline “helpful” model.
2. **Reward model**: trained to predict human preferences over pairs of outputs:

   * Given $(y_A, y_B)$ for prompt $x$, and human label (which is better), train $r_\phi$ so that
     $$
     P(y_A \succ y_B \mid x) = \sigma(r_\phi(x,y_A) - r_\phi(x,y_B)).
     $$
3. **RL step (usually PPO)**:

   * Policy $\pi_\theta$ initialized from SFT.
   * Reward:
     $$
     R(x,y) = r_\phi(x,y) - \beta \mathrm{KL}(\pi_\theta(\cdot\mid x) \mid \pi_{\text{SFT}}(\cdot\mid x)),
     $$
   * Optimize expected reward via PPO.

Pros:

* Can directly optimize non-differentiable preference signals.

Cons:

* RL training instability, reward hacking, expensive.

#### 7.3.3 Direct Preference Optimization (DPO)

Avoids explicit online RL; optimizes an objective implied by preferences and a reference policy.

Given preferred $y^+$ and rejected $y^-$ for prompt $x$, with reference policy $\pi_{\text{ref}}$:

DPO objective (simplified):

$$
\mathcal{L}_{\text{DPO}}(\theta) = - \log \sigma\left(
\beta \left[
\log \pi_\theta(y^+ \mid x) - \log \pi_\theta(y^- \mid x)
- \log \pi_{\text{ref}}(y^+ \mid x) + \log \pi_{\text{ref}}(y^- \mid x)
\right]\right).
$$

* Pure supervised optimization on log-probs, no rollout or credit assignment through sampling.
* Much simpler to implement than PPO; widely used in practice now.

(ORPO, RRHF, etc. are variations on direct preference / ranking style training with different regularizations and sampling strategies.)

---

### 7.4 Inference Acceleration

Key pain points: **latency** and **memory bandwidth**, especially for long contexts.

#### 7.4.1 KV Caching

In autoregressive decoding, at time step $t$:

* Self-attention uses all previous tokens’ keys and values $(K_{1:t}, V_{1:t})$.
* We can cache $K_{1:t-1}, V_{1:t-1}$ from earlier steps and only compute $K_t, V_t$ for the new token.
* Attention then is just with the concatenated cache, avoiding recomputing for all past tokens.

This turns per-step attention cost from $O(t^2)$ (if recomputed) to $O(t)$, and overall from $O(T^2)$ to $O(T^2)$ but with a much smaller constant and better cache use. For large $T$, caching is essential.

#### 7.4.2 FlashAttention

Exact attention algorithm that is IO-aware:

* Computes attention in tiles in GPU SRAM, not by materializing full $QK^\top$ and attention matrix.
* Fuses softmax and matmuls to reduce memory reads/writes.
* Overall complexity remains $O(T^2)$, but memory traffic is drastically reduced, giving large wall-clock speedups and allowing longer sequences.

#### 7.4.3 Speculative Decoding

Use a small **draft model** $p_\phi$ to propose a chunk of tokens, then verify/accept them with the large model $p_\theta$.

Rough sketch:

1. Draft model generates candidate continuation $(\tilde{x}_{t+1:t+k})$ quickly.
2. Large model evaluates joint probabilities and either accepts the whole block or partially corrects it.
3. Carefully designed to keep the overall distribution equal to $p_\theta$.

Benefit: fewer calls to the large model per generated token; speedup depends on how good the small model is.

#### 7.4.4 Multi-Token Prediction

Train model to predict multiple future tokens per step (e.g., predict $x_{t+1:t+k}$ from prefix). At inference, you can generate multiple tokens in fewer forward passes.

Tricky points:

* Need architecture and training objective that maintain generation quality.
* Still an active research area.

---

### 7.5 Quantization and Distillation

#### 7.5.1 Quantization

Represent weights (and sometimes activations) with lower precision than FP16/FP32.

* **8-bit, 4-bit** quantization are common trade-offs.
* Formats: NF4, FP4, INT4, etc.
* Approaches:

  * Post-training quantization (PTQ): calibrate scales on a small dataset, no retraining.
  * Quantization-aware training (QAT): simulate quantization during training to adapt.

Methods like GPTQ, AWQ, SmoothQuant are practical recipes for weight-only or weight+activation quantization for LLMs. Core idea: approximate high-precision matmul with low-precision operations while controlling error and preserving quality.

#### 7.5.2 Distillation

Train a smaller **student** model to mimic a larger **teacher** model.

Typical losses:

1. Match teacher token distributions:
   $$
   L_{\text{KD}} = \mathrm{KL}\big(p_{\text{teacher}}(\cdot\mid x)\ \mid\ p_{\text{student}}(\cdot\mid x)\big).
   $$
2. Optionally mix with ground-truth cross-entropy.

For LLMs:

* Can distill general behavior (from pretraining) or aligned behavior (from RLHF/DPO-tuned teacher).
* Often used to obtain small, fast models for real-time applications.

---

## 8. Generative Models

Now zooming out from LLMs to general deep generative modeling.

### 8.1 Autoencoders (AEs, DAEs, Sparse AEs, VAEs)

#### 8.1.1 Basic Autoencoder

Encoder:
$$
z = f_\theta(x)
$$
Decoder:
$$
\hat{x} = g_\phi(z)
$$

Train to minimize reconstruction loss:
$$
L(\theta,\phi) = \mathbb{E}_{x \sim \mathcal{D}} \big[\ell(x, \hat{x})\big].
$$

* If capacity is high and no constraints, AE can learn identity.
* Regularization (bottleneck, noise, sparsity) encourages learning useful latent representations.

#### 8.1.2 Denoising Autoencoder (DAE)

Corrupt input $\tilde{x} \sim q(\tilde{x} \mid x)$ (e.g., add noise, mask pixels), and reconstruct original $x$:

$$
L = \mathbb{E}_{x,\tilde{x}} \big[\ell\big(x, g_\phi(f_\theta(\tilde{x}))\big)\big].
$$

* Forces robustness to noise.
* DAEs are connected to score matching and diffusion ideas.

#### 8.1.3 Sparse Autoencoder

Encourage latent $z$ to be sparse, via penalty like:
$$
\Omega(z) = \lambda \sum_i |z_i|
\quad \text{or a KL penalty to a sparse prior.}
$$

This mimics dictionary learning, discovering parts-based representations.

#### 8.1.4 Variational Autoencoder (VAE)

Latent variable model with explicit generative story:

1. Prior: $z \sim p(z)$ (often $\mathcal{N}(0,I)$).
2. Decoder (likelihood): $x \sim p_\theta(x \mid z)$.

Intractable posterior $p_\theta(z \mid x)$ → approximate with $q_\phi(z \mid x)$ (encoder). Optimize **ELBO**:

$$
\log p_\theta(x) \ge
\mathbb{E}_{z \sim q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
- D_{\text{KL}}(q_\phi(z \mid x) \mid p(z)).
$$

Use **reparameterization trick**:
$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0,I),
$$
so gradients flow through $\mu,\sigma$.

Interpretation:

* First term: reconstruction (likelihood).
* Second term: regularizer pushing $q_\phi(z \mid x)$ toward prior $p(z)$, enabling sampling and interpolation.

---

### 8.2 GANs (Generative Adversarial Networks)

Two-player minimax game:

* **Generator** $G(z)$ maps noise $z \sim p(z)$ to fake samples.
* **Discriminator** $D(x)$ outputs probability of “real vs fake”.

Original GAN objective:
$$
\min_G \max_D \ \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)]
- \mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))].
$$

Generator often trained with non-saturating loss:
$$
\min_G \ -\mathbb{E}_{z}[\log D(G(z))].
$$

Problems:

* **Mode collapse**: $G$ produces limited diversity.
* Training instability, gradient issues.

#### 8.2.1 WGAN and WGAN-GP

Wasserstein GAN changes objective to approximate Earth Mover (Wasserstein-1) distance:

$$
W(p_{\text{data}}, p_G) \approx \max_{f \in \mathcal{F}_{\text{1-Lip}}}
\mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_G}[f(x)].
$$

$D$ becomes a **critic** $f$ with Lipschitz constraint. WGAN-GP enforces it with gradient penalty:

$$
\lambda \mathbb{E}_{\hat{x}} \big[(|\nabla_{\hat{x}} f(\hat{x})|_2 - 1)^2\big].
$$

Benefits:

* More stable training.
* Meaningful loss correlated with sample quality.

#### 8.2.2 StyleGAN

Key ideas:

* **Style mapping network**: map latent $z$ to style $w$.
* Modulate each conv layer with $w$.
* Separate content and style; multi-scale control over features.

Gives state-of-the-art image quality and control; widely used for face synthesis.

#### 8.2.3 DiffAugment

Data augmentation applied to both real and fake images **inside** the discriminator pipeline, so $D$ cannot trivially detect augment artifacts. Improves GAN performance, especially in low-data regimes.

---

### 8.3 Diffusion Models

Now the dominant class for high-quality image generation.

#### 8.3.1 Forward (Diffusion) Process

Define a Markov chain that gradually adds Gaussian noise:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t} x_{t-1},\ \beta_t I\right),
$$

with small $\beta_t$. Closed form:

$$
q(x_t \mid x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t} x_0,\ (1-\bar{\alpha}_t)I\right),
\quad \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s).
$$

As $t \to T$, $x_T$ approaches pure noise.

#### 8.3.2 Reverse (Denoising) Process

Goal: learn reverse transitions:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(\mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)\big),
$$

such that sampling $x_T \sim \mathcal{N}(0,I)$ and iteratively denoising yields $x_0 \sim p_{\text{data}}$.

Training objective can be simplified to predicting the noise $\epsilon$ that was added at each step:

* Sample $t$, $x_0$, and $\epsilon \sim \mathcal{N}(0,I)$.
* Form:
  $$
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon.
  $$
* Train $\epsilon_\theta(x_t, t)$ with:
  $$
  L(\theta) = \mathbb{E}_{t,x_0,\epsilon} |\epsilon - \epsilon_\theta(x_t, t)|^2.
  $$

This is the DDPM (Denoising Diffusion Probabilistic Model) formulation.

#### 8.3.3 Classifier-Free Guidance

For conditional generation (e.g., text prompt $c$):

* Train one model that can operate both **with** and **without** condition (drop $c$ with some probability).
* At inference, combine conditional and unconditional predictions:

  * Get $\epsilon_\theta(x_t, t, c)$ and $\epsilon_\theta(x_t, t, \varnothing)$,
  * Guided noise:
    $$
    \epsilon_{\text{guided}} = (1 + w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \varnothing)
    $$
    with guidance scale $w$.

This trades off fidelity to condition vs sample diversity.

#### 8.3.4 Latent Diffusion (Stable Diffusion)

Instead of operating in pixel space, operate in **VAE latent space**:

1. Train VAE encoder–decoder: $x \leftrightarrow z$.
2. Run diffusion in $z$-space:

   * Much smaller dimensionality → cheaper.
3. Decode final $z_0$ back to image.

This is the core of Stable Diffusion and many modern image generators.

---

### 8.4 Flow-Based Models

Models with **invertible** transformations and tractable log-likelihood.

Base idea:

* Start with simple base distribution $z_0 \sim p_0(z)$ (e.g., $\mathcal{N}(0,I)$).
* Apply invertible transformations:
  $$
  z_K = f_K \circ \dots \circ f_1(z_0).
  $$

For $x = z_K$, by change of variables:
$$
\log p(x) = \log p_0(z_0) - \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|.
$$

Design $f_k$ so that:

* They are expressive.
* Jacobian determinants are easy to compute.

Examples:

* **RealNVP**: affine coupling layers (half of dimensions transformed conditioned on the rest).
* **Glow**: adds invertible $1 \times 1$ convolutions for channel mixing.

Flows provide exact likelihoods and invertible mapping, but are less competitive in sample quality compared to diffusion/GANs for high-res images.

---

### 8.5 Token-Based Diffusion (for LLMs)

Idea: apply diffusion-style noising and denoising in **discrete token space** (or a relaxed embedding space):

* Noise is applied to token sequences (e.g., random replacement, masking).
* Model learns to reverse the corruption in multiple steps, like a discrete diffusion chain.
* Still an active research area; attempts to bring some of diffusion’s training advantages to language modeling.

---

> To make sure the core conceptual distinctions stick:

Can you write 1–2 sentences explaining how **VAE training** (ELBO) conceptually differs from **next-token MLE training in LLMs**? (Don’t worry about perfect wording; just highlight the main difference in objective/latent structure.)

---

## 9. Vision Architectures (Advanced)

### 9.1 Vision Transformer (ViT)

Core idea: treat an image as a **sequence of patches** and use a Transformer encoder.

1. **Patch embedding**

* Split image $x\in\mathbb{R}^{H\times W\times C}$ into non-overlapping patches of size $P\times P$.
* Each patch is flattened and linearly projected:
  $$
  z_0^i=W_E\,\text{vec}(\text{patch}_i)+b_E
  $$
* Prepend a learned CLS token $z_0^\text{[CLS]}$ for classification, add position embeddings.

2. **Transformer encoder**

* Apply $L$ layers of self-attention + FFN (as in Section 6).
* Final CLS representation $z_L^\text{[CLS]}$ goes to linear head for classification.

3. **Characteristics**

* Minimal inductive bias compared to CNNs (no convolution, no explicit locality).
* Needs strong regularization and lots of data (or good pretraining); DeiT shows it can be data-efficient with distillation and augmentations.

---

### 9.2 CLIP (Contrastive Pretraining)

Goal: learn a **joint embedding space** for images and text.

1. Two encoders:

   * Image encoder $f_\theta(x_\text{img})$ (CNN or ViT).
   * Text encoder $g_\phi(x_\text{text})$ (Transformer).

2. Contrastive loss (InfoNCE style):

For a batch of $N$ image–caption pairs $(x_i,c_i)$, let:
$$
u_i=f_\theta(x_i),\quad v_i=g_\phi(c_i)
$$
Normalize to unit length; logits via cosine similarity:
$$
\ell_{ij}=\frac{u_i^\top v_j}{\tau}
$$
Loss encourages matched pairs to be closer than mismatched ones:
$$
L=\frac12\big(L_{\text{img}\to\text{text}}+L_{\text{text}\to\text{img}}\big),
$$
with each term a cross-entropy over the softmax of $\ell_{ij}$.

3. Result: zero-shot classifier

* For class names or templates (“a photo of a {label}”), encode prompts and classify images by cosine similarity to text embeddings.

---

### 9.3 MAE (Masked Autoencoder for Vision)

Self-supervised **masked patch reconstruction**:

1. Randomly mask a high fraction (e.g., 75%) of image patches.
2. Encode visible patches with a ViT encoder.
3. Decode to reconstruct the full image (or pixel/feature targets).

Objective:
$$
L=\mathbb{E}\left[|x_{\text{masked}}-\hat{x}_{\text{masked}}|^2\right]
$$

Key ideas:

* Asymmetric: encoder sees few tokens; decoder is lightweight.
* Masking forces encoder to learn semantic, holistic structure.

---

### 9.4 SAM (Segment Anything Model)

High-level idea: a **general segmentation model** with prompts:

* Image encoder (ViT-like) produces dense feature map.
* Lightweight prompt encoder for points/boxes/text.
* Mask decoder combines image features + prompt embedding to predict segmentation masks.

Key concept: segmentation as “promptable” task, using foundation vision encoder pre-trained on huge mask dataset.

---

### 9.5 DETR and YOLO Family

#### 9.5.1 DETR (DEtection TRansformer)

* Treat detection as **set prediction**.
* Backbone (CNN or ViT) → feature map → Transformer encoder–decoder.
* Decoder has a fixed number of **object queries** (learned embeddings) that attend to encoder features and output bounding boxes + class labels.
* Loss is bipartite matching (Hungarian) between predicted boxes and ground-truth (set-based, permutation-invariant).

Advantages:

* Simpler pipeline (no anchors, NMS integrated into training).
  Challenges:
* Slow convergence; later variants (Deformable DETR, DAB, etc.) improve that.

#### 9.5.2 YOLO family

* Single-stage detectors: predict boxes and classes directly from feature maps.
* Modern versions (YOLOv5/7/8, RT-DETR-like) use:

  * Feature pyramids.
  * Anchor-free heads or refined anchor schemes.
  * Strong augmentations and optimized architectures.

You mainly need to remember: YOLO = fast, single-shot detection; DETR = transformer-based set prediction.

---

### 9.6 NeRF and Diffusion for Vision

1. **NeRF (Neural Radiance Fields)**

* Represent a 3D scene as a function:
  $$
  F_\theta:(\mathbf{x},\mathbf{d})\mapsto(\sigma,\mathbf{c})
  $$
  where $\mathbf{x}$ is 3D position, $\mathbf{d}$ is viewing direction, $\sigma$ is density, $\mathbf{c}$ is color.
* Render images by volume rendering along rays; train to match posed images.

2. **Diffusion for Vision**

* Apply diffusion to images or latents (as in Section 8.3).
* Control with text prompts (via cross-attention to text embeddings).
* Many vision tasks can be cast as conditional diffusion (inpainting, editing, super-resolution, etc.).

---

## 10. Multimodal Deep Learning

### 10.1 Vision–Language Models

#### 10.1.1 CLIP (revisited)

* Already covered; it’s the canonical image–text embedding model.

#### 10.1.2 BLIP, BLIP-2, Flamingo

* **BLIP**: unified vision–language pretraining with captioning + ITM (image-text matching) + contrastive losses.

* **BLIP-2**:

  * Freezes a visual encoder, learns a small Q-Former (query Transformer) that maps vision features to a compact set of tokens.
  * These tokens are fed into a frozen LLM via a small projection layer → efficient multimodal adaptation.

* **Flamingo**:

  * Perception backbone (vision encoder).
  * Perceiver-like cross-attention to produce visual tokens.
  * Insert those into an LLM with cross-attention; supports interleaved image–text sequences.

Core pattern: re-use a strong unimodal model (CLIP or ViT + LLM) and connect them with a light bridging module.

---

### 10.2 Large Multimodal Models (LMMs)

Examples: GPT-4o, Gemini.

* Use a **vision encoder** to produce patch embeddings.
* Map visual embeddings into LLM token space (via linear projection or adapter).
* The LLL then processes tokens from both text and image modalities, using standard self-attention.

Important notions:

* **Cross-attention between modalities**: text tokens can attend to visual tokens and vice versa.
* **Instruction tuning**: multimodal instruction data (e.g., “describe this plot”, “read this screenshot”) aligns the model to useful behaviors.

---

### 10.3 Image Captioning and VQA

* Encoder–decoder or encoder-only + decoding head.
* Typically:

  * Vision encoder → global or region-level features.
  * Text decoder attends to those features while generating captions or answering questions.

The architecture is often just “image → embeddings → text decoder with cross-attention”; differences lie in data and loss (captioning vs QA vs reasoning).

---

### 10.4 Video Transformers and Speech

1. **Video**

* Treat video frames as a spatiotemporal token sequence:

  * 3D patches (time × height × width) → tokens.
  * Use axial or factorized attention across space and time, or more efficient local windows.
* Applications: action recognition, video captioning, temporal localization.

2. **Speech**

* Models like wav2vec 2.0:

  * Self-supervised on raw waveforms or spectrograms via masking/contrastive tasks.
  * Then fine-tuned for ASR or other tasks.
* Whisper: encoder–decoder Transformer for multi-language ASR + translation, trained on huge transcribed audio dataset.

---

## 11. Reinforcement Learning (DL-heavy Parts)

Here we only focus on the pieces that overlap with deep learning architectures (detailed RL theory is in your separate RL sheet).

### 11.1 Policy Gradient and Actor–Critic

1. **Policy gradient**

For policy $\pi_\theta(a\mid s)$ and return $R_t$:
$$
\nabla_\theta J(\theta)=\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t),R_t\big].
$$

Use Monte Carlo or bootstrapped estimates of $R_t$; high variance, improved by baselines.

2. **Actor–critic**

* **Actor**: policy network $\pi_\theta$ (often a deep network).
* **Critic**: value network $V_\phi(s)$ or $Q_\phi(s,a)$.

Advantage:
$$
A_t=R_t-V_\phi(s_t).
$$

Actor update uses:
$$
\nabla_\theta J(\theta)\approx\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t\mid s_t),A_t].
$$

Critic trained by regression on value targets (TD learning).

---

### 11.2 PPO (Proximal Policy Optimization)

Widely used in RLHF.

1. **Clipped surrogate objective**

Let $r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$. PPO optimizes:

$$
L^\text{CLIP}(\theta)=\mathbb{E}_t\big[
\min\big(
r_t(\theta)A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\big)
\big].
$$

Interpretation:

* If $r_t$ moves too far from 1 (policy changed too much), the clipped term limits improvement.
* Acts like a trust-region method but simpler than TRPO.

2. **Implementation**

* Policy and value networks often share base, separate heads.
* Loss combines policy, value, and entropy terms.

---

### 11.3 Q-Learning and DQN

1. **Q-learning (tabular)**

Update rule:
$$
Q(s,a)\leftarrow Q(s,a)+\alpha\big(r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big).
$$

2. **Deep Q Network (DQN)**

Approximate $Q(s,a)$ with a neural network $Q_\theta(s,a)$.

Key tricks:

* **Target network**: $Q_{\theta^-}$ lagging copy of $Q_\theta$ for stable bootstrapping.
* **Experience replay**: sample mini-batches from replay buffer to break temporal correlations.
* Loss:
  $$
  L(\theta)=\mathbb{E}_{(s,a,r,s')}\big[(y-Q_\theta(s,a))^2\big],
  $$
  where $y=r+\gamma\max_{a'}Q_{\theta^-}(s',a')$ (or variants).

---

### 11.4 AlphaZero and MuZero Architectures (High Level)

1. **AlphaZero**

* Uses a deep residual network that outputs:

  * Policy logits $\pi_\theta(a\mid s)$.
  * Value estimate $v_\theta(s)$.
* Integrates with Monte Carlo Tree Search (MCTS):

  * Network guides search; search produces improved policy targets.
  * Supervised training on MCTS outputs.

2. **MuZero**

* Learns a **model** of environment dynamics in a learned latent space:

  * Representation network: $h_0=f_\theta(s)$
  * Dynamics network: $h_{k+1},r_{k+1}=g_\theta(h_k,a_k)$
  * Prediction network: value + policy from $h_k$.

No need for known environment dynamics; everything is learned. MCTS operates on latent states.

---

### 11.5 World Models and Offline RL

1. **World models (Dreamer style)**

* Learn a latent dynamics model:

  * Encoder: $z_t=e(x_t)$
  * Recurrent latent dynamics: $z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t)$
  * Decoder: reconstruct observations; also predict rewards.

* Train policy inside the learned world (imagination rollouts), then deploy in real environment.

* Decouples environment sampling from policy improvement.

2. **Offline RL**

* Learn policy from fixed dataset (no interaction).
* Deep networks approximate $Q$ or policy, but must handle **distributional shift**: learned policy visits states not present in dataset.

Approaches:

* Conservative Q-learning.
* Implicit behavior regularization.
* Policy constraints to stay close to behavior policy.

---

> **To keep this feeling like two-way learning:**
if you had to very briefly explain to someone how **PPO differs from simple policy gradient**, what’s the one main idea you’d mention (in plain words, not equations)?

---

## 12. Self-Supervised & Contrastive Learning

### 12.1 Contrastive Learning and InfoNCE

Goal: learn representations where **positive pairs** are close, **negatives** are far.

Given encoder $f_\theta$, for a batch of $N$ pairs $(x_i, x_i^+)$ (two views of same image), define:

* Embeddings: $z_i = f_\theta(x_i)$, $z_i^+ = f_\theta(x_i^+)$, normalized.
* Similarity: $s_{ij} = \frac{z_i^\top z_j}{\tau}$ (cosine with temperature $\tau$).

InfoNCE loss (SimCLR style) for sample $i$:

$$
L_i = -\log \frac{\exp(s_{i,i^+})}{\sum_{k \neq i} \exp(s_{i,k})}.
$$

Interpretation:

* Treats other samples in batch as negatives.
* Maximizes mutual information between views in representation space (loosely).

---

### 12.2 SimCLR and MoCo

1. **SimCLR**

* Two strong augmentations per image → views $(x_i, x_i^+)$.
* Encoder + MLP projection head.
* Large batch sizes give many negatives.
* No memory bank; everything is from current batch.

2. **MoCo**

* Momentum encoder for keys: $f_\text{key}$ parameters updated as EMA of query encoder.
* Maintain a **queue** (memory bank) of negative keys.
* Contrast each query with current positive + large set of stored negatives.
* Efficient for smaller batch sizes.

---

### 12.3 BYOL, SwAV, DINO

These move away from explicit negatives.

1. **BYOL**

* Online network and target network (EMA of online).
* Objective: predict target representation from online view.
* Surprisingly, no explicit negatives; collapse is avoided via architecture asymmetries and EMA.

2. **SwAV**

* Clustered contrastive learning.
* Assign samples to prototype vectors online via Sinkhorn-Knopp (balanced assignments).
* Different views of same image share cluster assignments.

3. **DINO / DINOv2**

* Student–teacher self-distillation.
* Teacher is EMA of student.
* Student matches teacher’s soft assignments over prototypes for different augmentations.
* Works extremely well as a general-purpose vision backbone.

---

### 12.4 Self-Distillation and Masked Autoencoding

* **Self-distillation**: model is trained to match its own (or EMA’s) predictions under perturbations.
* **Masked autoencoding (MAE / BERT for images)**:

  * Randomly mask tokens/patches.
  * Predict them from visible context.
  * Forces model to capture structure and semantics (similar to BERT in NLP).

---

## 13. Deep Learning Theory (High-Level Intuitions)

### 13.1 Universal Approximation and Overparameterization

* **Universal approx. theorem**: a single hidden layer MLP with non-polynomial activation can approximate any continuous function on a compact domain.
* Overparameterized deep nets: number of parameters $\gg$ number of data points; yet they generalize well.

Key points:

* Optimization becomes easier: many global minima with low training loss.
* Implicit bias of SGD (and architecture) selects “simple” solutions among many.

---

### 13.2 Double Descent and Sharp vs Flat Minima

1. **Double descent**

* Classical regime: test error is U-shaped vs model capacity.
* In modern deep nets, as you cross interpolation threshold (zero training error), test error can initially go up then **down again** with increasing capacity → “double descent.”

2. **Sharp vs flat minima**

* Flat minima: loss changes slowly with parameter perturbations.
* Sharp minima: loss increases quickly around minimum.
* Empirical correlation: flat minima → better generalization.
* Small-batch, noisy SGD tends to prefer flatter basins.

---

### 13.3 Lottery Ticket Hypothesis (revisited) and NTK

* LTH: large random networks contain sparse subnetworks that train well from initial weights.
* **Neural tangent kernel (NTK)**:

  * In infinite-width limit with small initialization, training dynamics linearize around initialization.
  * Network behaves like kernel regression with a fixed kernel (NTK).
  * Explains some phenomena for very wide nets, but realistic nets often leave the pure NTK regime (feature learning matters).

---

### 13.4 Depth vs Width and Inductive Biases

* Depth allows compositional hierarchies; some functions are exponentially more efficient to represent with depth.
* Width can approximate many functions but might require exponentially more units.
* Inductive biases from architecture:

  * CNNs: locality, translation equivariance.
  * Transformers: permutation-invariance plus positional encoding.
  * RNNs: sequential recurrences.

These biases strongly influence sample efficiency and generalization.

---

## 14. Deep Learning Systems

### 14.1 Distributed Training

You mainly need the **concepts** and when to use each.

1. **Data Parallelism**

* Replicate model on each device.
* Each processes a different mini-batch shard.
* Gradients averaged (all-reduce).
* Simple and widely used; limited by model size fitting on a single device.

2. **Model Parallelism**

* **Tensor parallelism**: split large tensors across devices (e.g., split weight matrices column-wise or row-wise).
* **Pipeline parallelism**: split layers into stages, pass micro-batches through as a pipeline.
* Trade computation vs communication; used for huge LLMs.

3. **FSDP / ZeRO**

* Shard parameters, gradients, optimizer states across data-parallel ranks.
* ZeRO stages:

  * Stage 1: shard optimizer states.
  * Stage 2: shard gradients.
  * Stage 3: shard parameters.
* FSDP automatically handles full sharding and all-gather/scatter around forward/backward.

Goal: fit very large models by spreading memory across many devices.

---

### 14.2 Serving and Inference

1. **Batch vs Online Decoding**

* **Batch inference**: many requests together, maximize throughput (e.g., offline scoring).
* **Online decoding**: interactive user requests; care about latency and jitter.

2. **Token Streaming and KV Cache**

* Stream tokens as they are generated to reduce perceived latency.
* KV cache used to avoid recomputation (Section 7).

3. **Model Parallel Inference**

* Same ideas as training, but tuned for low-latency serving.
* Often different parallelism layouts for training vs inference (e.g., tensor parallel at inference, pipeline at training).

4. **Sharded Serving (vLLM, TensorRT-LLM, FasterTransformer)**

* Efficient runtime that:

  * Packs sequences into memory-friendly layouts.
  * Manages KV cache sharing and paging.
  * Uses fused CUDA kernels and graph optimizations.

---

### 14.3 Quantization: Static, Dynamic, QAT

1. **Static quantization**

* Calibrate activation ranges on a calibration set.
* Quantize weights and activations to fixed scales/zero-points.
* Good for deployment when you can run calibration offline.

2. **Dynamic quantization**

* Weights quantized ahead of time; activations quantized on the fly per batch.
* Often used for CPU inference (e.g., linear layers).

3. **Quantization-Aware Training (QAT)**

* Simulate quantization during training (via fake quantization nodes).
* Model learns to be robust to quantization error.
* Best performance when aggressive (e.g., 4-bit) quantization is needed.

---

### 14.4 Compilers and Kernels

1. **Graph capture and fusion**

* TorchDynamo, AOTAutograd, XLA, TVM, ONNX Runtime capture computation graphs.
* They fuse operations, optimize memory, and generate efficient backend code.

2. **Custom kernels**

* CUDA or Triton kernels for specialized ops (attention, normalization, etc.).
* Key for squeezing out performance (FlashAttention, fused MLPs, etc.).

The interviewer usually cares that you can **reason** about when to use distributed training, KV caching, quantization, and compilation to meet latency/throughput constraints.

---

## 15. Safety, Alignment, and Responsible AI (LLM-Focused)

### 15.1 RLHF and Constitutional AI

* **RLHF**: already covered; align models with human preferences via reward models and RL.
* **Constitutional AI**:

  * Use a set of “constitutional” principles (e.g., safety, helpfulness) to generate preference data or critiques.
  * Model learns to self-criticize and revise outputs according to the principles, reducing human labeling.

---

### 15.2 Jailbreak Robustness and Toxicity Filters

* Jailbreaks: prompts that circumvent safety guardrails.
* Techniques:

  * Prompting and system instructions.
  * Fine-tuning on adversarial examples.
  * Post-hoc safety filters (classifiers) over model outputs and inputs.

Toxicity filters:

* Separate classifier that estimates toxicity / hate / self-harm risk.
* Can be used to block, rephrase, or ask for clarification.

---

### 15.3 Preference Modeling and Evaluation

* Align models to **multiple axes**: helpfulness, harmlessness, honesty, etc.
* Preference data: pairwise comparisons, scalar ratings, rubric-based eval.
* Automatic evaluation: LLM-as-judge, calibrated with human evaluation.

---

## 16. Evaluation Metrics (DL-Specific)

### 16.1 Sequence and Text Metrics

1. **BLEU**

* $n$-gram precision with a brevity penalty.
* Designed for machine translation.
* Favors exact matches; can be brittle.

2. **ROUGE**

* Recall-oriented; counts overlapping $n$-grams between candidate and reference.
* Used for summarization.

3. **METEOR**

* Considers synonyms and stemming.
* More semantically aware than pure n-gram counts, but less used now.

4. **Perplexity**

For language models, perplexity on a dataset:
$$
\text{PPL} = \exp\left( - \frac{1}{N} \sum_{t=1}^N \log p_\theta(x_t \mid x_{<t}) \right).
$$

* Lower is better.
* Equivalent to exponentiated average negative log-likelihood.

---

### 16.2 Vision and Generative Metrics

1. **FID (Fréchet Inception Distance)**

* Model real and generated features (from an Inception net) as Gaussians.
* Compute Fréchet distance:
$$
\text{FID} = |\mu_r - \mu_g|_2^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big).
$$
* Lower is better; measures distributional similarity.

2. **CLIP Score**

* CLIP similarity between image and caption.
* Used as proxy for text–image alignment and quality.

3. **WER/CER for speech**

* Word Error Rate / Character Error Rate:
$$
\text{WER} = \frac{S + D + I}{N},
$$
where $S$=substitutions, $D$=deletions, $I$=insertions, $N$=reference words.

---

### 16.3 Ranking Metrics

* **nDCG (normalized Discounted Cumulative Gain)**:

For ranked list with relevance labels $rel_i$, DCG@k:
$$
\text{DCG@k} = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i+1)}.
$$
Normalize by ideal DCG (sorted by relevance):
$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}.
$$

Used extensively in neural ranking, recsys, search.

---

## 17. Modern Research Trends (High-Level Map)

### 17.1 Retrieval-Augmented Models

* **RAG**: retrieve documents conditioned on query, feed as context to LLM.
* **Fusion-in-Decoder**: multiple retrieved docs passed into seq2seq decoder with cross-attention.
* **RETRO**: retrieval during pretraining; model conditions on retrieved neighbors.

Key idea: scale **knowledge** via retrieval instead of parameters.

---

### 17.2 Long-Sequence and Memory Models

* **Transformer-XL**: segment-level recurrence + relative positions.
* **Reformer**: LSH attention for sub-quadratic complexity.
* **Performer**: kernel-based linear attention.
* **Longformer, BigBird**: sparse attention patterns (local + global tokens).
* **State-space models (S4, S5, Mamba)**: continuous-time / sequence models with linear-time complexity and long-range memory.

Goal: handle very long context (10^4–10^6 tokens) efficiently.

---

### 17.3 Recurrent GPT / RWKV-Style and World Models for Reasoning

* **RWKV**: blends RNN-like recurrence with attention-style behavior; aims for linear-time inference and streaming.
* **World models for reasoning**:

  * Build internal latent “planning” or “simulation” layer (lookahead models).
  * Use LLMs or neural dynamics to imagine consequences, then act/answer.

Still an evolving area; core idea is **explicit modeling of future or hidden structure** rather than pure pattern completion.

---

### 17.4 Diffusion Transformers, MoE, Tool Use, Agents

1. **Diffusion Transformers (DiTs)**

* Replace UNet with pure Transformer backbone for diffusion.
* Better scaling properties and flexible conditioning.

2. **Mixture-of-Experts (MoE) routing**

* Already discussed; actively explored for compute-efficient scaling.
* Routing quality and stability are central research topics.

3. **Tool use and agents**

* LLMs augmented with tools: retrieval, code execution, calculators, external APIs.
* Agent frameworks:

  * Plan → call tools → reflect → act loops.
  * Memory, planning, and environment interaction are active research directions.

---

> To close the loop and make this “stick,” here’s a tiny self-check:

If an interviewer asks you, **“Why do people use retrieval-augmented generation (RAG) instead of just making an even bigger LLM?”**, what 2–3 concise points would you give?

