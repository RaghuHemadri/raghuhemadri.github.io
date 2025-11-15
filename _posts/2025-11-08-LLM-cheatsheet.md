---
layout: post
title: "LLM Interview: Fundamentals and Implementation"
date: 2025-11-08 09:00:00
description: 
tags: genai generative-ai ml
categories: ml
thumbnail: assets/img/llm_cheatsheet.png
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Contents
- [LLM Architectures, Training and Inference in PyTorch](#llm-architectures-training-and-inference-in-pytorch)
- [1 LLM Tokenizers: concept + code](#1-llm-tokenizers-concept--code)
- [2 Self-attention: concept + code](#2-self-attention-concept--code)
- [3 Transformer: concept + code](#3-transformer-concept--code)
- [4 LLM Decoder (response sampling): the main methods](#4-llm-decoder-response-sampling-the-main-methods)
- [5 KV Cache: concept + code](#5-kv-cache-concept--code)
- [6 Positional embeddings: concept (math-first, no code)](#6-positional-embeddings-concept-math-first-no-code)
- [7 PEFT (Parameter-Efficient Fine-Tuning): concept + code](#7-peft-parameter-efficient-fine-tuning-concept--code)
- [8 Quantization: concept + code](#8-quantization-concept--code)
- [9 RLHF \& DPO: concept + code](#9-rlhf--dpo-concept--code)
- [10 GRPO — Group Relative (Policy) Optimization](#10-grpo--group-relative-policy-optimization)
- [11 Speculative decoding (mechanics, math, guarantees)](#11-speculative-decoding-mechanics-math-guarantees)
- [12 Word embeddings (with tiny PyTorch snippets)](#12-word-embeddings-with-tiny-pytorch-snippets)
- [13 LLM Context Length Extension (how to make models read longer)](#13-llm-context-length-extension-how-to-make-models-read-longer)
- [14 FlashAttention (and a tiny PyTorch sketch)](#14-flashattention-and-a-tiny-pytorch-sketch)
- [15 Multi-Query Attention (MQA) + tiny PyTorch](#15-multi-query-attention-mqa--tiny-pytorch)
- [16 Rotary Positional Embedding (RoPE) + tiny PyTorch](#16-rotary-positional-embedding-rope--tiny-pytorch)
- [17 Information Retrieval (IR) fundamentals](#17-information-retrieval-ir-fundamentals)
- [18 Indexing (IR) + tiny PyTorch](#18-indexing-ir--tiny-pytorch)
- [19 Query Understanding (IR)](#19-query-understanding-ir)
- [20 Ranking \& Re-Ranking (IR)](#20-ranking--re-ranking-ir)
- [21 A/B experimentation \& testing in **LLMs**](#21-ab-experimentation--testing-in-llms)
- [22 LLM training regimes (the landscape)](#22-llm-training-regimes-the-landscape)
- [23 Data preprocessing for LLMs (what actually matters)](#23-data-preprocessing-for-llms-what-actually-matters)
- [24 Losses you’ll actually use (and why)](#24-losses-youll-actually-use-and-why)
- [25 Evaluation metrics (make numbers honest)](#25-evaluation-metrics-make-numbers-honest)
- [26 Putting it together (a minimal training recipe)](#26-putting-it-together-a-minimal-training-recipe)
- [27 Distributed Training for ML/LLMs](#27-distributed-training-for-mlllms)
- [28 ML/LLM **Inference Optimizations**](#28-mlllm-inference-optimizations)


---

# LLM Architectures, Training and Inference in PyTorch

Repository: [LLM Architectures, Training and Inference in PyTorch](https://github.com/RaghuHemadri/LLMs-From-Scratch.git)

Quick preview:
- A from-scratch PyTorch walk-through covering tokenizers, attention, Transformer blocks, training recipes (pretrain / SFT / PEFT), decoding strategies, KV cache patterns, RoPE/positional schemes, and inference optimizations.
- Includes didactic toy implementations, practical tips, and runnable snippets for experiments.
- Good entry points: README, notebooks / demos, and minimal model examples for hands-on testing.

Notes:
- Intended as an educational, implementation-focused companion to the markdown cheatsheet.
- Use the repo examples for experiment scaffolding and adapt tokenizer/model names when switching to HF models or quantized runtimes.
- Check license/usage in the repo before reuse.

# 1 LLM Tokenizers: concept + code

## Big picture

A tokenizer maps text → integer ids. Modern LLMs use **subword** tokenizers that:

* avoid OOV by composing unseen words from pieces,
* strike a balance between character- and word-level units,
* keep vocabulary small enough for fast softmax / embedding tables.

**Pipeline = normalization → pre-tokenization → model (BPE/WordPiece/Unigram) → post-processing (special tokens).**

---

## A) Byte-Pair Encoding (BPE)

**Idea.** Start from characters (often raw **bytes** to guarantee coverage). Repeatedly **merge the most frequent adjacent pair** to create a new symbol. Greedy, frequency-based.

**Training (greedy merges)**

1. Initialize vocab with all single bytes (or characters).
2. Count pair frequencies over corpus (within words).
3. Add the most frequent pair as a new token (merge it).
4. Re-encode corpus with that merge applied.
5. Repeat until you hit vocab size.

**Encoding.** Greedy longest-match using learned merges.

**Pros/Cons.**

* ✅ Very fast to train and use; simple; byte fallback solves OOV.
* ❌ Greedy merges can create odd boundaries; not a probabilistic model.

**Toy BPE (pure python, tiny)**

```python
from collections import Counter, defaultdict

def bpe_train(corpus_words, vocab_size, end_token="</w>"):
    # corpus_words: list[str] (already normalized, split on whitespace)
    words = [list(w) + [end_token] for w in corpus_words]
    vocab = {" ".join(w): freq for w, freq in Counter(tuple(w) for w in words).items()}

    merges = []
    def get_stats(vocab):
        stats = Counter()
        for tokenized, freq in vocab.items():
            symbols = tokenized.split()
            for i in range(len(symbols)-1):
                stats[(symbols[i], symbols[i+1])] += freq
        return stats

    def merge_vocab(pair, vocab):
        a, b = pair
        ab = a + b
        out = {}
        for tokenized, freq in vocab.items():
            symbols = tokenized.split()
            i = 0; new_syms = []
            while i < len(symbols):
                if i < len(symbols)-1 and symbols[i]==a and symbols[i+1]==b:
                    new_syms.append(ab); i += 2
                else:
                    new_syms.append(symbols[i]); i += 1
            out[" ".join(new_syms)] = out.get(" ".join(new_syms), 0) + freq
        return out

    while True:
        stats = get_stats(vocab)
        if not stats or len(stats)+256 >= vocab_size: break
        best = max(stats, key=stats.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
    return merges

def bpe_encode(word, merges, end_token="</w>"):
    symbols = list(word) + [end_token]
    merge_table = {"".join(k): "".join(k) for k in merges}  # not used directly
    for a,b in merges:
        i = 0
        while i < len(symbols)-1:
            if symbols[i]==a and symbols[i+1]==b:
                symbols[i:i+2] = [a+b]
            else:
                i += 1
    return symbols

# demo
corpus = "lower newer lowest wide widely news".split()
merges = bpe_train(corpus, vocab_size=1000)
print(bpe_encode("lowest", merges))
````

---

## B) WordPiece (WP)

**Idea.** Start with characters. Iteratively add the subword that **maximizes corpus likelihood improvement** under a simple language model with independent subword emissions (approx). Greedy but **likelihood-driven**; decoding uses **max-prob segmentation** (Viterbi or greedy longest-probable).

**Training sketch.**

* At each step, consider candidate joins $(x \oplus y)$ and pick the one maximizing $(\Delta \log P(\text{data}))$. Approximations make this tractable (counts + smoothing).
* Subwords often use a continuation marker (e.g., `##ing`) to mark non-initial pieces.

**Pros/Cons.**

* ✅ Tends to select linguistically useful units; robust across domains.
* ❌ More bookkeeping than BPE; still greedy and approximate.

**Toy WordPiece-like encoder (Viterbi over a fixed vocab)**

```python
import math

def viterbi_wp(text, vocab, logp):
    # vocab: set of subwords; logp: dict[subword]->log prob (negative)
    n = len(text)
    dp = [math.inf]*(n+1); back = [-1]*(n+1); token = [None]*(n+1)
    dp[0] = 0.0
    for i in range(n):
        if dp[i] == math.inf: continue
        for j in range(i+1, n+1):
            sub = text[i:j] if i==0 else "##"+text[i:j]
            if sub in vocab:
                cost = dp[i] + (-logp[sub])
                if cost < dp[j]:
                    dp[j] = cost; back[j] = i; token[j] = sub
    if dp[n] == math.inf: return None
    out = []
    i = n
    while i>0:
        out.append(token[i]); i = back[i]
    return list(reversed(out))

# Example tiny vocab and log-probs
vocab = {"t", "he", "the", "##re", "##refore", "##for", "a"}
logp  = {w: -math.log(1+len(w)) for w in vocab}  # toy scores
print(viterbi_wp("therefore", vocab, logp))
```

*(Real training learns `logp` from counts; this snippet just shows the WP **decoding** idea.)*

---

## C) Unigram Language Model (SentencePiece)

**Idea.** Maintain a **candidate inventory** of subwords (often seeded from many BPE/WP candidates). Assume each word is generated by a **mixture over segmentations** with subword probabilities $p(t)$. Use **EM** to (a) estimate $p(t)$ and (b) **prune** low-utility tokens to shrink the vocab.

**Model.**
For a word $w$, all segmentations $S(w)$ have prob
$$
P(w) = \sum_{s \in S(w)} \prod_{t \in s} p(t), \quad \sum_t p(t) = 1.
$$

EM:

* **E-step:** compute posterior usage of each token $t$ via forward-backward over each word.
* **M-step:** update $p(t) \propto \text{expected count}(t)$; prune a fraction with lowest loss impact.
* Iterate until target vocab size.

**Pros/Cons.**

* ✅ Globally probabilistic; yields smooth segmentations; robust domain transfer.
* ❌ Slightly heavier training; decoding still uses Viterbi/forward-backward.

**Toy Unigram LM (single EM step, minimal)**

```python
import math
from collections import defaultdict

def unigram_em_step(words, vocab_probs):
    # vocab_probs: dict[token] -> prob (sum to 1), tokens include continuation markers or not (SentencePiece avoids them)
    def forward_scores(w):
        n = len(w); f = [0.0]+[0.0]*n
        for i in range(1, n+1):
            s = 0.0
            for j in range(max(0, i-10), i):  # max token length = 10 for speed
                tok = w[j:i]
                if tok in vocab_probs:
                    s += f[j]*vocab_probs[tok]
            f[i] = s
        return f

    expected_counts = defaultdict(float)
    total_ll = 0.0
    for w in words:
        f = forward_scores(w)
        Z = f[len(w)] + 1e-12
        total_ll += math.log(Z)
        # backward to collect expected counts (inside-outside)
        n = len(w)
        b = [0.0]*(n+1); b[n] = 1.0
        for i in range(n-1, -1, -1):
            s = 0.0
            for j in range(i+1, min(n, i+10)+1):
                tok = w[i:j]
                if tok in vocab_probs:
                    s += vocab_probs[tok]*b[j]
            b[i] = s
        for i in range(n):
            for j in range(i+1, min(n, i+10)+1):
                tok = w[i:j]
                if tok in vocab_probs:
                    # posterior for this token occurrence
                    contrib = (f[i]*vocab_probs[tok]*b[j]) / Z
                    expected_counts[tok] += contrib
    # M-step (normalize)
    s = sum(expected_counts.values()) + 1e-12
    new_probs = {t: c/s for t,c in expected_counts.items()}
    return new_probs, total_ll

# demo (ridiculously tiny)
words = ["therefore", "there", "the", "a"]
init_vocab = {c: 1.0 for c in set("therefora")}  # char-level init
Z = sum(init_vocab.values()); vocab_probs = {k:v/Z for k,v in init_vocab.items()}
for _ in range(3):
    vocab_probs, ll = unigram_em_step(words, vocab_probs)
print(sorted(list(vocab_probs.items()))[:5])
```

*(Real SentencePiece also learns a larger candidate set and prunes low-contribution tokens between EM rounds.)*

---

## Practical bits you’ll be asked in interviews

* **Normalization:** Unicode NFKC, lowercasing, accent stripping (depends on model). Byte-level BPE often uses **no normalization** beyond UTF-8 bytes.
* **Pre-tokenization:** Split on whitespace/punct or treat as raw bytes; SentencePiece commonly avoids language-specific rules (language-agnostic).
* **Special tokens:** `<pad> <s> </s> <unk> <mask> <bos> <eos>` and task-specific sentinels.
* **Byte-level fallback:** Guarantees any input is encodable; common in GPT-style BPE (Radford).
* **Vocab size trade-off:** Larger vocab → shorter sequences but bigger embedding/softmax; smaller vocab → longer sequences but better sharing.
* **Evaluation:** tokenization speed, compression ratio (avg tokens/char), downstream perplexity or validation loss with the same model.

---

## Using Hugging Face quickly

```python
# pip install tokenizers
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# BPE
bpe_tok = Tokenizer(models.BPE(unk_token="[UNK]"))
bpe_tok.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=32000, special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])
bpe_tok.train(files=["corpus.txt"], trainer=trainer)

# WordPiece
wp_tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
wp_tok.pre_tokenizer = pre_tokenizers.Whitespace()
wp_trainer = trainers.WordPieceTrainer(vocab_size=30522, special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"])
wp_tok.train(files=["corpus.txt"], trainer=wp_trainer)

# Unigram (SentencePiece-like)
uni_tok = Tokenizer(models.Unigram())
uni_tok.pre_tokenizer = pre_tokenizers.Metaspace()  # SP-style space handling
uni_trainer = trainers.UnigramTrainer(vocab_size=32000, special_tokens=["<unk>","<s>","</s>"])
uni_tok.train(files=["corpus.txt"], trainer=uni_trainer)
```

---

## When to pick what

* **BPE (byte-level):** most GPT decoders; robust to arbitrary text; fastest to train; great default.
* **WordPiece:** compatible with BERT heritage; good morphological splits; common in encoder models.
* **Unigram LM:** strong cross-lingual performance; smoother segmentations; widely used in T5/ALBERT/LLAMA-(SP flavored) ecosystems.

---

> **Quick check (your turn, one sentence):**
If you were building a multilingual decoder-only LLM for noisy web text, which tokenizer would you start with and why?

---

# 2 Self-attention: concept + code

**What it is.**  
Each token turns into a **query** $q_i$, **key** $k_j$, **value** $v_j$. Token $i$ gathers info from all tokens $j$ via weights $\alpha_{ij}$ (similarity of $q_i$ and $k_j$):

$$
\alpha_{ij} = \text{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}} + m_{ij}\right),\quad
\text{Attn}(i) = \sum_j \alpha_{ij} v_j
$$

* $m_{ij}=0$ if allowed, $-\infty$ if masked (padding/causal).  
* **Why divide by $\sqrt{d_k}$?** To keep logits’ variance stable as $d_k$ grows (prevents softmax from peaking too hard).

**Multi-head:** project inputs to $H$ subspaces and run attention in parallel, then concat + project back. Heads let the model capture different relations (syntax, coref, position, etc.).

---

## Minimal, reusable PyTorch modules

```python
import torch, torch.nn as nn, torch.nn.functional as F

def make_padding_mask(lengths, T, device):
    # lengths: [B] valid lengths; True where we should MASK
    idx = torch.arange(T, device=device)[None, :].expand(len(lengths), T)
    return (idx >= lengths[:, None])  # [B, T]

def make_causal_mask(T, device):
    # True above the diagonal (to be masked)
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** -0.5
    def forward(self, Q, K, V, attn_mask=None):
        """
        Q,K,V: [B, H, T, Dk]
        attn_mask: [B, 1 or H, T, T] (True where to mask)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        P = torch.softmax(scores, dim=-1)                             # [B,H,T,T]
        out = torch.matmul(P, V)                                      # [B,H,T,Dk]
        return out, P  # return P if you want to inspect attention
        

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(self.d_k)

    def forward(self, x, padding_mask=None, causal=False):
        """
        x: [B,T,D]
        padding_mask: [B,T] (True where PAD tokens are) -> will be broadcast
        causal: bool
        """
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.d_k).permute(2,0,3,1,4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B,H,T,Dk]

        # Build combined mask (True = mask out)
        attn_mask = None
        if padding_mask is not None:
            pad = padding_mask[:, None, None, :].expand(B, self.h, T, T)  # mask keys at PAD
            attn_mask = pad if attn_mask is None else (attn_mask | pad)
        if causal:
            cm = make_causal_mask(T, x.device)[None, None, :, :].expand(B, self.h, T, T)
            attn_mask = cm if attn_mask is None else (attn_mask | cm)

        ctx, _ = self.attn(Q, K, V, attn_mask)          # [B,H,T,Dk]
        ctx = ctx.transpose(1,2).contiguous().view(B,T,D)
        return self.out(ctx)                             # [B,T,D]
````

**Usage (standalone):**

```python
B,T,D = 2, 6, 64
x = torch.randn(B, T, D)
lengths = torch.tensor([6, 4])                # second sequence has padding after step 4
pad_mask = make_padding_mask(lengths, T, x.device)
mha = MultiHeadSelfAttention(d_model=D, n_heads=4)
y = mha(x, padding_mask=pad_mask, causal=True)  # causal decoder-style attention
```

---

## Common masks at a glance

* **Padding mask:** block attending *to* PAD tokens (keys).
* **Causal mask:** block attending *to future* positions (upper triangle).
* **Cross-attention mask:** same shapes, but Q from decoder, K/V from encoder.

---

## Numerical & performance tips

* Use **pre-LayerNorm** around attention in deep stacks.
* For long sequences, use fused kernels (e.g., FlashAttention) to reduce memory from $O(T^2)$ to $O(T \cdot d)$ while keeping exact results.
* When debugging, print **max/min** logits and verify masked logits go to `-inf` before softmax.

---

## Tiny sanity check you can run

```python
with torch.no_grad():
    # A toy: make token 0 a perfect key/value and see head attends to it
    x = torch.randn(1, 5, 32)
    attn = MultiHeadSelfAttention(32, 4)
    y = attn(x, causal=False)  # just ensure no crash
```

---

> **Your turn (one quick question):**
In decoder-only language modeling, why do we need **both** a causal mask **and** a padding mask? (Answer in a sentence.)

---

# 3 Transformer: concept + code

## Big picture

A Transformer is a stack of blocks built from:

1. **Self-attention** (tokens talk to tokens),
2. **Position-wise MLP**,
   with **residual** connections and **LayerNorm** (usually **pre-LN** today).

Two common shapes:

* **Encoder–decoder** (seq2seq: translation, T5): encoder has self-attn; decoder has causal self-attn + **cross-attn** to encoder states.
* **Decoder-only** (GPT family): just causal self-attn + MLP blocks.

---

## Core math (scaled dot-product attention)

For a sequence $X\in\mathbb{R}^{T\times d_{\text{model}}}$,
$$
Q = XW_Q,\quad K=XW_K,\quad V=XW_V,\quad
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

* $M$ is a mask: $0$ for allowed, $-\infty$ for blocked (e.g., **causal** mask for GPT).
* **Multi-head:** split into $H$ heads, run attention in parallel, then concatenate and project.

**Block (pre-LN)**
$$
\begin{aligned}
Y &= X + \text{MHA}(\text{LN}(X)) \\
Z &= Y + \text{MLP}(\text{LN}(Y))
\end{aligned}
$$

**Complexity:** $O(T^2 d)$ memory/time due to the $T\times T$ attention matrix.

---

## Minimal PyTorch: decoder-only block

```python
import torch, torch.nn as nn, torch.nn.functional as F

def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()  # True above diag

class ScaledDotProduct(nn.Module):
    def __init__(self, dk):
        super().__init__(); self.scale = dk ** -0.5
    def forward(self, Q, K, V, mask=None):
        # Q,K,V: [B, H, T, Dh]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,T,T]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        P = torch.softmax(scores, dim=-1)                            # [B,H,T,T]
        return torch.matmul(P, V)                                    # [B,H,T,Dh]

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProduct(self.dh)
    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2,0,3,1,4)  # [3,B,H,T,Dh]
        Q, K, V = qkv[0], qkv[1], qkv[2]                                     # [B,H,T,Dh]
        out = self.attn(Q, K, V, attn_mask)                                  # [B,H,T,Dh]
        out = out.transpose(1,2).contiguous().view(B, T, D)                  # [B,T,D]
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttn(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)
    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)   # (see § Positional embeddings later)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # tied? set self.head.weight = self.tok.weight
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        mask = causal_mask(T, idx.device)[None, None, :, :]  # broadcast to [B,H,T,T]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)  # [B,T,V]
````

**Training loop (language modeling, next-token):**

```python
def lm_loss(logits, targets):
    # logits: [B,T,V]; targets: [B,T] (next-token ids)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# B,T = batch and context length; idx is input tokens; tgt is shifted-by-1
# opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1)
```

---

## Practical notes & pitfalls

* **Pre-LN vs Post-LN:** pre-LN stabilizes deep training; post-LN may give slightly better perplexity at small depth but is brittle.
* **Init & scale:** tie input/output embeddings; consider RMSNorm for speed; use scaled residuals in very deep nets.
* **Masks:** causal for decoder; padding for encoder; **cross-attn** in encoder–decoder lets decoder attend to encoder states with its own $Q$ and the encoder’s $K,V$.
* **Complexity pressure:** long contexts → $T^2$ cost; see FlashAttention, KV cache, sparse/linear attention, and speculative decoding (later sections).

---

> **Quick check (your turn, one line):**
Why do we divide by $\sqrt{d_k}$ in the attention scores $QK^\top/\sqrt{d_k}$?


---

# 4 LLM Decoder (response sampling): the main methods

We generate a sequence $y_{1:T}$ from an autoregressive LM

$$
p_\theta(y_{1:T}\mid x)=\prod_{t=1}^{T} p_\theta\left(y_t \mid x, y_{<t}\right),\quad
p_\theta(y_t \mid \cdot)=\mathrm{softmax}(z_t)
$$

where $z_t \in \mathbb{R}^{V}$ are logits over a vocabulary $(V)$.

All decoding methods modify $z_t$ $\Rightarrow$ get a new distribution $\tilde p(y_t \mid \cdot)$ $\Rightarrow$ **sample** or **argmax**.  
Below, $\mathcal{S}$ denotes the chosen support set of tokens at step $t$; we always renormalize on $\mathcal{S}$.

---

## A) Temperature scaling (entropy control)

Scale logits by $1/\tau$ before softmax:

$$
\tilde p_\tau(i) = \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.
$$

* $\tau<1$: sharper (low entropy), more deterministic.
* $\tau>1$: flatter (high entropy), more diverse.
* As $\tau\to 0$, $\tilde p_\tau \to \arg\max$; as $\tau\to \infty$, $\tilde p_\tau \to \text{uniform}$.

**View:** maximizes likelihood under a **tempered** model $q \propto p^{1/\tau}$.  
Expected log-prob decreases ~ linearly with $\tau$.

---

## B) Top-k sampling (truncate to k highest-prob tokens)

Let $\mathcal{S}_k$ be the indices of the $k$ largest $p(i)$. Define:

$$
\tilde p(i) = \frac{\mathbf{1}\{i\in\mathcal{S}_k\} \, p(i)}{\sum_{j\in\mathcal{S}_k} p(j)}.
$$

* Controls **support size** directly.
* Pair with temperature: first scale, then truncate.
* Too small $k$ → bland/repetitive; too large $k$ → incoherent tails.

---

## C) Nucleus / Top-p sampling (truncate by mass)

Let $\mathcal{S}_p$ be the **smallest** set such that $\sum_{i\in\mathcal{S}_p} p(i) \ge p$. Then

$$
\tilde p(i) = \frac{\mathbf{1}\{i\in\mathcal{S}_p\} \, p(i)}{\sum_{j\in\mathcal{S}_p} p(j)}.
$$

* Adaptively chooses support size by **probability mass**.
* Typical $p\in[0.85,0.95]$; add temperature to control entropy inside the nucleus.

---

## D) Typical sampling (match local surprisal to entropy)

Compute token surprisals $s(i)=-\log p(i)$ and distribution entropy $H=-\sum_i p(i)\log p(i)$.  
Keep tokens whose surprisal is **closest to $H$**—i.e., minimize $|s(i)-H|$—until cumulative mass $\ge p$; then renormalize.

* Intuition: drop both ultra-predictable and ultra-surprising tails; keep “typical set”.
* Good at avoiding dullness *and* nonsense; hyper-params: mass $p$ and (optionally) a band width around $H$.

---

## E) Min-p / $\epsilon$-sampling (floor the tail)

Keep tokens above a **probability floor** $\epsilon$:  
$\mathcal{S}=\{ i \mid p(i)\ge \epsilon\}$; renormalize.

* Prevents extremely low-probability tokens from ever being sampled (helps factuality/formatting).

---

## F) Repetition / frequency / presence penalties (discourage reuse)

Modify logits using the history counts $c_i$ (frequency) or presence $\mathbf{1}\{c_i>0\}$:

$$
\hat z_i = z_i - \lambda_f c_i - \lambda_p \mathbf{1}\{c_i>0\},\quad
\tilde p \propto \exp(\hat z/\tau).
$$

Another popular rule (OpenAI “repetition\_penalty”):

$$
\hat z_i =
\begin{cases}
z_i/\rho & \text{if } i \in \text{history and } z_i>0,\\
z_i\cdot \rho & \text{if } i \in \text{history and } z_i<0,\\
z_i & \text{otherwise.}
\end{cases}
$$

* Reduces loops; set gently (e.g., $\rho \in [1.05,1.2]$) to avoid drift.

---

## G) (Greedy) Beam search (maximize sequence probability)

Deterministic search for

$$
y^\star = \arg\max_{y_{1:T}} \sum_{t=1}^{T} \log p(y_t \mid x,y_{<t}).
$$

Keep $B$ partial hypotheses (“beams”); at each step expand each by all tokens, keep top $B$ by score.  
**Length bias:** longer sequences accrue more negative log-prob. Fix via **length normalization**:

$$
\mathrm{score}(y_{1:t}) = \frac{1}{(5+t)^\alpha/(5+1)^\alpha}\sum_{k=1}^{t}\log p(y_k\mid \cdot),
$$

with $\alpha\in[0,1]$.

* Pros: good for tasks with single precise target (MT w/ references).
* Cons: reduces diversity; in open-ended generation it can amplify dullness.
* **Diverse beam search:** add dissimilarity penalties between beams to spread them.

---

## H) Contrastive / anti-degeneration decoding (a.k.a. contrastive search)

Select token $i$ that balances **model confidence** and **degeneracy penalty** (self-similarity of hidden states):

$$
i^\star = \arg\max_i \left[ \lambda \log p(i\mid \cdot)\;-\;(1-\lambda)\, \max_{t'<t}\cos\left(h(i), h_{t'}\right)\right],
$$

where $h(i)$ is the hidden state if we append token $i$.

* Intuition: confident yet **novel** continuations.
* Often used with a small candidate set $C$ (e.g., top-k) for speed.

*(Related but different: “contrastive **decoding**” subtracts a weak LM: $\log p_{\text{strong}} - \alpha \log p_{\text{weak}}$.)*

---

## I) Sampling-within-beam (stochastic beams)

Hybrid of top-k/top-p **inside** each beam expansion; still keeps top $B$ partial sequences by sampled scores.

* Improves diversity while retaining some search structure.

---

## J) Constrained decoding (regex/grammar/lexical constraints)

Search over sequences that satisfy constraints $\mathcal{C}$ (e.g., JSON grammar, must-include phrases). Formally:

$$
y^\star = \arg\max_{y\in \mathcal{L}(\mathcal{C})} \sum_t \log p(y_t\mid \cdot).
$$

* Implement via finite-state automatons (FSA) or LL(1)/PEG parsers gating logits (masking illegal next tokens).

---

## K) Calibrated stopping & entropy heuristics

* **EOS rules:** stop when $p(\text{</s>})$ exceeds a threshold or repeated high EOS mass.
* **Entropy stop:** stop if $\mathsf{H}(p_t)$ falls below a floor for $m$ steps (model “knows” what it wants to say).
* **Length priors:** soft prior over $T$ (e.g., Gaussian) added to beam scores.

---

## L) Putting it together (practical “stack”)

At each step $t$:

1. Start with logits $z_t$.
2. Apply **penalties** (repetition/frequency/presence or constraint masks).
3. Apply **temperature**.
4. Apply **support selection** (top-p or top-k or typical).
5. Sample (or pick by a scoring rule like contrastive).
6. Optional: enforce constraints / JSON grammar before committing token.

**Typical defaults for helpful assistants:**

* temperature $=0.7$–$0.9$,
* top-p $=0.9$ (or top-k = 40–100),
* mild repetition penalty (1.05–1.15),
* typical sampling as an alternative to top-p when you want sharper on-topic outputs.

---

## When to use what (fast mental model)

* **Factual QA, low hallucination:** $\tau\downarrow$, top-p (0.8–0.9), repetition penalty on; consider contrastive search.
* **Creative writing / brainstorming:** $\tau\uparrow$, top-p (0.92–0.98), maybe typical sampling; minimal penalties.
* **Formal outputs (JSON, SQL):** constrained decoding (grammar mask) + low temperature.
* **Machine translation / summarization with references:** beam (with length norm) or sampling-within-beam.

---

## Short derivations that clarify intuition

1. **Temperature ≈ KL-regularization:**  
   Sampling from $q \propto p^{1/\tau}$ is equivalent to maximizing $\mathbb{E}_q[\log p]$  
   subject to $\mathsf{H}(q)$ being **higher** as $\tau$ increases (Lagrange multiplier on entropy).  
   So $\tau$ trades log-likelihood vs. diversity.

2. **Top-p minimizes tail risk under a mass budget:**  
   Among all supports of mass $\ge p$, choosing the **smallest** tail (highest-prob tokens)  
   minimizes the expected surprisal; renormalization preserves the “core” while dropping the risky tail.

3. **Typical set rationale:**  
   For i.i.d. draws from $p$, most mass lies where surprisal $s(i)\approx H$.  
   Selecting tokens near $H$ approximates sampling from the **asymptotic equipartition** region,  
   avoiding overly confident clichés and wild outliers.

4. **Contrastive search anti-loop term:**  
   If the next-token hidden state $h(i)$ is too similar to a past state, the cosine term grows,  
   reducing the score. This penalizes **self-retrieval** and thus repetitive continuations.

---

> **Quick check (one bite-sized question):** 
If your generations look **fluent but generic** (“safe, samey answers”), which two knobs would you try first,  
and in which direction would you move them? (Name the method and whether you’d increase/decrease.)

---

# 5 KV Cache: concept + code

## What it is (why it speeds up generation)

During autoregressive decoding at step $t$, self-attention needs keys/values from **all prior tokens** $1{:}t-1$. Recomputing those every step is wasteful. A **KV cache** stores per-layer, per-head:

$$
K_{1:t-1}\in\mathbb{R}^{(t-1)\times d_k},\quad V_{1:t-1}\in\mathbb{R}^{(t-1)\times d_v}
$$

so at step $t$ you only compute $q_t$ and then:

$$
\text{Attn}(t)=\mathrm{softmax}\Big(\frac{q_t K_{1:t}^\top}{\sqrt{d_k}}\Big)\,V_{1:t},\quad
K_{1:t}=[K_{1:t-1};k_t],\; V_{1:t}=[V_{1:t-1};v_t].
$$

**Win:** per step cost falls from $O(t,d)$ *plus recomputation of all past projections* to just computing the new $(k_t,v_t)$ and a matrix-vector product with cached $(K,V)$.

**Complexities (per layer; $H$ heads, hidden $d$, head dim $d_k=d/H$):**

* Compute: $O(H\cdot t \cdot d_k)$ for $q_tK^\top$ (matrix-vector), vs $O(H\cdot t^2 d_k)$ total over a whole forward without caching.
* Memory: $O(H\cdot T \cdot d_k)$ for $K$ and same for $V$. For batch $B$, $O(BHTd_k)$.
  Rule of thumb: bytes $\approx 2 \times B \times H \times T \times d_k \times \text{dtype\_bytes}$.

**MQA/GQA to shrink caches**

* **Multi-Query Attention (MQA):** share **K,V** across heads ⇒ $H$ drops to $1$ in the cache terms; huge memory savings with similar quality for decoders.
* **Grouped-Query Attention (GQA):** share $K,V$ within groups of size $g$: effective $H_{\text{cache}}=H/g$.

**Positions with caches**

* **Absolute positions:** store $K,V$ after adding positions at their time index.
* **RoPE (rotary):** apply rotation to current $Q,K$ at index $t$; cached $K$ already contain past rotations.
* **Sliding window / ALiBi / long-context tricks:** restrict attention to last $W$ tokens to cap cache at $W$.

**Paged / chunked caches (serving)**

* Allocate cache in fixed pages (e.g., 16–128 tokens) per sequence; enables dynamic growth, efficient KV “swapping,” and multi-tenant batching without memcpy.

---

## Minimal PyTorch-ish cache pattern (decoder-only, single head for clarity)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class KVCache:
    def __init__(self, max_len, d_k, device):
        self.K = torch.empty((0, d_k), device=device)  # [t, d_k]
        self.V = torch.empty((0, d_k), device=device)
        self.max_len = max_len
    def append(self, k_t, v_t):
        # k_t, v_t: [1, d_k]
        self.K = torch.cat([self.K, k_t], dim=0)
        self.V = torch.cat([self.V, v_t], dim=0)
        # optional sliding window:
        if self.K.size(0) > self.max_len:
            self.K = self.K[-self.max_len:]
            self.V = self.V[-self.max_len:]

class CausalHead(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_k, bias=False)
        self.scale = d_k ** -0.5

    def forward_one(self, x_t, cache: KVCache):
        # x_t: [1, d_model] current token hidden
        q = self.Wq(x_t)                # [1, d_k]
        k = self.Wk(x_t)                # [1, d_k]
        v = self.Wv(x_t)                # [1, d_k]
        # attend over cached + current
        K = torch.cat([cache.K, k], dim=0)          # [t, d_k]
        V = torch.cat([cache.V, v], dim=0)          # [t, d_k]
        scores = (q @ K.T) * self.scale             # [1, t]
        P = torch.softmax(scores, dim=-1)           # [1, t]
        ctx = P @ V                                 # [1, d_k]
        # update cache after use
        cache.append(k, v)
        return ctx
````

**Multi-head + MQA tweak**

* Standard MHA: maintain one `KVCache` **per head**.
* **MQA:** share **one** `KVCache` across heads; compute distinct queries per head, but the same `K,V` are used.

---

## Practical gotchas & tips

* **Precision:** caching in FP16/BF16 is common; logits in FP32 avoid softmax overflow.
* **Contiguity:** keep `K,V` contiguous in memory (or paged) to maximize bandwidth.
* **Batching variable lengths:** store ${seq\_len}[b]$ and mask attention to valid prefix.
* **EOS truncation:** stop appending once EOS is produced (per sequence) to free memory.
* **Cache reuse vs prompt sharing:** for repeated prompts (RAG, agents), **prefill** once and reuse caches across continuations (a giant time saver).
* **FlashAttention & caches:** FlashAttn v2+ supports paged KV; still need the same $O(T)$ cache footprint but compute is IO-optimized.
* **Long context:** sliding window ($W$) caps memory at $O(W)$, but model must be trained/fine-tuned to tolerate local-attention at generation.

---

> **Tiny numerics check (one question):**
Why does **MQA** reduce memory roughly by a factor of the number of heads $H$ in the cache, and what trade-off might you expect in attention expressivity?

---

# 6 Positional embeddings: concept (math-first, no code)

## Why we need them

Self-attention is permutation-invariant: for token states $X\in\mathbb{R}^{T\times d}$, attention uses only inner products of projected states, so shuffling rows leaves outputs unchanged. We inject order by adding a **position-dependent term** so the model can condition on index differences.

---

## A) Learned **absolute** positional embeddings

Add a learned vector $p_t\in\mathbb{R}^d$ to each token at index $t$:
$$
\tilde x_t = x_t + p_t,\qquad t=1,\dots,T_{\max}.
$$
Attention then depends on positions through $\tilde x_t$.
**Limits:** fixed table ⇒ poor extrapolation beyond $T_{\max}$; encodes *absolute* index, not relative distances.

---

## B) **Sinusoidal** (Vaswani et al.)

Closed-form, frequency-based embedding:
$$
p_t[2i]   = \sin\!\bigg(\frac{t}{\omega_i}\bigg),\quad
p_t[2i+1] = \cos\!\bigg(\frac{t}{\omega_i}\bigg),\quad
\omega_i = 10000^{2i/d}.
$$
Key property: any shift $(t+\Delta)$ is a **linear transform** of $p_t$:
$$
p_{t+\Delta} = R(\Delta)\,p_t,
$$
so relative offsets are representable. Still **absolute add**: $\tilde x_t=x_t+p_t$. Generalizes past training lengths better than learned-absolute but relative reasoning still indirect.

---

## C) **Relative positional bias** (Shaw et al.; T5)

Inject **pairwise** bias depending on offset $\delta = t-q$ directly into attention logits:
$$
\alpha_{q\to t} \propto \frac{q_q^\top k_t}{\sqrt{d_k}} + b_{\mathrm{rel}}(\mathrm{clip}(\delta)),
$$
where $b_{\mathrm{rel}}\in\mathbb{R}^{2K+1}$ is a learned table over binned offsets $[-K,\ldots,K]$.

* **T5** uses **relative bias only** (no absolute add), often with bucketing $(\delta\mapsto \text{bucket}(\delta))$ that is linear for small $|\delta|$ and logarithmic for large $|\delta|$.
  **Upshot:** models **distance** directly; extrapolates to longer sequences if buckets cover them.

---

## D) **Transformer-XL / Decomposed relative** (content & position)

Factor attention score into content–content, content–position, and a global “query to absolute” term:
$$
\text{score}(q,t)=
\underbrace{q_q^\top k_t}_{\text{content–content}}
+
\underbrace{q_q^\top r_{q-t}}_{\text{content–position}}
+
\underbrace{u^\top k_t}_{\text{global content bias}}
+
\underbrace{v^\top r_{q-t}}_{\text{global positional bias}},
$$
with $r_\Delta$ a learnable embedding of relative offset $\Delta$. This yields efficient **segment recurrence** and relative generalization.

---

## E) **RoPE** (Rotary Positional Embedding)

Apply a **rotation** in each 2-D subspace of $\mathbb{R}^{d_k}$ to **queries and keys** before dot products. For head dim $d_k$, split into pairs $(u_{2i}, u_{2i+1})$. Define the complex form $u^{(c)}=u_{2i}+j u_{2i+1}$. At position $t$,
$$
\mathrm{RoPE}_t(u) = R_t u
\quad\text{with}\quad
R_t = \operatorname{diag}\!\big(e^{j \theta_0 t},\dots,e^{j \theta_{d_k/2-1} t}\big),\quad
\theta_i=\frac{1}{\omega_i}.
$$
Then
$$
\big(\mathrm{RoPE}_q(q_q)\big)^\top \big(\mathrm{RoPE}_t(k_t)\big)
= q_q^\top R_{t-q} k_t,
$$
so the dot product depends only on **relative offset** $t-q$.
**Benefits:** elegant relative encoding, strong long-range generalization; ubiquitous in modern LLMs.
**Extensions:**

* **NTK/YaRN scaling:** rescale $\theta_i$ to stretch usable context (effectively lowers frequencies to support longer $T$).
* **Dynamic RoPE scaling:** learn per-head scales to adapt frequencies.

---

## F) **ALiBi** (Attention with Linear Biases)

No positional vectors—just add a **monotone linear penalty** to attention logits:
$$
\alpha_{q\to t} \propto \frac{q_q^\top k_t}{\sqrt{d_k}} - m_h\, (q-t)^+,
$$
with slope $m_h>0$ per head and $(\cdot)^+=\max(0,\cdot)$ for causal models.
**Intuition:** farther keys get penalized; earlier tokens attend locally by default while some heads get small slopes to allow global attention.
**Pros:** trivial, extrapolates to any length; **no cache position transforms**.
**Cons:** less expressive than RoPE at modeling periodic/structured patterns.

---

## G) **Rotary vs Relative-bias: deriving the relative effect**

* With **RoPE**, the logit is $\langle R_q q_q, R_t k_t\rangle = \langle q_q, R_{t-q} k_t\rangle$, an *implicit* relative mechanism implemented by a rotation matrix dependent on $t-q$.
* With **relative bias**, the logit is $\langle q_q, k_t\rangle + b(t-q)$, an *additive* term independent of content vectors.
* Thus RoPE **modulates the similarity geometry**, while relative bias **shifts** scores uniformly per offset. Many strong models combine them (e.g., RoPE + small learned bias).

---

## H) **Absolute vs Relative: length generalization**

* **Absolute (learned):** best in-range fit; struggles OOD (positions $>T_{\text{train}}$).
* **Sinusoidal:** fixed spectrum helps extrapolation but does not directly encode **differences**.
* **Relative (bias/XL/RoPE/ALiBi):** directly parameterize *distances*, so *extrapolate* better and support streaming/segment stitching.

---

## I) Multi-dimensional positions (brief)

* **2-D grids (ViT):** add separable row/col embeddings $(p^{\text{row}}_r+p^{\text{col}}_c)$ (absolute) or 2-D relative biases $b(\Delta r,\Delta c)$.
* **Graphs / sets:** use Laplacian eigenvectors (positional encodings as graph Fourier features) or random features of diffusion distances; same principle—inject structure-dependent “position”.

---

## J) Choosing for LLMs today (rules of thumb)

* **Decoder-only, long context:** **RoPE** (with NTK/YaRN/dynamic scaling) or **ALiBi** if you want simplicity and unbounded extrapolation.
* **Encoder–decoder (T5-style):** **relative bias buckets** are strong and stable.
* **Very long sequences with streaming:** consider **ALiBi** or **RoPE + sliding window**; for strict JSON/format tasks, bias toward **relative bias** to maintain locality.

---

### One-line check

> Why does RoPE make the attention score depend on $t - q$ (a relative offset) even though we apply a position transform to **each** vector separately? Answer by pointing to the key algebraic step.

---


# 7 PEFT (Parameter-Efficient Fine-Tuning): concept + code

## Why PEFT?

Full fine-tuning updates **all** model weights $\Theta$ (billions of params). PEFT freezes $\Theta$ and introduces a **small** set of trainable parameters $\Phi$, giving similar quality with far less memory/compute and better multi-task storage.

---

## A) LoRA (Low-Rank Adaptation)

**Idea.** For a weight $W\in\mathbb{R}^{d_\text{out}\times d_\text{in}}$ (e.g., the attention $W_Q, W_V$), learn a **low-rank delta**:
$$
W' := W + \Delta W,\quad \Delta W = BA,\quad A\in\mathbb{R}^{r\times d_\text{in}},\; B\in\mathbb{R}^{d_\text{out}\times r},\; r\ll\min(d_\text{in},d_\text{out}).
$$
Often scale with $\alpha$:
$$
W'x = Wx + \frac{\alpha}{r}\, B(Ax).
$$

* Train: only $A,B$ (and sometimes a bias) update; $W$ stays frozen.
* Placement: typically on $W_Q, W_V$ (sometimes $W_O$, MLP).
* **Params:** $O(r(d_\text{in}+d_\text{out}))$ per matrix.

**Why it works (intuition):** many task-specific shifts lie in a **low-intrinsic-rank** subspace.

---

## B) DoRA / LoRA+ (scaling tweaks)

* **LoRA+:** separate learning rates for $A$ vs $B$ to stabilize training (since $A$ hits inputs, $B$ outputs).
* **DoRA:** decompose $\Delta W = s \cdot \hat{\Delta W}$ with a learned **magnitude** $s$ and normalized direction; tends to be more robust across ranks.

---

## C) Prefix / Prompt Tuning (P-Tuning v2)

Add **learned virtual tokens** to each layer’s attention as extra keys/values:
$$
\text{Attn}(Q,[K;K_p],[V;V_p]),
$$
where $K_p,V_p$ come from a small MLP on a trainable prefix embedding.

* Trains only the prefix parameters; **no** base weights touched.
* Excellent for sequence-to-sequence and instruction-style prompts.

---

## D) Adapters (Houlsby / Pfeiffer)

Insert small bottleneck MLPs inside blocks:
$$
h \mapsto h + W_\uparrow\,\sigma(W_\downarrow\,\text{LN}(h)),
\quad W_\downarrow\in\mathbb{R}^{d\times b},\; W_\uparrow\in\mathbb{R}^{b\times d},\; b\ll d.
$$

* Parameter budget controlled by bottleneck $b$.
* Strong when you can modify architecture (vs black-box weights).

---

## E) IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Learn **per-channel gates** that rescale attention/MLP pathways:
$$
\text{Attn}(Q,K,V)=
\text{softmax}\!\left(\frac{(Q\odot l_q)(K\odot l_k)^\top}{\sqrt{d_k}}\right)(V\odot l_v),
$$
with trainable $l_q,l_k,l_v\in\mathbb{R}^{d}$.

* Extremely parameter-light (just vectors).
* Works well combined with other PEFT methods.

---

## F) QLoRA (Quantized LoRA)

* **Freeze base** in **4-bit** NF4 (or 8-bit), train LoRA on top in FP16/BF16.
* Memory drops by ~4–8× vs FP16 full FT; performance close to LoRA on FP16.
* Needs paged optimizers and double quantization for stability.

---

## G) Choosing a method

| Scenario                           | Good choice              | Why                                        |
| ---------------------------------- | ------------------------ | ------------------------------------------ |
| Small task budget, strong backbone | **LoRA / QLoRA**         | Best quality per parameter, simple to ship |
| Strict “no weight changes” serving | **Prefix/Prompt tuning** | Pure side-params, easy to swap             |
| Multi-domain bank of adapters      | **Adapters**             | Modular per-task blocks                    |
| Ultra-tiny param budget            | **IA³**                  | Only gating vectors                        |
| Multi-task with interference       | **Mix LoRA + prefix**    | Separate pathways reduce conflict          |

---

## H) Mathy training notes

* **Scale:** use $\alpha/r$ so effective update magnitude is stable across ranks.
* **Where to place:** attention projections are most impactful; adding to MLP sometimes helps longer answers.
* **Rank selection:** start $r\in[4,16]$ for 7–13B models; increase for harder shifts.
* **Regularization:** weight decay on $A,B$ is mild (1e-2–1e-3); dropout on adapter outputs can reduce overfit.
* **Merging for export:** after training, you can add $\Delta W$ into $W$ for a one-piece model (if not using quantized base).

---

## I) Minimal usage patterns (PyTorch/HF)

**LoRA on a causal LM (attention Q,V)**

```python
# pip install peft transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_id = "meta-llama/Llama-2-7b-hf"
tok = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")

peft_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","v_proj"],  # names depend on model
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(base, peft_cfg)
# Now train only LoRA params; base weights are frozen.
````

**QLoRA (4-bit base + LoRA)**

```python
from transformers import BitsAndBytesConfig
bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_cfg, device_map="auto")
# Then wrap with the same LoRA config as above.
```

**Prefix tuning with PEFT**

```python
from peft import PrefixTuningConfig, get_peft_model
cfg = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(base, cfg)
```

*(Exact `target_modules`/naming varies across model repos.)*

---

## J) Pitfalls & remedies

* **Catastrophic format drift:** too-high rank or LR can change style; lower $\alpha$, add adapter dropout, or restrict placements.
* **Low-signal task:** PEFT can underfit tiny datasets; warm up LR, increase rank, or allow MLP LoRA.
* **Quantization mismatch (QLoRA):** ensure compute dtype (BF16/FP16) for LoRA layers; keep layernorms in higher precision.
* **Multi-adapter interference:** use separate adapter names and **AdapterFusion** (learned mixing) or pick task per inference.

---

> **Quick check (1 sentence):**
Why does LoRA scale its update by $\alpha/r$? What happens if you keep $\alpha$ fixed and double $r$ without that scaling?

---

# 8 Quantization: concept + code

## What problem it solves

Shrink memory/latency by storing weights/activations with **fewer bits** while keeping accuracy.

* **Weight-only** (e.g., W4A16): biggest win for LLM inference (KV cache still fp16/bf16).
* **Weight + activation** (e.g., W8A8): useful on edge/TPU; needs careful calib.
* **PTQ** (Post-Training) vs **QAT** (Quantization-Aware Training).

---

## Core math (uniform affine quantizer)

Given real tensor $x$ and $b$-bit integers $q \in [q_{\min}, q_{\max}]$,
$$
\textstyle s=\frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad z=\Big\lfloor \frac{-x_{\min}}{s}\Big\rceil,
$$
$$
q=\operatorname{clip}\Big(\Big\lfloor \frac{x}{s}\Big\rceil+z;\ q_{\min}, q_{\max}\Big),\qquad
\hat x = s (q - z).
$$

* **Symmetric**: set $z=0$, $x_{\min}=-\alpha,\ x_{\max}=\alpha$.
* **Asymmetric**: nonzero $z$, better when distribution is skewed.
* **Per-tensor** vs **per-channel** scales (per-channel reduces error for conv/linear).

**Quantization error (uniform, high-resolution):**  
If clipping range is $[- \alpha, \alpha]$, step $\Delta = 2\alpha/(2^b-1)$, MSE $\approx \Delta^2/12$ (ignoring clipping tails). Hence **clipping** (choose $\alpha$) trades rounding vs tail loss.

---

## What’s commonly used for LLMs

* **RTN** (round-to-nearest): W4/W8 weight-only; fast baseline.
* **GPTQ** (PTQ, second-order): minimize layerwise output error using a Hessian approximation; excellent W3–W4 accuracy.
* **AWQ** (Activation-aware): identify “salient” channels (large activation impact), keep them in higher precision or larger scales; robust W4.
* **SmoothQuant** (W8A8): shift activation range into weights using per-channel scaling so activations quantize well.
* **KV-cache quant**: k/v projected to int8/float8 to reduce memory bandwidth with minimal quality drop.

---

## PTQ calibration (how ranges/scales are picked)

Given a **calibration set** $\mathcal{D}$ of hidden activations:

1. **MinMax/Percentile:** $\alpha = \text{percentile}(|x|, p)$.
2. **MSE/MAE search:** pick $\alpha$ minimizing $|x - \hat x(\alpha)|^2$.
3. **Entropy/KL:** choose histogram bins to minimize KL between float and quantized histos.

Then compute $s,z$ and quantize. For **per-channel**, do this per output channel of Linear: better for heavy-tailed weights.

---

## Tiny PyTorch snippets (didactic)

**1) Per-tensor symmetric INT8 quant/dequant helper**

```python
import torch

def quantize_int8(x):
    # symmetric per-tensor
    a = x.abs().max()
    s = a / 127.0 + 1e-12
    q = torch.clamp((x / s).round(), -128, 127).to(torch.int8)
    return q, s

def dequantize_int8(q, s):
    return q.float() * s
````

**2) Per-channel weight quant for a Linear layer (W8, A16)**

```python
import torch, torch.nn as nn

def quantize_per_channel_w8(W):  # W: [out, in]
    a = W.abs().amax(dim=1, keepdim=True)           # per-out-channel
    s = a / 127.0 + 1e-12
    q = torch.clamp((W / s).round(), -128, 127).to(torch.int8)
    return q, s.squeeze(1)

class W8A16Linear(nn.Module):
    def __init__(self, W_fp, bias=None):
        super().__init__()
        q, s = quantize_per_channel_w8(W_fp)
        self.register_buffer("W_q", q)
        self.register_buffer("s", s)       # [out]
        if bias is not None:
            self.register_buffer("b", bias)
        else:
            self.b = None
    def forward(self, x):                  # x: fp16/bf16
        # dequant on the fly; many runtimes fuse this
        W = (self.W_q.float().T * self.s).T
        y = x @ W.T
        return y if self.b is None else y + self.b
```

**3) bitsandbytes 4-bit (QLoRA-style)**

```python
# pip install bitsandbytes transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # normal-float4 (non-uniform)
    bnb_4bit_use_double_quant=True,      # quantize the scales too
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                             quantization_config=bnb_cfg,
                                             device_map="auto")
```

**4) GPTQ (conceptual outline)**

* For a Linear $y = W x$, collect activations $X=[x_1,\dots,x_n]$ and outputs $Y$.
* Minimize $|W X - \hat W X|_F^2$ where $\hat W$ is quantized.
* Column-wise solve using a Cholesky of $G = XX^\top$ (Hessian proxy):
  $$
  \hat w_c = \arg\min_{q \in \mathcal{Q}} |w_c - q|_{G}^2
  $$
  where $|v|_G^2 = v^\top G v$. Greedy update columns, compensating residuals.
* Intuition: preserve **post-layer** outputs, not raw weights.

---

## QAT (Quantization-Aware Training)

Insert **fake-quant** in the forward, use **STE** (straight-through estimator) through the rounding:
$$
\tilde q = \operatorname{round}(x/s),\quad \hat x = s,\tilde q,\quad
\frac{\partial \hat x}{\partial x} \approx 1_{|x| \le \alpha}.
$$

* Train end-to-end so model adapts to quantization noise.
* Best for W8A8/W4A8 when PTQ loses too much.

---

## Practical recipes (LLMs)

* **Server inference (GPU):** W4/W3 **weight-only** (GPTQ/AWQ) + fp16 activations; optionally int8 KV cache.
* **Edge/CPU:** W8A8 static/PTQ with SmoothQuant; fuse ops; per-channel scales.
* **Mixed precision:** keep **embeddings, layernorm, output head** in higher precision if quality dips.
* **Calibration set:** 128–512 prompts resembling deployment; run **prefill** to collect activations.

---

## Pitfalls & fixes

* **Outlier channels** (very large weights/acts) dominate scale → clip or route to higher precision (AWQ outlier handling).
* **RoPE sensitivity** at 4-bit: keep RoPE/positional projections higher precision or use per-head/grouped scales.
* **KV-cache int8** can hurt very long-context reasoning → try float8 (E4M3/E5M2) or leave V in bf16.

---

> **Quick check (one-liner):**
> Why does **per-channel** weight quantization usually outperform **per-tensor** for linear layers?

---

# 9 RLHF & DPO: concept + code

## 0) Setup recap

We start with an SFT model ($\pi_{\text{sft}}$) trained on high-quality $(x, y)$ pairs. We also keep a **reference policy** ($\pi_{\text{ref}}$) (often the frozen $\pi_{\text{sft}}$).

Human feedback data: for each prompt $x$, we have **pairwise preferences** $(y^+, y^-)$ where annotators preferred $y^+$ over $y^-$.

---

## A) Reward Modeling (pairwise Bradley–Terry)

We learn a scalar reward $r_\phi(x,y)$ so that

$$
P_\phi(y^+ \succ y^- \mid x) = \sigma\big(r_\phi(x,y^+) - r_\phi(x,y^-)\big),
\quad \sigma(u)=\frac{1}{1+e^{-u}}.
$$

**Loss (cross-entropy):**

$$
\mathcal{L}*{\text{RM}}(\phi)
= -\mathbb{E}*{x, (y^+,y^-)} \big[\log \sigma(r_\phi(x,y^+)-r_\phi(x,y^-))\big].
$$

At inference, the reward is used to score new samples.

**Minimal PyTorch RM (decoder as feature extractor)**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, backbone, hidden=768):
        super().__init__()
        self.backbone = backbone  # frozen or lightly tuned LM returning final hidden states
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # pool with last non-pad token (or mean-pool)
        mask = attention_mask.bool()
        idx = mask.sum(dim=1) - 1
        pooled = h[torch.arange(h.size(0)), idx]  # [B, hidden]
        return self.head(pooled).squeeze(-1)      # [B]

def pairwise_rm_loss(r_pos, r_neg):
    return F.binary_cross_entropy_with_logits(r_pos - r_neg, torch.ones_like(r_pos))
```

---

## B) RLHF with PPO (policy optimization with KL control)

Goal: **maximize reward while staying close to $\pi_{\text{ref}}$**.

### Objective (per trajectory $y\sim \pi_\theta(\cdot|x)$)

$$
J(\theta)=
\mathbb{E}\Big[
\underbrace{r_\phi(x,y)}*{\text{scalar from RM}}
-\beta,\underbrace{\mathrm{KL}\left(\pi*\theta(\cdot|x)\ |\ \pi_{\text{ref}}(\cdot|x)\right)}_{\text{regularizer}}
\Big].
$$

In tokenized form, with stepwise rewards ($r_t=0$) except final step (or dense heuristics), we compute **advantages** $A_t$ from returns and a learned value baseline $V_\psi$.

### PPO clipped surrogate

Let $\pi_\theta$ and old policy $\pi_{\theta_\text{old}}$ define ratio

$$
\rho_t=\frac{\pi_\theta(y_t\mid x, y_{<t})}{\pi_{\theta_\text{old}}(y_t\mid x, y_{<t})}.
$$

Then the PPO loss is

$$
\mathcal{L}_{\text{PPO}}(\theta)=
-\mathbb{E}\Big[
\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\big)

* c_v (V_\psi(s_t) - R_t)^2 + c_e \mathsf{H}[\pi_\theta(\cdot|s_t)]
  \Big],
  $$

and we **add a KL penalty** to $\pi_{\text{ref}}$ either:

* explicitly: $\mathcal{L} + \beta,\mathrm{KL}(\pi_\theta | \pi_{\text{ref}})$, or
* implicitly in the reward: $r_t^{\text{KL}} = r_t - \beta \log \frac{\pi_\theta(y_t|s_t)}{\pi_{\text{ref}}(y_t|s_t)}$.

Advantage via GAE:

$$
A_t = \sum_{l\ge 0} (\gamma\lambda)^l \delta_{t+l},
\quad
\delta_t=r_t + \gamma V_\psi(s_{t+1})-V_\psi(s_t).
$$

**High-level PPO loop (skeleton)**

```python
# 1) Collect rollouts:
#   sample y ~ pi_theta(.|x) using nucleus/top-k; cache logprobs, values, ref logprobs.

# 2) Compute rewards:
#   r = r_phi(x,y)  (scalar) -> distribute to final step or shape through tokens.
#   add KL shaping: r_t <- r_t - beta * (logp_pi - logp_ref).

# 3) Compute advantages A_t with GAE; normalize A.

# 4) Optimize for K epochs:
#   compute ratio rho_t, PPO clipped loss + value loss + entropy bonus.

# Notes: use microbatched generation, mixed precision, and gradient accumulation.
```

**Tuning tips (RLHF)**

* $\beta$ (KL coeff) is crucial: too small ⇒ mode collapse/over-optimization; too big ⇒ no change vs ref.
* Short rollouts (64–256 tokens) stabilize credit assignment.
* Keep prompt/task diversity high; periodically refresh $\pi_{\text{ref}}$ (or not—common to keep it fixed as SFT).

---

## C) DPO (Direct Preference Optimization)

**Idea.** Skip the explicit reward model and RL loop. Optimize $\pi_\theta$ **directly** on pairwise preferences to approximate the KL-regularized RL solution.

### Derivation sketch

Solve the **constrained** problem (maximum entropy RL view):

$$
\max_{\pi} \ \mathbb{E}*{x,y\sim \pi} [r(x,y)] \quad
\text{s.t.}\ \mathrm{KL}(\pi(\cdot|x)|\pi*{\text{ref}}(\cdot|x)) \le \epsilon.
$$

The optimal $\pi^\star$ has **Boltzmann** form:

$$
\pi^\star(y|x) \propto \pi_{\text{ref}}(y|x), e^{\beta r(x,y)}.
$$

With **pairwise** Bradley–Terry preference ($y^+ \succ y^-$ iff $r(x,y^+)>r(x,y^-)$), eliminate $r$ to get a discriminative objective in terms of **log-ratios**:

$$
\mathcal{L}*{\text{DPO}}(\theta)=
-\mathbb{E}*{x,(y^+,y^-)}
\Big[
\log \sigma\Big(
\beta\big[
\underbrace{\log\pi_\theta(y^+|x)-\log\pi_\theta(y^-|x)}_{\text{policy preference}}
-----------------------------------------------------------------------------------

\underbrace{\log\pi_{\text{ref}}(y^+|x)-\log\pi_{\text{ref}}(y^-|x)}_{\text{reference correction}}
\big]\Big)
\Big].
$$

Here $\beta>0$ controls sharpness (akin to inverse temperature).
We compute $\log\pi(\cdot|x)$ as **sum of token log-probs** along the response (optionally length-normalized or EOS-trimmed).

**Minimal DPO training loop**

```python
# Given batches of (x, y_pos, y_neg)
# Precompute ref logprobs with the frozen reference LM for efficiency.

def seq_logprob(model, input_ids, attention_mask, label_ids):
    # return scalar log p(y|x) by summing token logprobs where labels != -100
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp_tok = torch.log_softmax(logits, dim=-1)
    lp = logp_tok.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1)
    return (lp * (label_ids != -100)).sum(dim=-1)  # [B]

beta = 0.1
for batch in loader:
    lp_pos = seq_logprob(pi_theta, batch.pos_inp, batch.pos_mask, batch.pos_lbl)   # [B]
    lp_neg = seq_logprob(pi_theta, batch.neg_inp, batch.neg_mask, batch.neg_lbl)
    lp_pos_ref = batch.lp_pos_ref  # cached from pi_ref
    lp_neg_ref = batch.lp_neg_ref

    logits_pref = beta * ((lp_pos - lp_neg) - (lp_pos_ref - lp_neg_ref))
    loss = F.binary_cross_entropy_with_logits(logits_pref, torch.ones_like(logits_pref))
    opt.zero_grad(); loss.backward(); opt.step()
```

**Intuition:** push $\pi_\theta$ to assign **more** probability mass to $y^+$ than $y^-$ **relative to** how much the reference already did. If $\pi_{\text{ref}}$ already strongly prefers $y^+$, the gradient is small.

---

## D) RLHF vs DPO: when to use which?

| Aspect       | RLHF (PPO)                                          | DPO                                                   |
| ------------ | --------------------------------------------------- | ----------------------------------------------------- |
| Data need    | Prefs **or** scalar reward model usable OOD         | Requires pairwise prefs                               |
| Compute      | Heavier (rollouts, value fn, PPO epochs)            | Light (pure supervised-style on prefs)                |
| Control      | Fine-grained through reward shaping and KL schedule | One knob ($\beta$); less control at token granularity |
| Stability    | Can be tricky (credit assignment, KL tuning)        | Typically very stable, easy to scale                  |
| Expressivity | Can incorporate non-decomposable / tool-use rewards | Limited to preference pairs on text                   |

**Practical defaults**

* Start with **DPO** for simplicity/throughput on large preference sets.
* Use **RLHF/PPO** when you need **non-textual rewards** (e.g., tool correctness, safety policies, function outputs) or shaped token-level rewards.

---

## E) Implementation gotchas

* **Logprob computation:** mask prompts properly; exclude padding from sums; consider length-norm to reduce bias toward short replies.
* **Reward hacking (RLHF):** watch for reward overoptimization → monitor KL to ref and human evals; use early stopping or increase $\beta$.
* **Annotation noise:** aggregate preferences; use **pairwise Bradley–Terry with annotator reliabilities** or filter low-agreement items.
* **Safety:** incorporate refusal/guardrails via additional preference data or multi-objective (e.g., safety reward minus helpfulness violation).
* **Mixture-of-objectives:** combine SFT loss with DPO/RL loss (e.g., warm start or multi-task schedule).

---

> **Quick check (1–2 lines)**
> Why does DPO include the **reference log-prob terms** $\log\pi_{\text{ref}}(y^\pm|x)$ inside the sigmoid argument, and what would likely happen if you dropped them?

---

# 10 GRPO — Group Relative (Policy) Optimization

## Core idea

GRPO optimizes a policy **without** a value function or explicit reward model by using **grouped rollouts per prompt** and a **relative baseline** inside the group. It’s essentially REINFORCE with a strong control variate and PPO-style KL control.

Given a prompt $x$, sample a *group* ${y^{(i)}}*{i=1}^M \sim \pi*\theta(\cdot\mid x)$. Score each with a scalar reward $r^{(i)} = r(x,y^{(i)})$ (e.g., pass@k for code, unit tests, formatting/safety checks, length heuristics, or lightweight preference/rank).

### Relative advantage inside the group

Compute a baseline from the group, e.g.

$$
\bar r = \frac{1}{M}\sum_{i=1}^M r^{(i)},\qquad
A^{(i)} = \frac{r^{(i)}-\bar r}{\sigma_r + \epsilon},\quad
\sigma_r^2=\tfrac{1}{M}\sum_i (r^{(i)}-\bar r)^2.
$$

This yields **zero-mean**, scale-normalized advantages *per prompt*, removing the need for a critic $V_\psi$.

### Objective with KL control (PPO-lite)

Let $\rho^{(i)} = \exp!\big(\log\pi_\theta(y^{(i)}\mid x)-\log\pi_{\theta_\text{old}}(y^{(i)}\mid x)\big)$ be the sequence-level ratio (or token-sum). A GRPO step minimizes

$$
\mathcal{L}*{\text{GRPO}}(\theta)=
-\mathbb{E}*{x}\left[\frac{1}{M}\sum_{i=1}^M
\Big(
\min\big(\rho^{(i)} A^{(i)},\ \mathrm{clip}(\rho^{(i)},1-\epsilon,1+\epsilon)A^{(i)}\big)
\Big)\right]

* \beta,\mathrm{KL}!\left(\pi_\theta(\cdot|x),|,\pi_{\text{ref}}(\cdot|x)\right).
  $$

Notes:

* Like PPO, but **no value loss**; advantage is **group-relative**.
* KL term to a **reference** $\pi_{\text{ref}}$ (often the SFT model) stabilizes style and prevents reward hacking.
* You can use **token-level** ratios with the same idea (sum of clipped terms), but sequence-level is common for cheap rewards.

### Why it works

* **Baseline as control variate:** subtracting $\bar r$ reduces gradient variance (REINFORCE identity), and normalizing by $\sigma_r$ stabilizes across prompts/domains.
* **Group comparison** makes rewards **relative** to co-sampled alternatives for the same $x$, which is robust to noisy reward scales.

### Practical rewards for GRPO

* **Outcome-only:** pass/fail tests, exact-match, regex/JSON validity, safety rule satisfaction.
* **Shaped:** add brevity/format bonuses, tool-use success, or light-weight model-based scores.
* **Rank-from-rules:** rank the $M$ samples with deterministic criteria and map ranks to scores (e.g., ${+1,0,-1}$).

### Minimal loop (conceptual)

1. For each $x$: sample $M$ responses with $\pi_{\theta_\text{old}}$.
2. Compute $r^{(i)}$, then $A^{(i)}$ via group stats.
3. Optimize $\theta$ with the GRPO loss above (PPO clip + KL to $\pi_{\text{ref}}$).
4. Update $\theta_\text{old}\leftarrow \theta$ periodically.

---

## GRPO vs DPO vs RLHF (when to use what)

| Dimension                  | **GRPO**                                                                    | **DPO**                                                      | **RLHF (PPO)**                                            |
| -------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| **Data needed**            | No preferences; needs **cheap scalar reward** computable per sample         | **Pairwise prefs** $(y^+,y^-)$ per $x$                       | Either prefs → reward model **or** direct scalar rewards  |
| **Critic / value fn**      | **None** (group baseline)                                                   | None                                                         | **Yes** (value net or equivalent)                         |
| **Objective form**         | Policy gradient with group-relative **advantages** + PPO clip + KL          | Logistic **pairwise** loss on log-prob **ratios** w.r.t. ref | PPO surrogate with **token advantages** and KL shaping    |
| **Credit assignment**      | **Sequence-level** by default; cheap                                        | **Sequence-level** (pairwise)                                | Can be **token-level** with shaped rewards                |
| **Stability & complexity** | Simple; stable if KL/clip tuned; no critic headaches                        | Very simple; very stable; one $\beta$ knob                   | Most complex; sensitive to $\beta$, GAE, rollout length   |
| **Best for**               | Code, math, tools where **automatic tests** exist; scalable online training | Large **preference** datasets; instruction quality, style    | Rich, **non-decomposable** rewards and fine token shaping |
| **Weakness**               | Needs computable reward; coarse token credit unless shaped                  | Needs preference pairs; limited control at token level       | Heavier engineering; risk of over-optimization            |

**Rules of thumb**

* Have **tests/validators**? → **GRPO** first (cheap, no critic).
* Have **human preferences** only? → **DPO** (throughput champ).
* Need **fine token shaping** or tool outcomes with complex structure? → **RLHF/PPO**.

---

## Math sidebar: variance reduction

REINFORCE gradient for one sample:
$\nabla_\theta \log \pi_\theta(y|x),(r-\underbrace{b(x)}_{\text{baseline}})$.

Choosing $b(x)=\mathbb{E}[r|x]$ minimizes variance. GRPO’s $\bar r$ is an *unbiased* estimator of that baseline using the group, and dividing by $\sigma_r$ standardizes the scale across prompts, taming the PPO ratio dynamics.

---

## Pitfalls & tips

* **Group size $M$:** too small → high variance; too large → expensive. Common: $M=4!\sim!8$.
* **Reward saturation:** if most samples pass (or fail), add *soft* shaping terms or widen tests to keep variance.
* **KL schedule:** start with higher $\beta$ to anchor style, then anneal.
* **Clipping $\epsilon$:** 0.1–0.2 typical; larger when advantages are well-normalized.
* **Token-level shaping:** optionally distribute a scalar outcome over tokens (e.g., uniform or with a heuristic) if you need more granular control.

---

> **Quick check (your turn, one line):**
> What’s the main **statistical benefit** of subtracting the **group mean reward** $\bar r$ in GRPO before applying PPO clipping and KL?

---

# 11 Speculative decoding (mechanics, math, guarantees)

## Goal

Speed up sampling from a **large verifier model** $p$ (the “target” LM) by using a **cheaper draft/proposal model** $q$ to suggest multiple future tokens at once, then **accept–reject** those suggestions so the final samples are *exactly* from $p$.

---

## Setup

* Context (history) $h = (x, y_{<t})$.
* Verifier $p_\theta(y_t\mid h)$: big model (accurate, slow).
* Draft $q_\phi(y_t\mid h)$: small model (approximate, fast).
* We aim to produce a next-token (or next-block) sample whose marginal law equals $p_\theta$.

---

## Single-token speculative sampling (rejection sampling view)

**Proposal:** sample $y\sim q(\cdot\mid h)$.

**Accept with probability**

$$
a(h,y)=\min\!\left(1,\ \frac{p(y\mid h)}{q(y\mid h)}\right).
$$

If accepted, output $y$. If rejected, resample **from a corrected distribution** restricted to the complement:

$$
\tilde p(\cdot\mid h, \text{reject}) \propto p(\cdot\mid h)\,\mathbf{1}_{\cdot\neq y}.
$$

This is classical rejection sampling: by construction the output is **exactly** from $p$.

**Acceptance rate**

$$
\alpha(h)=\sum_y q(y\mid h)\,a(h,y)=\sum_y \min\!\big(q(y\mid h),\,p(y\mid h)\big).
$$

So $\alpha$ rises as $q$ aligns with $p$ (their overlap increases).

---

## Block speculative decoding (practical algorithms)

Rather than 1 token, draft **$m$** tokens in one cheap sweep:

$$
\hat y_{t:t+m-1}\sim q(\cdot\mid h).
$$

Then run **one forward** of $p$ over the *whole proposed block* to obtain the verifier conditionals

$$
p(\hat y_t\mid h),\; p(\hat y_{t+1}\mid h,\hat y_t),\;\dots,\; p(\hat y_{t+k}\mid h,\hat y_{t:t+k-1}).
$$

**Prefix acceptance rule.**

Scan left-to-right; for each position $j$ in the block compute

$$
a_j=\min\!\left(1,\ \frac{p(\hat y_{t+j}\mid h,\hat y_{t:t+j-1})}{q(\hat y_{t+j}\mid h,\hat y_{t:t+j-1})}\right).
$$

Independently sample $u_j\sim \mathrm{Uniform}(0,1)$.

* If $u_j\le a_j$, the token passes.
* Stop at the **first failure** $j^\star$; accept the **entire prefix** $\hat y_{t:t+j^\star-1}$.
* At $j^\star$, resample from a **correction distribution** that compensates for the rejected draft:

  $$
  r(\cdot)\ \propto\ p(\cdot\mid h,\hat y_{t:t+j^\star-1})\ -\ \min\!\Big(p(\cdot\mid h,\dots),\ q(\cdot\mid h,\dots)\Big).
  $$

  Emit one token from $r$, then continue generation *from $p$* (or restart a new speculative block).

**Why this is exact.**

Inductively, the probability that the output token equals $y$ is the sum of:  
(i) drafting $y$ and accepting it with $a$, plus  
(ii) drafting something else and then drawing $y$ from the correction $r$.  

Algebra collapses to $p(y\mid \cdot)$. Thus the produced sequence has the **same law as $p$**.

---

## Speedup intuition

Let:

* $\alpha$ = expected per-token acceptance rate,
* $m$ = draft block length,
* $C_p$ = cost of one forward of $p$ over a block of length $m$ (amortized via KV cache),
* $C_q$ = cost of one forward of $q$ over that block.

Roughly, each joint pass tries to “harvest” about $\alpha m$ accepted tokens from $q$ while paying one $p$-forward.

**Throughput gain (very rough):**

$$
\text{speedup} \approx \frac{1}{\frac{C_p}{\alpha m} + \frac{C_q}{\alpha m}}
\;\Big/\;1
=\frac{\alpha m}{C_p + C_q}.
$$

So increase $m$ until the **verification cost grows** faster (KV, bandwidth); pick $q$ so $\alpha$ is high (low KL) but $C_q$ stays low.

---

## Design choices & variants

1. **Choice of $q$.**

   * **Small distilled LM** (same tokenizer/positions) → high $\alpha$ at low cost.
   * **Temperature-matched** $q$ can lift $\alpha$ (too sharp/flat hurts overlap).
   * **Medusa-style heads:** add cheap “lookahead heads” on $p$ to propose futures (no second model).

2. **Tree / multi-branch proposals.**  
   Draft a small **beam/tree** from $q$; verify along paths with $p$; accept the longest valid prefix among branches. Increases $\alpha$ at extra $q$ cost.

3. **Block size & cadence.**  
   Larger $m$ reduces $p$ calls, but the chance that *some* token fails rises (prefix shorter). Tune $m$ where $\mathbb{E}[\text{accepted}]$ saturates.

4. **Caching.**

   * Reuse $p$’s KV cache built on the accepted prefix only.
   * $q$’s cache is ephemeral; you don’t carry it forward after verification.

5. **Constrained outputs.**  
   Apply grammar/regex **masks to both $q$ and $p$** so acceptance decisions respect constraints. The exactness proof still holds because masking changes both densities consistently.

6. **Latency vs throughput.**  
   Block verification adds **micro-batch latency** (wait for $q$ then a big $p$ step). For interactive latency, keep $m$ small or pipeline $q$ and $p$ across sequences.

---

## Guarantees & diagnostics

* **Unbiasedness:** By construction, the marginal of the emitted token equals $p$ (see acceptance + correction decomposition).
* **Acceptance rate bounds:**

  $$
  \alpha = 1 - \tfrac12 \lVert p - q\rVert_1
  \quad\text{(for single-token proposals)}
  $$

  so total-variation distance directly caps rejections.

* **KL calibration:** Lower $\mathrm{KL}(p\|q)$ typically raises $\alpha$. Distillation or temperature tuning for $q$ helps.

**What to monitor**

* Prefix length distribution, mean accepted tokens per verification, time per accepted token, $\alpha$ vs position (quality often drops late in sequences).

---

## Practical pitfalls & fixes

* **Mismatched tokenization/positions** → invalid comparisons ($p/q$). Use same vocab and positional scheme (RoPE scaling!).
* **$q$ too weak** → low $\alpha$, no speedup. Distill from $p$ on recent traffic; align decoding settings (top-p, temperature).
* **Bandwidth-bound verification** → computing $p(\hat y_{t:t+m-1}\mid \cdot)$ must be fused (FlashAttention, paged KV).
* **Large vocab corrections** → at rejection, computing the full correction distribution can be heavy; use **top-k caps** with proof-preserving normalization or sample via alias tables over the corrected support.

---

## Related ideas (mental map)

* **Lookahead heads (Medusa)**: train auxiliary heads to predict several steps ahead from current hidden state; verify with main head—no extra model $q$.
* **EAGLE / drafting with experts**: mixture proposals tailored to common next-token patterns; similar accept-verify loop.
* **Early-exit token heads**: predict top-prob tokens with shallow layers; verify with full stack (layer-drop speculative decoding).

---

> **Quick check (your turn, one line):**
If you increase the block size $m$ but your acceptance rate $\alpha$ drops, what *measurable* quantity should you track to decide whether the change actually improved throughput?

---

# 12 Word embeddings (with tiny PyTorch snippets)

## What they are (and why)

We map each token $w$ in a vocabulary of size $V$ to a vector $e_w \in \mathbb{R}^d$. Stack all vectors as rows of a matrix
$$
E \in \mathbb{R}^{V \times d}, \quad e_w = E[w,:].
$$
This turns discrete symbols into geometry, where dot products and angles carry meaning. Cosine similarity
$$
\cos(\theta)=\frac{e_u^\top e_v}{|e_u||e_v|}
$$
lets us talk about “closeness” of words.

In most modern models, $E$ is just a learned parameter table—the first layer of the network. It’s optimized by backprop from whatever loss you use (LM loss, contrastive loss, etc.). “Distributional semantics” emerges: words that appear in similar contexts get similar vectors.

## A minimal training view

Suppose we try to predict a context token $c$ from a center token $w$ (a toy skip-gram). Let $E$ be the input embeddings and $U \in \mathbb{R}^{V\times d}$ be “output” embeddings. The model score is
$$
s(c \mid w) = U_c^\top E_w,
$$
and the probability is softmax over all $c$:
$$
p(c\mid w) = \frac{\exp(U_c^\top E_w)}{\sum_{c'} \exp(U_{c'}^\top E_w)}.
$$
The loss for a training pair $(w,c)$ is $-\log p(c\mid w)$. Gradient pushes $E_w$ toward $U_c$ and away from other $U_{c'}$.

In large corpora we often use negative sampling instead of full softmax: maximize  
$$\log \sigma(U_c^\top E_w) + \sum_{j=1}^k \log \sigma(-U_{n_j}^\top E_w).$$

## Minimal PyTorch: lookup + one training step

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

V, d = 10_000, 128
E  = nn.Embedding(V, d)      # input embeddings
U  = nn.Embedding(V, d)      # output embeddings (for skip-gram-like training)
opt = torch.optim.Adam(list(E.parameters()) + list(U.parameters()), lr=1e-3)

# toy batch of center/context word ids
center = torch.tensor([12, 35, 999])          # [B]
context = torch.tensor([77,  5,  42])         # [B]

e = E(center)                 # [B, d]
u = U(context)                # [B, d]
scores = e @ U.weight.T       # [B, V] (naive full softmax; fine for tiny V)
loss = F.cross_entropy(scores, context)
opt.zero_grad()
loss.backward()
opt.step()
````

## Minimal cosine-similarity search

```python
# get top-5 neighbors for token t by cosine similarity
def topk_neighbors(t, k=5):
    W = E.weight / E.weight.norm(dim=1, keepdim=True)  # [V, d], L2-normalized
    v = W[t]                                           # [d]
    sims = (W @ v)                                     # [V]
    vals, idx = sims.topk(k+1)                         # +1 to skip itself
    return [(int(i), float(s)) for s, i in zip(vals[1:], idx[1:])]

print(topk_neighbors(12))
```

## Practical notes (quick, but meaningful)

* **Tokenization matters.** Subword tokenizers (BPE/WordPiece/Unigram) reduce OOV and let rare words share pieces; embeddings attach to subwords.
* **Regularization.** Norm clipping or weight decay can prevent runaway norms; sometimes we L2-normalize embeddings to focus on angles.
* **Tying weights.** In LMs, tie the input and output embeddings ($E \equiv U$) to save params and often help generalization.
* **Bias in geometry.** Embeddings reflect corpus statistics; measure and mitigate (e.g., projection away from bias subspaces) if needed.

---

> **Quick check (your turn):**
When training with the softmax loss above, what geometric pressure is applied to $E_w$ relative to $U_c$ and the other $U_{c'}$? Say it in one sentence.

---

# 13 LLM Context Length Extension (how to make models read longer)

## The core tension

Transformers learn a *positional geometry* during training. If you push them to longer sequences than they saw, the geometry (angles/frequencies of positions) drifts, so attention scores become miscalibrated. Extending context is about **keeping the learned geometry stable** while letting absolute positions grow.

We’ll focus on three practical levers:

1. **Scale positions at inference** (no extra training)
2. **Adjust the positional basis** (slightly change RoPE frequencies)
3. **Continue pretraining on long sequences** (teach the model the new regime)

---

## A. Position Interpolation (PI): compress positions to what the model knows

With Rotary Positional Embeddings (RoPE), each head/channel pair gets a frequency $ \omega_m $ and position $ p $ injects a rotation $ R(\omega_m p) $. If training max length is $ L_{\text{train}} $ and you want $ L_{\text{new}} $, define a scale

$$
s = \frac{L_{\text{train}}}{L_{\text{new}}} \in (0,1).
$$

Then **pretend** position $ p $ is $ p' = s p $ whenever you apply RoPE. You keep the relative phases similar to the training regime, so the attention kernel stays familiar.

* Upside: zero-shot, no retraining.
* Downside: very long ranges compress a lot; fine-grained local distinctions can blur.

### Tiny PyTorch (drop-in scale for RoPE)

```python
import torch

def rope_rotate(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D/2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,T,D/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y

def rope_cache(T, D, base=10_000.0, scale=1.0, device="cpu"):
    # scale<1 compresses positions: use p' = scale * p
    pos = torch.arange(T, device=device) * scale           # [T]
    half = D // 2
    freqs = base ** (-torch.arange(0, half, device=device) / half)  # [D/2]
    angles = pos[:, None] * freqs[None, :]                 # [T, D/2]
    return torch.cos(angles), torch.sin(angles)
````

Use:

```python
cos, sin = rope_cache(T=8192, D=head_dim, base=10_000.0, scale=4096/8192)
q = rope_rotate(q, cos, sin)
k = rope_rotate(k, cos, sin)
```

---

## B. Frequency (base) adjustment: keep local detail while growing range

RoPE uses a geometric bank of frequencies $ \omega_m = \text{base}^{-m/(D/2)} $. Larger **base** → **slower** frequencies → can represent longer periods. Instead of compressing positions, you can **lower frequencies** by increasing the base (or equivalently, multiply the angles by a factor $ < 1 $). Intuition: preserve local angles while making rotations change more slowly over distance.

A simple, safe recipe:

* Choose a target scale $ s = L_{\text{train}}/L_{\text{new}} $.
* **Either** scale positions ($ p' = s p $, as above), **or** scale angles ($ \theta' = s\theta $). Scaling angles is equivalent to increasing the base so that $ \omega'_m = s \omega_m $.

Pros/cons mirror PI: better large-range behavior with less local compression if you carefully tune $ s $; still zero-shot and simple.

> Practical tip: Combine *light* angle scaling with *light* position scaling to trade off local fidelity and global reach.

---

## C. Continued pretraining on long sequences (“teach the model the new world”)

Zero-shot tricks plateau. The robust path is: resume training with long sequences and a **length curriculum** (e.g., 2k → 8k → 32k). Two key details:

1. **Mixture of lengths.** Keep a fraction of short sequences so the model doesn’t forget fine local structure.
2. **Document packing with natural boundaries.** Long batches should maintain document coherence (avoid random concatenations that create “fake” long-range dependencies).

Loss stays standard LM cross-entropy:

$$
\mathcal{L} = -\sum_{t} \log p_\theta(x_t \mid x_{<t}),
$$

but the **data loader** ensures long contexts are frequent enough to shift the inductive bias.

### Microscopic loader sketch

```python
# Pseudocode: sample length buckets and pack without crossing doc boundaries
def sample_batch(docs, target_len):
    batch = []
    while len(batch_tokens(batch)) < B * target_len:
        doc = pick_doc_with_len>=target_len()
        batch.append(doc[:target_len])
    return pad_stack(batch)
```

---

## D. Architectural helps (to go *really* long)

* **Sliding-window (local) attention**: each token attends only to a band of width $ W $; complexity $ O(TW) $. Add **dilated**/**global** tokens to keep long-distance routes.
* **Attention sinks / global anchors**: dedicate a few positions with global receptive field; others attend locally plus these sinks, creating sparse long-range highways.
* **Segmented caching**: keep KV states for segments; bridge with sparse cross-segment attention.

### Minimal sliding mask

```python
def band_mask(T, W, device="cpu"):
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    M = (j <= i) & (j >= i - W)  # causal + window W
    return (~M).float().masked_fill(~M, float('-inf'))  # add to attention logits
```

---

## E. Sanity checks (you *must* measure)

* **Local faithfulness**: perplexity on short sequences should not degrade.
* **Long-range retrieval**: needle-in-a-haystack/Passkey tasks at multiple offsets (e.g., positions 1k, 8k, 32k).
* **Temporal drift**: plot cosine similarity of RoPE phases before/after scaling for the frequency bands—large band distortions predict failures.

---

## Mental model (one-liner)

RoPE gives each channel a rotating phasor; long-context tricks keep the *relative rotation* between any two positions close to what the model learned, either by **compressing positions**, **slowing rotations**, or **retraining** to accept new rotations.

---

> **Your turn (quick check):**
If you scale positions by $ s = L_{\text{train}}/L_{\text{new}} $, what happens to the *relative* angle between two positions $ p $ and $ q $ under RoPE, and why does that help zero-shot extrapolation?

---

# 14 FlashAttention (and a tiny PyTorch sketch)

## What problem it solves

Vanilla attention forms $A=\mathrm{softmax}(QK^\top/\sqrt{d})$ then multiplies $(AV)$. Materializing $A\in\mathbb{R}^{T\times T}$ costs $O(T^2)$ memory and bandwidth. **FlashAttention** avoids ever forming $A$. It streams over blocks of keys/values, doing the softmax **online** with exact math (not an approximation), maximizing use of on-chip SRAM and minimizing HBM traffic.

## The key math trick: online softmax

For a query row $i$, let logits over a block $B$ be $z_j = (q_i^\top k_j)/\sqrt{d}$.
Maintain per-row running statistics across blocks:

* running max $m = \max_j z_j$
* running normalizer $l = \sum_j e^{z_j - m}$
* running weighted sum $u = \sum_j e^{z_j - m} v_j$

When a new block arrives with block-max $m_b$ and block-sum $l_b=\sum_{j\in B} e^{z_j-m_b}$ and block-weighted $u_b=\sum_{j\in B} e^{z_j-m_b} v_j$,  
update **exactly**:
$$
m'=\max(m, m_b),\quad
l' = e^{m - m'}\,l + e^{m_b - m'}\,l_b,\quad
u' = e^{m - m'}\,u + e^{m_b - m'}\,u_b.
$$
After all blocks, the output row is $o_i = u'/l'$. This equals the full softmax result but never stores $T\times T$.

Causal masks are handled by restricting each block to keys $j\le i$.

## Minimal PyTorch: use fused op (best) or a tiny streamed sketch

### (A) Use PyTorch’s fused attention (fastest, minimal)

```python
import torch
import torch.nn.functional as F

B, H, T, D = 2, 8, 4096, 64
q = torch.randn(B, H, T, D, device="cuda")
k = torch.randn(B, H, T, D, device="cuda")
v = torch.randn(B, H, T, D, device="cuda")

# PyTorch will dispatch to a FlashAttention-style kernel on supported GPUs
o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)  # [B,H,T,D]
````

### (B) Educational micro-stream (CPU, tiny T) — shows the online softmax idea

```python
import torch
def flashlike_row(q_i, K, V, block=128, causal_end=None):
    # q_i: [D], K,V: [T,D]
    T = K.size(0)
    if causal_end is None: causal_end = T
    m = torch.tensor(-float('inf'))
    l = torch.tensor(0.0)
    u = torch.zeros_like(V[0])
    for s in range(0, causal_end, block):
        e = min(causal_end, s+block)
        z = (K[s:e] @ q_i) / (q_i.numel()**0.5)              # [e-s]
        m_b = z.max()
        l_b = torch.exp(z - m_b).sum()
        u_b = (torch.exp(z - m_b).unsqueeze(1) * V[s:e]).sum(0)
        m_new = torch.max(m, m_b)
        l = torch.exp(m - m_new)*l + torch.exp(m_b - m_new)*l_b
        u = torch.exp(m - m_new)*u + torch.exp(m_b - m_new)*u_b
        m = m_new
    return u / l

# tiny demo
T, D = 256, 32
q = torch.randn(T, D)
k = torch.randn(T, D)
v = torch.randn(T, D)
i = 200
out_i = flashlike_row(q[i], k, v, block=64, causal_end=i+1)  # causal
```

## Why it’s faster

* **I/O awareness:** fewer reads/writes to HBM; work is organized in tiles that fit SRAM.
* **Exactness:** it’s not a low-rank approximation; numerically identical (modulo fp errors) to full softmax attention.
* **Kernel fusion:** softmax, scaling, dropout, and matmuls are fused to reduce memory traffic.

## Shapes & complexity

Time remains $O(T^2 d)$ like standard attention, but **memory traffic** drops from $O(T^2)$ to $O(Td)$, which is the bottleneck on GPUs—hence big speedups and the ability to run long sequences.

---

> **Quick check (your turn):**
In the online softmax update, why do we keep a running **max** $m$ and rescale old/new sums by $e^{m - m'}$ and $e^{m_b - m'}$? State the reason in one sentence.

---

# 15 Multi-Query Attention (MQA) + tiny PyTorch

## Why MQA exists

At **decode time**, we cache keys/values for all past tokens. In standard multi-head attention (MHA) we have $H$ separate $(K,V)$ sets—so KV cache scales like $O(H,T,d_h)$. **MQA** keeps **per-head queries** but uses a **single shared** $(K,V)$ (or a few groups → **GQA**). Cache then scales like $O(T,d_k)$ (or $O(G,T,d_k)$ for $G$ groups), slashing memory and bandwidth.

## The math in one place

Given input $X\in\mathbb{R}^{T\times d_{\text{model}}}$:

* Queries remain per-head:
  $$
  Q_h = X W^Q_h,\quad Q_h\in\mathbb{R}^{T\times d_h},\quad h=1..H.
  $$
* **Shared** keys/values:
  $$
  K = X W^K,\quad V = X W^V,\quad K,V\in\mathbb{R}^{T\times d_h}.
  $$
  Attention per head:
  $$
  \text{Attn}_h=\mathrm{softmax}\!\left(\frac{Q_h K^\top}{\sqrt{d_h}} + M\right)V.
  $$
  Then concatenate $[\text{Attn}_1|\cdots|\text{Attn}_H]W^O$.

GQA is the middle ground: partition heads into $G$ groups; each group shares its own $(K,V)$.

## What changes (and what might you lose)

* **Big win:** KV cache and memory traffic shrink by $\approx H\times$ (MQA) or $\approx H/G$ (GQA). This is crucial for long contexts and fast serving.
* **Trade-off:** With shared $(K,V)$, heads can’t specialize via different key/value subspaces. In practice, GQA (e.g., 8–16 groups) often preserves quality while still saving a lot.

## KV cache scale (decode)

* MHA: cache $\approx 2 \cdot T \cdot H \cdot d_h$ floats (keys + values).
* MQA: cache $\approx 2 \cdot T \cdot d_h$ (independent of $H$).
* GQA: $\approx 2 \cdot T \cdot G \cdot d_h$.

## Minimal PyTorch module (MQA block only)

```python
import torch, torch.nn as nn
import torch.nn.functional as F

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads

        # Per-head Q: implement as single linear then reshape
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        # Shared K,V (single head dims)
        self.Wk = nn.Linear(d_model, self.dh, bias=False)
        self.Wv = nn.Linear(d_model, self.dh, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None, kv_cache=None):
        # x: [B,T,d_model]
        B, T, _ = x.shape
        q = self.Wq(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B,H,T,Dh]
        # Shared K,V
        k = self.Wk(x)                                               # [B,T,Dh]
        v = self.Wv(x)                                               # [B,T,Dh]

        # For decoding with cache:
        if kv_cache is not None:
            # kv_cache: dict with 'k','v' of shape [B, Tc, Dh]
            k = torch.cat([kv_cache['k'], k], dim=1)
            v = torch.cat([kv_cache['v'], v], dim=1)
            kv_cache['k'], kv_cache['v'] = k, v  # updated in-place

        # Expand shared K,V across heads *without* copying data (broadcast)
        k_exp = k.unsqueeze(1)                                       # [B,1, T',Dh]
        v_exp = v.unsqueeze(1)                                       # [B,1, T',Dh]

        # Scaled dot-product attention (causal mask optional)
        # scores: [B,H,T,T'] using broadcasted K
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) / (self.dh ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask  # e.g., causal: -inf above diagonal
        p = F.softmax(scores, dim=-1)
        o = torch.matmul(p, v_exp)                                   # [B,H,T,Dh]

        o = o.transpose(1, 2).contiguous().view(B, T, self.d)        # [B,T,d_model]
        return self.Wo(o), {'k': k, 'v': v}
````

**Notes**

* The broadcasted $(K,V)$ avoid per-head duplication.
* To make it **GQA**, create $G$ sets of $(W^K_g, W^V_g)$, assign $H/G$ heads per group, and broadcast within each group.

## Where MQA shines

* **Throughput & latency at inference** (fewer bytes moved for KV).
* **Long context** (KV cache fits memory).
* **Speculative/streaming decoding** benefits compound due to less KV traffic.

## When to prefer GQA

If you observe small quality drops with full MQA on difficult tasks (e.g., multi-hop reasoning), try $G\in{4,8,16}$. It often recovers quality while keeping big savings.

---

> **Quick check (your turn):**
Suppose $H=16$, $d_h=64$, context length $T$. Roughly how many floats are in the KV cache for MHA vs. MQA vs. GQA with $G=8$? Give the expressions (ignore batch/blocks), then say in a phrase which saves the most and why.

---

# 16 Rotary Positional Embedding (RoPE) + tiny PyTorch

## Idea in one line

Instead of adding a position vector, **rotate** each query/key channel pair by a position-dependent angle. This bakes **relative** offsets into the dot product, so attention inherently knows “how far apart” two tokens are.

## The math (compact but clear)

Split each head vector into even–odd pairs: for $x\in\mathbb{R}^{d}$,
$$
x = \big[(x^{(1)}_1,x^{(2)}_1),\dots,(x^{(1)}_{d/2},x^{(2)}_{d/2})\big].
$$

For position $p$ and frequency bank $\{\omega_m\}_{m=1}^{d/2}$,  
rotate each 2D pair by angle $\theta_m(p)=\omega_m p$:
$$
\mathrm{RoPE}_p(x)_m =
\begin{bmatrix}
\cos\theta_m(p) & -\sin\theta_m(p)\\
\sin\theta_m(p) & \ \cos\theta_m(p)
\end{bmatrix}
\begin{bmatrix}
x^{(1)}_m\\ x^{(2)}_m
\end{bmatrix}.
$$

Apply to **Q and K**:
$$
\tilde q_p=\mathrm{RoPE}_p(q),\quad \tilde k_t=\mathrm{RoPE}_t(k).
$$

Key property (why it encodes relative positions):
$$
\tilde q_p^\top \tilde k_t
= q^\top R(\theta(p))^\top R(\theta(t))\,k
= q^\top R\big(\theta(t)-\theta(p)\big)k,
$$
so scores depend on **$(t-p)$** (relative), not absolute positions. (Here $R$ is the block-diagonal rotation.)

Frequency schedule (geometric):
$$
\omega_m = \text{base}^{-\frac{m-1}{d/2}},\quad \text{base}\approx 10{,}000.
$$

High-index pairs rotate slowly → capture long-range; low-index rotate fast → capture local detail.

## Minimal PyTorch (drop-in for a head)

```python
import torch

def rope_cache(T, Dh, base=10_000.0, device="cpu", scale=1.0):
    # scale<1.0 can be used for context extension (position interpolation)
    pos = torch.arange(T, device=device) * scale              # [T]
    half = Dh // 2
    freqs = base ** (-torch.arange(0, half, device=device) / half)  # [Dh/2]
    angles = pos[:, None] * freqs[None, :]                    # [T, Dh/2]
    return torch.cos(angles), torch.sin(angles)               # [T, Dh/2] each

def apply_rope(x, cos, sin):
    # x: [B, H, T, Dh]; cos/sin: [T, Dh/2]
    x1, x2 = x[..., ::2], x[..., 1::2]                        # pair channels
    cos = cos[None, None, ...]                                 # [1,1,T,Dh/2]
    sin = sin[None, None, ...]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)             # interleave back
    return y

# usage inside attention
# q,k: [B,H,T,Dh]
cos, sin = rope_cache(T=q.size(2), Dh=q.size(-1), base=10_000.0)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
````

## Practical notes that matter

* **Q/K only.** Do not rotate $V$; the relative-position effect lives in $(qk^\top)$.
* **Causal attention unchanged.** RoPE is orthogonal to masking; combine as usual.
* **Context extension hooks.** To run beyond training length $L$, use `scale = L/L_new` in `rope_cache` (compress positions) or scale angles—both keep relative geometry stable for zero-shot extrapolation.
* **Numerics.** FP16 works; BF16 preferred for very long sequences due to angle precision.

## Intuition (narrative)

Think of each even–odd channel as a tiny complex number. RoPE multiplies it by $e^{i\omega_m p}$. When Q at position $p$ meets K at position $t$, their phases subtract like $e^{i\omega_m (t-p)}$. The model can thus “feel” how far a key is from its query across multiple frequency bands, the way music hears both bass (slow rotations) and treble (fast rotations).

---

> **Quick check (your turn):**
Why does applying the *same* rotation bank to Q and K yield relative-position awareness in the dot product, whereas adding positional vectors would mostly encode absolute positions?

---

# 17 Information Retrieval (IR) fundamentals

## The retrieval loop (bird’s-eye)

Query in → represent query & corpus → score each candidate → rank → evaluate. Most of IR is about *representations* and *scoring functions* that trade speed vs. relevance.

---

## Classic lexical IR (exact/weighted matches)

### Bag-of-words weighting

Let $tf_{t,d}$ be term $t$’s count in doc $d$, $df_t$ its document frequency, and $N$ corpus size.

* **tf-idf (cosine):**
  $$
  w_{t,d} = (1+\log tf_{t,d})\cdot \log\frac{N}{df_t+1},\quad
  \text{score}(q,d) = \frac{\sum_{t\in q} w_{t,q}w_{t,d}}{|w_q|\,|w_d|}.
  $$

* **BM25 (strong lexical baseline):**
  $$
  \mathrm{BM25}(q,d)=\sum_{t\in q} idf(t)\cdot
  \frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1\!\left(1-b+b\frac{|d|}{\text{avgdl}}\right)},
  \quad
  idf(t)=\log\frac{N-df_t+0.5}{df_t+0.5}.
  $$
  Typical $k_1\in[1.2,2.0]$, $b\in[0.6,0.8]$.
  Narrative: BM25 rewards repeated on-topic terms but saturates; normalizes by doc length so long docs don’t dominate.

Strengths: fast inverted indexes, exact term control, interpretable.  
Limitations: struggles with synonyms/paraphrases (“car” vs “automobile”).

---

## Neural IR (semantic matches)

### Dual-encoder (bi-encoder)

Encode query and doc separately:
$$
\mathbf{q}=f_\theta(q),\quad \mathbf{d}=g_\theta(d),\quad
\text{score}(q,d)=\mathbf{q}^\top \mathbf{d}\ \ \text{(often cosine)}.
$$
Train with in-batch negatives / hard negatives, loss:
$$
\mathcal{L}=-\log\frac{\exp(\mathbf{q}^\top \mathbf{d}^+/\tau)}{\sum_{d^-}\exp(\mathbf{q}^\top \mathbf{d}^-/\tau)}.
$$
Narrative: fast ANN search at scale; meaning-based matches emerge.

### Cross-encoder (re-ranker)

Concatenate “[CLS] q [SEP] d” and score with a transformer; highest accuracy, but $O(k)$ forward passes for top-$k$ candidates. Use after a fast first stage (lexical or dual-encoder).

### Hybrid

$\text{score}=\alpha\,\text{BM25}+(1-\alpha)\,\mathbf{q}^\top\mathbf{d}$. Often wins in practice: lexical precision + semantic recall.

---

## Indexing & search at scale (conceptual preview)

* **Inverted index** (lexical): term → postings list of (docID, freq, positions). Supports BM25 in sub-linear time.
* **Vector index** (ANN): product quantization, IVF, HNSW graphs to approximate nearest neighbors in $O(\log N)$ or $O(N^\rho)$ with tiny $\rho$.
* **Two-stage retrieval**: (1) retrieve $k\!\sim\!1000$ fast; (2) re-rank $k$ with cross-encoder or task-specific scorer.
* **Sharding & caching**: shard by docID or semantic partition; cache hot queries and neighbors.

(We’ll implement a minimal PyTorch vector index in Section 8.)

---

## Query understanding (bridge to Section 9)

* **Normalization:** tokenization, case-folding, stopword handling, stemming/lemmatization (for lexical pipelines).
* **Expansion/rewrite:** pseudo-relevance feedback (Rocchio), thesaurus/embedding synonyms, intent classification.
* **Disambiguation:** entity linking (“Apple”→company vs fruit) improves retrieval precision.

---

## Evaluation (don’t skip!)

Let $R_q$ be relevant set for query $q$, and $\pi_q@k$ the top-$k$ ranked list.

* **Precision@k:** $\lvert\pi_q@k \cap R_q\rvert/k$.
* **Recall@k:** $\lvert\pi_q@k \cap R_q\rvert/\lvert R_q\rvert$.
* **MRR:** $\frac{1}{|Q|}\sum_q 1/\text{rank}_q(\text{first relevant})$.
* **nDCG@k:** discounted gain with graded relevance:
  $$
  \mathrm{DCG}@k=\sum_{i=1}^k \frac{2^{rel_i}-1}{\log_2(i+1)},\quad
  \mathrm{nDCG}@k=\frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}.
  $$
  Narrative: use recall@k for first-stage retrievers (coverage), nDCG/MRR for ranking quality.

**Stat testing:** paired randomization/bootstrap across queries; report confidence intervals, not just means.

---

## Failure modes & fixes

* **Lexical miss (synonyms):** add neural/hybrid, query expansion.
* **Semantic drift (off-topic but similar):** add lexical filters or exact-match constraints.
* **Shortcut bias in training:** mine *hard negatives* (BM25-high but irrelevant) to sharpen the dual-encoder.
* **Long docs overwhelm:** split into passages; aggregate scores by max/mean.

---

## Mental model

Start **broad** with a fast, recall-oriented retriever (BM25 or dual-encoder + ANN). Then **focus** with a precise re-ranker (cross-encoder) and task constraints. Evaluate with query-level stats and proper tests.

---

> **Quick check (your turn):***  
You’re designing a two-stage retriever for technical Q&A. Name a **lexical** first stage and a **neural** second stage you’d pick, and in one sentence explain why that pairing balances recall and precision.

---

# 18 Indexing (IR) + tiny PyTorch

## Why indexing?

Retrieval = scoring many items quickly. Indexes trade a bit of *precompute + structure* for large *query-time* speedups.

We’ll do two minimal paths:

1. a **lexical inverted index** for BM25, and
2. a **vector (ANN-ish) index** in PyTorch: flat IP + a tiny IVF sketch.

---

## A) Lexical inverted index (BM25-ready)

### Idea (narrative)

Token → a **postings list** of documents (and term frequencies, maybe positions). At query time, touch only postings for query terms, accumulate BM25 scores.

### Math (BM25 recap)

$$
\text{BM25}(q,d)
= \sum_{t\in q}
\underbrace{\log\frac{N-df_t+0.5}{df_t+0.5}}_{idf(t)}
\cdot
\frac{tf_{t,d}(k_1+1)}{tf_{t,d}+k_1\!\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}.
$$

### Minimal code (pure Python; tiny & readable)

```python
from collections import defaultdict, Counter
import math

class InvertedIndex:
    def __init__(self, k1=1.5, b=0.75):
        self.postings = defaultdict(list)  # term -> [(docid, tf)]
        self.doclen = {}                   # docid -> |d|
        self.N = 0
        self.k1, self.b = k1, b
        self.df = defaultdict(int)
        self.avgdl = 0.0

    def add(self, docid, tokens):
        tf = Counter(tokens)
        L = sum(tf.values())
        self.doclen[docid] = L
        for t, c in tf.items():
            self.postings[t].append((docid, c))
        self.N += 1

    def finalize(self):
        self.df = {t: len(self.postings[t]) for t in self.postings}
        self.avgdl = sum(self.doclen.values()) / max(1, self.N)

    def bm25(self, query_tokens, topk=10):
        scores = defaultdict(float)
        for t in query_tokens:
            if t not in self.postings: 
                continue
            df = self.df[t]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1e-12)
            for docid, tf in self.postings[t]:
                L = self.doclen[docid]
                denom = tf + self.k1 * (1 - self.b + self.b * L / self.avgdl)
                scores[docid] += idf * (tf * (self.k1 + 1)) / denom
        # rank
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]

# tiny demo
docs = {
  0: "fast exact attention kernel",
  1: "approximate nearest neighbor search",
  2: "attention is all you need transformer",
}
tok = lambda s: s.lower().split()

idx = InvertedIndex()
for i, d in docs.items(): idx.add(i, tok(d))
idx.finalize()
print(idx.bm25(tok("attention kernel"), topk=5))
````

**Why this works:** Scoring touches only the postings of the query terms → sub-linear in corpus size for sparse queries.

---

## B) Vector indexing (semantic retrieval)

### Idea (narrative)

Encode docs to vectors $\mathbf{d}\in\mathbb{R}^D$, encode query to $\mathbf{q}$, use **inner product** (or cosine) and return top-$k$.

* **Flat** (exact): one matrix multiply $\mathbf{Q}\mathbf{D}^\top$.
* **ANN**: first route the query to a handful of **coarse clusters** (IVF), then search only those lists.

### Flat (exact) PyTorch index

```python
import torch

class FlatIPIndex:
    def __init__(self, dim, device="cpu", normalize=False):
        self.D = torch.empty(0, dim, device=device)
        self.normalize = normalize

    def add(self, X):
        X = X if not self.normalize else X / (X.norm(dim=1, keepdim=True) + 1e-12)
        self.D = torch.cat([self.D, X], dim=0)

    def search(self, q, k=10):
        q = q if not self.normalize else q / (q.norm(dim=1, keepdim=True) + 1e-12)
        scores = q @ self.D.T                      # [B, N]
        vals, idx = torch.topk(scores, k, dim=1)   # exact top-k
        return vals, idx

# tiny demo
torch.manual_seed(0)
dim = 64
index = FlatIPIndex(dim, device="cpu", normalize=True)
docs = torch.randn(1000, dim)
index.add(docs)
q = torch.randn(2, dim)
vals, idx = index.search(q, k=5)
print(idx)
```

**Notes**

* If you L2-normalize both sides, inner product = cosine similarity.
* Complexity: one GEMM, good on GPU for tens of thousands to millions (with batching).

---

### Tiny IVF (coarse quantizer) in pure PyTorch

**Goal:** reduce search to a few lists. We’ll:

1. learn $C$ coarse centroids by k-means (few iters),
2. assign each doc to its nearest centroid (build lists),
3. at query time, probe the top $n_{\text{probe}}$ centroids and search only those docs (flat within lists).

```python
import torch

def kmeans(X, C=64, iters=10):
    # X: [N, D]
    N, D = X.shape
    cent = X[torch.randperm(N)[:C]].clone()            # init by sampling
    for _ in range(iters):
        # assign
        dist = torch.cdist(X, cent)                    # [N, C]
        assign = dist.argmin(dim=1)                    # [N]
        # update
        for c in range(C):
            mask = (assign == c)
            if mask.any():
                cent[c] = X[mask].mean(dim=0)
    return cent, assign

class IVFIndex:
    def __init__(self, dim, C=64, device="cpu", normalize=True):
        self.dim, self.C, self.device = dim, C, device
        self.normalize = normalize
        self.centroids = None
        self.lists = [[] for _ in range(C)]
        self.store = []  # raw vectors (torch tensors)

    def build(self, X, iters=10):
        X = X.to(self.device)
        if self.normalize:
            X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
        self.centroids, assign = kmeans(X, C=self.C, iters=iters)
        self.store = [None] * len(X)
        for i in range(self.C): self.lists[i] = []
        for i, c in enumerate(assign.tolist()):
            self.lists[c].append(i)
            self.store[i] = X[i]

    def search(self, q, k=10, nprobe=8):
        if self.normalize:
            q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
        # 1) choose nprobe centroids
        d2c = torch.cdist(q, self.centroids)                # [B, C]
        probes = d2c.topk(nprobe, largest=False).indices    # [B, nprobe]
        all_scores, all_ids = [], []
        for b in range(q.size(0)):
            cand_ids = sum([self.lists[c] for c in probes[b].tolist()], [])
            if len(cand_ids) == 0:
                all_scores.append(torch.empty(0))
                all_ids.append(torch.empty(0, dtype=torch.long))
                continue
            Xcand = torch.stack([self.store[i] for i in cand_ids], dim=0)  # [M,D]
            s = q[b:b+1] @ Xcand.T                                         # [1,M]
            vals, idx = torch.topk(s, min(k, s.size(1)), dim=1)
            all_scores.append(vals.squeeze(0))
            all_ids.append(torch.tensor([cand_ids[j] for j in idx.squeeze(0)], dtype=torch.long))
        return all_scores, all_ids

# tiny demo
torch.manual_seed(0)
X = torch.randn(5000, 64)
idx = IVFIndex(64, C=64, device="cpu", normalize=True)
idx.build(X, iters=8)
q = torch.randn(2, 64)
scores, ids = idx.search(q, k=5, nprobe=8)
print(ids[0])
```

**Why this helps:** most of the corpus sits in **non-probed** lists, so each query scores only a small subset. Quality ↑ with larger `nprobe`, speed ↑ with smaller.

**Caveats**

* This IVF is intentionally minimal: no residuals (IVF-PQ), no re-ranking buffer, no GPU tiling—use Faiss for production.
* Rebuild/merge logic omitted; in practice you support incremental adds by assigning new vectors to nearest centroid on the fly.

---

## Practical glue

* **Chunking docs:** long docs → fixed-size passages (e.g., 256–512 tokens) with overlap; index passages; aggregate at doc level on output.
* **Metadata filters:** keep small inverted maps (e.g., year → docIDs) to quickly prune candidates *before* vector search.
* **Hybrid search:** compute BM25@K and ANN@K, take union, re-rank with a cross-encoder (Section 7).
* **Caching:** cache top-k for frequent queries and nearest centroid probes.

---

## Debugging & evaluation

* **Recall@k vs. exact flat:** compare IVF to FlatIP on a held-out set to choose `C` and `nprobe`.
* **Latency budget:** measure breakdown (encode q, route, score, Python overhead). Batch queries to amortize matmuls.
* **Drift checks:** if encoders update, **reindex** or store the projection epoch with vectors.

---

## Mental model

Lexical inverted indexes excel at precision with exact terms. Vector indexes give semantic reach. IVF says: “route first, then score locally,” making semantic search scale.

---

> **Quick check (your turn):**
If your flat index returns perfect top-k but is too slow, how would you tune an IVF index (which knobs, and in which direction) to approach the same recall while meeting a 5× latency target?

---

# 19 Query Understanding (IR)

## What “query understanding” really means

Turn a user’s raw string into a **search intent** and a **machine-usable query** that retrieves the right evidence. It’s a pipeline of light NLP + IR math + a bit of semantics:

1. **Normalize** the surface form.
2. **Interpret** structure (entities, operators, time, units, negation).
3. **Expand/clarify** to bridge vocab gaps.
4. **Rewrite** into the retrieval system’s dialect (BM25 terms, ANN vector, graph/Cypher, SQL).
5. **Disambiguate** when the query is underspecified (possibly interactively).

---

## 1) Normalization (make tokens comparable)

* **Case folding, Unicode normalize (NFKC).**
* **De-noise:** strip punctuation where safe, collapse whitespace; keep symbols (e.g., “C++”) if your domain needs them.
* **Stemming/lemmatization:** helps lexical recall; lemmatization is gentler for technical text.
* **Stopwords:** drop for BM25; keep for phrase/proximity models.

> Narrative: we reduce avoidable lexical variance without hurting meaning.

---

## 2) Structural interpretation (turn words into *meaning*)

* **Entities & types:** link “AlphaChip”, “Siddharth Garg” to canonical IDs; attach types (Paper, Person). Precision here massively helps ranking.
* **Operators:** detect explicit Boolean (`AND/OR/NOT`), **negation** (“not tensorflow”), **proximity** (`"flash attention"~3`), **field restrictors** (`title:`, `author:`).
* **Temporal intent:** “since 2023”, “last quarter” → explicit ranges.
* **Quantities & units:** “< 10 ms latency” → normalize numbers and convert units.
* **Comparatives/superlatives:** “best”, “cheapest” → rewrite as sorts/filters.
* **Task intent classification:** navigational (homepage?), informational, transactional; routes to different retrieval stacks or verticals.

---

## 3) Expansion (bridge vocabulary gaps—carefully)

### A) Pseudo-relevance feedback (Rocchio, vector-space view)

Start from initial retrieval, take top $D$ docs, compute their centroid, and move the query toward it:

$$
\vec q' = \alpha \vec q + \frac{\beta}{|D|}\sum_{d\in D}\vec d - \frac{\gamma}{|N|}\sum_{n\in N}\vec n.
$$

In bag-of-words, this becomes new term weights; keep top $K$ added terms to avoid drift.

### B) RM3 (query likelihood view)

Estimate a relevance model $p(t\mid R)$ from top docs, then interpolate:

$$
p(t\mid q') = (1-\lambda)\,p(t\mid q) + \lambda\,p(t\mid R).
$$

Great lexical gains; control $\lambda$ and prune to high-IDF terms.

### C) Embedding/synonym expansion

Nearest neighbors in embedding space (or WordNet/thesauri). Use **constrained** expansion: keep terms sharing entity/type or appearing in same field.

> Guardrail: expansion boosts recall but can drift off-topic—pair with **field filters** or **phrase constraints**.

---

## 4) Rewriting to your backends

### For BM25 / lexical

* Build a Boolean query: required terms (`+term`), optional boosts (`term^w`), phrases (`"flash attention"`), fields (`title:flash`), filters (year:2023-2025).

### For ANN / dual-encoder

* Create **one or more** semantic queries:

  * **Original** text.
  * **Clarified paraphrase** (expand acronyms, resolve pronouns).
  * **Entity-only** variant (drop function words).

* Search each; fuse by max or learned weights.

### For KG/SQL

* Semantic parse → typed slots (author, work, date).
* Emit a **safe** query (constrained grammar), e.g., Cypher or SQL with whitelisted predicates.

---

## 5) Disambiguation & interaction

* **Multi-intent detection:** short queries often bundle senses (“apple developer conference videos”). Branch into verticals; rank fusions.
* **Ask minimal clarifying Qs** when ambiguity is catastrophic (“Do you mean AlphaChip the 2021 paper or the 2023 system?”).
* **Session context:** inherit entities/time from conversation turns; decay context if the user pivots.

---

## 6) Language-modeling view of retrieval (useful lens)

Classic **Query Likelihood Model** with Dirichlet smoothing:

$$
\log p(q\mid d) = \sum_{t\in q} \log \frac{tf_{t,d} + \mu\, p(t\mid C)}{|d| + \mu},
$$

where $p(t\mid C)$ is corpus probability and $\mu$ (e.g., 1000–3000) controls smoothing.

Narrative: documents that could have *generated* the query get higher scores. This ties neatly into RM3 and also guides what expansions are probabilistically sensible.

---

## 7) Phrases, proximity, and structure matter

* **Exact phrases** (`"rotary positional embedding"`) fight semantic drift.
* **Proximity scoring**: add gains when query terms occur within window $w$.
* **Field boosts**: titles/abstracts carry stronger evidence than body text in scientific IR.

---

## 8) Robustness tricks you’ll actually use

* **Misspellings:** character n-grams or edit-distance candidates → then rank normally.
* **Acronyms:** learn bi-directional expansions from corpora (“HPWL” ↔ “half-perimeter wirelength”).
* **Negation handling:** rewrite `-tensorflow` as a **filter** in lexical; in neural, post-filter candidates containing negated terms.
* **Numeric ranges:** detect comparators ($<,>,=,\approx$) and units; normalize to canonical fields.

---

## 9) Evaluation & ablations (don’t skip)

* **Query-level** metrics: nDCG@k, MRR, Recall@k; report **paired** significance tests (bootstrap or randomization).
* **Ablate**: +entities, +phrases, +PRF, +embedding-expansion; measure drift errors.
* **Stratify** by query length and head/tail frequency—expansion often helps the tail.

---

## Failure modes → fixes

* **Over-expansion drift:** cap added terms $K$, raise IDF threshold, add phrase anchors.
* **Entity confusion:** require type agreement; use canonical IDs everywhere.
* **Neural off-topic:** hybrid filter: must match at least one high-IDF query term or entity.
* **Underspecified time/units:** ask a one-line clarification; default to conservative ranges with an explainable assumption.

---

## Mental model

Start from a **clean, typed** understanding of the query. Use **minimal expansion** to bridge vocabulary, but keep retrieval grounded with **phrases, fields, and entities**. Always rewrite into the concrete operators your backend understands.

---

> **Quick check (your turn):**
You run BM25 first, then a dual-encoder. For a short, ambiguous query like “flash attention speed,” name **one normalization**, **one structural interpretation**, and **one expansion** you’d apply before retrieval—and say in a phrase how each reduces error.

---

# 20 Ranking & Re-Ranking (IR)

## What “ranking” is really doing

You’re learning a scoring function $s_\theta(q,d)$ so that, **for a given query $q$**, relevant documents appear before non-relevant ones. Because users only see the **top of the list**, the objective should reflect **list quality**, not just individual scores.

Three lenses help:

1. **Pointwise:** regress $y_{qd}$ (click/relevance) from $s_\theta(q,d)$. Simple, but misaligned with ranks.
2. **Pairwise:** prefer $d^+$ over $d^-$: $s(q,d^+) > s(q,d^-)$.
3. **Listwise:** optimize a surrogate to list metrics like nDCG.

We’ll build intuition from pairwise → listwise → practical re-ranking.

---

## Pairwise view (clean geometry)

For a query $q$, sample a positive $d^+$ and a negative $d^-$. Train with a **margin** (hinge) or **logistic** loss:
$$
\mathcal{L}_{\text{hinge}}=\max\!\big(0,\ 1 - (s(q,d^+)-s(q,d^-))\big),\qquad
\mathcal{L}_{\text{logit}}=\log\!\big(1+\exp(-(s^+ - s^-))\big).
$$
Narrative: every update pushes $d^+$ up and $d^-$ down **relative to each other**, matching “who beats whom.”

**Tiny PyTorch (pairwise logistic)**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class BiEncoder(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.d = nn.Linear(d, d, bias=False)
    def encode_q(self, x): return self.q(x)
    def encode_d(self, x): return self.d(x)

def pairwise_loss(qe, de_pos, de_neg, tau=1.0):
    s_pos = (qe * de_pos).sum(-1) / tau
    s_neg = (qe * de_neg).sum(-1) / tau
    return F.softplus(-(s_pos - s_neg)).mean()  # log(1+e^{-(Δ)})

# toy step
model = BiEncoder(128)
opt = torch.optim.Adam(model.parameters(), 1e-3)
q = torch.randn(32,128); dpos = torch.randn(32,128); dneg = torch.randn(32,128)
loss = pairwise_loss(model.encode_q(q), model.encode_d(dpos), model.encode_d(dneg))
opt.zero_grad(); loss.backward(); opt.step()
````

---

## Listwise view (optimize the **list**, not pairs)

### nDCG (what we care about)

With graded relevance $\text{rel}*i \in {0,1,2,\dots}$ at rank $i$:
$$
\mathrm{DCG}@k = \sum*{i=1}^{k} \frac{2^{\text{rel}_i}-1}{\log_2(i+1)},\qquad
\mathrm{nDCG}@k = \frac{\mathrm{DCG}@k}{\mathrm{IDCG}@k}.
$$
Non-differentiable in ranks, so we use **smooth surrogates** or **LambdaRank**.

### ListNet / ListMLE (probability over permutations)

Turn scores $s_i$ into a **Plackett–Luce** distribution over permutations $\pi$:
$$
P(\pi\mid s) = \prod_{i=1}^{n} \frac{\exp(s_{\pi_i})}{\sum_{j=i}^{n} \exp(s_{\pi_j})}.
$$
Fit $P(\pi \mid s)$ to a **target** induced by relevance labels (e.g., ListMLE uses the single permutation sorted by labels). Loss = $-\log P(\pi^\star\mid s)$.

**Tiny PyTorch (ListMLE for one query)**

```python
def listmle_loss(scores, rel):
    # scores: [n], rel: [n] (higher is better)
    order = torch.argsort(rel, descending=True)
    s = scores[order]
    # -log Plackett–Luce likelihood
    # sum_i [ log(sum_{j>=i} exp(s_j)) - s_i ]
    logcumexp = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
    return (logcumexp - s).sum()

# example for a batch of queries with padding would loop queries separately
```

### LambdaRank intuition (nDCG-aware gradients)

Compute pairwise gradients $\lambda_{ij}$ proportional to the **change in nDCG** if $i$ and $j$ swap:
$$
\lambda_{ij} = -\sigma\cdot \frac{1}{1+\exp(s_i - s_j)} \cdot \big|\Delta\mathrm{nDCG}_{ij}\big| \cdot \operatorname{sign}(\text{rel}_i-\text{rel}*j).
$$
Then update scores with $\partial \mathcal{L}/\partial s_i = \sum_j \lambda*{ij}$.
Narrative: bigger rank mistakes near the top (large $\Delta$ nDCG) get bigger gradients.

---

## Re-ranking in practice (two-stage)

1. **Retriever** produces a candidate set $C_q$ (e.g., 200–1000 docs).
2. **Re-ranker** (more expressive, slower) scores only $C_q$ and returns top-$k$.

### Neural re-rankers

* **Cross-encoder:** encode concatenated $[q; d]$ and regress a relevance score; highest quality.
* **Interaction models:** kernel pooling / ColBERT (late interaction) balance speed vs fidelity.

**Tiny PyTorch (cross-encoder skeleton)**

```python
import torch, torch.nn as nn, torch.nn.functional as F
# stand-in MLP; plug in a transformer encoder in practice
class CrossEncoder(nn.Module):
    def __init__(self, d_in=768):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256,1))
    def forward(self, x):  # x: [B, L, d] pooled -> [B, d_in]
        pooled = x.mean(1)
        return self.mlp(pooled).squeeze(-1)  # [B]

# pairwise training on candidate pairs
def ce_pairwise_step(model, qd_pos, qd_neg):
    s_pos, s_neg = model(qd_pos), model(qd_neg)
    return F.softplus(-(s_pos - s_neg)).mean()
```

**Blending lexical + neural (hybrid re-rank)**
$$
\text{score} = \alpha\cdot \mathrm{BM25}(q,d) + (1-\alpha)\cdot s_\theta(q,d),
$$
tune $\alpha$ on a validation set; often improves robustness.

---

## Diversity & redundancy (not just the best one)

Users want a **useful slate**, not clones. A simple, effective approach is **MMR (Maximal Marginal Relevance)**:
$$
\text{MMR}(d) = \lambda,\text{rel}(d) - (1-\lambda)\max_{d'\in S}\ \text{sim}(d,d'),
$$
greedily build $S$ by picking the max MMR at each step.

**Tiny PyTorch (MMR with dot-product sim)**

```python
def mmr_select(scores, doc_embs, k=10, lam=0.7):
    # scores: [N], doc_embs: [N, D] (L2-normalized)
    selected, mask = [], torch.zeros_like(scores, dtype=torch.bool)
    for _ in range(k):
        if len(selected)==0:
            i = scores.argmax().item()
        else:
            S = torch.stack(selected)
            sim = doc_embs @ doc_embs[S].T             # [N, |S|]
            maxsim = sim.max(dim=1).values
            mmr = lam * scores - (1-lam) * maxsim
            mmr[mask] = -1e9
            i = mmr.argmax().item()
        selected.append(torch.tensor(i))
        mask[i]=True
    return torch.stack(selected).tolist()
```

---

## Counterfactual learning-to-rank (click bias is not labels)

Clicks are biased by position and presentation. Use **inverse propensity weighting** (IPW) or randomized interventions.

* **Position propensities** $p_k = \Pr(\text{examine rank }k)$ from randomized swaps.
* **Unbiased pairwise loss:**
  $$
  \mathcal{L} = \mathbb{E}\Big[ w_{i},w_{j},\log(1+\exp(-(s_i-s_j))) \Big],
  \quad w_i=\frac{\mathbf{1}*{\text{clicked }i}}{p*{\text{rank}(i)}}.
  $$
* **Self-normalized** variants (SNIPS) tame variance.

---

## Calibration, ties, and cutoffs

* **Score calibration:** map scores to probabilities via Platt scaling or isotonic regression on a labeled dev set—useful for thresholding and fusion.
* **Early cutoff:** if only top-10 matter, train with **top-k** objectives (e.g., weight pairs by whether either sits in top-k).
* **Ties:** break deterministically (docID) to get stable offline metrics.

---

## Evaluating re-rankers (what to actually report)

* First-stage **Recall@k** (coverage).
* Re-rank **MRR/nDCG@k** on a held-out labeled set.
* **Ablations:** pairwise vs listwise loss, +hybrid score, +MMR.
* **Latency budget:** report milliseconds per 1k candidates; ensure re-rank fits the p95 SLA.

---

## Mental model

Use a **fast, recall-oriented retriever** to form a candidate set; train a **pairwise/listwise re-ranker** whose gradients emphasize **top-positions**. Add **diversity (MMR)** to avoid redundancy, and de-bias with **propensity weighting** when training from clicks. Evaluate with **list metrics** and respect **latency**.

---

> **Quick check (your turn):**
You have a candidate set of 500 docs per query. You care about nDCG@10 and observe many near-duplicate answers at the top. In one or two sentences, which loss (pairwise or listwise) would you train your re-ranker with, and how would you incorporate diversity at inference without hurting relevance?

---

# 21 A/B experimentation & testing in **LLMs**

## What’s special vs generic ML A/B

LLMs are **stochastic generators** whose quality is judged **pairwise** and **subjectively**. That changes units, metrics, and stats:

* **Unit of randomization:** user, session, or **prompt** (task). Make it **sticky** per unit to avoid cross-contamination (memory, caches, learning).
* **Determinism:** fix decoding (e.g., `temperature=0.2`, `top_p=0.95`, seeded), or treat randomness as part of the policy and average over repeats.
* **Caching:** prompts and KV caches can leak speed advantages; measure **latency** and **token-cost** as guardrails.

---

## Core online designs

### A/B with blinded pairwise evaluation (gold standard)

For a sampled prompt \(x\), show users **one** variant (A or B) for impact metrics (engagement, task success). Separately, to estimate *quality*, run **blind side-by-side (SxS)** on a holdout audience or annotators:

* Present \((y_A, y_B)\) in **random order**; collect preference \(r \in \{-1,0,+1\}\).
* Estimate **win-rate** \(p = \Pr[B \succ A]\).

**Power (back-of-envelope):** with variance \(p(1-p)\), to detect \(\Delta = p-0.5\):

$$
n \approx \frac{(z_{1-\alpha/2}+z_{1-\beta})^2\, p(1-p)}{\Delta^2}.
$$

### Interleaving / dueling bandits (fast feedback)

For ranking/search/chat-suggestion, **interleave** items from A and B into one slate so each user interaction compares policies. Use team-draft interleaving; outcome = which policy’s items receive more credit. This reduces variance and speeds decisions.

### Multi-armed bandits for prompts/params

When exploring many prompts or decoding settings, use **Top-Two Thompson Sampling** or **Successive Halving** with *pairwise* rewards to prune quickly, then confirm with a locked A/B.

---

## Metrics that actually work

### Pairwise Bradley–Terry (BT) skill

Model preference of \(i\) over \(j\):

$$
\Pr[i \succ j] = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}.
$$

Fit \(\beta\) by logistic regression on pairwise outcomes; report \(\beta_B - \beta_A\) with CI. Handles ties by splitting or by Davidson extension.

### Judge reliability & bias

* **Blind** the variants and **shuffle** order.
* Track **inter-rater agreement** (Cohen’s \(\kappa\) / Krippendorff’s \(\alpha\)).
* Calibrate **LLM-as-judge** with a golden set; periodically **AA tests** to detect drift.

### Task KPIs

* **Exact/substring match**, **pass@k**, rubric scores (structured), **toxicity/safety** rates, **latency** (p95), **cost** (tokens). Treat safety as a **hard guardrail**.

---

## Offline evaluation before online

### Log-replay with counterfactuals

Given logged \((x_i, a_i, r_i)\) under policy \(b\), and candidate policy \(\pi\):

* **IPS:** \(\hat V = \frac1n \sum \frac{\pi(a_i \mid x_i)}{b(a_i \mid x_i)} r_i\).
* **Self-normalized IPS** for stability.
* **Doubly Robust**: add reward model \(\hat q(x,a)\) for lower variance.

For **ranking/slates**, use **slate IPS** (propensities for the whole slate, or position-wise with independence assumptions). Clip weights to control variance, then validate with small online canaries.

---

## Prompt & system evaluation pitfalls → fixes

* **Prompt leakage / contamination:** randomize *instructions* and *few-shot exemplars* across arms, or share a frozen template.
* **Context effects:** prior turns change difficulty; bucket by **conversation state**.
* **Length confound:** longer answers can look “better.” Normalize by length or include a **verbosity penalty** in the rubric.
* **Diversity:** ensure prompt sets cover skills (coding, reasoning, safety, multilingual); stratify analysis.

---

## Minimal, practical snippets

### (1) Bradley–Terry in PyTorch (two variants A,B; extendable)

```python
import torch, torch.nn.functional as F

# outcomes: +1 if B wins, 0 tie, -1 if A wins
y = torch.tensor([+1, -1, +1, 0, +1], dtype=torch.float32)
w_tie = 0.5  # split ties

beta = torch.nn.Parameter(torch.zeros(2))  # beta[0]=A, beta[1]=B
opt = torch.optim.LBFGS([beta], max_iter=100)

def closure():
    opt.zero_grad()
    pB = torch.sigmoid(beta[1]-beta[0])          # P(B beats A)
    # log-likelihood with tie-splitting
    logp = (y==+1)*torch.log(pB+1e-9) + (y==-1)*torch.log(1-pB+1e-9) + (y==0)*torch.log(w_tie*pB + w_tie*(1-pB)+1e-9)
    loss = -logp.mean()
    loss.backward()
    return loss

opt.step(closure)
delta = (beta[1]-beta[0]).item()
````

### (2) Win-rate & bootstrap CI

```python
import torch
def winrate_ci(labels, B=2000, alpha=0.05):  # labels in {-1,0,+1}
    x = torch.tensor(labels)
    w = (x==+1).float(); l=(x==-1).float(); t=(x==0).float()*0.5
    p = (w + t).mean()
    boots = []
    n = len(x)
    for _ in range(B):
        s = x[torch.randint(0, n, (n,))]
        w = (s==+1).float(); l=(s==-1).float(); t=(s==0).float()*0.5
        boots.append((w+t).mean())
    boots = torch.stack(boots)
    lo, hi = boots.quantile(alpha/2).item(), boots.quantile(1-alpha/2).item()
    return float(p), lo, hi
```

### (3) Self-normalized IPS for log-replay

```python
def snips(actions, rewards, pi_probs, b_probs):
    # actions: indices of taken actions under logging policy
    w = pi_probs.gather(1, actions[:,None]).squeeze(1) / (b_probs.gather(1, actions[:,None]).squeeze(1)+1e-9)
    w = torch.clamp(w, max=50.0)
    num = (w * rewards).sum()
    den = w.sum() + 1e-9
    return (num / den).item()
```

---

## Sequential testing & guardrails for LLMs

* Use **alpha-spending** or **always-valid p-values** (e.g., e-processes) when peeking.
* Hard **safety guardrails**: disallowed content rate must **not** increase (non-inferiority).
* **Cost/latency** as constraints; declare **efficient dominance** if quality ↑ and cost ↓.

---

## Reporting that convinces skeptics

* Primary: **win-rate** (BT delta with CI) on a **pre-registered prompt set**.
* Secondaries: pass@k (code), exact-match (QA), rubric scores, latency/cost.
* AA checks (A vs A) to calibrate false positives.
* Stratified effects by prompt type; *one* exploratory page, clearly labeled.

---

## Mental model

Treat an LLM configuration (weights + prompts + decoding) as a **policy**. Measure **pairwise quality** with blinded comparisons, keep **safety/cost** as guardrails, stabilize decisions with **variance reduction** and **sequential control**, and use **counterfactual replay** to de-risk before going live.

---

> **Quick check (your turn):**
You’re launching a new prompt for code generation. In one or two sentences, describe the **online design** (unit, metric, and blinding) and the **offline check** (counterfactual or test set) you’d run before rollout.

---

# 22 LLM training regimes (the landscape)

## A. Pretraining (next-token prediction)

Train on mixed-domain text with the **autoregressive** loss  
$$
\mathcal{L}_{\text{LM}}(\theta)=-\sum_{t}\log p_\theta(x_t\mid x_{<t}),
$$
optionally with **packing** (multiple docs per sequence, causal separators) and **curricula** (short→long).

Minimal loop (teacher forcing):

```python
logits = model(input_ids)                 # [B,T,V]
loss = F.cross_entropy(logits[:,:-1].flatten(0,1),
                       input_ids[:,1:].flatten(0,1))
loss.backward()
````

## B. Supervised finetuning (SFT) / Instruction tuning

Curate prompt→completion pairs. Same LM loss but on *instruction-shaped* data (templates, roles, system prompts). Add **formatting guards** (e.g., JSON schema). Often mix in safety exemplars.

## C. Preference learning (aligning behavior)

Two families:

* **RLHF (RM + PPO):** learn reward (r_\phi(x,y)) from pairwise prefs, then optimize policy (\pi_\theta) with a KL-regularized RL objective
  $$
  \max_\theta ;\mathbb{E}*{y\sim \pi*\theta}!\big[r_\phi(x,y)\big]-\beta,\mathrm{KL}(\pi_\theta\mid\mid\pi_0).
  $$

* **Direct Preference Optimization (DPO):** fit (\pi_\theta) **directly** to preferences vs a reference (\pi_0) without an explicit RM:
  $$
  \mathcal{L}*{\text{DPO}}
  = -\mathbb{E}\Big[\log\sigma!\Big(\beta\big[\log \tfrac{\pi*\theta(y^+\mid x)}{\pi_0(y^+\mid x)}-\log \tfrac{\pi_\theta(y^-\mid x)}{\pi_0(y^-\mid x)}\big]\Big)\Big].
  $$
  Narrative: DPO says “make the chosen response more likely than the rejected one, relative to a reference,” achieving alignment with pure supervised updates.

Tiny DPO step:

```python
def dpo_loss(logp_pos, logp_neg, logp0_pos, logp0_neg, beta=0.1):
    z = beta * ((logp_pos - logp0_pos) - (logp_neg - logp0_neg))
    return F.softplus(-z).mean()          # -log σ(z)
```

Related: **IPO/ORPO/KTO/GRPO** variants tweak the margin, temperature, or regularizer; **AWR/AWAC-style** do advantage-weighted MLE.

## D. Parameter-efficient finetuning (PEFT)

Freeze base weights, train **adapters** or low-rank deltas (LoRA: (W \leftarrow W + A B^\top) with small rank). Big wins when data is small or you must host many personas.

## E. Continual & domain adaptation

Mix a **small % of general data** (to prevent forgetting) with domain text; optionally **regularize** toward the old model (Fisher/LLR penalties) or distill from it.

---

# 23 Data preprocessing for LLMs (what actually matters)

## A. Quality filtering

* **Deduplication:** near-dup removal (MinHash/LSH, SimHash on 3–5-grams).
* **Boilerplate/markup stripping; language ID; profanity/safety filters.**
* **Heuristic quality models:** small classifier scoring perplexity bands, formatting, symbol ratios.

## B. Tokenization & packing

* Choose subword (BPE/WordPiece/Unigram). Track **bytes-per-token** (efficiency).
* **Packing:** concatenate docs with separators to reduce padding; preserve **doc boundaries** for loss masking.

Packing mask sketch:

```python
# attn_mask zeros across boundaries so tokens don't attend across docs
attn_mask = torch.ones(T,T, dtype=torch.bool)
attn_mask[boundary_positions[:,None] <= torch.arange(T)] = 0  # conceptually
```

## C. Document mixing & curricula

Balance domains with **mixture weights** (e.g., code/wiki/books/web). Use **length curricula** for long-context stability (Section 2).

## D. Decontamination

For eval sets (e.g., GSM8K, HumanEval), **hash** canonical problems and remove matches/near-matches from training to avoid leakage.

## E. SFT & preference data hygiene

* Normalize prompts (roles, tool-call format), redact secrets.
* Balance tasks; **swap positive/negative** orders to avoid position bias.
* For pairwise prefs, ensure **hard negatives** (close but wrong) and **inter-annotator checks**.

---

# 24 Losses you’ll actually use (and why)

## A. Cross-entropy (autoregressive)

As above; sometimes with **label smoothing** (\epsilon) to soften targets:
$$
\mathcal{L} = -(1-\epsilon)\log p_\theta(x_t) - \epsilon \sum_v \tfrac{1}{V}\log p_\theta(v).
$$

## B. Span corruption / denoising (T5/UL2)

Sample spans; replace with sentinels; predict the spans:
$$
\mathcal{L}=-\sum \log p_\theta(\text{span}\mid \text{masked input}).
$$
Narrative: better sample-efficiency for seq2seq tasks; less ideal for pure decoder-only unless adapted (prefix-LM).

## C. KL-regularized objectives (alignment)

During RLHF-style updates, include
$$
\mathcal{L}*{\text{KL}}=\beta,\mathrm{KL}(\pi*\theta(\cdot\mid x)\mid\mid\pi_0(\cdot\mid x))
$$
to keep outputs near a safe/reference policy (controls drift/verbosity).

## D. Contrastive / response ranking

Given ((y^+,y^-)) for prompt (x),
$$
\mathcal{L}=-\log \frac{\exp(s_\theta(x,y^+)/\tau)}{\exp(s_\theta(x,y^+)/\tau)+\exp(s_\theta(x,y^-)/\tau)},
$$
where (s_\theta) can be sequence logprob, a small judge head, or pooled repr.

## E. Auxiliary stability losses

* **Entropy bonus** to avoid collapse in RLHF.
* **Speculative/distillation**: match teacher tokens (q) (KL) while keeping base LM loss.

---

# 25 Evaluation metrics (make numbers honest)

## A. Intrinsic model quality

* **Perplexity / BPC:** ( \mathrm{PPL}=\exp(\tfrac{1}{N}\sum -\log p) ). Useful for pretraining, weakly correlated with task utility.
* **Next-token accuracy** on controlled corpora (sanity checks).
* **Calibration:** ECE / Brier; does the model’s confidence match reality?

## B. Task metrics (deterministic)

* **Exact match / F1** (QA), **ROUGE** (summarization), **BLEU** (MT; with caveats).
* **Code:** pass@k with unbiased estimator:
  $$
  \widehat{\text{pass@k}} = \mathbb{E}\Big[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\Big],
  $$
  where (n) samples, (c) correct; estimate via bootstrap over problems.
* **Math/Reasoning:** GSM8K/MathQA exact, **step-faithfulness** (chain-of-thought judged to match reasoning).

## C. Preference / human metrics

* **Win-rate / BT-skill** from blinded pairwise judgments (Section 11).
* **Rubric scoring:** structured multi-axis (helpful, harmless, honest), often LLM-as-judge **calibrated** on human gold.

## D. Safety, toxicity, bias

* Trigger rates on curated safety suites; **non-inferiority** vs baseline.
* **Leakage** (PII/tool abuse) rates on red-team prompts.

## E. Systems metrics (don’t bury these)

* **Latency** (p50/p95), **throughput** (tok/s), **cost** (prompt+completion tokens), **context hit-rate** with retrieval, **KV-cache** memory.

**Report with rigor:** stratify by task type, include **CIs** (bootstrap), and show **AA tests** to calibrate false positives.

---

# 26 Putting it together (a minimal training recipe)

1. **Curate & clean** mixed-domain corpus → dedup → tokenize → pack.
2. **Pretrain** with LM loss; monitor PPL + few task probes.
3. **Instruction-tune** on high-quality SFT; format-checkers in the loop.
4. **Align with preferences** (DPO if simple; RLHF if you need fine control of behaviors/cost).
5. **PEFT** per domain/product; keep a small mix of general data to avoid forgetting.
6. **Evaluate** with: task scores, win-rate, safety guardrails, and latency/cost.
7. **A/B** (Section 11) with blinded SxS and pre-registered OEC.

---

> **Quick check (your turn):**
If you switch from RLHF (PPO) to DPO for alignment, what *regularizer or reference* keeps the policy from drifting, and how does that show up in the DPO objective you’d implement?

---

# 27 Distributed Training for ML/LLMs

## Mental model

You want maximum **throughput** (tokens/sec) at a given **cost** (GPU-hours) without exploding **memory**. You get there by (i) splitting work across devices, (ii) minimizing/overlapping communication, and (iii) taming memory with sharding & recomputation.

We’ll climb from **data parallel** → **optimizer/state sharding** → **tensor & pipeline parallel** → **3D parallel**. Along the way: comms math, memory math, and micro-optimizations.

---

## A) Communication, at a glance (why the network matters)

Collectives on $p$ workers dominate scaling.

* **All-Reduce** (sum gradients across $p$ ranks): ring all-reduce cost (bytes $S$)

$$
T_{\text{allred}} \approx 
\underbrace{\frac{2(p-1)}{p}\frac{S}{\text{BW}}}_{\text{bandwidth term}}
\;+\;
\underbrace{(p-1)\cdot \alpha}_{\text{latency term}},
$$

with link bandwidth BW and per-hop latency $\alpha$.

* Rule of thumb: keep your **compute time per step** $\gg T_{\text{allred}}$, or **overlap** comm with compute (bucketed gradients, reduce-scatter in backprop).

Mixed precision (BF16/FP16) halves gradient bytes; gradient compression is rare for LLMs due to instability.

---

## B) Data Parallel (DP) = many copies, split the batch

Every rank holds full model; each sees a **different micro-batch**; gradients summed via all-reduce.

* **Throughput** roughly scales $\propto p$ until comms bite.
* **Memory**: full params + optimizer states + activations per rank.

**Minimal PyTorch (DDP)**

```python
# torchrun --nproc_per_node=8 train.py
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = YourModel().cuda()
model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

for xb, yb in loader:              # loader uses DistributedSampler
    xb, yb = xb.cuda(), yb.cuda()
    loss = model(xb, yb)           # backward inside
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
````

**Tricks that matter**

* **Gradient accumulation**: emulate large global batch without expanding per-GPU activations.
* **Bucketed all-reduce**: smaller buckets overlap better; too small increases latency hits.
* **Determinism**: set seeds & use DistributedSampler with fixed epoch; watch async ops.

---

## C) Memory bloat & Optimizer State Sharding (ZeRO/FSDP)

A big LLM spends most memory in **optimizer states** (Adam has $(m,v)$ plus params & grads). For parameters $P$ floats:

* Params: $P$
* Grads: $P$
* Adam states: $2P$

→ **~4×** model-size footprint per rank (in FP32 terms; BF16 reduces params/grads, not always optimizer states).

**ZeRO stages (intuition)**

* **Stage 1:** shard optimizer states across $p$ → each rank stores $\approx 2P/p$.
* **Stage 2:** shard grads too → each rank grads $\approx P/p$.
* **Stage 3:** shard **params** as well → each rank holds only its shard during fwd/bwd; do **gather/scatter** on the fly.

**FSDP (PyTorch)** ~ ZeRO-3 style with **per-module** sharding & prefetch. It *reshards* params between layers to keep peak memory low; overlaps all-gathers with compute.

**Minimal FSDP**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
model = FSDP(YourModel().cuda(),
             sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3-like
             mixed_precision=mp,
             device_id=local_rank)
```

**Activation checkpointing** (recompute instead of storing):

```python
from torch.utils.checkpoint import checkpoint

def block(x):
    # heavy layer stack
    return sublayers(x)

y = checkpoint(block, x)  # drop activations, recompute in backward
```

**TL;DR**: ZeRO-3/FSDP let you fit bigger models by sharding **everything** and paying comms during forward/backward.

---

## D) Model Parallel: **Tensor** & **Sequence** splitting

When a **single layer** doesn’t fit, split its math.

### Tensor Parallel (TP, Megatron-style)

Split large matmuls across devices.

* **Column parallel** ($Y = X W$), with $W=[W_1; W_2 \dots W_t]$.
  Each rank computes $Y_i = X W_i$; concatenate or all-reduce as needed.

* **Row parallel** ($Z = Y V$), with $V^\top=[V_1^\top; \dots; V_t^\top]$.
  Each rank computes partial $Z_i = Y_i V_i$; sum-reduce across ranks.

**Attention TP**: QKV projections and output projection split similarly; softmax stays local if sequence is not split.

**Sequence Parallel (SP)**: split **sequence length** $T$ across ranks; reduce activations that need global (layernorm stats, attention softmax if needed) with small collectives. Cuts activation memory without changing math.

**Costs**: TP adds **intra-layer** all-reduces; place TP groups **within the same node** (NVLink/NVSwitch) to avoid NIC bottlenecks.

---

## E) Pipeline Parallel (PP): cut the network into stages

Divide layers into $p$ **stages**; stream **micro-batches** through like an assembly line.

* **Bubble** (idle time) fraction for GPipe schedule:

$$
\text{bubble} \approx \frac{p-1}{m},
$$

with $m$ micro-batches per global batch. Increase $m$ to shrink bubbles (up to memory/network limits).

* **Schedules**:

  * **GPipe**: all forward then all backward per micro-batch; simple, bigger bubbles.
  * **1F1B (PipeDream-Flush)**: interleave fwd/back to reduce activation staleness and memory.

**Inter-stage traffic**: activations only (not params). Balance stage compute or you’ll bottleneck on the slowest stage.

---

## F) 3D Parallel (DP × TP × PP) for very large LLMs

Combine:

* **DP** across nodes (data shards),
* **TP** within node (NVSwitch),
* **PP** across nodes (stage the stack).

Add **ZeRO-1/2/3 or FSDP** on top to shard states. Real deployments pick a decomposition that matches **hardware topology**:

* TP size = GPUs per NVSwitch island,
* PP depth = layer count / (TPed layer size),
* DP = remainder.

**Throughput per step (back-of-envelope)**:

$$
T_{\text{step}} \approx
\max\Big(
\underbrace{T_{\text{compute}}}*{\text{per stage}},\
\underbrace{T*{\text{intra-TP comm}}}*{\text{NVLink}},\
\underbrace{T*{\text{inter-PP comm}}}_{\text{NIC}}
\Big)
\times \text{pipeline factor},
$$

with pipeline factor $\approx 1/(1-\text{bubble})$. Aim to keep inter-stage payloads within NIC budgets.

---

## G) Overlap & scheduling (turn red bars green)

* **Comm–comp overlap**: start **reduce-scatter** when a gradient bucket is ready; **all-gather** next layer’s params while computing the current one (FSDP prefetch).
* **CUDA graphs**: capture step to cut CPU launch overhead on small kernels.
* **Fused kernels**: FlashAttention, fused optimizers (FusedAdam), RMSNorm kernels reduce HBM trips.

---

## H) The data path (starve the GPU → lose)

* **Sharded dataloaders** (DistributedSampler), large **prefetch** queue.
* **Pinned memory** + non-blocking `to(device, non_blocking=True)`.
* **On-the-fly tokenize/pack** on CPU threads; avoid Python GIL hotspots (multiprocessing, compiled ops).

---

## I) Fault tolerance & elasticity

* **Periodic checkpoints**: shard-aware (FSDP/ZeRO) checkpoints with state dict consolidation.
* **Elastic training**: Randezvous on restarts, be careful—changing DP world size changes effective batch & LR; use **linear LR scaling** or warm restart.
* **Determinism**: fixed seeds, no async AMP cast races, consistent bucket ordering.

---

## J) Minimal configs that work

**(a) Fit bigger on the same GPUs:** FSDP FULL_SHARD + BF16 + activation checkpointing; grad-accumulate to reach target global batch.

**(b) Go faster across a node:** DDP + FlashAttention + fused optim + good bucket size; keep batch large enough to hide all-reduce.

**(c) Very large model across nodes:** TP within node, PP across nodes, DP across node groups; ZeRO-1/2 or FSDP for optimizer sharding; micro-batch count tuned to kill pipeline bubbles.

---

## K) What to measure (and fix first)

* **Utilization**: SM occupancy, **MFU** (model FLOP utilization). If low → kernel fusion, larger batch, fewer small ops.
* **Overlap**: timeline shows comm bars overlapping compute? If not, tune bucket sizes, enable reduce-scatter/backward overlap.
* **Imbalance** (PP): stage times; rebalance layers or interleave assignments.
* **Network**: NCCL BW vs theoretical; ensure correct topology env vars; isolate traffic (no PCIe oversub).

---

## L) Microscopic examples

**Bucketed gradient overlap (conceptual)**

```python
# set TORCH_DISTRIBUTED_DEBUG=DETAIL to verify buckets
torch.distributed.algorithms.ddp_comm_hooks.default_hooks.default as hook
# or register a fp16 compression hook for small gains:
ddp.register_comm_hook(state=None, hook=torch.distributed.algorithms.ddp_comm_hooks.default.fp16_compress_hook)
```

**Gradient accumulation (simulate big batch)**

```python
acc_steps = 8
for i, (xb, yb) in enumerate(loader):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(xb.cuda(), yb.cuda()) / acc_steps
    loss.backward()
    if (i+1) % acc_steps == 0:
        opt.step(); opt.zero_grad(set_to_none=True)
```

**Pipeline sketch (2 stages, pseudo)**

```python
# stage0_on_gpu0, stage1_on_gpu1; micro-batches mb[0..m-1]
# fwd 0..m-1 on stage0; stream to stage1; then 1F1B backward—use torch.distributed.pipelining in practice.
```

---

## M) Rules of thumb (kept short)

* Keep TP within NVLink islands; use PP for cross-node splits.
* Use FSDP/ZeRO-3 when memory-bound; DDP when compute-bound.
* Choose micro-batches to (i) fit memory, (ii) kill pipeline bubbles, (iii) keep GEMMs large.
* Overlap everything: reduce-scatter in backward, all-gather in forward.
* Profile before guessing; the timeline tells the truth.

---

> **Quick check (your turn):**
Suppose you move from pure DDP to FSDP (FULL_SHARD) on the same 8-GPU node to fit a $2\times$ larger model. In one or two sentences, explain **what gets sharded**, **what extra communication you now pay**, and **how you’d hide that cost** during forward/backward.

---

# 28 ML/LLM **Inference Optimizations**

## Mental model

Latency and cost come from three levers: **math**, **memory traffic**, and **machinery** (framework/serving). Good inference trims all three:

* shrink math (fewer FLOPs, fewer tokens),
* move fewer bytes (KV/state, activations),
* remove software overhead (launches, Python, RPC).

We’ll walk from model graph → kernels & numerics → attention/decoding tricks → serving system. Tiny code blocks are anchors; concepts are the priority.

---

## A) Graph-level: make the network easy to run

**Fuse and eliminate.** Export a static-ish graph, fold constants, and fuse patterns (bias+matmul+gelu, layernorm+pointwise).

```python
# PyTorch 2.x: ahead-of-time edges for stable shapes
import torch
m = MyDecoder().eval().to('cuda')
m = torch.compile(m, mode="max-autotune")   # gives graph capture + kernel fusion
````

* **Operator choice matters.** Use native fused kernels: `torch.nn.functional.scaled_dot_product_attention` (FlashAttention-style), `fused_adam` (server warmup/preload), fused RMSNorm/GELU if available.
* **CUDA Graphs**: capture one warm path to kill kernel-launch overhead in low-latency settings (steady shapes/batch sizes).

```python
static_in = torch.zeros((B,T,D), device='cuda')
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = m(static_in)  # graph-captured steady path
# later:
static_in.copy_(live_in); g.replay(); use(out)
```

**Pitfall**: dynamic control flow and ragged shapes fight capture; bucket requests by shape.

---

## B) Numerics & compression: keep accuracy, drop bytes

### Quantization

* **Weight-only INT8/INT4** for matmuls (LLM.int8/int4 kernels) → big memory savings with tiny perplexity hit if outlier handling is used.
* **SmoothQuant/AWQ** style: shift activation range into weights pre-quantization for stable INT8 activation quant.
* **KV-cache quant** (e.g., NF4/FP8) often works with minimal degradation; keep QK scores in higher precision.

Minimal weight-only example (conceptual):

```python
# Pseudocode: pack weight into int8 and dequantize on-the-fly
W_fp = lin.weight.data
scale = W_fp.abs().amax(dim=1, keepdim=True) / 127
W_i8  = torch.clamp((W_fp/scale).round(), -127, 127).to(torch.int8)
# Kernel: y = x @ dequant(W_i8, scale)   # custom fused GEMM
```

### Low-rank adapters at serve time

If you host many personas, **merge LoRA** into base weights offline (“merge-and-run”), or keep them separate with **adapter caches** (trade latency vs memory).

---

## C) Kernel-level: stop moving memory

### SDPA / FlashAttention

Always route attention through fused, IO-aware kernels:

```python
o = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=mask,
                                                     is_causal=True)
```

These avoid the $T \times T$ allocation and stream blocks with **online softmax** (Section 3).

### Layouts & tiling

* Keep contiguous $[B,H,T,D]$ for attention; avoid repeated transposes.
* Pre-transpose weights to match GEMM preferences (column-major for cuBLASLt plans).

### Memory planning

* Pre-allocate KV buffers (paged, see below).
* Reuse activation buffers across layers (frameworks do this; exporting helps).

---

## D) Attention & KV: where most LLM bytes live

### KV cache discipline

* **Multi-Query / Grouped-Query Attention**: share K/V across heads or groups → KV memory $\downarrow$ and bandwidth $\downarrow$.
* **Paged attention**: store KV in fixed-size pages; avoid giant contiguous buffers and enable **context windowing** & **eviction**.
* **Segmented/rolling caches**: keep recent tokens at high resolution; down-sample or drop distant ones for ultra-long sessions.

### Long-context tricks at inference

* **RoPE scaling** (Section 2): small angle/position scaling to extend length zero-shot.
* **Local + global tokens**: banded attention plus a few global “sink” tokens creates sparse highways at negligible cost.

---

## E) Decoding strategies: fewer useless tokens, fewer passes

### Speculative decoding (draft-then-verify)

Use a cheap **draft model** (or low-precision pass) to propose $k$ tokens; verify with the target model in one forward. If prefixes agree, you *skip* target-model steps.

Skeleton:

```python
# 1) Draft k tokens with small model D
draft_ids = D.generate_stepwise(prompt_ids, k)

# 2) Verify in one pass on target T
logits_T = T.forward(prompt_ids + draft_ids[:-1])  # overlaps
# 3) Accept longest matching prefix; continue
```

Works best when the draft is accurate and much cheaper.

### Early exit / confidence halting

Attach a small **exit head** to some layers; if token distribution is peaky (entropy below threshold), stop early for that token. Useful in constrained tasks or copy-heavy regimes.

### Constrained decoding

Regex/CFG/JSON constraints prune branches; fewer logits considered, fewer backtracks.

### Temperature & top-p

For latency-critical tasks, lower temperature and narrower $p$ reduce tail work (and can improve judge scores if verbosity hurts).

---

## F) Batching & scheduling: the serving superpower

### Continuous dynamic batching

Aggregate tokens across many requests **each step**. The scheduler builds a micro-batch of active sequences at the same decode step; KV paging makes joining/leaving cheap.

* **Prefill vs decode**: separate lanes. Prefill (first pass) is heavy but parallel; decode is light per step. Keep both pipelines saturated.
* **Bucketing by shape**: group by (dtype, head_dim, seq_bucket) for CUDA Graphs.

### Admission control

Protect **p95 latency** by capping batch or queue depth during spikes; defer long prompts to a prefill-only pool.

---

## G) System-level optimizations

* **Transport**: use **server-side streaming**; start emitting tokens as soon as logits stabilize.
* **RPC/runtime**: keep hot path in C++ where possible; avoid Python GIL on schedulers.
* **NUMA & pinning**: pin NIC/CPU threads to local NUMA nodes; avoid cross-socket jitter.
* **Cache the prompt**: if many users share a long system prompt/tools, precompute and reuse the **prefill KV**.

---

## H) Retrieval-augmented serving (RAG) that doesn’t thrash

* **Light retriever** (ANN) in-process; cache embeddings for frequent queries.
* **Slim contexts**: feed only the **answerable** passages (e.g., 2–6), not the union of everything relevant.
* **Grounding validators**: short re-ranker or string-match filter post-generation to avoid verbose detours.

---

## I) Measure → iterate (what to plot)

* **Token throughput**: prefill tok/s and decode tok/s (they bottleneck differently).
* **KV bytes moved** per token (before/after MQA/GQA).
* **SM utilization vs memory BW**: if BW-bound, prioritize IO-aware kernels and quantization; if SM-underutilized, fuse or increase batch.
* **p50/p95 latency** by prompt/response length buckets; watch long-tail.
* **Degradation vs savings**: for each quantization/speculative setup, run a **quality smoke suite** (win-rate or exact metrics); only keep Pareto-efficient points.

---

## J) A compact “cookbook”

* **Small model, strict latency (chat/agent):** SDPA + CUDA Graphs, continuous batching, MQA/GQA, KV-NF4, prompt-KV cache, constrained decoding.
* **Large model, high throughput (batch jobs):** weight-only INT4/INT8, aggressive batching, speculative decoding, prefill–decode split pools, adapter-merge offline.
* **Ultra-long context:** RoPE scaling + local/global sparse attention + paged KV; consider summarizing distant context into learned “memory tokens.”

---

## K) Tiny end-to-end sketch (PyTorch, conceptual)

```python
class FastLLM(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.blocks = torch.nn.ModuleList([...])  # use fused RMSNorm/GELU
    @torch.inference_mode()
    def forward(self, x, kv_cache=None):
        # x: [B,T], run prefill or decode step; use SDPA and MQA
        for blk in self.blocks:
            x, kv_cache = blk(x, kv_cache=kv_cache, use_sdpa=True, mqa=True)
        return x, kv_cache

model = torch.compile(FastLLM().eval().to('cuda'), mode="max-autotune")

# serving loop (pseudo): dynamic batching with paged KV
while True:
    batch = scheduler.form_decode_batch()        # active seqs
    ids, kv_pages = batch.tokens, batch.kv_pages
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, kv_pages = model(ids, kv_cache=kv_pages)
    next_ids = sample(logits, top_p=0.9, temp=0.2, constrained=True)
    scheduler.commit(next_ids, kv_pages)
```

---

## L) What usually breaks first (and how to fix it)

* **Jittery latency despite high GPU util** → queueing/scheduler imbalance; separate prefill and decode lanes.
* **Out-of-memory spikes** → fragmented KV; adopt paged KV and hard caps on max tokens per request.
* **Quality drop after quantization** → try weight-only first; calibrate per-channel scales; leave QK matmuls in higher precision.
* **Speculative decoding underperforms** → draft model too weak or misaligned; share tokenizer/positional scheme and retune accept length ($k$).

---

> **Quick check (your turn)**
You need to cut **p95 latency by ~30%** for a 7B chat model without changing responses much. In one or two sentences, name **two** complementary tactics you’d try first (one kernel/numerics level, one serving/system level), and briefly why they’re likely to help.

---