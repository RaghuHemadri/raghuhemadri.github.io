---
layout: post
title: "LLM Cheatsheet: Fundamentals and Implementation"
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

> **Quick check (one-liner): **
Why does **per-channel** weight quantization usually outperform **per-tensor** for linear layers?

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
\mathcal{L}_{\text{RM}}(\phi)
= -\mathbb{E}_{x, (y^+,y^-)} \big[\log \sigma(r_\phi(x,y^+)-r_\phi(x,y^-))\big].
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
````

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

* c_v (V_\psi - R_t)^2 + c_e \mathsf{H}[\pi_\theta(\cdot|s_t)]
  \Big],
  $$

and we **add a KL penalty** to $\pi_{\text{ref}}$ either:

* explicitly: $\mathcal{L} + \beta,\mathrm{KL}(\pi_\theta | \pi_{\text{ref}})$, or
* implicitly in the reward: $r^\text{KL}*t = r_t - \beta \log \frac{\pi*\theta(y_t|s_t)}{\pi_{\text{ref}}(y_t|s_t)}$.

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
\pi^\star(y|x) \propto \pi_{\text{ref}}(y|x), e^{\beta, r(x,y)}.
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

\underbrace{\log\pi_{\text{ref}}(y^+|x)+\log\pi_{\text{ref}}(y^-|x)}_{\text{reference correction}}
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
Why does DPO include the **reference log-prob terms** $\log\pi_{\text{ref}}(y^\pm|x)$ inside the sigmoid argument, and what would likely happen if you dropped them?

---


# 10 GRPO — Group Relative (Policy) Optimization

## Core idea

GRPO optimizes a policy **without** a value function or explicit reward model by using **grouped rollouts per prompt** and a **relative baseline** inside the group. It’s essentially REINFORCE with a strong control variate and PPO-style KL control.

Given a prompt $x$, sample a *group* $\{y^{(i)}\}_{i=1}^M \sim \pi_\theta(\cdot\mid x)$. Score each with a scalar reward $r^{(i)} = r(x,y^{(i)})$ (e.g., pass@k for code, unit tests, formatting/safety checks, length heuristics, or lightweight preference/rank).

### Relative advantage inside the group

Compute a baseline from the group, e.g.

$$
\bar r = \frac{1}{M}\sum_{i=1}^M r^{(i)},\qquad
A^{(i)} = \frac{r^{(i)}-\bar r}{\sigma_r + \epsilon},\quad
\sigma_r^2=\tfrac{1}{M}\sum_i (r^{(i)}-\bar r)^2.
$$

This yields **zero-mean**, scale-normalized advantages *per prompt*, removing the need for a critic $V_\psi$.

### Objective with KL control (PPO-lite)

Let $\rho^{(i)} = \exp\!\big(\log\pi_\theta(y^{(i)}\mid x)-\log\pi_{\theta_\text{old}}(y^{(i)}\mid x)\big)$ be the sequence-level ratio (or token-sum). A GRPO step minimizes

$$
\mathcal{L}_{\text{GRPO}}(\theta)=
-\mathbb{E}_{x}\left[\frac{1}{M}\sum_{i=1}^M
\Big(
\min\big(\rho^{(i)} A^{(i)},\ \mathrm{clip}(\rho^{(i)},1-\epsilon,1+\epsilon)A^{(i)}\big)
\Big)\right]
+ \beta\,\mathrm{KL}\!\left(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right).
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
* **Rank-from-rules:** rank the $M$ samples with deterministic criteria and map ranks to scores (e.g., $\{+1,0,-1\}$).

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
$\nabla_\theta \log \pi_\theta(y|x)\,(r-\underbrace{b(x)}_{\text{baseline}})$.

Choosing $b(x)=\mathbb{E}[r|x]$ minimizes variance. GRPO’s $\bar r$ is an *unbiased* estimator of that baseline using the group, and dividing by $\sigma_r$ standardizes the scale across prompts, taming the PPO ratio dynamics.

---

## Pitfalls & tips

* **Group size $M$:** too small → high variance; too large → expensive. Common: $M=4\!\sim\!8$.
* **Reward saturation:** if most samples pass (or fail), add *soft* shaping terms or widen tests to keep variance.
* **KL schedule:** start with higher $\beta$ to anchor style, then anneal.
* **Clipping $\epsilon$:** 0.1–0.2 typical; larger when advantages are well-normalized.
* **Token-level shaping:** optionally distribute a scalar outcome over tokens (e.g., uniform or with a heuristic) if you need more granular control.

---

> **Quick check (your turn, one line):**
What’s the main **statistical benefit** of subtracting the **group mean reward** $\bar r$ in GRPO before applying PPO clipping and KL?

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
