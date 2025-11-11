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
```

---

## B) WordPiece (WP)

**Idea.** Start with characters. Iteratively add the subword that **maximizes corpus likelihood improvement** under a simple language model with independent subword emissions (approx). Greedy but **likelihood-driven**; decoding uses **max-prob segmentation** (Viterbi or greedy longest-probable).

**Training sketch.**

* At each step, consider candidate joins $(x!\oplus!y)$ and pick the one maximizing $(\Delta \log P(\text{data}))$. Approximations make this tractable (counts + smoothing).
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

**Idea.** Maintain a **candidate inventory** of subwords (often seeded from many BPE/WP candidates). Assume each word is generated by a **mixture over segmentations** with subword probabilities $({p(t)})$. Use **EM** to (a) estimate $(p(t))$ and (b) **prune** low-utility tokens to shrink the vocab.

**Model.**
For a word $(w)$, all segmentations $(S(w))$ have prob
$$
P(w) = \sum_{s \in S(w)} \prod_{t \in s} p(t), \quad \sum_t p(t) = 1.
$$
EM:

* **E-step:** compute posterior usage of each token $(t)$ via forward-backward over each word.
* **M-step:** update $(p(t) \propto \text{expected count}(t))$; prune a fraction with lowest loss impact.
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

**Quick check (your turn, one sentence):**
If you were building a multilingual decoder-only LLM for noisy web text, which tokenizer would you start with and why?

---
---

Great—Section 7 next.

# 2 Self-attention: concept + code

**What it is.**
Each token turns into a **query** $(q_i)$, **key** $(k_j)$, **value** $(v_j)$. Token $i$ gathers info from all tokens $j$ via weights $(\alpha_{ij})$ (similarity of $q_i$ and $k_j$):

$$
\alpha_{ij}=\text{softmax}*j!\left(\frac{q_i^\top k_j}{\sqrt{d_k}} + m*{ij}\right),\quad
\text{Attn}(i)=\sum_j \alpha_{ij} v_j
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
```

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

**Your turn (one quick question):**
In decoder-only language modeling, why do we need **both** a causal mask **and** a padding mask? (Answer in a sentence.)

---
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

For a sequence $(X\in\mathbb{R}^{T\times d_{\text{model}}})$,
$$
Q = XW_Q,\quad K=XW_K,\quad V=XW_V,\quad
\text{Attn}(Q,K,V) = \text{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

* $(M)$ is a mask: $(0)$ for allowed, $(-\infty)$ for blocked (e.g., **causal** mask for GPT).
* **Multi-head:** split into $(H)$ heads, run attention in parallel, then concatenate and project.

**Block (pre-LN)**
$$
\begin{aligned}
Y &= X + \text{MHA}(\text{LN}(X)) \
Z &= Y + \text{MLP}(\text{LN}(Y))
\end{aligned}
$$

**Complexity:** $(O(T^2 d))$ memory/time due to the $(T\times T)$ attention matrix.

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
```

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
* **Masks:** causal for decoder; padding for encoder; **cross-attn** in encoder–decoder lets decoder attend to encoder states with its own Q and the encoder’s K,V.
* **Complexity pressure:** long contexts → $(T^2)$ cost; see FlashAttention, KV cache, sparse/linear attention, and speculative decoding (later sections).

---

### Quick check (your turn, one line):

Why do we divide by $(\sqrt{d_k})$ in the attention scores $(QK^\top/\sqrt{d_k})$?

---
---

Great—let’s tackle **8) LLM Decoder (response sampling)**. No code, just the “how it really works” with math and when to use what.

# 4 LLM Decoder (response sampling): the main methods

We generate a sequence $y_{1:T}$ from an autoregressive LM

$$
p_\theta(y_{1:T}\mid x)=\prod_{t=1}^{T} p_\theta!\left(y_t \mid x,; y_{<t}\right),\quad
p_\theta(y_t \mid \cdot)=\mathrm{softmax}(z_t)
$$
where $z_t \in \mathbb{R}^{V}$ are logits over a vocabulary $(V)$.

All decoding methods modify $z_t$ $\Rightarrow$ get a new distribution $\tilde p(y_t \mid \cdot)$ $\Rightarrow$ **sample** or **argmax**. Below, $\mathcal{S}$ denotes the chosen support set of tokens at step $t$; we always renormalize on $\mathcal{S}$.

---

## A) Temperature scaling (entropy control)

Scale logits by $1/\tau$ before softmax:
$$
\tilde p_\tau(i) ;=; \frac{\exp(z_i/\tau)}{\sum_j \exp(z_j/\tau)}.
$$

* $\tau!<!1$: sharper (low entropy), more deterministic.
* $\tau!>!1$: flatter (high entropy), more diverse.
* As $\tau!\to!0$, $\tilde p_\tau \to argmax$; as $\tau!\to!\infty$, $\tilde p_\tau \to $ uniform.

**View:** maximizes likelihood under a **tempered** model $q \propto p^{1/\tau}$. Expected log-prob decreases ~ linearly with $\tau$.

---

## B) Top-k sampling (truncate to k highest-prob tokens)

Let $\mathcal{S}_k$ be the indices of the $k$ largest $p(i)$. Define:
$$
\tilde p(i) ;=; \frac{\mathbf{1}{i\in\mathcal{S}*k} , p(i)}{\sum*{j\in\mathcal{S}_k} p(j)}.
$$

* Controls **support size** directly.
* Pair with temperature: first scale, then truncate.
* Too small $k$ → bland/repetitive; too large $k$ → incoherent tails.

---

## C) Nucleus / Top-p sampling (truncate by mass)

Let (\mathcal{S}*p) be the **smallest** set s.t. (\sum*{i\in\mathcal{S}_p} p(i) \ge p). Then
[
\tilde p(i) ;=; \frac{\mathbf{1}{i\in\mathcal{S}*p}, p(i)}{\sum*{j\in\mathcal{S}_p} p(j)}.
]

* Adaptively chooses support size by **probability mass**.
* Typical (p\in[0.85,0.95]); add temperature to control entropy inside the nucleus.

---

## D) Typical sampling (match local surprisal to entropy)

Compute token surprisals (s(i)=-\log p(i)) and distribution entropy (H=-\sum_i p(i)\log p(i)). Keep tokens whose surprisal is **closest to (H)**—i.e., minimize (|s(i)-H|)—until cumulative mass (\ge p); then renormalize.

* Intuition: drop both ultra-predictable and ultra-surprising tails; keep “typical set”.
* Good at avoiding dullness *and* nonsense; hyper-params: mass (p) and (optionally) a band width around (H).

---

## E) Min-p / (\epsilon)-sampling (floor the tail)

Keep tokens above a **probability floor** (\epsilon): (\mathcal{S}={ i \mid p(i)\ge \epsilon}); renormalize.

* Prevents extremely low-probability tokens from ever being sampled (helps factuality/formatting).

---

## F) Repetition / frequency / presence penalties (discourage reuse)

Modify logits using the history counts (c_i) (frequency) or presence (\mathbf{1}{c_i>0}):
[
\hat z_i ;=; z_i - \lambda_f c_i - \lambda_p ,\mathbf{1}{c_i>0};\quad
\tilde p \propto \exp(\hat z/\tau).
]
Another popular rule (OpenAI “repetition_penalty”):
[
\hat z_i =
\begin{cases}
z_i/\rho & \text{if } i \in \text{history and } z_i>0,\
z_i\cdot \rho & \text{if } i \in \text{history and } z_i<0,\
z_i & \text{otherwise.}
\end{cases}
]

* Reduces loops; set gently (e.g., (\rho \in [1.05,1.2])) to avoid drift.

---

## G) (Greedy) Beam search (maximize sequence probability)

Deterministic search for
[
y^\star ;=; \arg\max_{y_{1:T}} \sum_{t=1}^{T} \log p(y_t \mid x,y_{<t}).
]
Keep (B) partial hypotheses (“beams”); at each step expand each by all tokens, keep top (B) by score.
**Length bias:** longer sequences accrue more negative log-prob. Fix via **length normalization**:
[
\mathrm{score}(y_{1:t}) ;=; \frac{1}{(5+t)^\alpha/(5+1)^\alpha}\sum_{k=1}^{t}\log p(y_k\mid \cdot),
]
with (\alpha!\in![0,1]).

* Pros: good for tasks with single precise target (MT w/ references).
* Cons: reduces diversity; in open-ended generation it can amplify dullness.
* **Diverse beam search:** add dissimilarity penalties between beams to spread them.

---

## H) Contrastive / anti-degeneration decoding (a.k.a. contrastive search)

Select token (i) that balances **model confidence** and **degeneracy penalty** (e.g., self-similarity of hidden states):
[
i^\star ;=; \arg\max_i \left[ \lambda \log p(i\mid \cdot);-;(1-\lambda), \underbrace{\max_{t'<t}\cos!\left(h(i),, h_{t'}\right)}_{\text{degeneracy}}\right],
]
where (h(i)) is the hidden state if we append token (i).

* Intuition: confident yet **novel** continuations.
* Often used with a small candidate set (C) (e.g., top-k) for speed.

*(Related but different: “contrastive **decoding**” that subtracts a weak LM: (\log p_{\text{strong}} - \alpha \log p_{\text{weak}}).)*

---

## I) Sampling-within-beam (stochastic beams)

Hybrid of top-k/top-p **inside** each beam expansion; still keeps top (B) partial sequences by sampled scores.

* Improves diversity while retaining some search structure.

---

## J) Constrained decoding (regex/grammar/lexical constraints)

Search over sequences that satisfy constraints (\mathcal{C}) (e.g., JSON grammar, must-include phrases). Formally:
[
y^\star = \arg\max_{y\in \mathcal{L}(\mathcal{C})} \sum_t \log p(y_t\mid \cdot).
]

* Implement via finite-state automatons (FSA) or LL(1)/PEG parsers gating logits (masking illegal next tokens).

---

## K) Calibrated stopping & entropy heuristics

* **EOS rules:** stop when (p(\text{</s>})) exceeds a threshold or repeated high EOS mass.
* **Entropy stop:** stop if (\mathsf{H}(p_t)) falls below a floor for (m) steps (model “knows” what it wants to say).
* **Length priors:** soft prior over (T) (e.g., Gaussian) added to beam scores.

---

## L) Putting it together (practical “stack”)

At each step (t):

1. Start with logits (z_t).
2. Apply **penalties** (repetition/frequency/presence or constraint masks).
3. Apply **temperature**.
4. Apply **support selection** (top-p or top-k or typical).
5. Sample (or pick by a scoring rule like contrastive).
6. Optional: enforce constraints / JSON grammar before committing token.

**Typical defaults for helpful assistants:**

* temperature (=0.7)–(0.9),
* top-p (=0.9) (or top-k (=40)–(100)),
* mild repetition penalty (1.05–1.15),
* typical sampling as an alternative to top-p when you want sharper on-topic outputs.

---

## When to use what (fast mental model)

* **Factual QA, low hallucination:** (\tau!\downarrow), top-p (0.8)–(0.9), repetition penalty on; consider contrastive search.
* **Creative writing / brainstorming:** (\tau!\uparrow), top-p (0.92)–(0.98), maybe typical sampling; minimal penalties.
* **Formal outputs (JSON, SQL):** constrained decoding (grammar mask) + low temperature.
* **Machine translation / summarization with references:** beam (with length norm) or sampling-within-beam.

---

## Short derivations that clarify intuition

1. **Temperature ≈ KL-regularization:** Sampling from (q \propto p^{1/\tau}) is equivalent to maximizing (\mathbb{E}_q[\log p]) subject to (\mathsf{H}(q)) being **higher** as (\tau) increases (Lagrange multiplier on entropy). So (\tau) trades log-likelihood vs. diversity.

2. **Top-p minimizes tail risk under a mass budget:** Among all supports of mass (\ge p), choosing the **smallest** tail (highest-prob tokens) minimizes the expected surprisal; renormalization preserves the “core” while dropping the risky tail.

3. **Typical set rationale:** For i.i.d. draws from (p), most mass lies where surprisal (s(i)\approx H). Selecting tokens near (H) approximates sampling from the **asymptotic equipartition** region, avoiding overly confident clichés and wild outliers.

4. **Contrastive search anti-loop term:** If the next-token hidden state (h(i)) is too similar to a past state, the cosine term grows, reducing the score. This penalizes **self-retrieval** and thus repetitive continuations.

---

### Quick check (one bite-sized question)

If your generations look **fluent but generic** (“safe, samey answers”), which two knobs would you try first, and in which direction would you move them? (Name the method and whether you’d increase/decrease.)

---
---