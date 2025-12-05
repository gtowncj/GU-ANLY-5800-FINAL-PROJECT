# Lecture 08: Transformers

*Instructor: Chris Larson | Georgetown University | ANLY-5800 | Fall '25*

In Lecture 07 we saw how attention mechanisms solve the information bottleneck in sequence-to-sequence models. The Transformer architecture, introduced by Vaswani et al. (2017), takes this idea further by replacing recurrence entirely with self-attention, enabling fully parallel computation and forming the foundation of modern language models.

## The Limitations of Recurrent Models

Recall from Lecture 06 that RNNs process sequences sequentially:
\[
\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t)
\]

**Key limitations:**
1. **Sequential dependency:** Cannot parallelize over time steps
2. **Gradient pathways:** Even with LSTM/GRU, very long sequences suffer from degraded gradient flow
3. **Fixed computation per step:** All positions receive equal computational budget regardless of complexity

**The Transformer solution:** Replace sequential recurrence with parallel self-attention, where every position directly attends to every other position in a single operation.

---

## Self-Attention: The Core Mechanism

**Intuition from Lecture 07.** In Bahdanau attention, the decoder attends to encoder states. Self-attention applies this within a single sequence: each position attends to all positions (including itself) to build contextualized representations.

**Problem setting.** Given an input sequence of token embeddings $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_T]^T \in \mathbb{R}^{T \times d}$, we seek contextual representations $\mathbf{H} \in \mathbb{R}^{T \times d}$ where each position can attend to all others with content-dependent weights.

---

## Scaled Dot-Product Attention

Self-attention operates through three learned linear projections of the input.

**Definition 1.1 (Query, Key, Value projections).** Let $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d_k}$ be learned weight matrices. For input $\mathbf{X} \in \mathbb{R}^{T \times d}$, we compute:
$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V \in \mathbb{R}^{T \times d_k}.
$$

**Interpretation:**
- **Queries** $\mathbf{Q}$: "What am I looking for?" — each row $\mathbf{q}_i$ represents position $i$'s search pattern
- **Keys** $\mathbf{K}$: "What do I contain?" — each row $\mathbf{k}_j$ represents position $j$'s content signature
- **Values** $\mathbf{V}$: "What information do I provide?" — each row $\mathbf{v}_j$ is the actual content to retrieve

**Definition 1.2 (Scaled dot-product attention).** The attention operator computes:
$$
\operatorname{Att}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V} \in \mathbb{R}^{T \times d_k}.
$$

**Step-by-step:**
1. **Compute similarity scores:** $\mathbf{S} = \mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{T \times T}$ where $S_{ij} = \mathbf{q}_i^T \mathbf{k}_j$ measures compatibility
2. **Scale:** Divide by $\sqrt{d_k}$ to control variance (see below)
3. **Normalize:** Apply softmax row-wise to get attention weights $\mathbf{A} \in \mathbb{R}^{T \times T}$ where $\sum_j A_{ij} = 1$
4. **Aggregate:** Compute weighted sum $\mathbf{H} = \mathbf{A}\mathbf{V}$ where $\mathbf{h}_i = \sum_j A_{ij} \mathbf{v}_j$

**Why scaling by $\sqrt{d_k}$?**

**Lemma 1.1.** Under random initialization with $\mathbf{q}_i, \mathbf{k}_j \sim \mathcal{N}(0, \mathbf{I}/d_k)$, the dot product $\mathbf{q}_i^T \mathbf{k}_j$ has variance $\mathbb{E}[(\mathbf{q}_i^T \mathbf{k}_j)^2] = 1$. Without scaling, variance grows as $d_k$, causing softmax to saturate (all weight on one position).

**Proof sketch:** Each component contributes variance $1/d_k$; summing $d_k$ independent terms gives variance $d_k \cdot (1/d_k) = 1$ after scaling.

**Proposition 1.1 (Numerical stability).** For numerical stability, subtract row-wise maximum before softmax:
$$
\operatorname{softmax}(\mathbf{S})_i = \frac{\exp(S_{ij} - m_i)}{\sum_k \exp(S_{ik} - m_i)}, \quad m_i = \max_j S_{ij}.
$$
This is mathematically equivalent but prevents overflow/underflow.

---

## Multi-Head Attention

**Motivation.** A single attention mechanism learns one notion of similarity. Different relationships (syntactic, semantic, positional) may require different similarity metrics. Multi-head attention runs multiple attention operations in parallel, each learning different patterns.

**Definition 2.1 (Multi-head attention).** For $h$ heads with per-head dimension $d_h = d_k/h$:
$$
\operatorname{MHA}(\mathbf{X}) = \operatorname{Concat}(\mathbf{H}_1,\ldots,\mathbf{H}_h)\,\mathbf{W}^O
$$
where each head computes:
$$
\mathbf{H}_i = \operatorname{Att}(\mathbf{X}\mathbf{W}_i^Q,\,\mathbf{X}\mathbf{W}_i^K,\,\mathbf{X}\mathbf{W}_i^V) \in \mathbb{R}^{T \times d_h}
$$
with head-specific projections $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V \in \mathbb{R}^{d \times d_h}$ and output projection $\mathbf{W}^O \in \mathbb{R}^{(h \cdot d_h) \times d}$.

**Example (h=8 heads, d=512).** Each head operates in $d_h = 64$ dimensions. The 8 heads might learn to attend to:
- Head 1: Previous token (local syntax)
- Head 2: Subject-verb agreement (long-range syntax)
- Head 3: Semantic similarity
- Head 4: Positional patterns
- Heads 5-8: Other task-specific patterns

**Theorem 2.1 (Expressivity).** Multi-head attention with $h$ heads can represent $h$ distinct attention patterns and linearly combine them, strictly increasing expressivity over single-head attention for fixed total parameters.

**Proof idea:** Each head learns independent $\mathbf{W}_i^Q, \mathbf{W}_i^K$ defining distinct similarity metrics. The output projection $\mathbf{W}^O$ learns task-specific weights for combining head outputs. Single-head attention is the special case $h=1$.

---

## The Transformer Block

A Transformer block combines self-attention with position-wise feed-forward layers, connected via residual pathways and layer normalization.

### Feed-Forward Networks

**Definition 3.1 (Position-wise FFN).** Applied independently to each position:
$$
\operatorname{FFN}(\mathbf{z}) = \sigma(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$
where $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, and typically $d_{\text{ff}} = 4d$ (expansion ratio).

**Role:** While attention mixes information across positions, FFN processes each position independently, providing non-linear transformation capacity. The expansion allows the model to project into a higher-dimensional space for richer representations.

**Common activation:** GELU (Gaussian Error Linear Unit) $\sigma(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard Gaussian CDF, providing smooth non-linearity.

### Layer Normalization

**Definition 3.2 (LayerNorm).** For vector $\mathbf{x} \in \mathbb{R}^d$:
$$
\operatorname{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$
where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learned parameters.

**Purpose:** Stabilizes activations by normalizing to zero mean and unit variance per layer, improving gradient flow in deep networks.

### Pre-Norm Architecture

**Definition 3.3 (Pre-norm Transformer block).** Given input $\mathbf{H}^{(\ell)} \in \mathbb{R}^{T\times d}$:
$$
\begin{aligned}
\tilde{\mathbf{H}}^{(\ell)} &= \operatorname{LN}(\mathbf{H}^{(\ell)}) \\
\mathbf{U}^{(\ell)} &= \operatorname{MHA}(\tilde{\mathbf{H}}^{(\ell)}) \\
\mathbf{Z}^{(\ell)} &= \mathbf{H}^{(\ell)} + \mathbf{U}^{(\ell)} \quad \text{(residual connection)} \\
\hat{\mathbf{Z}}^{(\ell)} &= \operatorname{LN}(\mathbf{Z}^{(\ell)}) \\
\mathbf{F}^{(\ell)} &= \operatorname{FFN}(\hat{\mathbf{Z}}^{(\ell)}) \\
\mathbf{H}^{(\ell+1)} &= \mathbf{Z}^{(\ell)} + \mathbf{F}^{(\ell)} \quad \text{(residual connection)}
\end{aligned}
$$

**Residual connections** (from ResNets, He et al. 2016) create direct gradient pathways, enabling training of very deep networks (100+ layers in modern LLMs).

**Pre-norm vs. Post-norm:** The original Transformer used post-norm (LN after residual add). Pre-norm applies LN before each sub-layer, providing more stable gradients for deep stacks and often eliminating the need for learning rate warmup.

### Causal Masking for Autoregressive Models

**Definition 3.4 (Causal mask).** For autoregressive (left-to-right) generation, we prevent position $i$ from attending to future positions $j > i$ by adding a mask:
$$
M_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$
Applied before softmax: $\operatorname{softmax}(\mathbf{S} + \mathbf{M})$.

**Effect:** The $-\infty$ values become $\exp(-\infty) = 0$ after softmax, ensuring each position only attends to itself and previous positions. This maintains the autoregressive property needed for language modeling.

---

## Positional Encodings

**The permutation problem.** Self-attention is permutation-equivariant: if we shuffle the input sequence, the output shuffles identically. Without positional information, "cat sat on mat" and "mat on sat cat" produce the same representation. We must inject position information.

### Sinusoidal Positional Encodings

**Definition 4.1 (Sinusoidal encoding).** For position $t \in \{0, 1, \ldots, T-1\}$ and dimension index $i \in \{0, 1, \ldots, d-1\}$:
$$
\mathrm{PE}(t, 2k) = \sin\!\left(\frac{t}{10000^{2k/d}}\right), \quad \mathrm{PE}(t, 2k+1) = \cos\!\left(\frac{t}{10000^{2k/d}}\right)
$$
where $k = \lfloor i/2 \rfloor$.

**Intuition:** Each dimension uses a different frequency. Low dimensions oscillate quickly (capture local position), high dimensions oscillate slowly (capture global position). The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

**Key property (relative position).** Using trigonometric identities:
$$
\sin(t + \phi) = \sin(t)\cos(\phi) + \cos(t)\sin(\phi)
$$
The encoding at position $t + k$ can be expressed as a linear function of the encoding at position $t$, allowing the model to learn relative positions.

**Usage:** Add positional encodings to token embeddings:
$$
\mathbf{X}_{\text{input}} = \mathbf{X}_{\text{tokens}} + \mathbf{PE}
$$

### Alternative: Learned Positional Embeddings

**Definition 4.2 (Learned positions).** Treat position as a categorical variable with learned embedding matrix $\mathbf{P} \in \mathbb{R}^{T_{\max} \times d}$ where $T_{\max}$ is maximum sequence length.

**Advantages:** Flexible, can learn task-specific positional patterns.

**Disadvantages:** Cannot generalize to sequences longer than $T_{\max}$ seen during training.

### Rotary Position Embeddings (RoPE)

**Definition 4.3 (RoPE).** Instead of adding position information, rotate the query and key vectors in complex space as a function of position. For position $t$ and dimension pair $(2k, 2k+1)$:
$$
\begin{bmatrix} q_{2k}' \\ q_{2k+1}' \end{bmatrix} = \begin{bmatrix} \cos(t\theta_k) & -\sin(t\theta_k) \\ \sin(t\theta_k) & \cos(t\theta_k) \end{bmatrix} \begin{bmatrix} q_{2k} \\ q_{2k+1} \end{bmatrix}
$$
where $\theta_k = 10000^{-2k/d}$.

**Key advantage:** The dot product $\mathbf{q}_i^T \mathbf{k}_j$ naturally encodes relative position $|i-j|$ through the rotation angles, providing better length extrapolation. Used in modern LLMs (GPT-NeoX, LLaMA, PaLM).

---

## Tokenization Schemes

**The vocabulary problem.** Word-level vocabularies are too large (millions of words) and cannot handle unseen words (out-of-vocabulary, OOV). Character-level models have small vocabularies but produce very long sequences. Subword tokenization provides the optimal trade-off.

### Byte-Pair Encoding (BPE)

**Algorithm 5.1 (BPE Training).**
1. Initialize vocabulary with all characters (or bytes)
2. Repeat until vocabulary reaches target size:
   - Find the most frequent adjacent pair of symbols in the corpus
   - Merge this pair into a new symbol
   - Add new symbol to vocabulary

**Example:**
```
   Corpus: "low low low lower lower newest widest"
   Iteration 1: "lo" + "w" → "low" (most frequent pair)
   Iteration 2: "low" + "er" → "lower"
   Iteration 3: "new" + "est" → "newest"
   ...
```

**Encoding (inference):** Greedily apply longest-matching merge rules.

**Definition 5.1 (BPE).** A deterministic tokenization algorithm that builds vocabulary bottom-up by iteratively merging frequent character (or byte) pairs.

### WordPiece

**Definition 5.2 (WordPiece).** Similar to BPE but selects merges to maximize the likelihood of the training corpus under a unigram language model:
$$
\text{score}(x, y) = \frac{\log P(xy)}{\log P(x) + \log P(y)}
$$
Merge the pair with highest score. Used in BERT.

### SentencePiece

**Definition 5.3 (SentencePiece Unigram).** Starts with a large vocabulary and iteratively removes tokens that least decrease the likelihood:
$$
\mathcal{L} = \sum_{i=1}^{N} \log P(X_i)
$$
where $P(X_i) = \sum_{x \in S(X_i)} P(x)$ sums over all possible segmentations $S(X_i)$ of sentence $X_i$.

**Advantages:**
- Language-independent (treats text as raw Unicode or bytes)
- Reversible (can reconstruct original text exactly, including whitespace)
- Probabilistic (provides multiple segmentations with probabilities)

**Practical considerations:**
- **Vocabulary size:** Typically 32k-100k tokens balances sequence length and vocabulary coverage
- **Special tokens:** Add `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` for specific tasks
- **Normalization:** Handle Unicode, case, accents consistently
- **Rare words:** Subword tokenization ensures all words can be represented (no true OOV)

---

## Transformer Architectures: Encoder, Decoder, and Encoder–Decoder

The Transformer framework supports three architectural variants, each suited to different tasks.

### Encoder-Only (BERT-class)

**Architecture:** Stack of bidirectional self-attention blocks (no causal masking).

**Input:** Tokenized text with special tokens: `[CLS] token1 token2 ... [SEP]`

**Self-attention:** Each position attends to all positions (past and future).

**Training objective:** Masked Language Modeling (MLM) — randomly mask 15% of tokens and predict them:
$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log p_\theta(x^t \mid x_{\setminus \mathcal{M}})
$$

**Use cases:**
- Sentence classification (sentiment analysis)
- Token classification (NER, POS tagging)
- Sentence-pair tasks (entailment, similarity)
- Question answering (span extraction)

**Examples:** BERT, RoBERTa, ALBERT

### Decoder-Only (GPT-class)

**Architecture:** Stack of causal self-attention blocks (with causal masking).

**Input:** Tokenized text: `token1 token2 token3 ...`

**Self-attention:** Each position attends only to itself and previous positions.

**Training objective:** Autoregressive language modeling — predict next token:
$$
\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log p_\theta(x^t \mid x^{<t})
$$

**Use cases:**
- Text generation
- Few-shot learning (via prompting)
- Code generation
- Dialogue

**Examples:** GPT-2, GPT-3, GPT-4, LLaMA, PaLM

### Encoder-Decoder (T5, BART, Original Transformer)

**Architecture:** Encoder stack + decoder stack connected via cross-attention.

**Encoder:** Bidirectional self-attention processes input.

**Decoder:** Causal self-attention + cross-attention to encoder outputs.

**Definition 6.1 (Cross-attention).** The decoder attends to encoder representations. With decoder queries $\mathbf{Q}_{\text{dec}}$ and encoder keys/values $\mathbf{K}_{\text{enc}}, \mathbf{V}_{\text{enc}}$:
$$
\operatorname{Att}_{\times}(\mathbf{Q}_{\text{dec}},\mathbf{K}_{\text{enc}},\mathbf{V}_{\text{enc}}) = \operatorname{softmax}\!\left(\frac{\mathbf{Q}_{\text{dec}} \mathbf{K}_{\text{enc}}^T}{\sqrt{d_k}}\right) \mathbf{V}_{\text{enc}}
$$

**Decoder block structure:**
1. Causal self-attention (on decoder inputs)
2. Cross-attention (to encoder outputs)
3. Feed-forward network

**Use cases:**
- Machine translation
- Summarization
- Question answering (generative)
- Text-to-text tasks

**Examples:** Original Transformer, T5, BART, mT5

**Comparison:**

| Architecture | Attention Pattern | Training Objective | Best For |
|--------------|-------------------|-------------------|----------|
| Encoder-only | Bidirectional | MLM | Understanding tasks |
| Decoder-only | Causal | Next-token prediction | Generation tasks |
| Encoder-decoder | Both + cross | Conditional generation | Seq2seq tasks |

---

## Training Objectives: Causal vs. Non-Causal

The choice of training objective determines what the model learns and what tasks it excels at.

### Autoregressive (Causal) Language Modeling

**Definition 7.1 (Autoregressive objective).** For sequence $x_{1:T}$ and parameters $\theta$:
$$
\mathcal{L}_{\text{AR}}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x^t \mid x^{<t})
$$

**Factorization:** Uses the chain rule to model the joint distribution:
$$
p(x^{1:T}) = \prod_{t=1}^{T} p(x^t \mid x^{<t})
$$

**Properties:**
- Consistent probability distribution over sequences
- Enables natural text generation via sampling
- Each token prediction uses only past context
- Used by GPT models

### Masked Language Modeling (Non-Causal)

**Definition 7.2 (Masked language modeling).** With random mask set $\mathcal{M}$ (typically 15% of tokens):
$$
\mathcal{L}_{\text{MLM}}(\theta) = -\sum_{t \in \mathcal{M}} \log p_\theta(x^t \mid x_{\setminus \mathcal{M}})
$$

**Masking strategy (BERT):**
- 80% of time: Replace with `[MASK]` token
- 10% of time: Replace with random token
- 10% of time: Keep original token

**Properties:**
- Uses bidirectional context (sees both past and future)
- Does not define a proper generative model
- Better representations for understanding tasks
- Used by BERT, RoBERTa

**Trade-offs:**

| Aspect | Autoregressive (AR) | Masked LM (MLM) |
|--------|---------------------|-----------------|
| Context | Unidirectional (causal) | Bidirectional |
| Generation | Natural (sample left-to-right) | Requires iterative refinement |
| Understanding | Good | Excellent |
| Pre-training efficiency | Sees each token once | Masks only 15% |
| Probability model | Proper joint distribution | Conditional marginals |

---

## Computational Complexity

**Theorem 8.1 (Attention complexity).** For sequence length $T$ and model dimension $d$:
- **Time:** $O(T^2 d)$ — dominated by computing $\mathbf{Q}\mathbf{K}^T$ and applying attention weights
- **Memory:** $O(T^2 + Td)$ — storing attention matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$ and activations

**Bottleneck:** The $T^2$ term makes vanilla Transformers impractical for very long sequences (e.g., $T > 8192$).

**Solutions for long sequences:**
- **Sparse attention:** Attend to subset of positions (Longformer, BigBird)
- **Linear attention:** Approximate attention with kernels (Performer, Linear Transformer)
- **State-space models:** Replace attention entirely (S4, Mamba) — see Lecture 10
- **Hierarchical models:** Process in chunks with cross-chunk attention

---

## Implementation and Training Considerations

**Initialization:** Use careful weight initialization to maintain activation variance across layers. For residual branches, scale by $1/\sqrt{L}$ where $L$ is depth (ReZero, DeepNet schemes).

**Optimization:** Standard recipe:
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$)
- **Learning rate schedule:** Warmup for first 4000-10000 steps, then cosine decay
- **Gradient clipping:** Clip norm to 1.0 to prevent instability
- **Weight decay:** $\lambda = 0.01$ for regularization

**Regularization:**
- **Dropout:** Apply to attention weights and FFN activations (typically 0.1)
- **Label smoothing:** For classification, use soft targets $y_{\text{smooth}} = (1-\epsilon)y + \epsilon/K$
- **Layer dropout (DropPath):** Randomly drop entire layers during training

**Mixed precision training:** Use FP16 for forward/backward, FP32 for parameter updates. Provides 2-3x speedup with minimal accuracy loss.

**Attention as kernel averaging:** Softmax attention computes $\mathbf{h}_i = \sum_j \alpha_{ij} \mathbf{v}_j$ where $\alpha_{ij} \propto \exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})$. This is a kernel method with exponential kernel, retrieving a weighted combination of values.

---

## Mathematical Connections and Theory

### Connection to Hopfield Networks

**Proposition 8.1 (Self-attention as associative memory).** Recall from Lecture 06 that continuous Hopfield networks with softmax nonlinearity have exponential storage capacity. Self-attention is mathematically equivalent to one update step of a modern Hopfield network.

**Proof sketch:** A Hopfield network retrieves pattern $\mathbf{x}_{\text{new}} = \mathbf{X} \operatorname{softmax}(\beta \mathbf{X}^T \mathbf{x})$ from memory $\mathbf{X}$. Setting queries $\mathbf{Q} = \mathbf{X}$, keys $\mathbf{K} = \mathbf{X}$, values $\mathbf{V} = \mathbf{X}$, and $\beta = 1/\sqrt{d_k}$, we recover self-attention.

**Implication:** Transformers inherit the exponential memory capacity of modern Hopfield networks, explaining their ability to store and retrieve vast amounts of information.

### Permutation Equivariance

**Proposition 8.2 (Equivariance property).** Without positional encodings, self-attention is permutation equivariant: for any permutation $\pi$ of sequence positions,
$$
\operatorname{MHA}(\pi \cdot \mathbf{X}) = \pi \cdot \operatorname{MHA}(\mathbf{X})
$$

**Proof:** The operations $\mathbf{Q}\mathbf{K}^T$ and softmax are permutation-equivariant in rows. Multiplying by $\mathbf{V}$ preserves this property.

**Consequence:** Positional encodings are essential—they break the symmetry to inject order information.

### Universal Approximation

**Theorem 8.2 (Yun et al., 2020).** A Transformer with sufficient width and depth can approximate any sequence-to-sequence function with arbitrary precision, making it a universal approximator for sequence processing.

This extends the classical universal approximation theorem for feed-forward networks to the sequential setting.

---

## Summary and Practical Recipe

The Transformer architecture revolutionized NLP by replacing recurrence with self-attention, enabling:
- **Parallel computation:** All positions processed simultaneously
- **Long-range dependencies:** Direct connections between any pair of positions
- **Scalability:** Architecture scales to billions of parameters

**Building a Transformer language model:**

1. **Tokenization:** Train BPE/SentencePiece on corpus (vocab size 32k-100k)
2. **Embeddings:** Token embeddings + positional encodings (RoPE recommended)
3. **Architecture:** Stack $L$ decoder blocks (12-96 layers for modern LLMs)
   - Each block: Pre-norm → Causal MHA → Residual → Pre-norm → FFN → Residual
   - Hidden dim $d = 768$-$12288$, heads $h = 12$-$96$, FFN expansion $4d$
4. **Output:** Linear projection to vocabulary, optionally tied to input embeddings
5. **Training:** AdamW optimizer, cosine LR schedule with warmup, gradient clipping
6. **Objective:** Minimize autoregressive loss $\mathcal{L}_{\text{AR}}$
7. **Monitor:** Validation perplexity, use mixed precision (FP16), checkpoint regularly

**Key hyperparameters (GPT-3 scale):**
- Parameters: 175B
- Layers: 96
- Hidden dim: 12288
- Heads: 96
- Context length: 2048 (training), extendable with RoPE
- Batch size: 3.2M tokens
- Learning rate: $6 \times 10^{-5}$ with warmup

---

## References

- Vaswani et al. (2017) "Attention Is All You Need" — Original Transformer paper
- Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2019) "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Brown et al. (2020) "Language Models are Few-Shot Learners" (GPT-3)
- Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson (2018) "SentencePiece: A simple and language independent approach to subword tokenization"
- Press & Wolf (2017) "Using the Output Embedding to Improve Language Models"
- Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE)
- Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture" (Pre-norm)
