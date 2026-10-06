# Decoder-only-architecture-
# Decoder-Only Transformer (Mini GPT) from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karrothu-vamsi-krishna/Decoder-only-architecture-/blob/main/Decoder_transformer_model.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue)

A character-level, decoder-only Transformer (GPT-style) implemented from scratch in PyTorch and trained on the Tiny Shakespeare dataset. The notebook builds the model step by step, from a simple bigram baseline up to a multi-layer Transformer with self-attention, so each component can be understood on its own.

## What's inside

The notebook `Decoder_transformer_model.ipynb` covers:

1. **Data pipeline**: loading Tiny Shakespeare (~1.1M characters), building a 65-character vocabulary, encode/decode functions, and a 90/10 train/validation split.
2. **Batching**: `get_batch` creates random context windows with targets shifted by one position.
3. **Baseline**: a bigram language model (token embedding table only).
4. **Self-attention**: a single causal attention head (query, key, value, lower-triangular mask, softmax).
5. **Transformer components**:
   - Multi-head attention with output projection
   - Position-wise feed-forward network (4x expansion, ReLU)
   - Transformer `Block` with residual connections and pre-LayerNorm
6. **GPT language model**: token + positional embeddings, stacked blocks, final LayerNorm, linear head, and autoregressive `generate`.
7. **Training and evaluation**: AdamW optimizer, cross-entropy loss, and `estimate_loss` for train/validation tracking.

## Architecture

```
Input tokens
   │
Token embedding + Positional embedding
   │
┌──▼────────────────────────────┐
│ Transformer Block  (x n_layer)│
│   LayerNorm → Causal          │
│   Multi-Head Self-Attention   │
│   + residual                  │
│   LayerNorm → Feed-Forward    │
│   + residual                  │
└──┬────────────────────────────┘
   │
Final LayerNorm
   │
Linear head → logits over vocabulary
```

## Final configuration

| Hyperparameter | Value |
| --- | --- |
| Vocabulary size | 65 (characters) |
| Embedding size (`n_embd`) | 64 |
| Attention heads (`n_head`) | 4 |
| Layers (`n_layer`) | 4 |
| Context length (`block_size`) | 32 |
| Batch size | 16 |
| Optimizer | AdamW, lr = 1e-3 |
| Training steps | 5000 |
| Hardware | CPU (Colab, no GPU) |

## Results

| Model | Loss |
| --- | --- |
| Bigram baseline (10,000 steps) | ~2.45 (train) |
| Mini GPT, context 8, n_embd 32 (5,000 steps) | ~2.12 (train) |
| Mini GPT, context 32, n_embd 64 (5,000 steps) | ~1.67 train / ~1.85 val |

Validation loss over training for the final model:

| Step | Train loss | Val loss |
| --- | --- | --- |
| 0 | 4.4548 | 4.4541 |
| 1000 | 2.0849 | 2.1216 |
| 2000 | 1.8527 | 1.9676 |
| 3000 | 1.7422 | 1.9139 |
| 4000 | 1.6974 | 1.8452 |
| 4500 | 1.6733 | 1.8470 |

The starting loss of about 4.45 is close to random guessing over 65 characters (ln 65 ≈ 4.17), and it drops steadily as the model learns.

### Sample generated text

```
BRUTUS:
If how hy.

Therein, the true. Nay's thing my flance.
Ay my prryseed is this Mosshow way,
Seep in Befinger yieldly to time;
```

The model learns Shakespeare's structure (speaker names, line breaks, punctuation, many real words) but not full coherence, which is expected for a ~0.2M-parameter model trained for 5,000 steps.

## How to run

**On Colab (easiest):** click the **Open In Colab** badge above and run all cells.

**Locally:**

```bash
git clone https://github.com/karrothu-vamsi-krishna/Decoder-only-architecture-.git
cd Decoder-only-architecture-
pip install torch jupyter
jupyter notebook Decoder_transformer_model.ipynb
```

The dataset is downloaded automatically by the second cell (`wget`). On Windows, replace the `!wget` line with a manual download of `input.txt` from the [char-rnn repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

## Possible improvements

- Add dropout (the `dropout` variable is defined but not yet used in the layers)
- Train longer and scale up `n_embd`, `n_layer`, and `block_size` on a GPU
- Use a subword tokenizer (e.g. BPE) instead of characters
- Add learning-rate scheduling and gradient clipping
- Add top-k / temperature sampling in `generate`

## Acknowledgements

- Inspired by Andrej Karpathy's "Let's build GPT" walkthrough and the [nanoGPT](https://github.com/karpathy/nanoGPT) project.
- Dataset: Tiny Shakespeare, from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).
- Architecture based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).

## Author

**Vamsi Krishna Karrothu**
[GitHub](https://github.com/karrothu-vamsi-krishna) · [LinkedIn](https://linkedin.com/in/vamsi-krishna-karrothu-04995036a) · [X (Twitter)](https://x.com/karrothu_v6387)
