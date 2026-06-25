---
language: en
tags:
  - jepa
  - retrieval
  - multi-hop-reasoning
  - embedding-prediction
  - sequential
  - safetensors
license: apache-2.0
---

# Sequential JEPA Lens — Model Card

## Overview

**Sequential JEPA** (Joint Embedding Predictive Architecture) is a compact neural network trained to predict the **next embedding vector** in a sequence. 

| Parameter | Value |
|---|---|
| **Architecture** | Sequential JEPA (Transformer + Causal Attention) |
| **Parameters** | ~41.4M (configurable) |
| **Weight Format** | SafeTensors |
| **Inference runtime** | Rust (Candle) / PyTorch |
| **Training** | PyTorch 2.x, torch.compile, bfloat16 mixed precision |

---

## Architecture

### Key Components

#### DeltaNet (Causal Multi-Head Attention)
- Flash Attention 2 via `scaled_dot_product_attention(is_causal=True)`
- Q/K/V/O projections without bias

#### SwiGLU MLP
- Gated MLP with SiLU activation: `down(silu(gate(x)) * up(x))`
- 4× expansion
- No bias for stability

#### GatedBlock
- Pre-norm architecture (RMSNorm before each sub-layer)
- **Scalable residual**: `res_scale = 1 / √(2 × layer_idx)`
- Dropout 10% during training

#### Prediction Head
- Deep predictor with residual skip: `output = x + head(x)`
- Double expansion → intermediate RMSNorm + GELU
- Final L2-normalization of the output

---

## Training

### Loss Function

Combination of two components:

1. **Positional Weighted Cosine Similarity Loss**
   - Position 0 gets the highest weight, decaying exponentially.
   - `loss = 1 - weighted_mean(cosine_similarity)`

2. **SigReg (Signature Regularization)**, λ=0.01
   - Prevents embeddings from collapsing to a single point.
   - Random projections + variance penalty.

### Data Augmentation (on GPU)

- **Gaussian Noise**: σ=0.02 (simulates embedding model variance)
- **Message Dropout**: 10% — random position replaced with a noise L2-normalized vector.

---

## License

Apache License 2.0
