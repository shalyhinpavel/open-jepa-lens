# Open JEPA Lens

An open-source implementation of a **Sequential JEPA (Joint-Embedding Predictive Architecture) Lens**.

> [!NOTE]
> **Clarification:** This repository implements a *sequential/temporal predictor lens* operating in latent embedding spaces for sequential modeling. It is **not** associated with the astrophysics research paper *"Lens-JEPA: Physics Informed Joint Embedding Predictive Architecture for Gravitational Lensing"* (NeurIPS ML4PS 2025).

This repository contains the core training framework for building lightweight, stateful predictors operating entirely in embedding space (L2-normalized vectors). By predicting the *latent representations* of future states instead of raw pixels or tokens, this architecture provides a highly efficient, edge-ready engine for tasks like multi-hop reasoning, anomaly detection, and temporal tracking.

## Philosophy: Bring Your Own Data (BYOD)

This framework is agnostic to the base encoder. You can use any frozen model (e.g., standard text embedding models or Vision Transformers) to extract raw embeddings from your data, and then train the JEPA Lens on top of those sequences.

1. **Extract**: Use your preferred model to convert your sequences (video frames, text dialogs) into vectors.
2. **Train**: Use `train_jepa_sequential.py` to train the lightweight predictive lens.
3. **Deploy**: Convert the PyTorch weights to Safetensors and run them in any Rust/C++ engine at the edge.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dummy Data (for testing)

If you just want to see the pipeline run without preparing your own dataset:

```bash
python core/generate_dummy_data.py --output dummy_vectors.npy --samples 1000 --seq-len 16 --dim 640
```
This will generate 1000 random sequences of 16 steps, with a vector dimension of 640.

### 3. Train the JEPA Lens

```bash
python core/train_jepa_sequential.py --data dummy_vectors.npy --save-dir ./checkpoints --dim 640
```

The script features:
- Causal DeltaNet Attention (Flash Attention 2)
- Multi-processing PyTorch DataLoader
- Zero-CPU-blocking GPU data augmentation
- Mixed precision training (bfloat16)

### 4. Convert to Safetensors

Once training completes, you can convert the `.pt` checkpoint to `.safetensors` for deployment in Rust engines (e.g., using Candle):

```bash
python core/convert_to_safetensors.py --src checkpoints/core_lens_sequential.pt --dst checkpoints/jepa_lens.safetensors
```

## Documentation

- `docs/MODEL_CARD_JEPA.md`: Architecture and technical details of the JEPA Lens.
- `RUNPOD_TRAINING_GUIDE.md`: Instructions for scaling training on cloud GPUs like RunPod.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

