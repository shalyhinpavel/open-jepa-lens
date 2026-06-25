# Open JEPA Lens — Advanced RunPod Training Guide

This guide describes the full cycle of training the Sequential JEPA Lens on the RunPod platform using a powerful GPU (e.g., NVIDIA H100 / H200).

---

## STEP 1: Setting up a Clean Environment

```bash
# 1. Create structure
mkdir -p /workspace/open-jepa-lens/{models,data,core}

# 2. Update PyTorch for Blackwell / Hopper (CUDA 12.8+)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy safetensors tqdm
```

---

## STEP 2: Generate Vectors

You need to encode your dataset into `.npy` vectors.
If you don't have a dataset yet, you can use the dummy generator to test the pipeline:

```bash
cd /workspace/open-jepa-lens
python core/generate_dummy_data.py --output data/dummy_vectors.npy --samples 50000
```

---

## STEP 3: Start Training

```bash
cd /workspace/open-jepa-lens
python core/train_jepa_sequential.py --data data/dummy_vectors.npy --save-dir checkpoints/
```

---

## Monitoring

While training is running, open a second terminal:
```bash
watch -n 1 nvidia-smi
# OR install nvitop for beautiful visualization:
pip install nvitop && nvitop
```
