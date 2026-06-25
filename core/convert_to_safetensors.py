import os
import sys
import argparse
import torch

try:
    from safetensors.torch import save_file
except ImportError:
    print("❌ Error: 'safetensors' package is not installed.")
    print("Please install it in your environment: pip install safetensors torch")
    sys.exit(1)

def convert_pt_to_safetensors(pt_path, sf_path):
    if not os.path.exists(pt_path):
        print(f"❌ Error: PyTorch checkpoint not found at: {pt_path}")
        return False

    print(f"📂 Loading PyTorch checkpoint: {pt_path}...")
    try:
        # Load weights on CPU to avoid CUDA dependency
        checkpoint = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load PyTorch checkpoint: {e}")
        return False

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Clean keys if they have compilation prefixes
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k == "proj_matrix":
            continue
        new_key = k.replace("_orig_mod.", "")
        # Convert to FP32 as expected by candle-core in Rust
        cleaned_dict[new_key] = v.to(torch.float32)

    # Validate keys against what the Rust engine expects
    expected_keys = [
        "pos_emb.weight",
        "input_proj.weight",
        "input_norm.weight",
        "ln_f.weight",
        "pred_head.0.weight",
        "pred_head.1.weight",
        "pred_head.3.weight"
    ]
    # Check for the 6 blocks
    for i in range(6):
        expected_keys.extend([
            f"blocks.{i}.ln_1.weight",
            f"blocks.{i}.ln_2.weight",
            f"blocks.{i}.mlp.gate.weight",
            f"blocks.{i}.mlp.up.weight",
            f"blocks.{i}.mlp.down.weight",
            f"blocks.{i}.attn.q_proj.weight",
            f"blocks.{i}.attn.k_proj.weight",
            f"blocks.{i}.attn.v_proj.weight",
            f"blocks.{i}.attn.o_proj.weight",
        ])

    missing_keys = [k for k in expected_keys if k not in cleaned_dict]
    if missing_keys:
        print(f"⚠️ Warning: Checkpoint is missing {len(missing_keys)} keys expected by Rust engine:")
        for mk in missing_keys[:5]:
            print(f"  - {mk}")
        if len(missing_keys) > 5:
            print(f"  - ... and {len(missing_keys) - 5} more")
    else:
        print("✅ Checkpoint matches the expected JEPA structure perfectly!")

    # Save as safetensors
    os.makedirs(os.path.dirname(sf_path) or ".", exist_ok=True)
    print(f"💾 Saving to Safetensors format: {sf_path}...")
    try:
        save_file(cleaned_dict, sf_path)
        print("✨ Conversion completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to save Safetensors file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Sovereign JEPA weights from PyTorch (.pt) to Candle Safetensors (.safetensors)")
    parser.add_argument("--src", default="checkpoints/core_lens_sequential.pt", help="Path to source PyTorch file")
    parser.add_argument("--dst", default="checkpoints/jepa_lens_v6.safetensors", help="Path to destination Safetensors file")
    
    args = parser.parse_args()
    convert_pt_to_safetensors(args.src, args.dst)
