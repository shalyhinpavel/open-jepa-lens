import numpy as np
import argparse
import os

def generate_dummy_data(output_path, num_samples=1000, sequence_length=16, dim=640):
    print(f"Generating {num_samples} dummy sequences of length {sequence_length} with dim {dim}...")
    
    # Generate random normally distributed data
    data = np.random.randn(num_samples, sequence_length, dim).astype(np.float32)
    
    # L2 normalize the vectors (simulate embeddings on the unit sphere)
    norms = np.linalg.norm(data, axis=-1, keepdims=True)
    data = data / (norms + 1e-6)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, data)
    print(f"✅ Saved dummy data to {output_path} (Shape: {data.shape})")
    print("You can now run training: python core/train_jepa_sequential.py --data", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy .npy vectors for testing the JEPA pipeline")
    parser.add_argument("--output", default="dummy_vectors.npy", help="Output path")
    parser.add_argument("--samples", type=int, default=1000, help="Number of sequences")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length")
    parser.add_argument("--dim", type=int, default=640, help="Embedding dimension")
    
    args = parser.parse_args()
    generate_dummy_data(args.output, args.samples, args.seq_len, args.dim)
