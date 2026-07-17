"""
Open JEPA Lens — Sequential JEPA Trainer (H200 / CUDA)
Adapted for standard embeddings (e.g., dim=640 or 1024).
Uses High-Performance PyTorch Dataset + DataLoader with Multi-Processing Workers,
GPU-based fast augmentation, pin_memory, and non_blocking transfers.

Usage:
    python core/train_jepa_sequential.py --data dummy_vectors.npy --save-dir ./checkpoints
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from dataclasses import dataclass

@dataclass
class JEPAConfig:
    n_embd: int = 640    # Default embedding dimension
    n_head: int = 10     # 640 / 10 = 64 per head (clean division)
    n_layer: int = 6
    dropout: float = 0.1
    max_seq_len: int = 512 # Supported maximum sequence length for positional embedding

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
        self.up = nn.Linear(config.n_embd, config.n_embd * 4, bias=False)
        self.down = nn.Linear(config.n_embd * 4, config.n_embd, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class GatedBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.attn = FlashAttention(config)
        self.res_scale = 1.0 / math.sqrt(2.0 * max(1, layer_idx))
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln_1(x)) * self.res_scale)
        x = x + self.dropout(self.mlp(self.ln_2(x)) * self.res_scale)
        return x

def compute_sigreg_loss(z, proj_matrix):
    """
    Variance regularization loss à la VICReg.
    Variance is calculated across projections of embeddings to prevent representation collapse.
    """
    B, T, D = z.shape
    z_flat = z.view(B * T, D)
    projections = z_flat @ proj_matrix
    mean = projections.mean(dim=0)
    var = projections.var(dim=0, unbiased=False)
    return mean.pow(2).mean() + (var - 1.0).pow(2).mean()

class SequentialJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        
        # Fixed random projection matrix for VICReg-style variance regularization
        self.register_buffer(
            'proj_matrix',
            F.normalize(torch.randn(config.n_embd, 64), dim=0)
        )
        
        self.input_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.input_norm = RMSNorm(config.n_embd)
        
        self.blocks = nn.ModuleList([GatedBlock(config, i+1) for i in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        
        self.pred_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2, bias=False),
            RMSNorm(config.n_embd * 2),
            nn.GELU(),
            nn.Linear(config.n_embd * 2, config.n_embd, bias=False),
        )

    def forward(self, x, targets=None):
        B, T, D = x.shape
        
        pos = torch.arange(T, device=x.device)
        x_input = x + self.pos_emb(pos)
        
        x = self.input_norm(self.input_proj(x_input))
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        preds = x + self.pred_head(x)
        preds = F.normalize(preds, dim=-1)

        loss, cos_sim = None, None
        if targets is not None:
            targets_norm = F.normalize(targets, dim=-1)
            cos_sim_all = F.cosine_similarity(preds, targets_norm, dim=-1)
            cos_sim = cos_sim_all.mean()

            pos_weights = torch.tensor(
                [1.0 / (2.0 ** i) for i in range(T)],
                device=preds.device, dtype=preds.dtype
            )
            pos_weights = pos_weights / pos_weights.sum()

            weighted_sim = (cos_sim_all * pos_weights.unsqueeze(0)).sum(dim=-1).mean()
            sim_loss = 1.0 - weighted_sim

            sigreg_loss = compute_sigreg_loss(preds, self.proj_matrix)
            loss = sim_loss + 0.01 * sigreg_loss
        return preds, loss, cos_sim


# ================================================================
# High-Performance PyTorch Dataset
# ================================================================
class SequentialDataset(Dataset):
    def __init__(self, path, split='train'):
        print(f"📥 Loading sequences from {path} for {split}...")
        raw = np.load(path)  # (N, WINDOW, DIM)
        
        mask = ~np.isnan(raw).any(axis=(1, 2))
        first_nonzero = np.abs(raw[:, 0, :]).sum(axis=1) > 0
        mask = mask & first_nonzero
        clean = raw[mask]
        
        np.random.seed(42)
        n = len(clean)
        idx = np.arange(n)
        np.random.shuffle(idx)
        split_point = int(n * 0.9)
        
        if split == 'train':
            self.data = clean[idx[:split_point]]
        else:
            self.data = clean[idx[split_point:]]
        
        print(f"📊 {split.capitalize()} set: {len(self.data):,} clean sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# ================================================================
# GPU-Accelerated Batch Augmentation (0ms CPU blocking)
# ================================================================
def augment_batch(batch, device):
    B, T, D = batch.shape
    # Clone to avoid in-place modification warnings, and reduce noise magnitude to 0.005
    batch = batch.clone() + torch.randn_like(batch) * 0.005
    
    drop_mask = torch.rand(B, device=device) < 0.1
    if drop_mask.any():
        drop_pos = torch.randint(0, T - 1, (B,), device=device)
        active_indices = torch.where(drop_mask)[0]
        active_positions = drop_pos[active_indices]
        noise = F.normalize(torch.randn(len(active_indices), D, device=device), dim=-1)
        batch = batch.clone()
        batch[active_indices, active_positions] = noise
    return batch

# ================================================================
# Training Loop
# ================================================================
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Device: {device}")

    config = JEPAConfig(n_embd=args.dim)
    model = SequentialJEPA(config).to(device)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune', fullgraph=True)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🧠 Sequential JEPA: {param_count/1e6:.2f}M parameters (dim={config.n_embd})")

    BATCH = 256
    ACCUM_STEPS = 4
    LR_MAX = 1e-3
    LR_MIN = 1e-5
    WARMUP_STEPS = 500
    EPOCHS = 300
    NUM_WORKERS = min(8, os.cpu_count() or 4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MIN, weight_decay=0.01, betas=(0.9, 0.98))
    
    data_path = args.data
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "core_lens_sequential.pt")

    train_dataset = SequentialDataset(path=data_path, split='train')
    val_dataset = SequentialDataset(path=data_path, split='val')
    
    # GradScaler is removed since bf16 training does not require loss scaling
    
    steps_per_epoch = len(train_dataset) // BATCH
    if steps_per_epoch == 0:
        print("Dataset too small for batch size. Using 1 step per epoch.")
        steps_per_epoch = 1
    total_steps = max(1, EPOCHS * steps_per_epoch)
    
    def get_lr(step):
        if step < WARMUP_STEPS:
            return LR_MIN + (LR_MAX - LR_MIN) * step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))
    
    global_step = 0
    best_val_sim = 0.0
    patience_counter = 0
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

    model.train()
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        epoch_sims = []
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(train_loader):
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            batch = batch.to(device, non_blocking=True)
            batch = augment_batch(batch, device)
            
            x = batch[:, :-1, :]
            y = batch[:, 1:, :]
            
            if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                _, loss, cos_sim = model(x, targets=y)
                loss = loss / ACCUM_STEPS
            
            loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            
            epoch_sims.append(cos_sim.item())
            
            if step % 200 == 0:
                avg_sim = np.mean(epoch_sims[-200:]) if len(epoch_sims) >= 200 else np.mean(epoch_sims)
                print(f"Step {step:05d}/{steps_per_epoch} | CosSim: {cos_sim.item():.4f} | AvgSim: {avg_sim:.4f} | LR: {lr:.2e}")

        model.eval()
        val_sims = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_batch = v_batch.to(device, non_blocking=True)
                vx = v_batch[:, :-1, :]
                vy = v_batch[:, 1:, :]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    _, _, v_sim = model(vx, targets=vy)
                val_sims.append(v_sim.item())
        
        val_mean = np.mean(val_sims) if len(val_sims) > 0 else 0.0
        epoch_mean = np.mean(epoch_sims) if len(epoch_sims) > 0 else 0.0
        print(f"✨ Validation CosSim: {val_mean:.4f} | Epoch AvgSim: {epoch_mean:.4f}")
        
        if val_mean > best_val_sim:
            best_val_sim = val_mean
            patience_counter = 0
            state_dict = model.state_dict()
            clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            torch.save({
                'model': clean_dict,
                'config': vars(config)
            }, save_path)
            print(f"💾 New best! Saved to {save_path} (CosSim: {val_mean:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️ Validation didn't improve. Patience: {patience_counter}/2")
            if patience_counter >= 2:
                print("\n🛑 Early stopping triggered! Validation stopped improving 2 times in a row.")
                break
        
        model.train()

    print(f"\n🏁 Training complete! Best Val CosSim: {best_val_sim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sequential JEPA model")
    parser.add_argument("--data", default="dummy_vectors.npy", help="Path to .npy vectors (N, WINDOW, DIM)")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--dim", type=int, default=640, help="Dimension of input vectors")
    args = parser.parse_args()
    train(args)
