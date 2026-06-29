"""
training/pretrain_ae.py

Autoencoder pre-training on LOBSTER LOB snapshots.

Trains a convolutional AE on the depth-profile vectors produced by
data/process_lobster.py and saves the encoder weights for downstream
use as a frozen feature extractor in the RL agent.

Architecture
------------
Input: 2K-dim LOB depth vector (K ask sizes + K bid sizes, K=10 levels)
       Produced by process_lobster.py as lob_snapshots.npy, shape (N, 2K).

Encoder:
    Reshape    → (batch, 1, 2K)          # treat depth profile as 1D signal
    Conv1d(1→32, kernel=3, padding=1)   # (batch, 32, 2K)
    ReLU
    Conv1d(32→16, kernel=3, padding=1)  # (batch, 16, 2K)
    ReLU
    Flatten                              # (batch, 16 * 2K)
    Linear(16*2K → latent_dim)          # (batch, latent_dim)

Decoder (mirror):
    Linear(latent_dim → 16*2K)
    ReLU
    Unflatten → (batch, 16, 2K)
    ConvTranspose1d(16→32, kernel=3, padding=1)
    ReLU
    ConvTranspose1d(32→1, kernel=3, padding=1)
    Flatten → (batch, 2K)               # reconstruction

Loss: MSE reconstruction loss.

Output: checkpoints/ae_encoder_<latent_dim>.pt
        Contains encoder state_dict only (decoder discarded after training).

Usage
-----
    # Train all three latent dim variants
    python training/pretrain_ae.py --latent_dim 16
    python training/pretrain_ae.py --latent_dim 8
    python training/pretrain_ae.py --latent_dim 32

    # Full run with custom data path
    python training/pretrain_ae.py \\
        --snapshots data/processed/lob_snapshots.npy \\
        --latent_dim 16 \\
        --epochs 100 \\
        --batch_size 2048 \\
        --lr 1e-3

Week 5 deliverable (run before Week 6 RL training).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# ── Architecture ──────────────────────────────────────────────────────────────

class LOBEncoder(nn.Module):
    """
    Convolutional encoder for LOB depth profiles.

    Treats the 2K-dim depth vector as a 1D signal and applies two
    convolutional layers to extract local structure (adjacent price
    levels tend to be correlated), then projects to a latent vector.

    Parameters
    ----------
    input_dim  : int — 2 * n_levels (default 20 for K=10)
    latent_dim : int — bottleneck dimension (ablate: 8, 16, 32)
    """

    def __init__(self, input_dim: int = 20, latent_dim: int = 16):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self._flat_dim  = 16 * input_dim   # channels × length after conv

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),   # (B, 32, input_dim)
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # (B, 16, input_dim)
            nn.ReLU(),
        )
        self.proj = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (B, input_dim) — normalised depth profile

        Returns
        -------
        Tensor shape (B, latent_dim) — latent representation
        """
        h = x.unsqueeze(1)          # (B, 1, input_dim)
        h = self.conv(h)             # (B, 16, input_dim)
        h = h.flatten(start_dim=1)  # (B, 16 * input_dim)
        return self.proj(h)          # (B, latent_dim)


class LOBDecoder(nn.Module):
    """
    Transposed-convolutional decoder — mirrors LOBEncoder.

    Parameters
    ----------
    input_dim  : int — output dimension = 2 * n_levels
    latent_dim : int — bottleneck dimension (must match encoder)
    """

    def __init__(self, input_dim: int = 20, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self._flat_dim = 16 * input_dim

        self.proj = nn.Sequential(
            nn.Linear(latent_dim, self._flat_dim),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor shape (B, latent_dim)

        Returns
        -------
        Tensor shape (B, input_dim) — reconstructed depth profile
        """
        h = self.proj(z)                               # (B, 16 * input_dim)
        h = h.view(-1, 16, self.input_dim)             # (B, 16, input_dim)
        h = self.deconv(h)                             # (B, 1, input_dim)
        return h.squeeze(1)                            # (B, input_dim)


class LOBAutoencoder(nn.Module):
    """Full autoencoder — used only during pre-training."""

    def __init__(self, input_dim: int = 20, latent_dim: int = 16):
        super().__init__()
        self.encoder = LOBEncoder(input_dim, latent_dim)
        self.decoder = LOBDecoder(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ── Data loading ──────────────────────────────────────────────────────────────

def load_snapshots(
    path:       str,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    seed:       int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load LOB snapshots from .npy file and split into train/val/test.

    The snapshot array produced by process_lobster.py has shape (N, 2K)
    where each row is [ask_sizes_L1..LK, bid_sizes_L1..LK] normalised
    to proportions (each side sums to 1).

    If the file does not exist, generates synthetic Gaussian data so the
    training loop can be exercised without real data.

    Parameters
    ----------
    path      : str   — path to lob_snapshots.npy
    val_frac  : float — fraction for validation (default 0.1)
    test_frac : float — fraction for test (default 0.1)
    seed      : int   — reproducibility seed

    Returns
    -------
    train_loader, val_loader, test_loader, input_dim
    """
    p = Path(path)
    if p.exists():
        arr = np.load(p).astype(np.float32)
        print(f"Loaded {arr.shape[0]} snapshots from {path}  "
              f"(shape {arr.shape})")
    else:
        print(f"WARNING: {path} not found — generating synthetic data.")
        print("Run data/process_lobster.py first for real results.")
        rng = np.random.default_rng(seed)
        arr = rng.dirichlet(np.ones(20), size=10000).astype(np.float32)
        print(f"Synthetic data: {arr.shape}")

    input_dim = arr.shape[1]   # = 2 * n_levels

    # Normalise rows to [0, 1] range (already proportions, but clip for safety)
    n_levels = input_dim // 2

    asks, bids = arr[:, :n_levels], arr[:, n_levels:]

    # Avoid division by zero if an entire side is empty
    asks = asks / np.maximum(asks.sum(axis=1, keepdims=True), 1e-8)
    bids = bids / np.maximum(bids.sum(axis=1, keepdims=True), 1e-8)

    arr = np.concatenate([asks, bids], axis=1).astype(np.float32)

    # Sanity checks
    assert np.allclose(arr[:, :n_levels].sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(arr[:, n_levels:].sum(axis=1), 1.0, atol=1e-6)

    dataset = TensorDataset(torch.from_numpy(arr))

    n      = len(dataset)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_trn  = n - n_val - n_test

    g = torch.Generator().manual_seed(seed)
    trn, val, tst = random_split(dataset, [n_trn, n_val, n_test], generator=g)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=2048, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    return (
        make_loader(trn, shuffle=True),
        make_loader(val, shuffle=False),
        make_loader(tst, shuffle=False),
        input_dim,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     LOBAutoencoder,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    """One epoch of MSE reconstruction training. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        loss = nn.functional.mse_loss(x_hat, x)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:  LOBAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate MSE reconstruction loss. Returns mean loss."""
    model.eval()
    total_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        x_hat, _ = model(x)
        total_loss += nn.functional.mse_loss(x_hat, x).item() * len(x)
    return total_loss / len(loader.dataset)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_encoder(
    encoder:    LOBEncoder,
    latent_dim: int,
    out_dir:    str,
    metrics:    dict,
) -> Path:
    """
    Save encoder state_dict to checkpoints/ae_encoder_<latent_dim>.pt.

    The saved file contains:
        state_dict  : encoder weights (load with LOBEncoder.load_state_dict)
        input_dim   : expected input dimension
        latent_dim  : bottleneck dimension
        metrics     : final train/val/test losses

    Returns path to saved checkpoint.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ae_encoder_{latent_dim}.pt"

    torch.save({
        "state_dict": encoder.state_dict(),
        "input_dim":  encoder.input_dim,
        "latent_dim": encoder.latent_dim,
        "metrics":    metrics,
    }, path)

    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, input_dim = load_snapshots(
        args.snapshots, seed=args.seed
    )
    print(f"input_dim={input_dim}  "
          f"train={len(train_loader.dataset)}  "
          f"val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)}")

    latent_dims = (
        [args.latent_dim] if args.latent_dim is not None
        else [8, 16, 32]
    )

    all_results = {}

    for latent_dim in latent_dims:
        print(f"\n{'='*60}")
        print(f"Training AE  latent_dim={latent_dim}")
        print(f"{'='*60}")

        model = LOBAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=10
        )

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        history       = []

        t0 = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            trn_loss = train_one_epoch(model, train_loader, optimiser, device)
            val_loss = evaluate(model, val_loader, device)
            scheduler.step(val_loss)

            history.append({"epoch": epoch, "train": trn_loss, "val": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % args.log_every == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                print(
                    f"  epoch {epoch:4d}/{args.epochs}  "
                    f"train={trn_loss:.6f}  val={val_loss:.6f}  "
                    f"best_val={best_val_loss:.6f}  "
                    f"elapsed={elapsed:.1f}s"
                )

            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # ── Restore best weights and evaluate on test set ─────────────────
        model.load_state_dict(best_state)
        tst_loss = evaluate(model, test_loader, device)
        print(f"\n  Test MSE: {tst_loss:.6f}")

        # ── Reconstruction quality check ──────────────────────────────────
        # Sample one batch and print per-dimension reconstruction error
        model.eval()
        with torch.no_grad():
            (x_sample,) = next(iter(test_loader))
            x_sample    = x_sample.to(device)
            x_hat, z    = model(x_sample)
            per_dim_err = (x_hat - x_sample).abs().mean(dim=0).cpu().numpy()
            print(f"  Per-dim MAE (first 5): "
                  f"{per_dim_err[:5].round(5).tolist()}")
            print(f"  Latent norm (mean):  "
                  f"{z.norm(dim=1).mean().item():.4f}")

        metrics = {
            "latent_dim":    latent_dim,
            "best_val_loss": best_val_loss,
            "test_loss":     tst_loss,
            "epochs_trained": len(history),
        }
        all_results[latent_dim] = metrics

        # ── Save encoder ──────────────────────────────────────────────────
        ckpt_path = save_encoder(
            model.encoder, latent_dim, args.out_dir, metrics
        )
        print(f"  Saved encoder → {ckpt_path}")

        # ── Save training history ─────────────────────────────────────────
        hist_path = Path(args.out_dir) / f"ae_history_{latent_dim}.json"
        with open(hist_path, "w") as f:
            json.dump({"config": vars(args), "history": history,
                       "metrics": metrics}, f, indent=2)
        print(f"  Saved history → {hist_path}")

    # ── Summary across all latent dims ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'latent_dim':>12} {'val_loss':>12} {'test_loss':>12}")
    print("-" * 38)
    for ld, m in all_results.items():
        print(f"{ld:>12} {m['best_val_loss']:>12.6f} {m['test_loss']:>12.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-train LOB autoencoder on LOBSTER snapshots."
    )
    parser.add_argument(
        "--snapshots",
        type=str,
        default="data/processed/lob_snapshots.npy",
        help="Path to lob_snapshots.npy from process_lobster.py",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension to train (default: train all three: 8, 16, 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="Directory for saved encoder weights (default: checkpoints/)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print training log every N epochs (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available",
    )

    args = parser.parse_args()
    main(args)