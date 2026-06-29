"""
Main training entry point.

Supports all agents x encoder combinations via Hydra config.
Logs to W&B: episode rewards, quantile distributions, inventory histograms.

Usage:
    python training/train.py agent=qrdqn encoder=autoencoder reward=asymmetric alpha=0.10

Week 5–7 deliverable.
"""
# TODO: implement
import torch

from pretrain_ae import LOBEncoder
ckpt = torch.load("checkpoints/ae_encoder_16.pt")
encoder = LOBEncoder(ckpt["input_dim"], ckpt["latent_dim"])
encoder.load_state_dict(ckpt["state_dict"])
encoder.eval()