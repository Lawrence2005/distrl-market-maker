"""
Autoencoder pre-training script.

Trains convolutional AE on LOBSTER LOB snapshots offline.
Saves encoder weights to checkpoints/ae_encoder_<latent_dim>.pt

Architecture:
    Encoder: Conv1D(32) -> ReLU -> Conv1D(16) -> ReLU -> Linear(latent_dim)
    Decoder: mirrors encoder

Loss: MSE reconstruction loss on LOB depth vectors.
Latent dim ablation: 8, 16, 32.

Week 5 deliverable (run before Week 6 training).
"""
# TODO: implement
