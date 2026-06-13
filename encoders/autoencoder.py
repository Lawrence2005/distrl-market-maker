"""
Autoencoder (AE) encoder — unsupervised pre-training.

Pre-trained on LOBSTER LOB snapshots offline (minimize reconstruction MSE).
Encoder weights frozen after pre-training.
Latent space visualizable via PCA/t-SNE for regime interpretation.

Directly motivated by Gašperov survey (2021) which lists AEs as promising.
Week 5 deliverable.
"""
# TODO: implement (pre-training script in training/pretrain_ae.py)
