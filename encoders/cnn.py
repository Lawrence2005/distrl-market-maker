"""
1D CNN encoder on LOB snapshot.

Architecture: Conv1D(32, k=3) -> ReLU -> Conv1D(16, k=3) -> ReLU -> Linear(latent_dim)
Input: concatenated bid/ask depth vectors (K levels per side), shape (2K,)
Trained end-to-end with RL agent (no pre-training).

Attribution: our own engineering design choice.
Motivated by Gašperov & Kostanjčar (2021) whose parallel SGU + LOB
architecture shows that learned feature extraction improves performance.

Week 5 deliverable.
"""
# TODO: implement
