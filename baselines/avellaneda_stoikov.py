"""
Avellaneda-Stoikov (2008) closed-form market maker.

Computes:
  reservation_price r = s - q * gamma * sigma^2 * (T - t)
  optimal_spread    delta* = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)

Core AS assumptions (documented for later ablation):
  (a) Reference price: driftless arithmetic Brownian motion
  (b) Fill intensity: depends ONLY on distance delta from reference price
  (c) Statistical independence: arrivals and reference price

Reference: Avellaneda & Stoikov (2008).
Week 3 deliverable.
"""
# TODO: implement
