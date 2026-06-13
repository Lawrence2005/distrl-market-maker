"""
Handcrafted feature encoder.

Market features (Spooner et al. 2018; Huang et al. 2015):
  - Bid-ask spread
  - Mid-price move (last N steps)
  - Book/queue imbalance  I=(V_b-V_a)/(V_b+V_a)
  - Signed volume
  - Realised volatility
  - RSI
  - LOB depth at K levels per side  (K=3 default; standard LOB practice)

Private features:
  - Inventory q                     (Spooner et al. 2018)
  - Active quoting distances δ_b,δ_a (Spooner et al. 2018)
  - Outstanding bid/ask price       (Sun et al. 2022)
  - Trade prices (last T steps)     (Patel 2018)
  - Available cash                  (Patel 2018)
  - Time remaining τ=(T-t)/T        (Gašperov survey 2021)

Note: VPIN toxicity removed — not in any of the 19 papers' reading notes.
Week 4 deliverable.
"""
# TODO: implement
