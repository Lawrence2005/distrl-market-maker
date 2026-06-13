"""
GLFT — Guéant-Lehalle-Fernandez-Tapia (2012) market maker.

Extends AS with hard inventory constraints [-Q, Q].
Two closed-form approximations:

  FOIC (First Order Inventory Control):
    Linearised optimal quotes. Fast; used on real trading desks.

  LIIC (Linear Inventory-based quote skewing):
    quotes = mid +/- delta*/2 - k*q
    Simplest GLFT variant; direct inventory skew.

Primary analytical benchmark throughout this project.
Reference: Guéant, Lehalle & Fernandez-Tapia (2012).
Week 3 deliverable.
"""
# TODO: implement
