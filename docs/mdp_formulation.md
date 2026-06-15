# MDP Formulation — Distributional Deep RL Market Maker

> **Status:** Week 1 deliverable  
> **Refers to:** Research Plan (Weeks 4–5), Evaluation Framework

Single source of truth for all MDP design decisions. Update this document first if anything changes, then the code.

---

## 1. Problem Framing

**Why RL for market making:**
1. The market is interactive — optimal features change with regime; static encoding is insufficient (Beysolow II).
2. RNNs hold positions too long for tick-level decisions (Beysolow II).
3. Market making is discrete-time stochastic inventory control — RL is the natural framework (Gašperov et al. 2021).

**MDP tuple:** `(S, A, R, P, γ=0.99)`

---

## 2. State Space

### 2.1 Market features

| Feature | Symbol | Source |
|---------|--------|--------|
| Bid-ask spread | `s` | Spooner et al. (2018) |
| Mid-price move | `Δm` | Spooner et al. (2018) |
| Book/queue imbalance | `I=(V_b−V_a)/(V_b+V_a)` | Spooner et al. (2018); Huang et al. (2015) |
| Signed volume | `sv` | Spooner et al. (2018) |
| Realised volatility | `σ_r` | Spooner et al. (2018) |
| RSI | `RSI` | Spooner et al. (2018) |
| LOB depth at K levels | `D_t` | Standard LOB practice |

### 2.2 Private features

| Feature | Symbol | Source |
|---------|--------|--------|
| Inventory | `q` | Spooner et al. (2018) |
| Active quoting distances | `δ_b, δ_a` | Spooner et al. (2018) |
| Outstanding bid/ask price | `p_o^b, p_o^a` | Sun et al. (2022) |
| Trade prices (last T steps) | `P_t^tr` | Patel (2018) |
| Available cash | `m_t` | Patel (2018) |
| Time remaining | `τ=(T−t)/T` | Gašperov survey (2021) |

> K=3 levels per side is a reasonable default. **Note:** VPIN toxicity was removed — it does not appear in any of the 19 papers' reading notes.

### 2.3 State Encoders and Recurrent Variants

Each neural agent (DQN, PPO, QR-DQN, IQN) is tested in two architectural modes: **snapshot mode**, where a single LOB observation is passed through an external encoder before reaching the agent's head, and **recurrent mode**, where an LSTM is integrated inside the agent's network architecture (DRQN-style), giving the agent explicit temporal memory across timesteps within an episode. SARSA (tile coding) is the only exception — incompatible with both neural encoders and recurrent integration.

#### Encoder 1 — Handcrafted (baseline)
- Identity transform on the feature vector above
- Config: `training/configs/encoder/handcrafted.yaml`

#### Encoder 2 — 1D CNN
```
Conv1D(32,k=3) → ReLU → Conv1D(16,k=3) → ReLU → Linear(latent_dim)
Input: (2K,) LOB depth snapshot | Output: 16–32 dims
```
Engineering choice — no single paper explicitly proposes this. Motivated by Gašperov & Kostanjčar (2021) parallel SGU+LOB architecture showing learned feature extraction helps. **Kumar (2019) is NOT a source for CNN.**

#### Encoder 3 — Autoencoder (AE)
```
Same Conv1D stack; decoder mirrors encoder
Pre-training: unsupervised MSE on LOBSTER snapshots (offline)
Output: 8–32 dims (latent_dim ablated) | Frozen after pre-training
```
Directly motivated by Gašperov survey (2021) which explicitly lists autoencoders as promising.

#### Recurrent Variant — LSTM integrated inside agent architecture
```
Input:  sequence of T=30 LOB snapshots, shape (T, input_dim)
LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)
Output: hidden state h_t, shape (128,) — feeds directly into Q-head or policy head
```
The LSTM hidden state (h_t, c_t) is carried forward across timesteps within an episode and reset to zero at episode start. This gives the agent explicit temporal memory — it can detect Hawkes-process order flow clustering, price momentum, and regime shifts from the history of LOB observations. This variant requires a sequence replay buffer that stores T-step sequences of raw LOB observations rather than single transitions; the LSTM hidden state is recomputed from the raw sequence during each training update to avoid stale-state bias from old network weights. Shared backbone implemented in `agents/recurrent_base.py`; instantiated separately as DRQN, RecurrentQRDQN, RecurrentIQN, RecurrentPPO. Primary citations: Sun et al. (2022); Hausknecht & Stone (2015).

**Ablation matrix:**
```
                    HC    CNN    AE    Recurrent (LSTM)
SARSA                ✓      ✗      ✗         ✗
DQN / DRQN           ✓      ✓      ✓         ✓
PPO / Recurrent PPO  ✓      ✓      ✓         ✓
QR-DQN (CVaR_α)      ✓      ✓      ✓         ✓
IQN (CVaR_α)         ✓      ✓      ✓         ✓
```
**Total: 1 (SARSA) + 4 agents × 3 snapshot + 4 agents × 1 recurrent = 17 variants**

---

## 3. Action Space

```
δ_b, δ_a ∈ {−4,−3,−2,−1,0,+1,+2,+3,+4} ticks  →  9×9 = 81 actions
bid = mid − δ_b × tick_size
ask = mid + δ_a × tick_size
```

Justification: Gašperov & Kostanjčar (2021) finds discrete actions standard in RL MM. Sun et al. (2022) finds 11–21 levels per side optimal; we use 9 as a starting point.

---

## 4. Reward Functions

### A — Asymmetric dampened (PRIMARY)
```
r_t = η · spread_PnL_t − (1−η) · |Δq_t|
η = 0.5 (default); sweep {0.3, 0.5, 0.7}
```
Disincentivises trend-following. Source: Spooner et al. (2018).

### B — Quadratic inventory penalty
```
r_t = spread_PnL_t − λ · q_t²
λ = 0.1 (default); sweep {0.01, 0.1, 1.0}
```
Source: Ganesh et al. JPMorgan (2019).

### C — Sparse liquidity reward
```
r_t = +1.0  if both sides fill
    = −0.5  if |q_t| > 0.8·Q_max
    =  0.0  otherwise
```
Source: Sun et al. (2022).

### Terminal (all formulations)
```
r_terminal = −κ · |q_T| · σ_T   (κ=1.0)
```

---

## 5. Distributional RL and CVaR Policy

**Distributional Bellman:** `Z^π(s,a) =^D R(s,a) + γ·Z^π(S',A')`

**QR-DQN** (Dabney et al. 2018):
```
Z_θ(s,a) = (1/N)·Σ δ_{θ_i(s,a)},  N=200
Loss: ρ_τ(u) = |τ − 1{u<0}|·L_κ(u),  τ_i=(2i−1)/(2N)
Architecture: dueling network + prioritised replay
```

**IQN** (Dabney et al. 2018):
```
φ(τ) = ReLU(Σ cos(πjτ)·w_j + b)
Z_θ(s,a,τ) = f(φ(τ) ⊙ ψ(s), a),  τ~U([0,1]) during training
```

**CVaR policy:**
```
CVaR_α(Z) = (1/α)·∫_0^α F^{−1}(τ)dτ
QR-DQN: action = argmax_a mean(θ_i for i ≤ floor(α·N))
IQN:    action = argmax_a mean(Z(s,a,τ) for τ~U([0,α]))
α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}
```

**Caveat (Lim & Auer 2021):** CVaR is not time-consistent. Our wrapper applies the risk objective at policy selection only — a tractable approximation, not the theoretically exact solution. Acknowledge in write-up.

---

## 6. Episode Structure

```
Simulator:     ABIDES-Gym
Length:        3900 steps (one trading day)
Inventory:     |q_t| ≤ Q_max = 10
Background:    50 noise traders, 10 momentum, 5 informed

Arrivals: Hawkes — λ(t) = μ + Σ_j α·exp(−β·(t−t_j))
Primary citation:    Bacry, Mastromatteo & Muzy (2015)
Empirical motivation: Huang et al. (2015)
```

---

## 7. Analytical Baselines

**AS (Avellaneda & Stoikov 2008):**
```
r = s − q·γ·σ²·(T−t)
δ* = γσ²(T−t) + (2/γ)·ln(1+γ/k)
```
Three violated assumptions: (1) driftless BM, (2) fill intensity depends only on δ, (3) arrivals and price independent.

**GLFT (Guéant et al. 2012) — primary benchmark:**
```
FOIC: linearised closed-form  →  baselines/glft.py::GLFT_FOIC
LIIC: bid = mid − δ*/2 − k·q  →  baselines/glft.py::GLFT_LIIC
```
RL must beat GLFT to justify added complexity. Same benchmark as Gašperov & Kostanjčar (2021) — results directly comparable.

---

## 8. Market Regimes

| Regime | σ | Arrivals | Drift | Informed % | Primary test |
|--------|---|----------|-------|------------|--------------|
| Low-Vol | σ_low≈0.5bps | Poisson | 0 | 5% | GLFT recovery; SARSA>Q-learning |
| High-Vol | σ_high≈2.0bps | Hawkes | 0 | 10% | CVaR advantage; recurrent>snapshot |
| Trending | σ_mid | Hawkes | μ≠0 | 20% | Inventory exploitation; recurrent temporal memory |
| Flash Crash | σ_high+5σ | Spike | μ<0 | 40% | CVaR vs. DQN drawdown |
| OOD Transfer | σ_high (trained σ_low) | Hawkes | 0 | 10% | Recurrent IQN vs. snapshot QR-DQN degradation |

---

## 9. Scoped Out

**Macro/micro hierarchical agent (Patel 2018):** Patel's own results show multi-agent underperforms micro-only. Recurrent integration inside QR-DQN/IQN subsumes the temporal LOB awareness of the micro-agent without coordination overhead — and adds the distributional objective on top.

**Adversarial RL (Spooner & Savani 2020):** Requires co-evolving adversary in ABIDES — a separate research project. CVaR achieves robustness tractably. Flash crash stress test is our robustness check.

---

## 10. References

| Citation | Role |
|----------|------|
| Avellaneda & Stoikov (2008) | AS baseline |
| Guéant et al. (2012) | GLFT baseline — FOIC/LIIC |
| Huang et al. (2015) | Queue imbalance feature; Hawkes motivation |
| Bacry et al. (2015) | Hawkes process model and calibration |
| Spooner et al. (2018) | State features; asymmetric reward η; SARSA>Q-learning |
| Beysolow II | RL motivation |
| Ganesh et al. JPMorgan (2019) | Quadratic reward |
| Gašperov et al. survey (2021) | RL motivation; autoencoders explicitly listed |
| Gašperov & Kostanjčar signals (2021) | CNN encoder motivation; GLFT benchmarks |
| Sun et al. (2022) | DRQN-LSTM recurrent architecture; sparse reward; action space |
| Hausknecht & Stone (2015) | Original DRQN architecture |
| Kumar (2019) | DRQN>DQN only — no CNN comparison |
| Patel (2018) | Macro/micro scoped out; private state features |
| Bellemare et al. (2017) | C51; distributional Bellman equation |
| Dabney et al. QR-DQN (2018) | QR-DQN |
| Dabney et al. IQN (2018) | IQN |
| Lim & Auer (2021) | CVaR time-consistency caveat |
| Spooner & Savani (2020) | ARL scoped out; CVaR wrapper justification |
| Byrd et al. (2019) | ABIDES simulator |
| Amrouni et al. | ABIDES-Gym |