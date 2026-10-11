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

### 2.3 Encoders

Three snapshot encoders expose `encoder.encode(obs) -> torch.Tensor` and are used as external modules by any neural agent (DQN, PPO, QR-DQN, IQN). The recurrent variant integrates an LSTM *inside* each agent's network architecture (DRQN-style), giving the agent explicit temporal memory across timesteps within an episode. SARSA (tile coding) is incompatible with both neural encoders and recurrent integration.

#### Encoder 1 — Handcrafted (baseline)
- Identity transform on the feature vector above
- Config: `training/configs/encoder/handcrafted.yaml`

#### Encoder 2 — 1D CNN
```
Conv1D(32,k=3) → ReLU → Conv1D(16,k=3) → ReLU → Linear(latent_dim)
Input: (2K,) LOB depth snapshot, convolved | Output: latent_dim (16–32) market dims,
       with the 6 private features (§2.2) concatenated on raw and unconvolved
       → latent_dim + 6 total, exposed to the agent
```
Engineering choice — no single paper explicitly proposes this. Motivated by Gašperov & Kostanjčar (2021) parallel SGU+LOB architecture showing learned feature extraction helps. **Kumar (2019) is NOT a source for CNN.** Until it was fixed, this encoder (and the AE encoder below) never saw the private features at all — the agent was blind to its own inventory, quote distances, and time remaining. See `encoders/base.py::PRIVATE_FEATURE_DIM`.

#### Encoder 3 — Autoencoder (AE)
```
Same Conv1D stack; decoder mirrors encoder
Pre-training: unsupervised MSE on LOBSTER snapshots (offline), market features only
Output: 8–32 market dims (latent_dim ablated) | Frozen after pre-training
        + 6 private features concatenated on raw, as in Encoder 2
```
Directly motivated by Gašperov survey (2021) which explicitly lists autoencoders as promising.

#### Recurrent Variant — LSTM integrated inside agent architecture
```
Input:  sequence of T=30 LOB snapshots, shape (T, input_dim)
LSTM(input_size=input_dim, hidden_size=128, num_layers=1, batch_first=True)
Output: hidden state h_t, shape (128,) — feeds directly into Q-head or policy head
```
The LSTM hidden state (h_t, c_t) is carried forward across timesteps within an episode and reset to zero at episode start. This gives the agent explicit temporal memory — it can learn to detect order flow clustering, price momentum, and regime shifts from the history of LOB observations, rather than treating each timestep as independent. This variant requires a sequence replay buffer that stores T-step sequences of raw LOB observations; the LSTM hidden state is recomputed from the raw sequence during each training update to avoid stale-state bias from old network weights.

Shared backbone in `agents/recurrent_base.py`; instantiated separately as DRQN, RecurrentQRDQN, RecurrentIQN, RecurrentPPO. Primary citations: Sun et al. (2022); Hausknecht & Stone (2015).

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

**As actually implemented** (`envs/lob_env.py`): one-sided, independently
chosen offsets per side, `δ_b, δ_a ∈ TICK_OFFSETS = {0,10,20,...,100} ticks`
→ 11x11 = 121 actions (`bid = mid − δ_b·tick_size`, `ask = mid + δ_a·tick_size`,
δ never negative — bid always at/below mid, ask always at/above). Spacing was
widened from the original `{0,...,10}` (i.e. $0-$0.10 max offset) to
`{0,10,...,100}` ($0-$1.00) after finding the narrower range was 3-10x
tighter than a single-step price move at this market's real ~$1000 mid
(AS(2008)-typical sigma 0.0003-0.001/step implies $0.30-$1.00/step) — level
*count* (11) is unchanged, so this doesn't affect network output layers,
replay buffer shapes, or SARSA's tile coding.

**Order size**: every submitted order is `order_size` shares (default 100,
matching ABIDES background agents' typical lot size — see
`abides_markets/models/order_size_model.py`). This was previously hardcoded
to 1 share, making the reward's spread-capture term ~2 orders of magnitude
too small relative to the per-step inventory mark-to-market term (see §5) —
not a documented design choice, a bug.

---

## 4. Reward Functions

Per-step reward is computed in `envs/lob_env.py::_compute_reward`. The
asymmetric reward (A) is this project's own formulation and the primary reward
for every reported experiment.

**Per-step quantities**
```
ψ_a     = matched_ask · (ask_price − mid)     # ask half-spread capture
ψ_b     = matched_bid · (mid − bid_price)     # bid half-spread capture
inv_pnl = q_prev · (mid − prev_mid)           # mark-to-market inventory PnL
PnL     = ψ_a + ψ_b + inv_pnl
```
`q_prev` is the inventory before this step's fills. Shares that transacted this
step are already compensated through ψ_a / ψ_b at their fill price.

### A — Asymmetric adverse-inventory penalty (PRIMARY, this project's formulation)
```
r_t = PnL_t − η · max(0, −inv_pnl_t)
η = 0.5 (default); ablation {0.1, 0.5, 1.0}
```
Only inventory losses are penalised, with weight η. Inventory gains count in full.

*Sign-sensitivity ablation.* The reverse-sign variant (config `reward=asymmetric_spooner`)
computes `PnL − η · max(0, inv_pnl)`, which penalises inventory gains and leaves
losses unpenalised. It is run only to test whether results depend on the sign
(`scripts/run_reward_comparison_eval.py`). It is not the project's reward and is not
a replication of any published reward.

### B — Sparse liquidity reward (ablation)
```
r_t = +1.0  if both sides fill this step (round-trip)
    = −0.5  if a fill crosses the spread (adverse selection)
    =  0.0  otherwise
```
Source: not yet verified against the paper (see `paper/literature_claims.md`, row 24).

### C — Quadratic inventory penalty (ablation)
```
r_t = PnL_t − λ · q_prev²        λ = 0.1 (default)
```

### Terminal (all formulations)
```
r_terminal = −κ · |q_T| · σ_T   (κ = 1.0), added at episode end
```
Implemented in `envs/lob_env.py::_terminal_reward`.

---

## 5. Distributional RL and CVaR Policy

**Distributional Bellman:** `Z^π(s,a) =^D R(s,a) + γ·Z^π(S',A')`

**QR-DQN** (Dabney et al. 2018):
```
Z_θ(s,a) = (1/N)·Σ δ_{θ_i(s,a)},  N=32 (all reported runs; Dabney et al. 2018 use N=200 on Atari)
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
Length:        390 steps (one 6.5 h trading day at one step per minute)
Inventory:     |q_t| ≤ Q_max = 1000 shares (order_size = 100 shares/order, ~10 orders' worth)
Background:    50 noise traders, 10 momentum, 5 informed

Arrivals: ABIDES rmsc04's own background-agent order flow (unmodified)
```

---

## 7. Analytical Baselines

**AS (Avellaneda & Stoikov 2008):**
```
r = s − q·γ·σ²·(T−t)
δ* = γσ²(T−t) + (2/γ)·ln(1+γ/k)
```
Three violated assumptions: (1) driftless BM, (2) fill intensity depends only on δ, (3) arrivals and price independent.

**GLFT (Guéant et al. 2012) — primary benchmark:** exact ODE solution (Prop. 5), `baselines/glft.py::GLFTBaseline`.
RL must beat GLFT to justify added complexity. Same benchmark as Gašperov & Kostanjčar (2021) — results directly comparable.

---

## 8. Market Regimes

| Regime | σ | Arrivals | Drift | Informed % | Primary test |
|--------|---|----------|-------|------------|--------------|
| Low-Vol | σ_low≈0.5bps | rmsc04 default | 0 | 5% | GLFT recovery; SARSA>Q-learning |
| High-Vol | σ_high≈2.0bps | rmsc04 default | 0 | 10% | CVaR advantage; recurrent>snapshot |
| OOD Transfer | σ_high (trained σ_low) | rmsc04 default | 0 | 10% | Recurrent IQN vs. snapshot QR-DQN degradation |

(Trending and Flash Crash regimes were removed from the project — see git
history. Trending had no real drift mechanism under real ABIDES (rmsc04's
fundamental is mean-reverting, not driftable); Flash Crash's ABIDES-native
megashock mechanism proved numerically unstable during calibration.)

---

## 9. Scoped Out

**Macro/micro hierarchical agent (Patel 2018):** Patel's own results show the multi-agent approach underperforms its macro-agent alone (arXiv 1812.10252, Fig. 12). Recurrent integration (LSTM inside DQN/QR-DQN/IQN) subsumes the temporal LOB awareness of the micro-agent without coordination overhead.

**Adversarial RL (Spooner & Savani 2020):** Requires co-evolving adversary in ABIDES — a separate research project. CVaR achieves robustness tractably. No fixed-adversary robustness check is currently in scope (the flash-crash stress test that served this role was removed — see §8).

---

## 10. References

| Citation | Role |
|----------|------|
| Avellaneda & Stoikov (2008) | AS baseline |
| Guéant et al. (2012) | GLFT baseline |
| Huang et al. (2015) | Queue imbalance feature |
| Spooner et al. (2018) | State features; eligibility-trace SARSA / Q-learning lineage |
| Beysolow II | RL motivation |
| Ganesh et al. JPMorgan (2019) | Risk-Averse |
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