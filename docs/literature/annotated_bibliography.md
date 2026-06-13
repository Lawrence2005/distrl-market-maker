# Literature Review — 19 Papers Across 5 Pillars

| Pillar | Authors & Year | Title (Short) | Core Contribution | Key Equation / Result | Relevance to This Project | Read Priority |
|----------|----------|----------|----------|----------|----------|----------|
| **1 — Classical Theory** | **Avellaneda & Stoikov (2008)** | *HFT in a Limit Order Book* | Closed-form optimal bid/ask under inventory risk.<br><br>**Three core assumptions:**<br>1. Reference price follows **driftless arithmetic BM**.<br>2. Fill intensities depend **only** on distance δ — not queue position/time.<br>3. Order arrivals and reference price are **statistically independent**.<br><br>Poisson arrivals:<br>`λ(δ) = A·e^{-kδ}`<br><br>These assumptions define where RL will outperform. | `r = s − q·γ·σ²·(T−t)`<br>`δ* = γσ²(T−t) + (2/γ)ln(1+γ/k)` | The theoretical benchmark every RL agent is measured against; defines reservation price and optimal spread. | ★★★★★ Must-read first |
| **1 — Classical Theory** | **Guéant, Lehalle & Fernandez-Tapia (2012)** | *GLFT — Dealing with Inventory Risk* | The GLFT model.<br><br>Rigorous extension of AS:<br>- HJB→ODE via change-of-variables<br>- Hard inventory constraints `[−Q,Q]`<br>- Proves AS admissibility<br><br>Two closed-form approximations:<br><br>**FOIC (First Order Inventory Control)**<br>Linearised optimal quotes — fast, used on real trading desks.<br><br>**LIIC (Linear Inventory-based Skewing)**<br>Simplest variant; direct inventory skew.<br><br>GLFT strictly generalises AS — AS is the unconstrained limit of GLFT. | HJB→ODE via change-of-variables.<br>Spectral decomposition → closed-form quotes.<br><br>FOIC:<br>`q_bid/ask = mid ± δ*/2 − (linear inventory term)`<br><br>LIIC:<br>simplest linear skew.<br><br>Backtested at Crédit Agricole Cheuvreux. | **PRIMARY ANALYTICAL BENCHMARK**<br><br>Implement both FOIC and LIIC variants.<br>Makes results directly comparable to Gašperov & Kostanjčar (2021).<br><br>RL must beat GLFT to justify added complexity. | ★★★★★ Must-read (= GLFT benchmark) |
| **1 — Classical Theory** | **Huang, Lehalle & Rosenbaum (2015)** | *Queue-Reactive Model* | LOB event intensities depend on queue sizes (not just price distance); calibrated to LOBSTER data. | Queue imbalance:<br>`(V_bid − V_ask)/(V_bid + V_ask)`<br>predicts short-term price direction. | Justifies LOBSTER calibration of background agents; motivates queue imbalance as state feature; validates Hawkes arrivals. | ★★★★☆ Read for simulation design |
| **2 — RL Market Making** | **Gašperov et al. (2021)** | *RL Approaches to Optimal MM (Survey)* | Comprehensive survey; RL superior in risk-adjusted returns.<br><br>Explicitly lists **autoencoders** as promising state representation.<br>Notes Dempster & Romahi's genetic algorithm approach to feature selection.<br>Dense rewards preferred.<br>NNs dominate (FFNN/DQN/DDQN/DRQN/ESN). | Taxonomy:<br>- model-free/model-based<br>- single/multi-agent<br>- LOB/dealer<br><br>Open problems:<br>- risk sensitivity<br>- multi-agent dynamics | Positions our contribution in literature; standard metrics and evaluation framework; read for breadth. | ★★★★★ Read early for context |
| **2 — RL Market Making** | **Spooner, Savani & McBurney (2018)** | *Market Making via RL* | TD RL with eligibility traces (SARSA, Q-learning, R-learning).<br><br>**Key results:**<br>1. SARSA > Q-learning for MM<br>2. Asymmetrically dampened reward improves stability<br>3. Tile coding more stable than NNs<br>4. RL naturally rediscovers inventory skewing | Reward:<br>`η·spread_PnL − (1−η)·|Δq|`<br><br>State:<br>- inventory<br>- quoting distances<br>- spread<br>- imbalance<br>- volatility<br>- RSI<br><br>Metrics:<br>`Normalised daily PnL = P&L / avg spread`<br>`MAP = (1/T)Σ|q_t|` | Primary MDP formulation reference; AS recovery methodology; code available on GitHub. | ★★★★★ Must-read |
| **2 — RL Market Making** | **Kumar, P.** | *Deep RL for Market Making* | DRQN outperforms DQN and temporal-difference RL on Sharpe ratio.<br>DQN does not generalise well across sessions. | `DRQN > DQN > TD-RL` on Sharpe. | Source for DRQN-LSTM justification. | ★★★★☆ Read for DRQN justification |
| **2 — RL Market Making** | **Gašperov & Kostanjčar (2021)** | *MM with Signals via DRL* | Two supervised SGUs (price range + trend) combined with DRL.<br>ARL + neuroevolution outperform standard approaches.<br>PDP used for interpretability.<br>MAP-Elites and novelty search suggested. | State:<br>- inventory<br>- SGU trend prediction<br>- SGU range prediction<br><br>Benchmarks:<br>- FOIC<br>- LIIC<br>(GLFT variants) | State-space design inspiration.<br>CNN/AE encoders replace handcrafted SGU signals. | ★★★★☆ Read for state design |
| **2 — RL Market Making** | **Sun, Huang & Yu (2022)** | *MM Strategy Optimization via DRL* | DRQN-LSTM architecture.<br>LSTM extracts temporal LOB patterns.<br>DQN learns state-action values.<br><br>State:<br>- internal (cash, inventory, orders)<br>- external (LOB via LSTM)<br><br>Benchmarked against naive RMM and Q-learning. | Architecture:<br>`LOB → LSTM → Q-head`<br><br>Reward:<br>`{+1, −0.5, 0}`<br><br>Optimal action space:<br>11–21 discrete levels.<br>Dueling > standard DQN. | **PRIMARY SOURCE** for LSTM implementation.<br>Direct architecture adopted. | ★★★★★ Must-read |
| **2 — RL Market Making** | **Spooner & Savani (2020)** | *Robust MM via Adversarial RL* | MM vs. adversary controlling `(σ, A, k)` in a zero-sum game.<br>Strategic adversary dominates fixed/random adversaries.<br><br>Important finding:<br>Direct risk-sensitive RL is numerically unstable. | State:<br>`(t,q)`<br><br>Reward:<br>`MtM − |q|` | Basis for robustness discussion.<br>ARL explicitly scoped out and replaced by CVaR + stress testing. | ★★★★☆ Read for robustness |
| **2 — RL Market Making** | **Ganesh et al. (JPMorgan, 2019)** | *RL for MM in Dealer Market* | Multi-agent simulator with market makers and investors.<br>RL learns competitor pricing and exploits drift through inventory positioning.<br>Boltzmann softmax important for risk-averse behaviour. | Reward:<br>`spread_PnL − λ·q²`<br><br>Metrics:<br>- episode return<br>- terminal wealth<br>- MAP<br>- MDD<br>- PnLMAP | Reward design, multi-agent dynamics, adverse selection decomposition. | ★★★★☆ Read for reward design |
| **2 — RL Market Making** | **Beysolow II** | *Market Making via RL (Springer)* | Policy gradient approaches were unprofitable.<br>DQN profitable but unstable.<br><br>Key lessons:<br>1. Markets are interactive.<br>2. RNNs hold positions too long.<br>3. Reward design is critical. | Negative-result study. | Motivation for SARSA baseline and careful reward shaping. | ★★★☆☆ |
| **2 — RL Market Making** | **Patel (2018)** | *Optimizing MM using MARL* | Hierarchical macro/micro-agent architecture.<br>Macro agent underperforms; pushes orders to wrong side of book.<br>Micro agent uses deep LOB state. | Two-agent DQN framework. | Explicitly scoped out.<br>DRQN-LSTM subsumes temporal benefits without coordination overhead. | ★★★☆☆ |
| **3 — Distributional RL** | **Bellemare, Dabney & Munos (2017)** | *C51 — Distributional Perspective on RL* | Introduces value distribution `Z` instead of scalar `Q`.<br>Preserves multimodality and improves stability. | `Z^π(x,a) =^D R(x,a) + γZ^π(X',A')` | Foundational distributional RL paper; ancestor of QR-DQN and IQN. | ★★★★★ Must-read |
| **3 — Distributional RL** | **Dabney, Rowland, Bellemare & Munos (2018)** | *QR-DQN* | Quantile regression distributional RL.<br>No need to specify `V_min/V_max`.<br>Improves performance and sample efficiency. | Quantile loss:<br>`ρ_τ(u)=|τ−1{u<0}|·|u|`<br><br>`CVaR_α=(1/⌊αN⌋)Σ_{i≤αN}θ_i(s,a)` | **PRIMARY ALGORITHM** for implementation. | ★★★★★ Must-read |
| **3 — Distributional RL** | **Dabney, Ostrovski, Silver & Munos (2018)** | *IQN* | Learns full quantile function via cosine embedding.<br>Supports arbitrary distortion risk measures. | `Z_θ(x,a,τ)=f(φ(τ)⊙ψ(x),a)` | Secondary algorithm; richer risk-sensitive policies and better OOD generalisation. | ★★★★★ Must-read |
| **3 — Distributional RL** | **Lim & Auer (2021)** | *Distributional RL for Risk-Sensitive Policies* | Shows CVaR is not time-consistent.<br>Introduces augmented-state formulation restoring Bellman structure. | Augmented state:<br>`(s, α)` | Theoretical grounding and caveat for CVaR implementation. | ★★★★☆ |
| **4 — Simulation** | **Byrd, Hybinette & Balch (2019)** | *ABIDES* | Agent-based financial market simulator with nanosecond resolution and NASDAQ-style messaging. | Deterministic seeding and configurable latency. | Primary simulator. | ★★★★★ Must-read |
| **4 — Simulation** | **Amrouni et al.** | *ABIDES-Gym* | Gym wrapper for ABIDES with market-making environment. | Observation = LOB + inventory + time.<br>Action = bid/ask offsets. | Main implementation environment. | ★★★★★ Must-read |
| **4 — Simulation** | **Bacry, Mastromatteo & Muzy (2015)** | *Hawkes Processes in Finance* | Canonical reference for self-exciting order flow.<br>Explains clustering and burstiness of arrivals. | `λ(t)=μ+Σ_jφ(t−t_j)`<br><br>`φ(t)=αe^{−βt}`<br><br>`ρ=α/β<1` | Primary Hawkes-process citation. | ★★★★☆ |
| **4 — Simulation** | **Huang, Lehalle & Rosenbaum (2015)** | *Queue-Reactive Model (simulation)* | Calibration target for ABIDES background agents.<br>Connects queue-reactive dynamics with Hawkes-style clustering. | Queue-reactive intensities calibrated to LOBSTER data. | Used for simulator calibration. | ★★★★☆ |
---

# Recommended Reading Order

## Phase 1 — Foundations (Must Read First)

1. Avellaneda & Stoikov (2008)
2. Guéant, Lehalle & Fernandez-Tapia (2012)
3. Spooner et al. (2018)
4. Gašperov et al. (2021)

## Phase 2 — Architecture Design

5. Sun et al. (2022)
6. Kumar
7. Gašperov & Kostanjčar (2021)
8. Ganesh et al. (2019)

## Phase 3 — Distributional RL

9. Bellemare et al. (2017)
10. Dabney et al. (QR-DQN, 2018)
11. Dabney et al. (IQN, 2018)
12. Lim & Auer (2021)

## Phase 4 — Simulation

13. ABIDES (2019)
14. ABIDES-Gym
15. Bacry et al. (2015)
16. Huang et al. (2015)

## Phase 5 — Robustness & Design Decisions

17. Spooner & Savani (2020)
18. Patel (2018)
19. Beysolow II