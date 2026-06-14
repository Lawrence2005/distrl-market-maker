# Scientific Evaluation Framework — Metrics × Models × Regimes

## Evaluation Metrics (What We Measure and Why)

| Category | Metric | Formula / Definition | Why It Matters | Literature Source |
|----------|----------|----------|----------|----------|
| **P&L Performance** | **Annualized Sharpe Ratio** | `E[P&L] / std(P&L) × √252` | Primary risk-adjusted return metric; standard in all RL market-making papers. | Gašperov et al. (2021); Spooner & Savani (2020) |
| **P&L Performance** | **Mean Episode P&L** | Average total P&L per trading episode | Raw return benchmark; must be positive to be viable. | All RL MM papers |
| **P&L Performance** | **Normalized Daily PnL** | `Total episode P&L / Average market bid-ask spread` | Normalizes profit across different securities/markets for fair cross-asset comparison; primary metric in Spooner (2018). Allows benchmarking without raw-dollar distortion. | Spooner et al. (2018); Kumar (DRQN MM) |
| **P&L Performance** | **PnLMAP Ratio** | `Total episode PnL / MAP` | Profit earned per unit of average inventory held. High PnLMAP = strong capital efficiency with low inventory risk. | Gašperov & Kostanjčar Signals (2021) |
| **P&L Performance** | **Terminal Wealth** | Final marked-to-market portfolio value at episode end:<br>`cash + inventory × final mid-price` | Captures whether the agent exits cleanly or leaves stranded inventory. Complements episode P&L. | Gašperov & Kostanjčar Signals (2021); Ganesh et al. (2019) |
| **P&L Performance** | **Winning Number (Win Rate)** | Fraction of episodes (or timesteps) with positive cumulative P&L | Simple and interpretable: what percentage of the time is the agent profitable? Useful for communicating results to non-technical audiences. | Sun, Huang & Yu (2022) |
| **Tail Risk** | **CVaR₀.₁₀ of P&L** | `(1/⌊0.1N⌋) · Σ_{i≤0.1N} θ_i`<br>(average of worst 10% quantile outputs) | Primary distributional metric; quantifies tail risk and is directly linked to QR-DQN outputs. | Dabney et al. QR-DQN (2018); Lim & Auer (2021) |
| **Tail Risk** | **Return-vs-CVaR Efficient Frontier** | Plot mean P&L vs. CVaR₀.₁₀ across α-sweep `{0.05, 0.10, 0.25, 0.50, 1.0}` | Shows whether risk-aversion genuinely trades return for safety. | Lim & Auer (2021); Dabney et al. IQN (2018) |
| **Tail Risk** | **Max Drawdown (MDD)** | `max_t(peak_PnL_t − PnL_t)` | Worst peak-to-trough loss; critical stress-testing metric. | Spooner & Savani (2020) |
| **Inventory Management** | **Mean Absolute Position (MAP)** | `MAP = (1/T) · Σ_t |q_t|` | Average absolute inventory over an episode. Lower MAP = better inventory management. | Spooner et al. (2018); Gašperov & Kostanjčar (2021); Kumar |
| **Inventory Management** | **Inventory Turnover Rate** | Number of inventory sign changes per episode | Measures how effectively the agent neutralizes directional exposure. | Ganesh et al. (2019); Guéant et al. (2012) |
| **Inventory Management** | **Inventory Distribution (σ_q)** | Standard deviation of inventory `q` over an episode | Risk-averse agents should maintain tighter inventory distributions. | Spooner et al. (2018); AS (2008) |
| **Inventory Management** | **Inventory Tail Exceedance** | Fraction of time `|q| > 0.8 · Q_max` | Measures how frequently the agent approaches position limits. | Guéant et al. (2012) |
| **Market Quality** | **Quoted Spread Distribution** | Distribution of `(ask_t − bid_t)` across episode | Tight spreads imply liquidity provision; overly tight spreads increase adverse-selection risk. | AS (2008); Huang et al. (2015) |
| **Market Quality** | **Fill Rate** | `# limit orders filled / # limit orders placed` | Measures quote competitiveness. Excessively high fill rates may indicate adverse selection. | Spooner et al. (2018); Sun et al. (2022) |
| **Market Quality** | **Price Impact Signature** | `E[mid_price_change \| agent order size = v]` vs. `v` | Measures whether the agent destabilizes the market. Good market makers should exhibit low impact. | Huang et al. (2015); Byrd et al. ABIDES (2019) |
| **Adverse Selection** | **Adverse Selection Cost** | `Effective spread − Quoted spread` (VPIN decomposition) | Measures how much spread revenue is lost to informed traders. | Ganesh et al. (2019); Gašperov Signals (2021) |
| **AS Theory Alignment** | **Quote-Skew R² (AS Recovery)** | `R²(agent_bid_offset ~ GLFT_reservation_price_skew)`<br>Run separately against AS and GLFT reference curves. | Quantifies whether the learned policy rediscovered optimal inventory management. GLFT is the primary benchmark; AS serves as the idealized reference. | AS (2008); Guéant et al. (2012); Spooner et al. (2018); Gašperov & Kostanjčar (2021) |
| **Robustness** | **OOD Performance Degradation** | `(Sharpe_in_distribution − Sharpe_OOD) / Sharpe_in_distribution` | Measures generalization failure when tested under unseen market conditions. | Spooner & Savani (2020); Dabney et al. IQN (2018) |
| **Robustness** | **Flash Crash Max Drawdown** | Max drawdown during synthetic `5σ` flash-crash event | Distributional agents should exhibit smaller drawdowns than expected-value agents. | Spooner & Savani (2020) |
| **State Representation** | **Latent Space Regime Separability** | Silhouette score of PCA/t-SNE clusters on AE latent vectors labeled by regime | Measures encoder interpretability and representation quality. | Gašperov & Kostanjčar (2021); Kumar |

---

## Market Regimes (What Conditions We Test In)

| Regime | Volatility σ | Arrival Process | Drift μ | Informed Trader % | Primary Hypothesis Being Tested |
|----------|----------|----------|----------|----------|----------|
| **Low-Vol Baseline** | `σ_low ≈ 0.5 bps/step` | Poisson (`λ` calibrated) | `μ = 0` | ~5% | **AS/GLFT Recovery:** Agent should converge to GLFT-like inventory skewing. SARSA should outperform Q-learning because on-policy advantages are clearest in stable, low-noise environments. Poisson arrivals used here; Hawkes arrivals reserved for high-volatility regimes. |
| **High-Vol Stochastic** | `σ_high ≈ 2.0 bps/step` | Hawkes Process (Bacry et al., 2015) | `μ = 0` | ~10% | **Distributional Advantage:** CVaR agents should outperform DQN on tail-risk metrics. Hawkes arrivals create bursty order flow and fat-tailed returns, making distributional objectives especially valuable. |
| **Trending Market** | `σ_mid` | Hawkes Process | `μ ≠ 0` | ~20% | **Inventory Exploitation:** Agent should build directional inventory positions. Tests management of adverse selection under elevated informed-trader activity. |
| **Flash Crash (Stress)** | `σ_high + 5σ shock` | Spike (simulated) | Sudden `μ < 0` | ~40% | **Robustness:** Compare max drawdown of CVaR agents vs. DQN during severe market dislocations. |
| **OOD Transfer** | `σ_high` (trained on `σ_low`) | Hawkes | `μ = 0` | ~10% | **Generalization:** IQN should degrade less than QR-DQN under out-of-distribution conditions. |

---

## State Representation Ablation Design

### 4 Encoders × 4 Neural Agents = 16 Variants + SARSA = 17 Total Variants

| Encoder Type | Architecture | Pre-training | State Dimension | Expected Advantage | Literature Basis |
|----------|----------|----------|----------|----------|----------|
| **Handcrafted Features (Baseline)** | Raw 15-dimensional feature vector | None; directly computed from ABIDES-Gym observations | ~15 dims | Highly interpretable; computationally efficient; closely aligned with AS intuition. Expected to struggle with complex LOB patterns. | Spooner et al. (2018); Huang et al. (2015) |
| **1D CNN Encoder** | `Conv1D(32,k=3) → ReLU → Conv1D(16,k=3) → ReLU → Linear(latent_dim)`<br><br>Input = concatenated bid/ask depth vectors (price × quantity across K levels) | End-to-end training jointly with RL agent | 16–32 dims (tuned) | Captures spatial LOB structure and depth imbalance. More expressive than handcrafted features while remaining lightweight. | Kumar; Gašperov & Kostanjčar (2021) |
| **Autoencoder (AE) Encoder** | Convolutional autoencoder pre-trained on LOBSTER snapshots. Encoder frozen after pre-training. Decoder mirrors encoder architecture. | Unsupervised pre-training via reconstruction MSE minimization on LOBSTER data.<br><br>Latent dimension ablation: `{8,16,32}` | 8–32 dims | Learns compact market representations independent of reward signal. Potentially stronger OOD generalization and latent-space interpretability. | Gašperov & Kostanjčar (2021); Kumar; broader representation-learning literature |
| **LSTM Encoder** | `LSTM(hidden=128)` over rolling window `T=30` sequential LOB snapshots → fixed-size hidden state `h_t`.<br><br>Compatible with DQN, PPO, QR-DQN, and IQN.<br><br>Requires sequence replay buffer. | End-to-end training with truncated BPTT (`T=30`).<br><br>Hidden state reset at episode start.<br><br>Replay buffer stores raw sequences; hidden states recomputed during training to avoid stale-state bias. | 128-dimensional hidden state | Captures temporal order-flow dynamics including Hawkes clustering, momentum, and mean reversion.<br><br>Primary ablation question: does temporal memory consistently outperform snapshot encoders? | Sun, Huang & Yu (2022); Kumar; Huang et al. (2015) |

---

## Experimental Matrix

### Agents

1. SARSA + Tile Coding
2. DQN
3. PPO
4. QR-DQN (CVaR)
5. IQN (CVaR)

### Encoders

1. Handcrafted Features
2. CNN
3. Autoencoder
4. LSTM

### Regimes

1. Low-Vol Baseline
2. High-Vol Stochastic
3. Trending Market
4. Flash Crash
5. OOD Transfer

### Core Research Questions

1. Do distributional agents (QR-DQN, IQN) improve tail-risk metrics relative to DQN?
2. Does CVaR produce a meaningful return-risk frontier?
3. Does the LSTM encoder consistently outperform snapshot-based encoders?
4. Can agents rediscover GLFT-style inventory control?
5. How robust are learned policies under flash-crash and OOD conditions?
6. Does AE pre-training improve convergence and generalization?
7. How closely do learned policies align with analytical market-making theory?