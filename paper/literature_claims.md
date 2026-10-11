# Literature claim validation register

Every peer-paper claim the preprint will cite, checked against the paper's text
(not the Master Sheet summary) before it goes into the draft.

**Status key:** SUPPORTED = paper says it, wording can be used as stated ·
QUALIFY = supported only with the narrower wording given ·
WRONG = paper says otherwise, do not use ·
UNVERIFIED = full text not checked, do not cite until checked.

Full texts for the arXiv papers were pulled from arXiv and read locally. Sources
marked "search snippet" were read only from a search result summary.

## Critical findings for our own code and docs

1. **Our asymmetric reward is our own formulation, and differs in sign from Spooner et al. (2018) eq. 6.**
   We keep our reward as the experiment. The paper's form dampens inventory gains and keeps losses; ours penalises inventory losses and keeps gains. The sign was flipped in commit fa2334f (2026-07-27), so the campaign results use the flipped form. The paper must present our reward as ours and must not describe it as Spooner's. The reverse-sign variant is only a sign-sensitivity check (`scripts/run_reward_comparison_eval.py`), not a replication.

2. **`docs/mdp_formulation.md` §9 misattributes Patel's result.** It says Patel shows
   "multi-agent underperforms micro-only." Patel (arXiv 1812.10252, Fig. 12) says the
   multi-agent approach "under-performs in comparison to the macro-agent." Patel also
   uses Bitcoin order-book data, not a simulator. Fix the doc before citing.

3. **`CLAUDE.md` overstates Spooner & Savani (2020).** It says "direct risk-sensitive
   RL is numerically unstable per Spooner & Savani 2020." The paper (§1.2, arXiv
   2003.01820) says variance-based and distributional risk-sensitive methods "can
   suffer from numerical instability," "especially true when using exponential utility
   functions." The "apply CVaR at selection, not in the Bellman update" design is our
   choice, not a result of that paper. Cite it as QUALIFY below.

## Register

| # | Cited as (sheet) | Source checked | Status | What the paper actually says | Use in preprint |
|---|---|---|---|---|---|
| 1 | Avellaneda & Stoikov (2008) | Not on arXiv; only GLFT's summary read | UNVERIFIED | — | Cite for the closed form only after reading the original (Quantitative Finance). |
| 2 | Guéant, Lehalle & Fernandez-Tapia (2012), "proves AS admissibility" | arXiv 1105.3115 | **WRONG** | Says that without inventory limits, optimality of the AS quotes has no proof, and "their admissibility appears to be an open problem." | Do not claim GLFT proves AS admissibility. Say GLFT adds hard inventory bounds [−Q, Q] and solves the problem with a change of variables and an ODE system (Prop. 5, with market impact). |
| 3 | GLFT: HJB → ODE via change of variables; hard bounds [−Q, Q] | arXiv 1105.3115 | SUPPORTED | Change of variables to an ODE system; Prop. 5 "Solution with market impact"; inventory bound Q. | Use as stated. |
| 4 | GLFT: "backtested at Crédit Agricole Cheuvreux" | arXiv 1105.3115 | UNVERIFIED | Not checked. | Drop unless found in the text. |
| 5 | Spooner, Savani & McBurney (2018), "Market Making via RL" (author list) | arXiv 1804.04216 | **WRONG** | Authors are Spooner, Fearnley, Savani & Koukorinis (AAMAS 2018). | Use the correct author list. |
| 6 | Spooner et al. (2018): on-policy SARSA > off-policy Q-learning | arXiv 1804.04216, §5 | **WRONG as stated** | Plain Q-learning and SARSA both gave "no useful results" in the basic setup. Eligibility traces (Q(λ), SARSA(λ)) gave "occasionally generated profits out-of-sample," still "unsatisfactory." The paper's conclusion credits eligibility traces and the combined agent's risk-adjusted performance, not a SARSA-over-QL ranking. | Write: "eligibility traces were needed for a profitable agent; plain Q-learning and SARSA failed in the basic setup." |
| 7 | Spooner et al. (2018): asymmetric reward η·spread − (1−η)\|Δq\| | arXiv 1804.04216, eq. 6 | **WRONG** | Form is `Ψ − max[0, η·Inv·Δm]` (see Critical finding 1). | Use the paper's eq. 6. |
| 8 | Spooner et al. (2018): dampening disincentivises trend-following | arXiv 1804.04216, §4.2 | SUPPORTED | "Engineered to disincentivise trend-following and bolster spread capture." | Use as stated. |
| 9 | Spooner & Savani (2020): STRATEGIC adversary strictly dominates FIXED and RANDOM | arXiv 2003.01820, §1.1 | SUPPORTED | "Training against a STRATEGIC adversary strictly dominates the other two settings... in terms of... the Sharpe ratio." | Use as stated (note: this is about adversarial training, which we do not do). |
| 10 | Spooner & Savani (2020): numerical instability → avoid direct risk-sensitive RL | arXiv 2003.01820, §1.2 | QUALIFY | "Can suffer from numerical instability… especially true when using exponential utility." | Use this narrower wording. Do not generalise it to CVaR. |
| 11 | Ganesh et al. (2019), JPMorgan: learns competitor pricing, skews quotes, exploits drift | arXiv 1911.05892, abstract | SUPPORTED | "Learns about its competitor's pricing strategy… manage inventory by skewing prices… exploit any significant market price drift." | Use as stated. |
| 12 | Patel (2018): decaying ε-greedy; DDQN micro-agent | arXiv 1812.10252 | SUPPORTED (ε-greedy, DDQN); macro DQN UNVERIFIED | ε-greedy with decay; micro-agent described with DDQN. | Use ε-greedy and DDQN only. |
| 13 | Patel (2018): multi-agent underperforms micro-only | arXiv 1812.10252, Fig. 12 | **WRONG** | "Under-performs in comparison to the macro-agent." | Do not use the micro-only comparison (see Critical finding 2). |
| 14 | Dabney et al. (2018) QR-DQN: N = 200 quantiles, quantile Huber loss | arXiv 1710.10044, §5 and eq. 9–10 | SUPPORTED with qualifier | N = 200 is their Atari hyperparameter, not a theoretical requirement. Quantile Huber loss `ρ^κ_τ(u) = |τ − δ{u<0}| L_κ(u)` is eq. 10. | Say "N = 200 as in Dabney et al.'s Atari setup." |
| 15 | Dabney et al. (2018) IQN: CVaR via τ ~ U([0, α]); cosine embedding | arXiv 1806.06923 | SUPPORTED | Distortion risk measures are applied by reshaping the sampling distribution of τ. CVaR(η, τ) = ητ. | Use as stated. |
| 16 | Bellemare, Dabney & Munos (2017), C51: instability in the control setting | arXiv 1707.06887 | SUPPORTED | "Significant distributional instability" in the control setting, "not a contraction in total variation, KL, or Kolmogorov distance." | Use as stated. |
| 17 | Hausknecht & Stone (2015) DRQN: random vs sequential updates | arXiv 1507.06527, §4 | QUALIFY | Both update types "are viable" in their experiments. Random updates start at random points and zero the LSTM state at each update. | Use for our replay-fix justification, and note that the paper does not rank the two. Our windows stay inside one episode with the hidden state zeroed at window start, matching random updates. |
| 18 | Byrd, Hybinette & Balch (2019), ABIDES: nanosecond resolution; deterministic seeding | arXiv 1904.12066 | SUPPORTED | Nanosecond resolution at NASDAQ granularity; "a single PRNG seed… the entire simulation to be guaranteed identical." | Use as stated. |
| 19 | Amrouni et al. (2021), ABIDES-Gym: Gym wrapper, market-making env, observation and action spaces, async→sync step() | arXiv 2110.14771 | QUALIFY / partly UNVERIFIED | Gym wrapper for a discrete-event simulator is confirmed. Their example action space is for other environments; "async→sync" is not found in the text. | Cite for the wrapper only. Describe our env from `envs/lob_env.py`. |
| 20 | Guéant & Manziuk (2019): beating the curse of dimensionality; scaling to portfolios | arXiv 1910.13205, abstract | SUPPORTED | "Scalable to portfolios of a few dozen bonds, hence beating the curse of dimensionality." | Use the abstract's wording. The "finite difference scaling issues" phrase is UNVERIFIED, so drop it. |
| 21 | Lim & Auer (2021): CVaR time-consistency; new distributional Bellman operator | arXiv: none; NeurIPS 2022 full text (proceedings.neurips.cc, 13 pp.) | **Authors corrected to Lim & Malik (2022)**; claim SUPPORTED | Standard action-selection for CVaR in distributional RL (Dabney et al. 2018a; Keramati et al. 2020) "converges to neither the dynamic, Markovian CVaR nor the static CVaR even if the optimal CVaR policy is stationary and Markov." | Cite Lim & Malik (2022). **This applies to our wrapper**: our CVaR selection is that standard rule, so describe it as a heuristic, not a CVaR-optimal policy. |
| 22 | Gašperov, Begušić, Posedel Šimović & Kostanjčar (2021), RL survey: RL superior in risk-adjusted returns | *Mathematics* 9(21), 2689, DOI 10.3390/math9212689 (not arXiv) | UNVERIFIED (search summary only) | Summary says RL gives better risk-adjusted returns. Full text not read. | Cite the journal version. Verify the claim in full text. |
| 23 | Gašperov & Kostanjčar (2021): market making with signals (SGUs + DRL) | IEEE Access, DOI 10.1109/ACCESS.2021.3074782 (not arXiv) | UNVERIFIED | Full text not accessed. Its benchmark claims (FOIC/LIIC) are not used, since we dropped those variants. | Cite only for the state-design idea, after reading the text. |
| 24 | Sun, Huang & Yu (2022): DRQN-LSTM for MM | Not found on arXiv or by search | UNVERIFIED | Could not locate the paper. | Do not cite until the paper is found. This claim supports our recurrent design, so it matters. |
| 25 | Bacry, Mastromatteo & Muzy (2015), Hawkes processes in finance | arXiv 1502.04592 (title and authors confirmed) | UNVERIFIED content | Authors and title match. Claims not read. | Related-work mention only. Hawkes arrivals are out of scope. |
| 26 | Huang, Lehalle & Rosenbaum (2015), queue-reactive model | arXiv 1312.0563 (authors confirmed) | UNVERIFIED content | Authors and title match. The "calibrated to LOBSTER" claim is not read, and we did not calibrate to LOBSTER. | Cite for queue-reactive dynamics only, after reading the text. |
| 27 | Kumar (2020), Sample Paper 1: DRQN beats TD-RL benchmark; DQN generalises poorly | `paper1.txt` (AAMAS 2020 extended abstract) | QUALIFY | Abstract: the DRQN approach "outperforms a prominent benchmark strategy… which uses temporal-difference reinforcement learning." Motivation: DQN "struggle[s] to generalize when evaluated in different environments" (a motivation, not a result). | Cite the benchmark win. Present the DQN generalisation point as motivation. |
| 28 | Beysolow II (book, Springer): policy gradient unprofitable; DQN unstable | Book, no arXiv | UNVERIFIED | Not checked. | Drop unless the book is read. |
| 29 | Spooner & Savani (2020) and our CVaR framing | arXiv 2003.01820 | QUALIFY | See Critical finding 3. | Cite only the narrower claim from row 10. |

## Notes for drafting

- Rows 2, 5, 6, 7, 13, and 21 contain factual errors in the sheet. The sheet was
  wrong in each case, so do not carry these citations over.
- Rows 22–26 and 28 need full-text reading before any sentence is attached to them.
- Row 24 (Sun, Huang & Yu) cannot be validated from here. Either find the paper or
  remove the citation.

## Added on the drafting pass

- **Lim & Malik (2022), full text checked** (row 21 above): our CVaR selection rule is the one they show can fail to converge to CVaR-optimal behaviour. The paper treats it as a heuristic.
- **Sun, Huang & Yu (2022)** is still not located, so it is not cited anywhere in `paper/main.tex`.
- **Gašperov survey and signals paper:** cited only in the Related Work's survey sentence (`gasperov2021survey`). Read the full text before any finding is attributed to them.
- **IQN CVaR implementation (internal, not a literature claim):** `IQNAgent.act` takes the bottom α·K of samples already drawn from U[0,α], so its effective tail is α². Recorded as a TODO in Methods §CVaR.
