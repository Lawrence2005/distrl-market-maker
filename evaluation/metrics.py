"""
All evaluation metrics.

Implements every metric in the Evaluation Framework (Sheet 2 of research plan):

P&L Performance:
  - annualized_sharpe(returns)
  - mean_episode_pnl(pnl_series)
  - normalized_daily_pnl(pnl, avg_spread)   [Spooner 2018]
  - terminal_wealth(cash, inventory, final_price)
  - win_rate(pnl_series)                     [Sun 2022]
  - pnlmap_ratio(pnl, map_value)             [Gašperov signals 2021]

Tail Risk:
  - cvar(returns, alpha)                     [QR-DQN quantile avg]
  - max_drawdown(pnl_series)
  - return_cvar_frontier(models, alpha_sweep)

Inventory:
  - mean_absolute_position(inventory_series) [MAP, Spooner 2018]
  - inventory_std(inventory_series)
  - inventory_tail_exceedance(inventory_series, Q_max)

Market Quality:
  - fill_rate(orders_placed, orders_filled)
  - price_impact_signature(mid_prices, order_sizes)
  - adverse_selection_cost(effective_spread, quoted_spread)

Week 8–9 deliverable.
"""
# TODO: implement
