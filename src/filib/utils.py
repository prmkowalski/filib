"""Module with functions used throughout the library."""

__all__ = [
    "get_factor_data",
    "combine_factors",
    "get_performance",
    "print_progress",
]

from contextlib import suppress
from datetime import datetime, timezone
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

with suppress(ImportError):
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = (12, 8)
import pandas as pd


def get_factor_data(
    factor: pd.DataFrame,
    price_data: pd.DataFrame,
    periods: Optional[List[int]] = None,
    split: Union[int, Sequence[float]] = 3,
    long_short: bool = False,
    leverage: float = 1,
    name: str = "",
) -> pd.DataFrame:
    """Return merged data: factor values, quantiles, weights and returns."""
    prices = price_data.xs("close", axis=1, level=1).filter(factor.columns)
    if factor.index.tz != prices.index.tz:
        raise ValueError("The time zone of `factor` and `prices` don't match.")
    factor.loc[datetime.now(timezone.utc)] = float("nan")
    factor.replace([float("-inf"), float("inf")], float("nan"), inplace=True)
    factor = factor.resample(prices.index.freq).ffill()[prices.index[0] :]
    periods = [1] if not periods else [1] + sorted(periods)
    deltas = [period * prices.index.to_series().diff().mode() for period in periods]
    deltas = [
        (
            delta.to_string(index=False).replace(":", "h", 1).replace(":", "m") + "s"
        ).replace(" dayss", "D")
        for delta in deltas
    ]
    forward_returns = {
        delta: -prices.diff(-period) / prices
        for period, delta in dict(zip(periods, deltas)).items()
    }
    index = factor.index.intersection(prices.index)
    factor_data = pd.concat(forward_returns, axis=1).reindex(index).stack()
    factor_data["factor"] = factor.stack()
    if isinstance(split, int):
        factor_quantile = 1 + factor_data.groupby(level=0)["factor"].transform(
            lambda x: pd.qcut(x, split, labels=False, duplicates="drop")
        )
    elif isinstance(split, (list, tuple, set)):
        factor_quantile = 1 + factor_data.groupby(level=0)["factor"].transform(
            lambda x: pd.cut(x, split, labels=False, duplicates="drop")
        )
        split = len(split) - 1
    else:
        raise ValueError(f"Factor `{name}` split type {type(split)} is not supported.")
    factor_data["factor_quantile"] = factor_quantile
    quantiles = [1, split] if long_short else list(range(1, split + 1))
    factor_data["weights"] = (
        factor_data[factor_data["factor_quantile"].isin(quantiles)]
        .groupby(level=0)["factor"]
        .transform(lambda x: (x - x.mean()) / (x - x.mean()).abs().sum())
    )
    factor_data["weights"].fillna(0, inplace=True)
    for period in forward_returns:
        factor_data[f"{name}_{period}"] = (
            factor_data["weights"] * factor_data[period] * leverage
        )
    factor_data.rename_axis(index=["date", "asset"], inplace=True)
    factor_data.name = name
    return factor_data


def combine_factors(
    factor_data: Dict[str, pd.DataFrame], combination: str
) -> pd.Series:
    """Return single factor by applying the provided weighting scheme."""
    if combination == ("sum_of_weights").lower():
        combined_factor = (
            pd.concat([factor["weights"] for factor in factor_data.values()], axis=1)
            .sum(axis=1)
            .unstack()
        )
    else:
        raise ValueError(f"Combination `{combination}` not available.")
    return combined_factor


def get_performance(
    factor_data: pd.DataFrame, plot: bool = True
) -> Tuple[str, pd.Series]:
    """Return factor performance analytics."""
    factor_index = list(factor_data.columns).index("factor")
    periods = factor_data.columns[:factor_index]
    returns_column = factor_data.columns[factor_index + 3]
    returns = factor_data[returns_column].groupby(level=0).sum()
    cum_returns = (1 + returns).cumprod()
    factor_stats = pd.concat(
        [
            factor_data.groupby("factor_quantile")["factor"].min(),
            factor_data.groupby("factor_quantile")["factor"].max(),
            factor_data.groupby("factor_quantile")["factor"].mean(),
            factor_data.groupby("factor_quantile")["factor"].size(),
            factor_data.groupby("factor_quantile").mean()[periods] * 10000,
        ],
        keys=["Min", "Max", "Mean", "Size", "Returns (bps)"],
        axis=1,
    )
    ic = pd.DataFrame()
    for period in periods:
        correlation = (
            factor_data[["factor", period]]
            .dropna()
            .groupby(level=0)
            .corr(method="spearman")
            .prod(axis=1)
            .mean()
        )
        ic.loc["- Information Coefficient:", period] = correlation
    autocorr = (
        factor_data["factor"]
        .unstack()
        .rank(axis=1)
        .apply(lambda col: col.autocorr())
        .mean()
    )
    duration = returns.index[-1] - returns.index[0]
    years = duration.days / 365.25
    try:
        sharpe = returns.mean() / returns.std()
        annualized_sharpe = sharpe * (len(returns) / years) ** 0.5
    except ZeroDivisionError:
        annualized_sharpe = float("nan")
    x = factor_data.groupby(level=0).mean()[periods[0]].dropna()
    y = factor_data.groupby(level=0).sum().loc[x.index, returns_column]
    beta = (len(x) * sum(x * y) - sum(x) * sum(y)) / (
        len(x) * sum(x ** 2) - sum(x) ** 2
    )
    alpha = (sum(y) - beta * sum(x)) / len(x)
    annualized_alpha = alpha * (len(returns) / years)
    win_rate = len(returns[returns > 0]) / len(returns)
    risk_reward = returns[returns > 0].mean() / -returns[returns < 0].mean()
    try:
        profit_factor = sum(returns[returns > 0]) / -sum(returns[returns < 0])
    except ZeroDivisionError:
        profit_factor = float("nan")
    cagr = (cum_returns[-1] / cum_returns[0]) ** (1 / years) - 1
    annualized_volatility = returns.std() * (len(returns) / years) ** 0.5
    drawdown = 1 - cum_returns.div(cum_returns.cummax())
    dd_duration = drawdown[drawdown == 0].index.to_series().diff()
    past_returns = (
        returns.groupby([returns.index.year, returns.index.month]).sum().unstack()
    )
    past_returns["Total"] = past_returns.sum(numeric_only=True, axis=1)
    past_returns = past_returns.applymap("{0:.1%}".format).replace("nan%", "")
    log = (
        f"Factor `{factor_data.name}` Analytics:\n"
        f"\n"
        f"{factor_stats.round(3).to_string()}\n"
        f"\n"
        f"{ic.round(3).to_string()}\n"
        f"- Factor Rank Autocorrelation: {autocorr:.2f}\n"
        f"\n"
        f"- Annualized Sharpe Ratio: {annualized_sharpe:.2f}\n"
        f"- Annualized Alpha (Beta): {annualized_alpha:.3f} ({beta:.3f})\n"
        f"- Win Rate: {win_rate:.2%}\n"
        f"- Risk / Reward: {risk_reward:.2f}\n"
        f"- Profit Factor: {profit_factor:.2f}\n"
        f"\n"
        f"- Start Date: {returns.index[0].date()}\n"
        f"- End Date: {returns.index[-1].date()}\n"
        f"- Duration: {duration} ({years:.1f} years)\n"
        f"- Rebalance every: {returns_column.rsplit('_', 1)[-1]}\n"
        f"\n"
        f"- Compound Annual Growth Rate: {cagr.max():.2%}\n"
        f"- Annualized Volatility: {annualized_volatility:.2%}\n"
        f"- Maximum Drawdown: -{drawdown.max():.2%}\n"
        f"- Maximum Drawdown Duration: {dd_duration.max()}\n"
        f"\n"
        f"- Past Returns:\n"
        f"\n"
        f"{past_returns.to_string(index_names=False)}\n"
    )
    if "plt" in globals() and plot:
        cum_returns.plot(title="Cumulative Returns", legend=True, linewidth=3)
        benchmark_rets = factor_data[periods[0]].groupby(level=0).sum()
        benchmark_cum_rets = (1 + benchmark_rets).cumprod().rename("benchmark")
        benchmark_cum_rets.plot(legend=True, secondary_y=True, color="gray")
        plt.show()
    summary = pd.Series(
        {
            "ic": ic.iloc[0, 0],
            "autocorr": autocorr,
            "sharpe": annualized_sharpe,
            "beta": beta,
            "alpha": annualized_alpha,
            "win": win_rate,
            "rr": risk_reward,
            "profit": profit_factor,
            "cagr": cagr,
        }
    ).rename(factor_data.name)
    return log, summary


def print_progress(
    current: int, total: int, prefix: str = "", suffix: str = "", bar_size: int = 30
) -> None:
    """Print string-based progress bar if connected to a console."""
    if sys.stdout.isatty() or "ipykernel" in sys.modules and total != 0:
        progress = current / total
        filled_length = int(progress * bar_size)
        bar = "█" * filled_length + "-" * (bar_size - filled_length)
        print(
            f"\r{prefix} |{bar}| {current}/{total} [{progress:.0%}] {suffix}",
            end="\033[K",
        )
        if current == total:
            print()
