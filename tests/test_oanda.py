from filib.oanda import Oanda, find_instruments, G10_USD, FOREX
from filib.helpers import *


class MyFactors(Oanda):
    """Four-factor test model."""

    @swap_sign
    def relative_strenght_index(self):
        factor = rsi(self.close, 14)
        split = [0, 30, 70, 100]
        return factor, split

    # Below are the factors from 101 Formulaic Alphas by Zura Kakushadze
    # https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2701346

    def alpha008(self):
        factor = -1 * rank(
            (
                (ts_sum(self.open, 5) * ts_sum(self.returns, 5))
                - delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10)
            )
        )
        return factor

    def alpha026(self):
        factor = -1 * ts_max(
            correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3
        )
        return factor

    def alpha034(self):
        factor = rank(
            (
                (1 - rank((stddev(self.returns, 2) / stddev(self.returns, 5))))
                + (1 - rank(delta(self.close, 1)))
            )
        )
        return factor


def test_workflow():

    # Initialize parameters
    model = MyFactors(
        instruments=G10_USD,  # Define universe
        symbol="USD",  # Optional, specify symbol to arrange price data
        granularity="D",  # Time period between each candle and between each rebalance
        count=500,  # Number of historical OHLCV candles to return for analysis
        periods=(1, 2, 3),  # Optional, specify periods for factor decay analysis
        split=3,  # Number of quantiles to split combined factor data
        long_short=True,  # Trade only top and bottom factor quantile
        combination="sum_of_weights",  # Formula for combining factors together
        leverage=3,  # Multiplier for the portfolio positions
    )
    assert isinstance(model, Oanda)
    assert len(model) == 4
    print(model)

    # Run commands from the proposed workflow
    model.performance()
    model.select(
        rules="abs(ic) > .01 or profit > 1",  # Example query expression
        swap_to="cagr",  # Align the signs of selected factors to specified metric
        inplace=True,  # Modify model to contain only selected factors
    )
    model.rebalance()

    # No objects to concatenate
    model.select(rules="abs(ic) > 1")
    model.performance()

    # Update attributes
    model.instruments = find_instruments("EUR", FOREX)
    model["momentum"] = lambda self: self.returns
    model.performance("momentum")
