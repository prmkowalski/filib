from filib.oanda import Oanda
from filib.helpers import *


class SampleFactors(Oanda):
    """Four-factor test model."""

    def momentum(self):
        factor = self.returns
        split = [-1, -.003, .003, 1]
        return factor, split

    # Below are the factors from 101 Formulaic Alphas by Zura Kakushadze
    # https://papers.ssrn.com/sol3/Papers.cfm?abstract_id=2701346

    def alpha008(self):
        factor = (-1 * rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) - delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
        return factor

    def alpha026(self):
        factor = (-1 * ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3))
        return factor

    def alpha034(self):
        factor = (rank(((1 - rank((stddev(self.returns, 2) / stddev(self.returns, 5)))) + (1 - rank(delta(self.close, 1))))))
        return factor


def test_strategy():

    # Initialize parameters
    test = SampleFactors(
        instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'NZD_USD',
                       'USD_CAD', 'USD_CHF', 'USD_NOK', 'USD_SEK'],
        symbol = 'USD',
        granularity = 'H1',
        count = 500,
        periods = (1, 2, 3),
        split = 3,
        long_short = True,
        combination = 'sum_of_weights',
        leverage = 7,
    )
    assert isinstance(test, Oanda)

    # Run commands from the proposed workflow
    test.performance()
    test.select(
        rules = 'abs(ic) > .01 or profit > 1',
        swap = 'cagr'
    )
    test.performance()
    test.rebalance()

    # No objects to concatenate
    test.select(rules = 'abs(ic) > 1')

    # Update attributes
    test.instruments = 'USD_CAD', 'USD_CHF', 'USD_NOK', 'USD_SEK'
    test.performance()
