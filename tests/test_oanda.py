import os

from filib.oanda import Oanda
from filib.helpers import *


class SampleFactors(Oanda):

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
        granularity = 'D',
        count = 500,
        symbol = 'USD',
        periods = (1, 2, 3),
        split = 3,
        accountID = os.environ['OANDA_ACCOUNT_ID'],
        leverage = 7,
        long_short = True,
        combination = 'sum_of_weights')
    assert isinstance(test, Oanda)

    # Run commands from the proposed workflow
    test.performance()
    test.select(
        rules = 'abs(ic) > .01 or profit > 1',
        swap = 'cagr')
    test.rebalance()
