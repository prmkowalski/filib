filib
=====

.. image:: https://img.shields.io/pypi/pyversions/filib
    :target: https://pypi.org/project/filib/
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/v/filib
    :target: https://pypi.org/project/filib/
    :alt: PyPI

.. image:: https://img.shields.io/pypi/status/filib
    :target: https://pypi.org/project/filib/
    :alt: PyPI - Status

.. image:: https://img.shields.io/github/license/makr3la/filib
    :target: https://github.com/makr3la/filib/blob/master/LICENSE
    :alt: GitHub

.. image:: https://github.com/makr3la/filib/workflows/CI/badge.svg
    :target: https://github.com/makr3la/filib/actions?query=workflow%3ACI+branch%3Amaster
    :alt: CI - Status

`Factor Investing <https://en.wikipedia.org/wiki/Factor_investing>`_
LIBrary is a lightweight algorithmic trading Python library built for easy
testing of predictive factors and portfolio rebalance via
`Oanda <https://www.oanda.com/>`_. Inspired by and compatible with
`Quantopian Open Source <https://www.quantopian.com/opensource>`_.

`Changelog » <https://github.com/makr3la/filib/releases>`_

Installation
------------

Install with `pip <https://pip.pypa.io/en/stable/>`_:

.. code:: bash

    $ pip install filib

Usage
-----

Proposed workflow contains three steps. Here's an example:

1. Assemble
^^^^^^^^^^^

Begin with imports, then hypothesize and create predictive factors as class methods:

.. code:: python

    from filib.oanda import Oanda  # Currently only Oanda is available
    from filib.helpers import *    # Optional functions, useful for factor generation


    class MyFactors(Oanda):

        def momentum(self):  # HYPOTHESIS: there is persistence in an asset's performance
            factor = self.returns
            split = [-1, -.003, .003, 1]  # List of thresholds or int to split equally
            return factor, split  # Always return at least factor and follow this order

        def relative_strenght_index(self):  # H: signal oversold or overbought assets
            import pandas as pd
            import pandas_ta as ta  # Example Technical Analysis Library
            factor = pd.DataFrame({
                instrument: self.price_data[instrument].ta.rsi(length=14)
                for instrument, _ in self.price_data
            }) * -1  # Short low factor values and long high factor values
            split = [-100, -70, -30, 0]
            return factor, split

        def big_mac_index(self):  # H: simplified Purchasing Power Parity theory
            import quandl  # Example Financial, Economic and Alternative Data Library
            iso_codes = get_iso_codes(self.price_data)
            codes = [f'ECONOMIST/BIGMAC_{COUNTRY}.5' for COUNTRY in iso_codes]
            factor = quandl.get(codes).dropna(how='all', axis=1)
            factor.columns = [iso_codes[c.split('_')[1].split()[0]] for c in factor]
            factor.index = factor.index.tz_localize('UTC')  # Convert time zone to UTC
            return factor  # If not specified split = 3 by default

2. Research
^^^^^^^^^^^

Initialize parameters (during the first run you will be asked to provide credentials):

.. code:: python

    research = MyFactors(
        instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'NZD_USD',
                       'USD_CAD', 'USD_CHF', 'USD_NOK', 'USD_SEK'],  # Define universe
        granularity = 'D',    # Time range between rebalancing and between each candle
        count = 500,          # Number of historical candles to return for analysis
        symbol = 'USD',       # Optional, specify symbol to arrange your price data
        periods = (1, 2, 3),  # Optional, specify periods to analyze factor decay
        split = 3,            # Number of quantiles to split your combined factor data
        long_short = True,    # Trade only top and bottom quantile of combined factor
        combination = 'sum_of_weights')  # Select factor combination method

Check the performance of factors combined together:

.. code::

    >>> research.performance()
    ...
    MyFactors - INFO - Factor `MyFactors_combined` Analytics:

                       Min    Max   Mean   Size Returns (bps)
                    factor factor factor factor            1D     2D     3D
    factor_quantile
    1.0             -1.131  0.009 -0.238   1299        -0.551 -1.292 -0.845
    2.0             -0.206  0.331  0.004   1213        -1.411 -2.811 -3.009
    3.0              0.000  1.125  0.247   1232        -0.663 -1.289 -3.189

                                   1D     2D     3D
    - Information Coefficient:  0.005 -0.001 -0.007
    - Factor Rank Autocorrelation: 0.09

    - Annualized Sharpe Ratio: -0.46
    - Annualized Alpha (Beta): -0.011 (0.119)
    - Win Rate: 48.32%
    - Risk / Reward: 0.99
    - Profit Factor: 0.92

    - Start Date: 2018-08-07
    - End Date: 2020-03-10
    - Duration: 581 days 00:00:00 (1.6 years)
    - Rebalance every: 1D

    - Compound Annual Growth Rate: -1.39%
    - Annualized Volatility: 2.94%
    - Maximum Drawdown: -4.09%
    - Maximum Drawdown Duration: 23 days 00:00:00
    ...

Alternatively set selection rules with a
`query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_
string to evaluate. Available metrics are:

- **ic**: Information Coefficient based on Spearman's Rank Correlation
- **autocorr**: Factor Rank Autocorrelation
- **sharpe**: Annualized Sharpe Ratio
- **beta**: Annualized Beta as exposure to trading universe
- **alpha**: Annualized Alpha as excess returns over trading universe
- **win**: Win Rate
- **rr**: Risk / Reward Ratio
- **profit**: Profit Factor = (sum of earnings) / (sum of losses)
- **cagr**: Compound Annual Growth Rate

Then analyze the performance of individual factors and select those that meet the rules:

.. code::

    >>> research.select(
    ...     rules = 'abs(ic) > .01 or profit > 1',  # Example query expression
    ...     swap = 'cagr')  # Align the signs of selected factors to specified metric
    ...
    MyFactors - INFO - Factors with signs that meet the rules `abs(ic) > .01 or profit > 1`:

    momentum                  -1.0
    relative_strenght_index    1.0


    MyFactors - INFO - Factor `MyFactors_selected` Analytics:

                       Min    Max   Mean   Size Returns (bps)
                    factor factor factor factor            1D     2D     3D
    factor_quantile
    1.0             -1.000  0.026 -0.107   1815        -1.972 -3.022 -4.282
    2.0             -0.152  1.000  0.095    757         2.189  3.523  3.425
    3.0              0.000  0.880  0.222    551         1.257 -0.077  1.388

                                   1D    2D     3D
    - Information Coefficient:  0.017  0.01  0.016
    - Factor Rank Autocorrelation: 0.04

    - Annualized Sharpe Ratio: 0.30
    - Annualized Alpha (Beta): 0.011 (0.025)
    - Win Rate: 44.71%
    - Risk / Reward: 0.92
    - Profit Factor: 1.06

    - Start Date: 2018-08-07
    - End Date: 2020-03-10
    - Duration: 581 days 00:00:00 (1.6 years)
    - Rebalance every: 1D

    - Compound Annual Growth Rate: 1.02%
    - Annualized Volatility: 3.61%
    - Maximum Drawdown: -4.34%
    - Maximum Drawdown Duration: 371 days 00:00:00
    ...

3. Trade
^^^^^^^^

Execute or schedule a script to rebalance your portfolio based on selected factors:

**PLEASE USE AT YOUR OWN RISK - THIS CAN TRADE REAL MONEY - NO WARRANTY IS GIVEN**

.. code:: python

    # strategy.py
    from filib.oanda import Oanda


    class SelectedFactors(Oanda):

        def momentum(self):
            factor = self.returns * -1.0  # Sign from the research
            split = [-1, -.003, .003, 1]
            return factor, split

        def relative_strenght_index(self):
            import pandas as pd
            import pandas_ta as ta
            factor = pd.DataFrame({
                instrument: self.price_data[instrument].ta.rsi(length=14)
                for instrument, _ in self.price_data
            }) * -1
            split = [-100, -70, -30, 0]
            return factor, split


    if __name__ == "__main__":

        strategy = SelectedFactors(
            instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'NZD_USD',
                           'USD_CAD', 'USD_CHF', 'USD_NOK', 'USD_SEK'],
            granularity = 'D',
            count = 250,
            symbol = 'USD',
            split = 3,
            accountID = '',  # Your Oanda's account ID for creating orders
            leverage = 7,    # Set the leverage for the portfolio positions
            long_short = True,
            combination = 'sum_of_weights')

        strategy.rebalance(live=True)  # Actually place orders

Check portfolio positions and generated orders in a log file or by dry run:

.. code::

    >>> strategy.rebalance()
    SelectedFactors - INFO - Portfolio from `2020-03-11 00:00:00+00:00`:

    CHF    -38.4%
    SEK     -8.1%
    EUR     -3.5%
    GBP      0.0%
    NOK      0.0%
    NZD      0.0%
    AUD     13.9%
    CAD     15.7%
    JPY     20.4%

    - Account NAV: 10000.00 EUR
    - Position Value: 0.00
    - Needed Orders:

    USD_CHF    30410
    USD_SEK     6430
    EUR_USD    -2430
    AUD_USD    16860
    USD_CAD   -12450
    USD_JPY   -16140

Contributing
------------

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.
