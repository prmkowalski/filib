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

.. image:: https://img.shields.io/github/license/prmkowalski/filib
    :target: https://github.com/prmkowalski/filib/blob/master/LICENSE
    :alt: GitHub

.. image:: https://github.com/prmkowalski/filib/workflows/CI/badge.svg
    :target: https://github.com/prmkowalski/filib/actions?query=workflow%3ACI+branch%3Amaster
    :alt: CI - Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black

`Factor Investing <https://en.wikipedia.org/wiki/Factor_investing>`_
LIBrary is a lightweight algorithmic trading Python library built for easy testing of
predictive factors and portfolio rebalance via
`Oanda <https://developer.oanda.com/rest-live-v20/introduction/>`_.
Inspired by and compatible with
`Quantopian Open Source <https://github.com/quantopian>`_.

    **NOTE**: This library is currently in alpha stage. Until it becomes stable
    I strongly recommend using practice account for testing and trading. You can also
    expect major changes without warnings, mostly responses to
    `Issues <https://github.com/prmkowalski/filib/issues>`_.

`Changelog » <https://github.com/prmkowalski/filib/releases>`_

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

Begin with imports, create hypotheses and write functions with predictive factors:

.. code:: python

    from filib.oanda import Oanda  # Currently only Oanda FOREX is available
    from filib.helpers import *  # Optional, useful for factor generation


    class MyFactors(Oanda):
        def momentum(self):  # THEORY: Persistence in asset performance
            factor = self.returns  # Write down your factor formula
            return factor  # By default split factor data to 3 quantiles

        @swap_sign  # Short high and long low factor values
        def relative_strenght_index(self):  # THEORY: Oversold / overbought indicator
            factor = rsi(self.close, 14)
            split = [0, 30, 70, 100]  # List of thresholds or int to split equally
            return factor, split  # Follow this order: factor, split

        def big_mac_index(self):  # THEORY: Simplified Purchasing Power Parity
            import quandl  # Financial, Economic and Alternative Data

            iso_codes = get_iso_codes(self.price_data)
            codes = [f"ECONOMIST/BIGMAC_{COUNTRY}.5" for COUNTRY in iso_codes]
            factor = quandl.get(codes).dropna(how="all", axis=1)
            factor.columns = [iso_codes[c.split("_")[1].split()[0]] for c in factor]
            factor.index = factor.index.tz_localize("UTC")  # Convert time zone to UTC
            return factor

2. Research
^^^^^^^^^^^

Initialize parameters (during the first run you will be asked to provide credentials):

.. code:: python

    model = MyFactors(
        instruments=["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "NZD_USD", "USD_CAD",
                     "USD_CHF", "USD_NOK", "USD_SEK"],  # Define universe
        symbol="USD",  # Optional, specify symbol to arrange price data
        granularity="D",  # Time period between each candle and between each rebalance
        count=500,  # Number of historical OHLCV candles to return for analysis
        periods=(1, 2, 3),  # Optional, specify periods for factor decay analysis
        split=3,  # Number of quantiles to split combined factor data
        long_short=True,  # Trade only top and bottom factor quantile
        combination="sum_of_weights",  # Formula for combining factors together
        leverage=3,  # Multiplier for the portfolio positions
    )

Check the performance of factors combined together:

.. code::

    >>> model.performance()
    Collecting price data: |██████████████████████████████| 9/9 [100%] in 4.0 s
    Preparing factor data: |██████████████████████████████| 3/3 [100%] in 12.0 s

    MyFactors - INFO - Factor `MyFactors_combined` Analytics:

                    Min    Max    Mean   Size    Returns (bps)
                    factor factor factor factor            1D     2D     3D
    factor_quantile
    1.0             -1.003  0.000 -0.237   1499        -1.337 -2.068 -2.320
    2.0             -0.243  0.210  0.005   1461        -2.582 -3.299 -5.138
    3.0             -0.027  0.973  0.238   1459         0.892 -0.835 -2.266

                                1D     2D     3D
    - Information Coefficient:  0.037  0.001  0.0
    - Factor Rank Autocorrelation: 0.05

    - Annualized Sharpe Ratio: 0.76
    - Annualized Alpha (Beta): 0.080 (0.042)
    - Win Rate: 52.55%
    - Risk / Reward: 1.02
    - Profit Factor: 1.15

    - Start Date: 2018-07-11
    - End Date: 2020-05-27
    - Duration: 686 days 00:00:00 (1.9 years)
    - Rebalance every: 1D

    - Compound Annual Growth Rate: 7.78%
    - Annualized Volatility: 10.44%
    - Maximum Drawdown: -11.49%
    - Maximum Drawdown Duration: 434 days 00:00:00
    ...

Alternatively set selection rules with a
`query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_
string to evaluate. Available metrics:

- **ic**:       Information Coefficient based on Spearman Rank Correlation
- **autocorr**: Factor Rank Autocorrelation
- **sharpe**:   Annualized Sharpe Ratio
- **beta**:     Annualized Beta as exposure to trading universe
- **alpha**:    Annualized Alpha as excess returns over trading universe
- **win**:      Win Rate
- **rr**:       Risk / Reward Ratio
- **profit**:   Profit Factor = (sum of earnings) / (sum of losses)
- **cagr**:     Compound Annual Growth Rate

Then analyze the performance of individual factors and select those that meet the rules:

.. code::

    >>> model.select(
    ...     rules="abs(ic) > .01 or profit > 1",  # Example query expression
    ...     swap_to="cagr",  # Align the signs of selected factors to specified metric
    ...     inplace=True,  # Modify model to contain only selected factors
    ... )
    Preparing performance: |██████████████████████████████| 3/3 [100%] in 6.2 s

    MyFactors - INFO - Factors with signs that meet the rules `abs(ic) > .01 or profit > 1`:

    big_mac_index             -1.0
    momentum                   1.0
    relative_strenght_index    1.0

3. Trade
^^^^^^^^

Check portfolio positions based on selected factors and generated submitted orders:

**PLEASE USE AT YOUR OWN RISK - THIS CAN TRADE REAL MONEY - NO WARRANTY IS GIVEN**

.. code::

    >>> model.rebalance(
    ...     accountID="",  # Your Oanda Account Identifier
    ...     live=True,  # Actually place orders
    ... )
    MyFactors - INFO - Portfolio from `2020-05-28 00:00:00+00:00`:

    NOK    -19.5%
    SEK    -15.3%
    CHF    -15.2%
    AUD      0.0%
    EUR      0.0%
    GBP      0.0%
    NZD      9.0%
    CAD     15.3%
    JPY     25.8%

    - Account NAV: 8423.77 EUR
    - Position Value: 25382.12
    - Submitted Orders:

    USD_JPY   -7240
    NZD_USD    4050
    USD_CAD   -4280
    USD_CHF    4260
    USD_NOK    5490
    USD_SEK    4280

Contributing
------------

Pull requests are welcome. For major changes, please open an issue first to discuss
what you would like to change.
