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

Check
`requirements.txt <https://github.com/makr3la/filib/blob/master/requirements.txt>`_
for dependencies.

Usage
-----

Write and execute a script based on following template:

.. code:: python

    from filib.oanda import Oanda  # Currently only Oanda is available
    from filib.helpers import *    # Optional functions, useful for factor generation


    class MyFactors(Oanda):

        def momentum(self):  # Hypothesis: there is persistence in an asset's performance
            factor = self.returns
            split = [-1, -.003, .003, 1]  # List of thresholds or int to split equally
            return factor, split  # Always return atleast factor and follow this order

        def relative_strenght_index(self):
            import pandas as pd
            import pandas_ta as ta  # Sample Technical Analysis Library
            factor = pd.DataFrame({
                instrument: self.price_data[instrument].ta.rsi(length=14)
                for instrument, _ in self.price_data
            }) * -1  # Short low factor values and long high factor values
            split = [-100, -70, -30, 0]
            return factor, split

        def big_mac_index(self):
            import quandl  # Sample Financial, Economic and Alternative Data Library
            iso_codes = get_iso_codes(self.price_data)
            codes = [f'ECONOMIST/BIGMAC_{COUNTRY}.5' for COUNTRY in iso_codes]
            factor = quandl.get(codes).dropna(how='all', axis=1)
            factor.columns = [iso_codes[c.split('_')[1].split()[0]] for c in factor]
            factor.index = factor.index.tz_localize('UTC')  # Convert time zone to UTC
            return factor  # If not specified split = 3 by default


    if __name__ == "__main__":

        strategy = MyFactors(  # Initialize strategy
            instruments = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD', 'NZD_USD',
                           'USD_CAD', 'USD_CHF', 'USD_NOK', 'USD_SEK'],  # Define universe
            granularity = 'D',    # Time range between rebalancing and between each candle
            count = 500,          # Number of historical candles to return for analysis
            symbol = 'USD',       # Optional, specify symbol to arrange your price data
            periods = (1, 2, 3),  # Optional, specify periods to analyze factor decay
            split = 3,            # Number of quantiles to split your combined factor data
            accountID = '',       # Your Oanda's account ID for creating orders
            target = .8,          # Set portfolio value target
            long_short = True,    # Trade only top and bottom quantile of combined factor
        )

        strategy.performance()  # Check performance of your factors combined together
        strategy.performance('momentum')  # Check performance for individual factors
        [strategy.performance(factor) for factor in strategy.factors]

        strategy.rebalance()  # Generate orders needed to rebalance your portfolio
        strategy.rebalance(live=True)  # Actually place orders
        # PLEASE USE AT YOUR OWN RISK - THIS CAN TRADE REAL MONEY - NO WARRANTY IS GIVEN

Contributing
------------

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.
