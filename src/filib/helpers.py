"""Module with a library of useful functions for factor generation."""

__all__ = [
    'get_iso_codes', 'correlation', 'delay', 'delta', 'product', 'rank',
    'scale', 'stddev', 'ts_sum', 'ts_min', 'ts_max', 'ts_rank'
]

import pandas as pd


def get_iso_codes(price_data=None):
    iso_codes = {
        'AUS': 'AUD',  # Australia
        'CAN': 'CAD',  # Canada
        'CHE': 'CHF',  # Switzerland
        'CHN': 'CNH',  # China (People's Republic of)
        'CZE': 'CZK',  # Czech Republic
        'DNK': 'DKK',  # Denmark
        'EA19': 'EUR',  # Euro area (19 countries)
        'EUR': 'EUR',  # European Union
        'GBR': 'GBP',  # United Kingdom
        'HKG': 'HKD',  # Hong Kong
        'HUN': 'HUF',  # Hungary
        'IND': 'INR',  # India
        'JPN': 'JPY',  # Japan
        'MEX': 'MXN',  # Mexico
        'NOR': 'NOK',  # Norway
        'NZL': 'NZD',  # New Zealand
        'POL': 'PLN',  # Poland
        'SAU': 'SAR',  # Saudi Arabia
        'SIN': 'SGD',  # Singapore
        'SWE': 'SEK',  # Sweden
        'THA': 'THB',  # Thailand
        'TUR': 'TRY',  # Turkey
        'USA': 'USD',  # United States
        'ZAF': 'ZAR',  # South Africa
    }
    if price_data is not None:
        iso_codes = {country: currency
                     for country, currency in iso_codes.items()
                     if currency in price_data.columns.levels[0]}
    return iso_codes


def correlation(x, y, d):
    """Return time-serial correlation of x and y for the past d days."""
    return x.rolling(d).corr(y)


def delay(x, d):
    """Return value of x d days ago."""
    return x.shift(d)


def delta(x, d):
    """Return today's value of x minus the value of x d days ago."""
    return x.diff(d)


def product(x, d):
    """Return time-series product over the past d days."""
    return x.rolling(d).apply(lambda x: x.prod(), raw=True)


def rank(x):
    """Return cross-sectional rank."""
    return x.rank(axis=1, pct=True)


def scale(x, a=1):
    """Return rescaled x such that sum(abs(x)) = a (the default is a = 1)."""
    return x.mul(a).div(abs(x).sum())


def stddev(x, d):
    """Return moving time-series standard deviation over the past d days."""
    return x.rolling(d).std()


def ts_sum(x, d):
    """Return time-series sum over the past d days."""
    return x.rolling(d).sum()


def ts_min(x, d):
    """Return time-series min over the past d days."""
    return x.rolling(d).min()


def ts_max(x, d):
    """Return time-series max over the past d days."""
    return x.rolling(d).max()


def ts_rank(x, d):
    """Return time-series rank in the past d days."""
    return x.rolling(d).apply(
        lambda na: pd.Series(na).rank().to_numpy()[-1], raw=True)
