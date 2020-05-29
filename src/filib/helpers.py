"""Module with a library of useful functions for factor generation."""

__all__ = [
    "get_iso_codes",
    "correlation",
    "delay",
    "delta",
    "product",
    "rank",
    "scale",
    "stddev",
    "ts_sum",
    "ts_min",
    "ts_max",
    "ts_rank",
    "z_score",
    "rsi",
]

from typing import Dict, Optional

import pandas as pd


def get_iso_codes(price_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Return {country ISO 3166 alpha-3: currency ISO 4217} codes."""
    iso_codes = {
        "AUS": "AUD",  # Australia
        "CAN": "CAD",  # Canada
        "CHE": "CHF",  # Switzerland
        "CHN": "CNH",  # China (People's Republic of)
        "CZE": "CZK",  # Czech Republic
        "DNK": "DKK",  # Denmark
        "EA19": "EUR",  # Euro area (19 countries)
        "EUR": "EUR",  # European Union
        "GBR": "GBP",  # United Kingdom
        "HKG": "HKD",  # Hong Kong
        "HUN": "HUF",  # Hungary
        "IND": "INR",  # India
        "JPN": "JPY",  # Japan
        "MEX": "MXN",  # Mexico
        "NOR": "NOK",  # Norway
        "NZL": "NZD",  # New Zealand
        "POL": "PLN",  # Poland
        "SAU": "SAR",  # Saudi Arabia
        "SIN": "SGD",  # Singapore
        "SWE": "SEK",  # Sweden
        "THA": "THB",  # Thailand
        "TUR": "TRY",  # Turkey
        "USA": "USD",  # United States
        "ZAF": "ZAR",  # South Africa
    }
    if price_data is not None:
        iso_codes = {
            country: currency
            for country, currency in iso_codes.items()
            if currency in price_data.columns.levels[0]
        }
    return iso_codes


def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-serial correlation of x and y for the past d days."""
    return x.rolling(d).corr(y)


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return value of x d days ago."""
    return x.shift(d)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return today's value of x minus the value of x d days ago."""
    return x.diff(d)


def product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-series product over the past d days."""
    return x.rolling(d).apply(lambda x: x.prod(), raw=True)


def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Return cross-sectional rank."""
    return x.rank(axis=1, pct=True)


def scale(x: pd.DataFrame, a: float = 1) -> pd.DataFrame:
    """Return rescaled x such that sum(abs(x)) = a (the default is a = 1)."""
    return x.mul(a).div(abs(x).sum())


def stddev(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return moving time-series standard deviation over the past d days."""
    return x.rolling(d).std()


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-series sum over the past d days."""
    return x.rolling(d).sum()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-series min over the past d days."""
    return x.rolling(d).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-series max over the past d days."""
    return x.rolling(d).max()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Return time-series rank in the past d days."""
    return x.rolling(d).apply(lambda na: pd.Series(na).rank().to_numpy()[-1], raw=True)


def z_score(x: pd.DataFrame, d: int = 20) -> pd.DataFrame:
    """Return moving time-series standard score over the past d days."""
    return x.apply(lambda z: (z - z.rolling(d).mean()) / z.rolling(d).std())


def rsi(x: pd.DataFrame, d: int = 14) -> pd.DataFrame:
    """Return Relative Strength Index indicator over the past d days."""
    change = x.diff()
    upward, downward = change.copy(), change.copy()
    upward[change <= 0] = 0
    downward[change > 0] = 0
    avg_gain = upward.ewm(d, adjust=False).mean()
    avg_loss = abs(downward.ewm(d, adjust=False).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    return rsi
