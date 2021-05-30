"""Module used for backtesting and trading via OANDA v20 REST API."""

__all__ = [
    "find_instruments",
    "get_price_data",
    "Oanda",
    "MAJORS",
    "EXOTICS",
    "FOREX",
    "INDICES",
    "COMMODITIES",
    "METALS",
    "BONDS",
    "ALL_SYMBOLS",
    "G10_USD",
    "EM_USD",
    "ALL_USD",
]

import configparser
from datetime import datetime
from functools import lru_cache
from inspect import getmembers, isfunction
import json
import logging
import os
import pickle
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlencode
import urllib.request as ur

import pandas as pd

try:
    from pandas import json_normalize
except ImportError:
    from pandas.io.json import json_normalize

from .helpers import swap_sign
from .utils import get_factor_data, combine_factors, get_performance, print_progress

Factor = Callable[..., Tuple[pd.DataFrame, Optional[Union[int, Sequence[float]]]]]


def _get_hostname_headers() -> Tuple[str, Dict[str, str]]:
    """Return the V20 REST server hostname and the header fields for HTTP requests."""
    try:
        hostname = os.environ["OANDA_HOSTNAME"]
        token = os.environ["OANDA_TOKEN"]
    except KeyError:
        config = configparser.ConfigParser()
        config_filepath = os.path.join(os.path.dirname(__file__), "config.ini")
        try:
            with open(config_filepath, "r") as config_file:
                config.read_file(config_file)
                hostname = config.get("oanda", "hostname")
                token = config.get("oanda", "token")
        except FileNotFoundError:
            logger = logging.getLogger(__name__)
            logger.error(
                f"OANDA v20 REST API config file is not found. "
                f"Please answer to generate it:"
            )
            account_type = input("- What is your account type? `Live` or `Practice`?\n")
            if account_type.lower() in ["live", "l"]:
                hostname = "https://api-fxtrade.oanda.com"
            elif account_type.lower() in ["practice", "p"]:
                hostname = "https://api-fxpractice.oanda.com"
            else:
                raise ValueError(f"Type `{account_type}` not available.")
            token = input("- Provide your personal access token:\n")
            config["oanda"] = {"hostname": hostname, "token": token}
            with open(config_filepath, "w") as config_file:
                config.write(config_file)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Connection": "Keep-Alive",
        "AcceptDatetimeFormat": "RFC3339",
    }
    return hostname, headers


def find_instruments(symbol: str, universe: List[str]) -> List[str]:
    """Return the universe of instruments containing the given symbol."""
    instruments = []
    for instrument in universe:
        base, quote = instrument.split("_")
        if symbol in (base, quote):
            instruments.append(instrument)
    return instruments


def get_price_data(
    instruments: Sequence[str],
    symbol: Optional[str] = None,  # Run _arrange_price_data
    save: bool = False,  # Serialize price_data for faster retrieval
    granularity: str = "D",
    count: int = 500,
    end: Union[str, float] = datetime.utcnow().timestamp(),
    **kwargs,  # See https://developer.oanda.com/rest-live-v20/instrument-ep/
) -> pd.DataFrame:
    """Return historical OHLCV candles."""
    freq = {
        "S5": "5S",  # 5 second candlesticks, minute alignment
        "S10": "10S",  # 10 second candlesticks, minute alignment
        "S15": "15S",  # 15 second candlesticks, minute alignment
        "S30": "30S",  # 30 second candlesticks, minute alignment
        "M1": "T",  # 1 minute candlesticks, minute alignment
        "M2": "2T",  # 2 minute candlesticks, hour alignment
        "M4": "4T",  # 4 minute candlesticks, hour alignment
        "M5": "5T",  # 5 minute candlesticks, hour alignment
        "M10": "10T",  # 10 minute candlesticks, hour alignment
        "M15": "15T",  # 15 minute candlesticks, hour alignment
        "M30": "30T",  # 30 minute candlesticks, hour alignment
        "H1": "H",  # 1 hour candlesticks, hour alignment
        "H2": "2H",  # 2 hour candlesticks, day alignment
        "H3": "3H",  # 3 hour candlesticks, day alignment
        "H4": "4H",  # 4 hour candlesticks, day alignment
        "H6": "6H",  # 6 hour candlesticks, day alignment
        "H8": "8H",  # 8 hour candlesticks, day alignment
        "H12": "12H",  # 12 hour candlesticks, day alignment
        "D": "B",  # 1 day candlesticks, day alignment
        "W": "W-MON",  # 1 week candlesticks, aligned to start of week
    }
    granularity = granularity.upper()
    if granularity not in freq:
        raise ValueError(
            f"Granularity `{granularity}` not available - "
            f"choose from {list(freq.keys())}."
        )
    h = str(hash(f"{instruments} {symbol} {granularity} {count} {kwargs}"))
    try:
        with open(h + ".pickle", "rb") as f:
            price_data = pickle.load(f)
    except FileNotFoundError:
        count_list = [5000] * (count // 5000)
        if count % 5000 != 0:
            count_list.append(count % 5000)
        objs = []
        prefix = "Collecting price data:"
        start_time = time.time()
        for i, instrument in enumerate(instruments):
            print_progress(i, len(instruments), prefix, f"`{instrument}`")
            if instrument not in ALL_SYMBOLS:
                raise ValueError(f"Instrument `{instrument}` not available.")
            to_time = end
            responses = []
            for c in count_list:
                hostname, headers = _get_hostname_headers()
                endpoint = f"/v3/instruments/{instrument}/candles"
                params = {
                    "granularity": granularity,
                    "count": c,
                    "to": to_time,
                    **kwargs,
                }
                url = hostname + endpoint + "?" + urlencode(params)
                req = ur.Request(url, headers=headers)
                with ur.urlopen(req) as r:
                    df = json_normalize(json.loads(r.read()), "candles").set_index(
                        "time"
                    )
                    to_time = df.index[0]
                    df.index = pd.to_datetime(df.index, utc=True)
                    df.drop("complete", axis=1, inplace=True)
                    columns = {
                        **{c: "open" for c in df.columns if c.endswith(".o")},
                        **{c: "high" for c in df.columns if c.endswith(".h")},
                        **{c: "low" for c in df.columns if c.endswith(".l")},
                        **{c: "close" for c in df.columns if c.endswith(".c")},
                    }
                    df.rename(columns=columns, inplace=True)
                    df = df.resample(freq[granularity]).agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    df = df.astype(float)
                responses.append(df)
                time.sleep(0.1)
            objs.append(pd.concat(responses).sort_index())
        suffix = f"in {time.time() - start_time:.1f} s"
        print_progress(len(instruments), len(instruments), prefix, suffix)
        price_data = pd.concat(objs, axis=1, keys=instruments)
        price_data = _arrange_price_data(price_data, symbol)
        price_data = price_data.ffill().dropna()
        price_data.index.freq = price_data.index.inferred_freq
        if save:
            with open(h + ".pickle", "wb") as f:
                pickle.dump(price_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return price_data


def _arrange_price_data(price_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Arrange the instruments to be quoted in the given symbol."""
    arranged = pd.DataFrame()
    for instrument, price in price_data:
        base, quote = instrument.split("_")
        if base == symbol:
            if price in ["open", "high", "low", "close"]:
                arranged[(quote, price)] = price_data[(instrument, price)] ** -1
            else:
                arranged[(quote, price)] = price_data[(instrument, price)]
        elif quote == symbol:
            arranged[(base, price)] = price_data[(instrument, price)]
        else:
            arranged[(instrument, price)] = price_data[(instrument, price)]
    arranged.columns = pd.MultiIndex.from_tuples(arranged.columns)
    return arranged


class Oanda:
    """Base class for creating factor model."""

    def __init__(
        self,
        instruments: Sequence[str],
        symbol: Optional[str] = None,
        granularity: str = "D",
        count: int = 500,
        periods: Optional[List[int]] = None,
        split: Union[int, Sequence[float]] = 3,
        long_short: bool = False,
        combination: str = "sum_of_weights",
        leverage: float = 1,
    ) -> None:
        self.instruments = instruments
        self.symbol = symbol
        self.granularity = granularity
        self.count = count
        self.periods = periods
        self.split = split
        self.long_short = long_short
        self.combination = combination
        self.leverage = leverage
        self.esteem = None
        self.factors = {
            name: function
            for name, function in dict(
                getmembers(self.__class__, predicate=isfunction)
            ).items()
            if name not in dict(getmembers(__class__, predicate=isfunction))
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(f"{self.__class__.__name__}.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(file_handler)

    @property
    def instruments(self) -> Sequence[str]:
        """Universe of instrument symbols."""
        return self._instruments

    @instruments.setter
    def instruments(self, value):
        self._instruments = value
        type(self).price_data.fget.cache_clear()
        type(self).open.fget.cache_clear()
        type(self).high.fget.cache_clear()
        type(self).low.fget.cache_clear()
        type(self).close.fget.cache_clear()
        type(self).volume.fget.cache_clear()
        type(self).returns.fget.cache_clear()
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def symbol(self) -> Optional[str]:
        """Optional symbol to arrange price data."""
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value
        type(self).price_data.fget.cache_clear()
        type(self).open.fget.cache_clear()
        type(self).high.fget.cache_clear()
        type(self).low.fget.cache_clear()
        type(self).close.fget.cache_clear()
        type(self).volume.fget.cache_clear()
        type(self).returns.fget.cache_clear()
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def granularity(self) -> str:
        """Time period between each candle and between each rebalance."""
        return self._granularity

    @granularity.setter
    def granularity(self, value):
        self._granularity = value
        type(self).price_data.fget.cache_clear()
        type(self).open.fget.cache_clear()
        type(self).high.fget.cache_clear()
        type(self).low.fget.cache_clear()
        type(self).close.fget.cache_clear()
        type(self).volume.fget.cache_clear()
        type(self).returns.fget.cache_clear()
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def count(self) -> int:
        """Number of historical OHLCV candles."""
        return self._count

    @count.setter
    def count(self, value):
        self._count = value
        type(self).price_data.fget.cache_clear()
        type(self).open.fget.cache_clear()
        type(self).high.fget.cache_clear()
        type(self).low.fget.cache_clear()
        type(self).close.fget.cache_clear()
        type(self).volume.fget.cache_clear()
        type(self).returns.fget.cache_clear()
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def periods(self) -> Optional[List[int]]:
        """Optional forward periods for factor decay analysis."""
        return self._periods

    @periods.setter
    def periods(self, value):
        self._periods = value
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def split(self) -> Union[int, Sequence[float]]:
        """Number of quantiles to split combined factor data."""
        return self._split

    @split.setter
    def split(self, value):
        self._split = value
        type(self).combined_factor_data.fget.cache_clear()

    @property
    def long_short(self) -> bool:
        """True if trade only top and bottom factor quantile, False if all."""
        return self._long_short

    @long_short.setter
    def long_short(self, value):
        self._long_short = value
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def combination(self) -> str:
        """Formula for combining factors together."""
        return self._combination

    @combination.setter
    def combination(self, value):
        self._combination = value
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    def leverage(self) -> float:
        """Multiplier for the portfolio positions."""
        return self._leverage

    @leverage.setter
    def leverage(self, value):
        self._leverage = value
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    @property
    @lru_cache()
    def price_data(self) -> pd.DataFrame:
        """Historical OHLCV candles data."""
        return get_price_data(
            instruments=self.instruments,
            granularity=self.granularity,
            count=self.count,
            symbol=self.symbol,
        )

    @property
    @lru_cache()
    def open(self) -> pd.DataFrame:
        """Opening price data."""
        return self.price_data.xs("open", axis=1, level=1)

    @property
    @lru_cache()
    def high(self) -> pd.DataFrame:
        """High price data."""
        return self.price_data.xs("high", axis=1, level=1)

    @property
    @lru_cache()
    def low(self) -> pd.DataFrame:
        """Low price data."""
        return self.price_data.xs("low", axis=1, level=1)

    @property
    @lru_cache()
    def close(self) -> pd.DataFrame:
        """Closing price data."""
        return self.price_data.xs("close", axis=1, level=1)

    @property
    @lru_cache()
    def volume(self) -> pd.DataFrame:
        """Tick volume data."""
        return self.price_data.xs("volume", axis=1, level=1)

    @property
    @lru_cache()
    def returns(self) -> pd.DataFrame:
        """Percentage change in the closing price data."""
        return self.close.pct_change()[1:]

    @property
    @lru_cache()
    def factor_data(self) -> Dict[str, pd.DataFrame]:
        """Individual factor values, quantiles, weights and returns."""
        factor_data = {}
        prefix = "Preparing factor data:"
        start_time = time.time()
        for i, (name, function) in enumerate(self.factors.items()):
            print_progress(i, len(self.factors), prefix, f"`{name}`")
            try:
                factor, s = function(self)
            except ValueError:
                factor, s = function(self), 3
            except TypeError:
                raise TypeError(f"`{name}` must return atleast factor.")
            factor_data[name] = get_factor_data(
                factor=factor,
                price_data=self.price_data,
                periods=self.periods,
                split=s,
                long_short=self.long_short,
                leverage=self.leverage,
                name=name,
            )
        suffix = f"in {time.time() - start_time:.1f} s"
        print_progress(len(self.factors), len(self.factors), prefix, suffix)
        return factor_data

    @property
    @lru_cache()
    def combined_factor(self) -> pd.DataFrame:
        """Single factor obtained by combination."""
        return combine_factors(self.factor_data, self.combination)

    @combined_factor.setter
    def combined_factor(self, value):
        self._combined_factor = value
        type(self).combined_factor_data.fget.cache_clear()

    @property
    @lru_cache()
    def combined_factor_data(self) -> pd.DataFrame:
        """Combined factor values, quantiles, weights and returns."""
        return get_factor_data(
            factor=self.combined_factor,
            price_data=self.price_data,
            periods=self.periods,
            split=self.split,
            long_short=self.long_short,
            leverage=self.leverage,
            name=f"{self.__class__.__name__}_combined",
        )

    @property
    @lru_cache()
    def summaries(self) -> List[pd.Series]:
        """Past performance summaries of individual factors."""
        summaries = []
        prefix = "Preparing performance:"
        start_time = time.time()
        for i, factor in enumerate(self.factor_data):
            print_progress(i, len(self.factor_data), prefix, f"`{factor}`")
            summaries.append(self.performance(factor, False))
        suffix = f"in {time.time() - start_time:.1f} s"
        print_progress(len(self.factor_data), len(self.factor_data), prefix, suffix)
        return summaries

    def __repr__(self):
        return f"{__class__.__name__}.{self.__class__.__name__}" f"{list(self.factors)}"

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, name: str):
        return self.factors[name]

    def __setitem__(self, name: str, function: Factor):
        self.factors[name] = function
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    def __delitem__(self, name: str):
        del self.factors[name]
        type(self).factor_data.fget.cache_clear()
        type(self).combined_factor.fget.cache_clear()
        type(self).combined_factor_data.fget.cache_clear()
        type(self).summaries.fget.cache_clear()

    def __iter__(self):
        return iter(self.factors.items())

    def performance(
        self, factor: Optional[str] = None, output: bool = True
    ) -> pd.Series:
        """Return and show past performance summary."""
        if factor:
            log, summary = get_performance(self.factor_data[factor], output)
            if output:
                self.logger.info(log)
            return summary
        log, summary = get_performance(self.combined_factor_data, output)
        if output:
            self.logger.info(log)
        return summary

    def select(
        self,
        rules: str,
        swap_to: Optional[str] = None,
        inplace: bool = False,
    ) -> None:
        """Select factors that meet the given rules and optionally swap signs.

        Available metrics:
        - ic:       Information Coefficient based on Spearman Rank Correlation
        - autocorr: Factor Rank Autocorrelation
        - sharpe:   Annualized Sharpe Ratio
        - beta:     Annualized Beta as exposure to trading universe
        - alpha:    Annualized Alpha as excess returns over trading universe
        - win:      Win Rate
        - rr:       Risk / Reward Ratio
        - profit:   Profit Factor = (sum of earnings) / (sum of losses)
        - cagr:     Compound Annual Growth Rate
        """
        select = pd.concat(self.summaries, axis=1).T.query(rules)
        signs = pd.Series(1, select.index) if not swap_to else select[swap_to]
        signs[signs > 0], signs[signs < 0] = 1, -1
        if signs.empty:
            self.logger.error(f"No factor satisfies the rules `{rules}`.")
        else:
            self.logger.info(
                f"Factors with signs that meet the rules `{rules}`:\n"
                f"\n"
                f"{signs.to_string()}\n"
            )
            if inplace:
                factors = {}
                for name, sign in signs.items():
                    if sign == 1:
                        factors[name] = self.factors[name]
                    elif sign == -1:
                        factors["-" + name] = swap_sign(self.factors[name])
                self.factors = factors
                type(self).factor_data.fget.cache_clear()
                type(self).combined_factor.fget.cache_clear()
                type(self).combined_factor_data.fget.cache_clear()
                type(self).summaries.fget.cache_clear()

    def rebalance(
        self,
        accountID: str = os.getenv("OANDA_ACCOUNT_ID"),
        live: bool = False,
        keep_current_trades: bool = False,
    ) -> None:
        """Generate and place orders to rebalance portfolio."""
        weights = self.combined_factor_data["weights"]
        positions = weights.loc[weights.index.get_level_values("date")[-1]]
        positions.sort_values(inplace=True)
        hostname, headers = _get_hostname_headers()
        endpoint = f"/v3/accounts/{accountID}"
        url = hostname + endpoint
        req = ur.Request(url, headers=headers)
        with ur.urlopen(req) as r:
            account = pd.read_json(r.read())["account"].apply(
                pd.to_numeric, errors="ignore"
            )
            trades = json_normalize(account.trades).apply(
                pd.to_numeric, errors="ignore"
            )
        orders = pd.Series(name=self.__class__.__name__, dtype=int)
        for instrument in self.instruments:
            if instrument not in FOREX:
                raise Exception("Currently only FOREX pairs are available for trading.")
            base, quote = instrument.split("_")
            params = {"instruments": instrument}
            url = hostname + endpoint + "/pricing?" + urlencode(params)
            req = ur.Request(url, headers=headers)
            with ur.urlopen(req) as r:
                pricing = (
                    json_normalize(json.loads(r.read()), "prices")
                    .apply(pd.to_numeric, errors="ignore")
                    .squeeze()
                )
            base_home_conversion_factor = (
                (pricing.closeoutAsk + pricing.closeoutBid) / 2
            ) * (
                (
                    pricing["quoteHomeConversionFactors.positiveUnits"]
                    + pricing["quoteHomeConversionFactors.negativeUnits"]
                )
                / 2
            )
            nav = account.NAV / base_home_conversion_factor
            for asset in positions.loc[positions != 0].index:
                if asset in base or asset == instrument:
                    orders[instrument] = int(
                        round(positions[asset] * nav * self.leverage, -1)
                    )
                elif asset in quote:
                    orders[instrument] = int(
                        round(-positions[asset] * nav * self.leverage, -1)
                    )
        for index, trade in trades.iterrows():
            if keep_current_trades and trade.instrument in orders.index:
                orders[trade.instrument] = orders[trade.instrument] - trade.currentUnits
            elif live:
                url = hostname + endpoint + f"/trades/{trade.id}/close"
                req = ur.Request(url, headers=headers, method="PUT")
                with ur.urlopen(req) as r:
                    pass
        orders = orders.loc[orders != 0]
        if live:
            url = hostname + endpoint + "/orders"
            for instrument, units in orders.items():
                params = {
                    "order": {
                        "units": units,
                        "instrument": instrument,
                        "timeInForce": "FOK",
                        "type": "MARKET",
                        "positionFill": "DEFAULT",
                    }
                }
                data = json.dumps(params).encode("ascii")
                req = ur.Request(url, data=data, headers=headers)
                with ur.urlopen(req) as r:
                    pass
        self.logger.info(
            f"Portfolio from `{weights.index.levels[0][-1]}`:\n"
            f"\n"
            f"{positions.apply('{0:.1%}'.format).to_string(header=False)}\n"
            f"\n"
            f"- Account NAV: {account.NAV:.2f} {account.currency}\n"
            f"- Position Value: {account.positionValue:.2f}\n"
            f"- {'Submitted Orders' if live else 'Needed Orders'}:\n"
            f"\n"
            f"{orders.to_string() if orders.any() else 'None'}\n"
        )


MAJORS = [
    "AUD_JPY",
    "AUD_USD",
    "EUR_AUD",
    "EUR_CHF",
    "EUR_GBP",
    "EUR_JPY",
    "EUR_USD",
    "GBP_CHF",
    "GBP_JPY",
    "GBP_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
]

EXOTICS = [
    "AUD_CAD",
    "AUD_CHF",
    "AUD_HKD",
    "AUD_NZD",
    "AUD_SGD",
    "CAD_CHF",
    "CAD_HKD",
    "CAD_JPY",
    "CAD_SGD",
    "CHF_HKD",
    "CHF_JPY",
    "CHF_ZAR",
    "EUR_CAD",
    "EUR_CZK",
    "EUR_DKK",
    "EUR_HKD",
    "EUR_HUF",
    "EUR_NOK",
    "EUR_NZD",
    "EUR_PLN",
    "EUR_SEK",
    "EUR_SGD",
    "EUR_TRY",
    "EUR_ZAR",
    "GBP_AUD",
    "GBP_CAD",
    "GBP_HKD",
    "GBP_NZD",
    "GBP_PLN",
    "GBP_SGD",
    "GBP_ZAR",
    "HKD_JPY",
    "NZD_CAD",
    "NZD_CHF",
    "NZD_HKD",
    "NZD_JPY",
    "NZD_SGD",
    "SGD_CHF",
    "SGD_HKD",
    "SGD_JPY",
    "TRY_JPY",
    "USD_CNH",
    "USD_CZK",
    "USD_DKK",
    "USD_HKD",
    "USD_HUF",
    "USD_INR",
    "USD_MXN",
    "USD_NOK",
    "USD_PLN",
    "USD_SAR",
    "USD_SEK",
    "USD_SGD",
    "USD_THB",
    "USD_TRY",
    "USD_ZAR",
    "ZAR_JPY",
]

FOREX = MAJORS + EXOTICS

INDICES = [
    "AU200_AUD",  # Australia 200
    "CN50_USD",  # China A50
    "EU50_EUR",  # Europe 50
    "FR40_EUR",  # France 40
    "DE30_EUR",  # Germany 30
    "HK33_HKD",  # Hong Kong 33
    "IN50_USD",  # India 50
    "JP225_USD",  # Japan 225
    "NL25_EUR",  # Netherlands 25
    "SG30_SGD",  # Singapore 30
    "TWIX_USD",  # Taiwan Index
    "UK100_GBP",  # UK 100
    "NAS100_USD",  # US Nas 100
    "US2000_USD",  # US Russ 2000
    "SPX500_USD",  # US SPX 500
    "US30_USD",  # US Wall St 30
]

COMMODITIES = [
    "BCO_USD",  # Brent Crude Oil
    "XCU_USD",  # Copper
    "CORN_USD",  # Corn
    "NATGAS_USD",  # Natural Gas
    "SOYBN_USD",  # Soybeans
    "SUGAR_USD",  # Sugar
    "WTICO_USD",  # West Texas Oil
    "WHEAT_USD",  # Wheat
]

METALS = [
    "XAU_USD",  # Gold
    "XAU_AUD",  # Gold_AUD
    "XAU_CAD",  # Gold_CAD
    "XAU_CHF",  # Gold_CHF
    "XAU_EUR",  # Gold_EUR
    "XAU_GBP",  # Gold_GBP
    "XAU_HKD",  # Gold_HKD
    "XAU_JPY",  # Gold_JPY
    "XAU_NZD",  # Gold_NZD
    "XAU_SGD",  # Gold_SGD
    "XAU_XAG",  # Gold_Silver
    "XPD_USD",  # Palladium
    "XPT_USD",  # Platinum
    "XAG_USD",  # Silver
    "XAG_AUD",  # Silver_AUD
    "XAG_CAD",  # Silver_CAD
    "XAG_CHF",  # Silver_CHF
    "XAG_EUR",  # Silver_EUR
    "XAG_GBP",  # Silver_GBP
    "XAG_HKD",  # Silver_HKD
    "XAG_JPY",  # Silver_JPY
    "XAG_NZD",  # Silver_NZD
    "XAG_SGD",  # Silver_SGD
]

BONDS = [
    "DE10YB_EUR",  # Bund
    "UK10YB_GBP",  # UK 10Y Gilt
    "USB10Y_USD",  # US 10Y T-Note
    "USB02Y_USD",  # US 2Y T-Note
    "USB05Y_USD",  # US 5Y T-Note
    "USB30Y_USD",  # US T-Bond
]

ALL_SYMBOLS = FOREX + INDICES + COMMODITIES + METALS + BONDS

G10_USD = [
    "AUD_USD",
    "EUR_USD",
    "GBP_USD",
    "NZD_USD",
    "USD_CAD",
    "USD_CHF",
    "USD_JPY",
    "USD_NOK",
    "USD_SEK",
]

EM_USD = [
    "USD_CNH",
    "USD_CZK",
    "USD_HUF",
    "USD_INR",
    "USD_MXN",
    "USD_PLN",
    "USD_THB",
    "USD_TRY",
    "USD_ZAR",
]

ALL_USD = G10_USD + EM_USD + ["USD_DKK", "USD_HKD", "USD_SAR", "USD_SGD"]
