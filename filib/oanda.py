"""Module used for backtesting and trading via OANDA v20 REST API."""

__all__ = [
    'find_instruments', 'get_price_data', 'Oanda', 'MAJORS', 'EXOTICS',
    'FOREX', 'INDICES', 'COMMODITIES', 'METALS', 'BONDS', 'ALL_SYMBOLS',
    'G10_USD', 'EM_USD', 'ALL_USD'
]

import configparser
from datetime import datetime
from inspect import getmembers, isfunction
import logging
import os
import pickle
import time
import warnings

import pandas as pd
import v20

from .utils import get_factor_data, combine_factors, get_performance


def _get_api():
    config = configparser.ConfigParser()
    config_filepath = os.path.join(os.path.dirname(__file__), 'config.ini')
    try:
        with open(config_filepath, 'r') as config_file:
            config.read_file(config_file)
            hostname = config.get('oanda', 'hostname')
            token = config.get('oanda', 'token')
    except FileNotFoundError:
        logger = logging.getLogger(__name__)
        logger.error(
            f'OANDA v20 REST API config file is not found. '
            f'Please answer to generate it:')
        account_type = input(
            '- What is your account type? `Live` or `Practice`?\n')
        if account_type.lower() in ['live', 'l']:
            hostname = 'api-fxtrade.oanda.com'
        elif account_type.lower() in ['practice', 'p']:
            hostname = 'api-fxpractice.oanda.com'
        else:
            raise ValueError(f'Account type `{account_type}` not available.')
        token = input('- Provide your personal access token:\n')
        config['oanda'] = {'hostname': hostname, 'token': token}
        with open(config_filepath, 'w') as config_file:
            config.write(config_file)
    api = v20.Context(hostname=hostname, token=token)
    return api


def find_instruments(symbol, universe):
    instruments = []
    for instrument in universe:
        base, quote = instrument.split('_')
        if symbol in (base, quote):
            instruments.append(instrument)
    return instruments


def get_price_data(instruments, price='M', granularity='D', count=500,
                   end=datetime.utcnow().timestamp(), symbol=None,
                   save=False, dailyAlignment=0, alignmentTimezone='UTC',
                   weeklyAlignment='Monday'):
    freq = {
        'S5': '5S',  # 5 second candlesticks, minute alignment
        'S10': '10S',  # 10 second candlesticks, minute alignment
        'S15': '15S',  # 15 second candlesticks, minute alignment
        'S30': '30S',  # 30 second candlesticks, minute alignment
        'M1': 'T',  # 1 minute candlesticks, minute alignment
        'M2': '2T',  # 2 minute candlesticks, hour alignment
        'M4': '4T',  # 4 minute candlesticks, hour alignment
        'M5': '5T',  # 5 minute candlesticks, hour alignment
        'M10': '10T',  # 10 minute candlesticks, hour alignment
        'M15': '15T',  # 15 minute candlesticks, hour alignment
        'M30': '30T',  # 30 minute candlesticks, hour alignment
        'H1': 'H',  # 1 hour candlesticks, hour alignment
        'H2': '2H',  # 2 hour candlesticks, day alignment
        'H3': '3H',  # 3 hour candlesticks, day alignment
        'H4': '4H',  # 4 hour candlesticks, day alignment
        'H6': '6H',  # 6 hour candlesticks, day alignment
        'H8': '8H',  # 8 hour candlesticks, day alignment
        'H12': '12H',  # 12 hour candlesticks, day alignment
        'D': 'B',  # 1 day candlesticks, day alignment
        'W': 'W-MON',  # 1 week candlesticks, aligned to start of week
    }
    granularity = granularity.upper()
    if granularity not in freq:
        raise ValueError(f'Granularity `{granularity}` not available - '
                         f'choose from {list(freq.keys())}.')
    h = str(hash(f'{instruments} {price} {granularity} {count} {symbol}'))
    try:
        with open(h + '.pickle', 'rb') as f:
            price_data = pickle.load(f)
    except FileNotFoundError:
        count_list = [5000] * (count // 5000)
        if count % 5000 != 0:
            count_list.append(count % 5000)
        objs = []
        for instrument in instruments:
            if instrument not in ALL_SYMBOLS:
                raise ValueError(f'Instrument `{instrument}` not available.')
            toTime = end
            responses = []
            for c in count_list:
                api = _get_api()
                r = api.instrument.candles(instrument, price=price,
                                           granularity=granularity,
                                           count=c, toTime=toTime,
                                           dailyAlignment=dailyAlignment,
                                           alignmentTimezone=alignmentTimezone,
                                           weeklyAlignment=weeklyAlignment)
                responses.append(r)
                toTime = r.get('candles')[0].time
                time.sleep(0.1)
            for r in responses:
                price_o, price_h, price_l, price_c, volume = {}, {}, {}, {}, {}
                for candle in r.get('candles'):
                    if price == 'M':
                        price_o[candle.time] = candle.mid.o
                        price_h[candle.time] = candle.mid.h
                        price_l[candle.time] = candle.mid.l
                        price_c[candle.time] = candle.mid.c
                        volume[candle.time] = candle.volume
                    elif price == 'A':
                        price_o[candle.time] = candle.ask.o
                        price_h[candle.time] = candle.ask.h
                        price_l[candle.time] = candle.ask.l
                        price_c[candle.time] = candle.ask.c
                        volume[candle.time] = candle.volume
                    elif price == 'B':
                        price_o[candle.time] = candle.bid.o
                        price_h[candle.time] = candle.bid.h
                        price_l[candle.time] = candle.bid.l
                        price_c[candle.time] = candle.bid.c
                        volume[candle.time] = candle.volume
                df = pd.DataFrame([price_o, price_h, price_l, price_c, volume],
                                  dtype=float).T
                df.index = pd.to_datetime(df.index, utc=True)
                df = df.resample(freq[granularity]).agg(
                    {0: 'first', 1: 'max', 2: 'min', 3: 'last', 4: 'sum'})
                df.columns = pd.MultiIndex.from_product(
                    [[r.get('instrument')],
                     ['open', 'high', 'low', 'close', 'volume']])
                objs.append(df)
        price_data = pd.concat(objs).sort_index().groupby(level=0).first()
        price_data = _arrange_price_data(price_data, symbol)
        price_data = price_data.ffill().dropna()
        price_data.index.freq = price_data.index.inferred_freq
        if save:
            with open(h + '.pickle', 'wb') as f:
                pickle.dump(price_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return price_data


def _arrange_price_data(price_data, symbol):
    arranged = pd.DataFrame()
    for instrument, price in price_data:
        base, quote = instrument.split('_')
        if base == symbol:
            if price in ['open', 'high', 'low', 'close']:
                arranged[(quote, price)] = price_data[(instrument, price)]**-1
            else:
                arranged[(quote, price)] = price_data[(instrument, price)]
        elif quote == symbol:
            arranged[(base, price)] = price_data[(instrument, price)]
        else:
            arranged[(instrument, price)] = price_data[(instrument, price)]
    arranged.columns = pd.MultiIndex.from_tuples(arranged.columns)
    return arranged


class Oanda:
    def __init__(self, instruments, granularity='D', count=500, symbol=None,
                 save=False, periods=None, split=3, accountID=None,
                 leverage=1, long_short=False, combination='sum_of_weights'):
        self.name = str(self.__class__).split('.')[-1].split("'")[0]
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(f'{self.name}.log')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        self.api = _get_api()
        self.instruments = instruments
        self.periods = periods
        self.split = split
        self.accountID = accountID
        self.leverage = leverage
        self.long_short = long_short
        self.combination = combination
        self.price_data = get_price_data(instruments, granularity=granularity,
                                         count=count, symbol=symbol, save=save)
        self.open = self.price_data.xs('open', axis=1, level=1)
        self.high = self.price_data.xs('high', axis=1, level=1)
        self.low = self.price_data.xs('low', axis=1, level=1)
        self.close = self.price_data.xs('close', axis=1, level=1)
        self.volume = self.price_data.xs('volume', axis=1, level=1)
        self.returns = self.close.pct_change()[1:]
        self.factor_data = {}
        for name, function in getmembers(self.__class__, predicate=isfunction):
            if self.name in str(function):
                start_time = time.time()
                try:
                    factor, s = function(self)
                except ValueError:
                    factor, s = function(self), 3
                except TypeError:
                    raise TypeError(f'`{name}` must return atleast factor.')
                self.factor_data[name] = get_factor_data(
                    factor, self.price_data, periods, s, leverage, long_short,
                    name)
                elapsed = time.time() - start_time
                self.logger.info(
                    f'Factor `{name}` initialized in {elapsed:.1f} s.')
        self.factors = list(self.factor_data.keys())
        self.combined_factor = combine_factors(self.factor_data, combination)
        self.combined_factor_data = get_factor_data(
            self.combined_factor, self.price_data, periods, split, leverage,
            long_short, f'{self.name}_combined')
        self.pd = self.price_data
        self.o = self.open
        self.h = self.high
        self.l = self.low
        self.c = self.close
        self.v = self.volume
        self.fd = self.factor_data
        self.cfd = self.combined_factor_data

    def performance(self, factor=None):
        warnings.filterwarnings('ignore', category=UserWarning)
        if factor:
            log, summary = get_performance(self.factor_data[factor])
            self.logger.info(log)
            return summary
        log, summary = get_performance(self.combined_factor_data)
        self.logger.info(log)
        return summary

    def select(self, rules, swap=None):
        summaries = [self.performance(factor) for factor in self.factors]
        select = pd.concat(summaries, axis=1).T.query(rules)
        sign = pd.Series(1, select.index) if not swap else select[swap]
        sign[sign > 0], sign[sign < 0] = 1, -1
        select_factor_data = {
            f'{int(sign[name])}{name}'.replace('1', '', 1):
            sign[name] * factor_data.loc[:, 'factor':]
            for name, factor_data in self.factor_data.items()
            if name in select.index}
        if not select_factor_data:
            self.logger.info(f'No factor satisfies the rules `{rules}`.')
            return None
        self.logger.info(f"Factors with signs that meet the rules `{rules}`:\n"
                         f"\n"
                         f"{sign.to_string()}\n")
        combined_factor = combine_factors(select_factor_data, self.combination)
        combined_factor_data = get_factor_data(
            combined_factor, self.price_data, self.periods, self.split,
            self.leverage, self.long_short, f'{self.name}_selected')
        log, summary = get_performance(combined_factor_data)
        self.logger.info(log)
        return sign.rename('selection'), summary

    def rebalance(self, live=False, keep_current_trades=False):
        weights = self.combined_factor_data['weights']
        positions = weights.loc[weights.index.get_level_values('date')[-1]]
        positions.sort_values(inplace=True)
        account = self.api.account.get(self.accountID).get('account')
        orders = pd.Series(name=self.name)
        for asset in positions.loc[positions != 0].index:
            for instrument in self.instruments:
                base, quote = instrument.split('_')
                pricing = self.api.pricing.get(
                    self.accountID, instruments=instrument).get('prices')[-1]
                base_home_conversion_factor = (
                    ((pricing.closeoutAsk + pricing.closeoutBid) / 2) *
                    ((pricing.quoteHomeConversionFactors.positiveUnits +
                      pricing.quoteHomeConversionFactors.negativeUnits) / 2))
                nav = account.NAV / base_home_conversion_factor
                if asset in base or asset == instrument:
                    orders[instrument] = int(round(
                        positions[asset] * nav * self.leverage, -1))
                elif asset in quote:
                    orders[instrument] = int(round(
                        -positions[asset] * nav * self.leverage, -1))
        for trade in self.api.trade.list_open(self.accountID).get('trades'):
            if keep_current_trades and trade.instrument in orders.index:
                orders[trade.instrument] = (
                    orders[trade.instrument] - trade.currentUnits)
            elif live:
                self.api.trade.close(self.accountID, trade.id)
        orders = orders.loc[orders != 0]
        if live:
            for instrument, units in orders.items():
                self.api.order.market(
                    self.accountID, instrument=instrument, units=units)
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
    'AUD_JPY', 'AUD_USD', 'EUR_AUD', 'EUR_CHF', 'EUR_GBP', 'EUR_JPY',
    'EUR_USD', 'GBP_CHF', 'GBP_JPY', 'GBP_USD', 'NZD_USD', 'USD_CAD',
    'USD_CHF', 'USD_JPY'
]

EXOTICS = [
    'AUD_CAD', 'AUD_CHF', 'AUD_HKD', 'AUD_NZD', 'AUD_SGD', 'CAD_CHF',
    'CAD_HKD', 'CAD_JPY', 'CAD_SGD', 'CHF_HKD', 'CHF_JPY', 'CHF_ZAR',
    'EUR_CAD', 'EUR_CZK', 'EUR_DKK', 'EUR_HKD', 'EUR_HUF', 'EUR_NOK',
    'EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_SGD', 'EUR_TRY', 'EUR_ZAR',
    'GBP_AUD', 'GBP_CAD', 'GBP_HKD', 'GBP_NZD', 'GBP_PLN', 'GBP_SGD',
    'GBP_ZAR', 'HKD_JPY', 'NZD_CAD', 'NZD_CHF', 'NZD_HKD', 'NZD_JPY',
    'NZD_SGD', 'SGD_CHF', 'SGD_HKD', 'SGD_JPY', 'TRY_JPY', 'USD_CNH',
    'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN',
    'USD_NOK', 'USD_PLN', 'USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB',
    'USD_TRY', 'USD_ZAR', 'ZAR_JPY'
]

FOREX = MAJORS + EXOTICS

INDICES = [
    'AU200_AUD',  # Australia 200
    'CN50_USD',  # China A50
    'EU50_EUR',  # Europe 50
    'FR40_EUR',  # France 40
    'DE30_EUR',  # Germany 30
    'HK33_HKD',  # Hong Kong 33
    'IN50_USD',  # India 50
    'JP225_USD',  # Japan 225
    'NL25_EUR',  # Netherlands 25
    'SG30_SGD',  # Singapore 30
    'TWIX_USD',  # Taiwan Index
    'UK100_GBP',  # UK 100
    'NAS100_USD',  # US Nas 100
    'US2000_USD',  # US Russ 2000
    'SPX500_USD',  # US SPX 500
    'US30_USD',  # US Wall St 30
]

COMMODITIES = [
    'BCO_USD',  # Brent Crude Oil
    'XCU_USD',  # Copper
    'CORN_USD',  # Corn
    'NATGAS_USD',  # Natural Gas
    'SOYBN_USD',  # Soybeans
    'SUGAR_USD',  # Sugar
    'WTICO_USD',  # West Texas Oil
    'WHEAT_USD',  # Wheat
]

METALS = [
    'XAU_USD',  # Gold
    'XAU_AUD',  # Gold_AUD
    'XAU_CAD',  # Gold_CAD
    'XAU_CHF',  # Gold_CHF
    'XAU_EUR',  # Gold_EUR
    'XAU_GBP',  # Gold_GBP
    'XAU_HKD',  # Gold_HKD
    'XAU_JPY',  # Gold_JPY
    'XAU_NZD',  # Gold_NZD
    'XAU_SGD',  # Gold_SGD
    'XAU_XAG',  # Gold_Silver
    'XPD_USD',  # Palladium
    'XPT_USD',  # Platinum
    'XAG_USD',  # Silver
    'XAG_AUD',  # Silver_AUD
    'XAG_CAD',  # Silver_CAD
    'XAG_CHF',  # Silver_CHF
    'XAG_EUR',  # Silver_EUR
    'XAG_GBP',  # Silver_GBP
    'XAG_HKD',  # Silver_HKD
    'XAG_JPY',  # Silver_JPY
    'XAG_NZD',  # Silver_NZD
    'XAG_SGD',  # Silver_SGD
]

BONDS = [
    'DE10YB_EUR',  # Bund
    'UK10YB_GBP',  # UK 10Y Gilt
    'USB10Y_USD',  # US 10Y T-Note
    'USB02Y_USD',  # US 2Y T-Note
    'USB05Y_USD',  # US 5Y T-Note
    'USB30Y_USD',  # US T-Bond
]

ALL_SYMBOLS = FOREX + INDICES + COMMODITIES + METALS + BONDS

G10_USD = [
    'AUD_USD', 'EUR_USD', 'GBP_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF',
    'USD_JPY', 'USD_NOK', 'USD_SEK'
]

EM_USD = [
    'USD_CNH', 'USD_CZK', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_PLN',
    'USD_THB', 'USD_TRY', 'USD_ZAR'
]

ALL_USD = G10_USD + EM_USD + ['USD_DKK', 'USD_HKD', 'USD_SAR', 'USD_SGD']
