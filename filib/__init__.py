"""Factor Investing Library"""

__all__ = ['helpers', 'oanda', 'utils']
__author__ = 'Paweł Kowalski'

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
     # package is not installed
    pass
