"""Factor Investing Library"""

__all__ = ["helpers", "oanda", "utils"]
__author__ = "Paweł Kowalski"

from contextlib import suppress

from pkg_resources import get_distribution, DistributionNotFound

with suppress(DistributionNotFound):
    __version__ = get_distribution(__name__).version
