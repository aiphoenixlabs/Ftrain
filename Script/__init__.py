import logging
from .api import train, merge, test

__version__ = "1.0.0"
__author__ = "FTRAIN Engine Team"
__all__ = ["train", "merge", "test", "__version__"]
logging.getLogger(__name__).addHandler(logging.NullHandler())
