
from __future__ import annotations

import logging
from typing import Final

from .api import merge, test, train

__version__: Final[str] = "1.1.0"
__author__: Final[str] = "FTRAIN Engine Team"

# Public package surface.
__all__: Final[tuple[str, ...]] = (
    "train",
    "merge",
    "test",
    "__version__",
    "__author__",
)

_logger = logging.getLogger(__name__)

if not any(isinstance(handler, logging.NullHandler) for handler in _logger.handlers):
    _logger.addHandler(logging.NullHandler())
_logger.propagate = True
