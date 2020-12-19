__version__ = "2.0.1"

from . import config

logger = config.logger

# Flatten namespace by importing contents of all modules of pymaid
try:
    from .fetch import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.fetch:\n' + str(error))

try:
    from .client import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.client:\n' + str(error))

try:
    from .upload import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.upload:\n' + str(error))

try:
    from .cluster import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.cluster:\n' + str(error))

try:
    from .user_stats import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.user_stats:\n' + str(error))

try:
    from .core import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.core:\n' + str(error))

try:
    from .utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.utils:\n' + str(error))

try:
    from .morpho import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.morpho:\n' + str(error))

try:
    from .connectivity import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.connectivity:\n' + str(error))

"""
try:
    from .snapshot import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.snapshot:\n' + str(error))
"""
