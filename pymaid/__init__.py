__version__ = "0.103"

from . import config

logger = config.logger

# Flatten namespace by importing contents of all modules of pymaid
try:
    from .fetch import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.fetch:\n' + str(error))

# Flatten namespace by importing contents of all modules of pymaid
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
    from .morpho import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.morpho:\n' + str(error))

try:
    from .plotting import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.plotting:\n' + str(error))

# This needs to be AFTER plotting b/c in plotting vispy is imported first
# and we set the backend!
try:
    from .scene3d import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.scene3d:\n' + str(error))

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
    from .graph import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph:\n' + str(error))

try:
    from .graph_utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph_utils:\n' + str(error))

try:
    from .resample import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.resample:\n' + str(error))

try:
    from .intersect import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.intersect:\n' + str(error))

try:
    from .connectivity import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.connectivity:\n' + str(error))

try:
    from .utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.utils:\n' + str(error))
