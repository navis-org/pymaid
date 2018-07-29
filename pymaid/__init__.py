__version__ = "0.88"

from pymaid import config

logger = config.logger

# Flatten namespace by importing contents of all modules of pymaid

try:
    from pymaid.fetch import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.fetch:\n' + str(error))

try:
    from pymaid.cluster import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.cluster:\n' + str(error))

try:
    from pymaid.morpho import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.morpho:\n' + str(error))

try:
    from pymaid.plotting import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.plotting:\n' + str(error))

try:
    from pymaid.scene3d import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.scene3d:\n' + str(error))

try:
    from pymaid.user_stats import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.user_stats:\n' + str(error))

try:
    from pymaid.core import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.core:\n' + str(error))

try:
    from pymaid.graph import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph:\n' + str(error))

try:
    from pymaid.graph_utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.graph_utils:\n' + str(error))

try:
    from pymaid.resample import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.resample:\n' + str(error))

try:
    from pymaid.intersect import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.intersect:\n' + str(error))

try:
    from pymaid.connectivity import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.connectivity:\n' + str(error))

try:
    from pymaid.utils import *
except Exception as error:
    logger.warning(str(error))
    logger.warning('Error importing pymaid.utils:\n' + str(error))
