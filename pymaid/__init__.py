__version__ = "0.74"

import logging

# Set up logging
module_logger = logging.getLogger('pymaid_init')
module_logger.setLevel(logging.INFO)
if len( module_logger.handlers ) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

# Flatten namespace by importing contents of all modules of pymaid

try:
    from pymaid.fetch import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.fetch:\n' + str(error))

try:
    from pymaid.cluster import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.cluster:\n' + str(error))

try:
    from pymaid.morpho import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.morpho:\n' + str(error))

try:
    from pymaid.plotting import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.plotting:\n' + str(error))

try:
    from pymaid.user_stats import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.user_stats:\n' + str(error))

try:
    from pymaid.core import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.core:\n' + str(error))

try:
    from pymaid.graph import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.graph:\n' + str(error))

try:
    from pymaid.graph_utils import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.graph_utils:\n' + str(error))

try:
    from pymaid.resample import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.resample:\n' + str(error))

try:
    from pymaid.intersect import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.intersect:\n' + str(error))

try:
    from pymaid.connectivity import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.connectivity:\n' + str(error))
