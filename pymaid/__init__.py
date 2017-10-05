__version__ = "0.64"

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
    from pymaid.pymaid import *
except Exception as error:
    module_logger.warning('Error importing pymaid.pymaid:\n' + str(error))

try:
    from pymaid.cluster import *
except Exception as error:
    module_logger.warning('Error importing pymaid.cluster:\n' + str(error))

try:
    from pymaid.morpho import *
except Exception as error:
    module_logger.warning('Error importing pymaid.morpho:\n' + str(error))

try:
    from pymaid.plotting import *
except Exception as error:
    module_logger.warning('Error importing pymaid.plotting:\n' + str(error))

try:
    from pymaid.user_stats import *
except Exception as error:
    module_logger.warning('Error importing pymaid.user_stats:\n' + str(error))

try:
    from pymaid.core import *
except Exception as error:
    module_logger.warning('Error importing pymaid.core:\n' + str(error))

try:
    from pymaid.igraph_catmaid import *
except Exception as error:
    module_logger.warning('Error importing pymaid.igraph_catmaid:\n' + str(error))  

try:
    from pymaid.rmaid import *
except Exception as error:
    module_logger.warning(str(error))
    module_logger.warning('Error importing pymaid.rmaid. This may be due to rpy2 not being installed. No worries, pymaid will still work though!' )
