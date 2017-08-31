__version__ = "0.61"


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
	module_logger.exception('Error importing pymaid.pymaid: ', error)	

try:
	from pymaid.cluster import *
except Exception as error:
	module_logger.exception('Error importing pymaid.cluster: ', error)	

try:
	from pymaid.morpho import *
except Exception as error:
	module_logger.exception('Error importing pymaid.morpho: ', error)	

try:
	from pymaid.plotting import *
except Exception as error:
	module_logger.exception('Error importing pymaid.plotting: ', error)	

try:
	from pymaid.user_stats import *
except Exception as error:
	module_logger.exception('Error importing pymaid.user_stats: ', error)	

try:
	from pymaid.core import *
except Exception as error:
	module_logger.exception('Error importing pymaid.core: ', error)	

try:
	from pymaid.igraph_catmaid import *
except Exception as error:
	module_logger.exception('Error importing pymaid.igraph_catmaid: ', error)	

try:
	from pymaid.rmaid import *
except Exception as error:
	module_logger.exception('Error importing pymaid.rmaid: ', error)	
