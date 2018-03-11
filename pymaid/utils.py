#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
#    Copyright (C) 2017 Philipp Schlegel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

import collections
import six
import numpy as np

#from pymaid import morpho, core, plotting, graph, graph_utils, core
import pymaid

def _type_of_script():
    """ Returns context in which pymaid is run. """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

def is_jupyter():
    """ Test if pymaid is run in a Jupyter notebook."""
    return _type_of_script() == 'jupyter'

def set_loggers(level='INFO'):
    """Helper function to set levels for all associated module loggers."""
    pymaid.morpho.module_logger.setLevel(level)
    pymaid.graph.module_logger.setLevel(level)
    pymaid.plotting.module_logger.setLevel(level)
    pymaid.graph_utils.module_logger.setLevel(level)
    pymaid.core.module_logger.setLevel(level)
    pymaid.fetch.module_logger.setLevel(level)
    pymaid.resample.module_logger.setLevel(level)
    pymaid.intersect.module_logger.setLevel(level)
    pymaid.cluster.module_logger.setLevel(level)
    pymaid.user_stats.module_logger.setLevel(level)
    pymaid.connectivity.module_logger.setLevel(level)

def _make_iterable(x, force_type=None):
    """ Helper function. Turns x into a np.ndarray, if it isn't already. For
    dicts, keys will be turned into array.
    """
    if not isinstance(x, collections.Iterable) or isinstance(x, six.string_types):
        x = [ x ]

    if isinstance(x, dict):
        x = list(x)

    if force_type:
        return np.array( x ).astype(force_type)
    else:
        return np.array( x )

def _is_iterable(x):
    """ Helper function. Returns True if x is an iterable but not str or
    dictionary.
    """
    if isinstance(x, collections.Iterable) and not isinstance( x, six.string_types ):
        return True
    else:
        return False

def _eval_conditions(x):
    """ Splits list of strings into positive (no ~) and negative (~) conditions
    """

    x = _make_iterable(x, force_type=str)

    return [ i for i in x if not i.startswith('~') ], [ i[1:] for i in x if i.startswith('~') ]