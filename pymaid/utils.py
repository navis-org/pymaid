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
import json
import pandas as pd

#from pymaid import morpho, core, plotting, graph, graph_utils, core
import pymaid

__all__ = ['neuron2json', 'json2neuron', 'set_loggers', 'set_pbars']

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

def set_pbars(hide=None, leave=None):
    """ Set global progress bar behaviours.

    Parameters
    ----------
    hide :      bool, optional
                Set to True to hide all progress bars.
    leave :     bool, optional
                Set to False to clear progress bars after they have finished.

    Returns
    -------
    Nothing

    """

    mods = [ pymaid.morpho, pymaid.graph, pymaid.graph_utils, pymaid.core,
             pymaid.fetch, pymaid.resample, pymaid.intersect, pymaid.cluster,
             pymaid.user_stats, pymaid.connectivity, pymaid.plotting ]

    if isinstance(hide, bool):
        for m in mods:
            m.pbar_hide = hide

    if isinstance(leave, bool):
        for m in mods:
            m.pbar_leave = leave


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

def _make_non_iterable(x):
    """ Helper function. Turns x into non-iterable, if it isn't already. Will
    raise error if len(x) > 1.
    """
    if not _is_iterable(x):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        raise ValueError('Iterable must not contain more than one entry.')

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

def neuron2json(x, **kwargs):
    """ Generate JSON formatted ``str`` respresentation of CatmaidNeuron/List.

    Notes
    -----
    Nodes and connectors are serialised using pandas' ``to_json()``. Most 
    other items in the neuron's __dict__ are serialised using 
    ``json.dumps()``. Properties not serialised: `._remote_instance`,
    `.graph`, `.igraph`. 

    Important
    ---------
    For safety, the :class:`~pymaid.CatmaidInstance` is not serialised as
    this would expose your credentials. Parameters attached to a neuronlist
    are currently not preserved.


    Parameters
    ----------
    x :         {CatmaidNeuron, CatmaidNeuronList}
    **kwargs
                Parameters passed to ``json.dumps()`` and 
                ``pandas.DataFrame.to_json()``.

    Returns
    -------
    str

    See Also
    --------
    :func:`~pymaid.json2neuron`
                Read json back into pymaid neurons.

    """

    if not isinstance(x, (pymaid.CatmaidNeuron, pymaid.CatmaidNeuronList)):
        raise TypeError('Unable to convert data of type "{0}"'.format(type(x)))

    if isinstance(x, pymaid.CatmaidNeuron):
        x = pymaid.CatmaidNeuronList([x])

    data = []
    for n in x:
        this_data = {'skeleton_id':n.skeleton_id}

        if 'nodes' in n.__dict__:
            this_data['nodes'] = n.nodes.to_json()
        
        if 'connectors' in n.__dict__:
            this_data['connectors'] = n.connectors.to_json(**kwargs)

        for k in n.__dict__:
            if k in ['nodes','connectors', 'graph','igraph', '_remote_instance']:
                continue            
            try:
                this_data[k] = n.__dict__[k]
            except:
                module_logger.error('Lost attribute "{0}"'.format(k))

        data.append(this_data)

    return json.dumps( data, **kwargs )

def json2neuron(s, **kwargs):
    """ Load neuron from JSON string.

    Parameters
    ----------
    s :         {str}
                JSON-formatted string.
    **kwargs
                Parameters passed to ``json.loads()`` and 
                ``pandas.DataFrame.read_json()``.

    Returns
    -------
    :class:`~pymaid.CatmaidNeuronList`

    See Also
    --------
    :func:`~pymaid.to_json`
                Turn neuron into json.

    """

    if not isinstance(s, str):
        raise TypeError('Need str, got "{0}"'.format(type(s)))

    data = json.loads(s, **kwargs)

    nl = pymaid.CatmaidNeuronList( [] )

    for n in data:
        # Make sure we have all we need
        REQUIRED = ['skeleton_id']

        missing = [ p for p in REQUIRED if p not in n ]

        if missing:
            raise ValueError('Missing data: {0}'.format(','.join(missing)))

        cn = pymaid.CatmaidNeuron( int( n['skeleton_id'] ) )

        if 'nodes' in n:
            cn.nodes = pd.read_json( n['nodes'] )
            cn.connectors = pd.read_json( n['connectors'] )

        for key in n:
            if key in ['skeleton_id','nodes','connectors']:
                continue
            setattr(cn, key, n[key])

        nl += cn

    return nl