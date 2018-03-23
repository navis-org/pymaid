#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
#    Copyright (C) 2017 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along


""" This module contains functions for intersections.
"""

import time
import logging
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import ConvexHull

from pymaid import fetch, core, utils

from tqdm import tqdm
if utils.is_jupyter():
    from tqdm import tqdm_notebook, tnrange
    tqdm = tqdm_notebook
    trange = tnrange

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if len( module_logger.handlers ) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

try:
    from pyoctree import pyoctree
except:
    module_logger.warning("Module pyoctree not found. Falling back to scipy's ConvexHull for intersection calculations.")

__all__ = sorted([ 'in_volume'])

# Default settings for progress bars
pbar_hide = False
pbar_leave = True

def in_volume(x, volume, inplace=False, mode='IN', remote_instance=None):
    """ Test if points are within a given CATMAID volume.

    Important
    ---------
    This function requires `pyoctree <https://github.com/mhogg/pyoctree>`_
    which is only an optional dependency of PyMaid. If pyoctree is not
    installed, we will fall back to using scipy ConvexHull instead of ray
    casting. This is slower and may give wrong positives for concave meshes!

    Parameters
    ----------
    x :               {list of tuples, CatmaidNeuron, CatmaidNeuronList}

                      1. List/np array -  ``[ ( x,y,z ), ( ... ) ]``
                      2. DataFrame - needs to have 'x','y','z' columns

    volume :          {str, list of str, core.Volume}
                      Name of the CATMAID volume to test OR core.Volume dict
                      as returned by e.g. :func:`~pymaid.get_volume()`
    inplace :         bool, optional
                      If False, a copy of the original DataFrames/Neuron is
                      returned. Does only apply to CatmaidNeuron or
                      CatmaidNeuronList objects. Does apply if multiple
                      volumes are provided
    mode :            {'IN','OUT'}, optional
                      If 'IN', parts of the neuron that are within the volume
                      are kept.
    remote_instance : CATMAID instance, optional
                      Pass if volume is a volume name

    Returns
    -------
    CatmaidNeuron
                      If input is CatmaidNeuron or CatmaidNeuronList, will
                      return parts of the neuron (nodes and connectors) that
                      are within the volume
    list of bools
                      If input is list or DataFrame, returns boolean: ``True``
                      if in volume, ``False`` if not
    dict
                      If multiple volumes are provided as list of strings,
                      results will be returned as dict of above returns.

    Examples
    --------
    >>> # Advanced example (assumes you already set up a CATMAID instance)
    >>> # Check with which antennal lobe glomeruli a neuron intersects
    >>> # First get names of glomeruli
    >>> all_volumes = remote_instance.fetch( remote_instance._get_volumes() )
    >>> right_gloms = [ v['name'] for v in all_volumes if v['name'].endswith('glomerulus') ]
    >>> # Neuron to check
    >>> n = pymaid.get_neuron('name:PN unknown glomerulus', remote_instance = remote_instance )
    >>> # Get intersections
    >>> res = pymaid.in_volume( n, right_gloms, remote_instance = remote_instance )
    >>> # Extract cable
    >>> cable = { v : res[v].cable_length for v in res  }
    >>> # Plot graph
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame( list( cable.values() ),
    ...                    index = list( cable.keys() )
    ...                   )
    >>> df.boxplot()
    >>> plt.show()

    """

    remote_instance = fetch._eval_remote_instance(remote_instance)

    if isinstance(volume, (list, dict, np.ndarray)) and not isinstance(volume, core.Volume):
        #Turn into dict
        if not isinstance(volume, dict):
            volume = { v['name'] : v for v in volume }

        data = dict()
        for v in tqdm(volume, desc='Volumes', disable=pbar_hide, leave=pbar_leave):
            data[v] = in_volume(
                x, volume[v], remote_instance=remote_instance, inplace=False, mode=mode)
        return data

    if isinstance(volume, str):
        volume = fetch.get_volume(volume, remote_instance)

    if isinstance(x, pd.DataFrame):
        points = x[['x', 'y', 'z']].as_matrix()

    elif isinstance(x, core.CatmaidNeuron):
        n = x

        if not inplace:
            n = n.copy()

        in_v = in_volume(n.nodes[['x', 'y', 'z']].as_matrix(), volume, mode=mode)

        # If mode is OUT, invert selection
        if mode == 'OUT':
            in_v = ~np.array(in_v)

        n.nodes = n.nodes[ in_v ]
        n.connectors = n.connectors[
            n.connectors.treenode_id.isin(n.nodes.treenode_id.tolist())]

        # Fix root nodes
        n.nodes.loc[~n.nodes.parent_id.isin(
            n.nodes.treenode_id.tolist() + [None]), 'parent_id'] = None

        # Reset indices of node and connector tables (important for igraph!)
        n.nodes.reset_index(inplace=True,drop=True)
        n.connectors.reset_index(inplace=True,drop=True)

        # Theoretically we can end up with disconnected pieces, i.e. with more than 1 root node
        # We have to fix the nodes that lost their parents
        n.nodes.loc[ ~n.nodes.parent_id.isin( n.nodes.treenode_id.tolist() ),
                          'parent_id' ] = None

        n._clear_temp_attr()

        if not inplace:
            return n
        else:
            return

    elif isinstance(x, core.CatmaidNeuronList):
        nl = x

        if not inplace:
            nl = nl.copy()

        for n in nl:
            n = in_volume(n, volume, inplace=True, mode=mode)

        if not inplace:
            return nl
        else:
            return
    else:
        points = x

    try:
        return _in_volume_ray( points, volume )
    except:
        module_logger.warning('Package pyoctree not found. Falling back to ConvexHull.')
        return _in_volume_convex( points, volume, approximate=False )

def _in_volume_ray(points, volume):
    """ Uses pyoctree's raycsasting to test if points are within a given
    CATMAID volume.
    """

    if 'pyoctree' in volume:
        # Use store octree if available
        tree = volume['pyoctree']
    else:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume['vertices'], dtype=float),
                                 np.array(volume['faces'], dtype=np.int32)
                                 )
        volume['pyoctree'] = tree

    # Generate rays for points
    mx = np.array(volume['vertices']).max(axis=0)
    mn = np.array(volume['vertices']).min(axis=0)

    rayPointList = np.array(
                [[[p[0], p[1], mn[2]], [p[0], p[1], mx[2]]] for p in points], dtype=np.float32)

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # by those that occur "above" the point
    intersections = [len([i for i in tree.rayIntersection(ray) if i.p[
                         2] >= points[k][2]])for k, ray in enumerate( tqdm(rayPointList,
                                                                           desc='Intersecting',
                                                                           leave=False,
                                                                           disable=pbar_hide
                                                                        ))]

    # Count odd intersection
    return [i % 2 != 0 for i in intersections]


def _in_volume_convex(points, volume, remote_instance=None, approximate=False, ignore_axis=[]):
    """ Uses scipy to test if points are within a given CATMAID volume.
    The idea is to test if adding the point to the cloud would change the
    convex hull.
    """

    remote_instance = fetch._eval_remote_instance(remote_instance)

    if type(volume) == type(str()):
        volume = fetch.get_volume(volume, remote_instance)

    verts = volume['vertices']

    if not approximate:
        intact_hull = ConvexHull(verts)
        intact_verts = list(intact_hull.vertices)

        if isinstance(points, list):
            points = np.array(points)
        elif isinstance(points, pd.DataFrame):
            points = points.to_matrix()

        return [list(ConvexHull(np.append(verts, list([p]), axis=0)).vertices) == intact_verts for p in points]
    else:
        bbox = [(min([v[0] for v in verts]), max([v[0] for v in verts])),
                (min([v[1] for v in verts]), max([v[1] for v in verts])),
                (min([v[2] for v in verts]), max([v[2] for v in verts]))
                ]

        for a in ignore_axis:
            bbox[a] = (float('-inf'), float('inf'))

        return [False not in [bbox[0][0] < p.x < bbox[0][1], bbox[1][0] < p.y < bbox[1][1], bbox[2][0] < p.z < bbox[2][1], ] for p in points]


