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

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

from . import fetch, core, utils, graph_utils, config

# Set up logging -> has to be before try statement!
logger = config.logger

try:
    from pyoctree import pyoctree
except ImportError:
    pyoctree = None
    logger.warning("Module pyoctree not found. Falling back to scipy's \
                            ConvexHull for intersection calculations.")

__all__ = sorted(['in_volume'])


def in_volume(x, volume, inplace=False, mode='IN', remote_instance=None):
    """ Test if points/neurons are within a given CATMAID volume.

    Important
    ---------
    This function requires `pyoctree <https://github.com/mhogg/pyoctree>`_
    which is only an optional dependency of pymaid. If pyoctree is not
    installed, we will fall back to using scipy ConvexHull instead of ray
    casting. This is slower and may give wrong positives for concave meshes!

    Parameters
    ----------
    x :               list of tuples | numpy.array | pandas.DataFrame | CatmaidNeuron | CatmaidNeuronList

                      - list/numpy.array is treated as list of x/y/z
                        coordinates. Needs to be shape (N,3): e.g.
                        ``[[x1, y1, z1], [x2, y2, z2], ..]``
                      - ``pandas.DataFrame`` needs to have ``x, y, z`` columns

    volume :          str | pymaid.Volume | list or dict of either
                      :class:`pymaid.Volume` or name of a CATMAID volume to
                      test. Multiple volumes can be given as list
                      (``[volume1, volume2, ...]``) or dict
                      (``{'label1': volume1, ...}``) of either str or
                      :class:`pymaid.Volume`.
    inplace :         bool, optional
                      If False, a copy of the original DataFrames/Neuron is
                      returned. Does only apply to CatmaidNeuron or
                      CatmaidNeuronList objects. Does apply if multiple
                      volumes are provided.
    mode :            'IN' | 'OUT', optional
                      If 'IN', parts of the neuron that are within the volume
                      are kept.
    remote_instance : CATMAID instance, optional
                      Pass if ``volume`` is a volume name.

    Returns
    -------
    CatmaidNeuron
                      If input is CatmaidNeuron or CatmaidNeuronList, will
                      return subset of the neuron(s) (nodes and connectors)
                      that are within given volume.
    list of bools
                      If input is a set of coordinates, returns boolean:
                      ``True`` if in volume, ``False`` if not in order.
    dict
                      If multiple volumes are provided, results will be
                      returned in dictionary with volumes as keys::

                        {'volume1': in_volume(x, volume1),
                         'volume2': in_volume(x, volume2),
                         ... }

    Examples
    --------
    Advanced example: Check with which antennal lobe glomeruli a neuron
    intersects.

    >>> # First prepare some volume names
    >>> gloms = ['v14.DA1','v14.DA2', 'v14.DA3', 'v14.DA4l' ,'v14.DL4',
    ...          'v14.VA2', 'v14.DC3', 'v14.VM7v', 'v14.DC4', 'v14.DC1',
    ...          'v14.DM5', 'v14.D', 'v14.VM2', 'v14.VC4', 'v14.VL1',
    ...          'v14.DM3', 'v14.DL1', 'v14.DP1m']
    >>> # Get neuron to check
    >>> n = pymaid.get_neuron('name:PN unknown glomerulus',
    ...                       remote_instance = remote_instance )
    >>> # Calc intersections with each of the above glomeruli
    >>> res = pymaid.in_volume(n, gloms, remote_instance=remote_instance)
    >>> # Extract cable
    >>> cable = {v: res[v].cable_length for v in res}
    >>> # Plot graph
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame(list( cable.values() ),
    ...                   index = list( cable.keys() )
    ...                   )
    >>> df.boxplot()
    >>> plt.show()

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # If we are given multiple volumes
    if isinstance(volume, (list, dict, np.ndarray)):
        # Force into dict
        if not isinstance(volume, dict):
            temp = {v: v for v in volume if isinstance(v, str)}
            temp.update({v.name: v for v in volume if isinstance(v, core.Volume)})
            volume = temp

        data = dict()
        for v in config.tqdm(volume, desc='Volumes', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            data[v] = in_volume(x, volume[v], remote_instance=remote_instance,
                                inplace=False, mode=mode)
        return data

    if isinstance(volume, str):
        volume = fetch.get_volume(volume, remote_instance)

    # Make copy if necessary
    if isinstance(x, (core.CatmaidNeuronList, core.CatmaidNeuron)):
        if inplace is False:
            x = x.copy()

    if isinstance(x, pd.DataFrame):
        points = x[['x', 'y', 'z']].values
    elif isinstance(x, core.CatmaidNeuron):
        in_v = in_volume(x.nodes[['x', 'y', 'z']].values, volume)

        # If mode is OUT, invert selection
        if mode == 'OUT':
            in_v = ~np.array(in_v)

        x = graph_utils.subset_neuron(x, x.nodes[in_v].treenode_id.values,
                                      inplace=True)

        if inplace is False:
            return x
        return
    elif isinstance(x, core.CatmaidNeuronList):
        for n in x:
            _ = in_volume(n, volume, inplace=True, mode=mode)

        if inplace is False:
            return x
        return
    else:
        points = x

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Points must be array of shape (N,3).')

    if pyoctree:
        return _in_volume_ray(points, volume)
    else:
        logger.warning(
            'Package pyoctree not found. Falling back to ConvexHull.')
        return _in_volume_convex(points, volume, approximate=False)


def _in_volume_ray(points, volume):
    """ Uses pyoctree's raycsasting to test if points are within a given
    CATMAID volume.
    """

    tree = getattr(volume, 'pyoctree', None)

    if not tree:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume.vertices, dtype=float),
                                 np.array(volume.faces, dtype=np.int32)
                                 )
        volume.pyoctree = tree

    # Get min max of volume
    mx = np.array(volume.vertices).max(axis=0)
    mn = np.array(volume.vertices).min(axis=0)

    # Get points outside of bounding box
    out = (points > mx).any(axis=1) | (points < mn).any(axis=1)
    isin = ~out
    in_points = points[~out]

    # Perform ray intersection on points inside bounding box
    rayPointList = np.array([[[p[0], p[1], mn[2]], [p[0], p[1], mx[2]]] for p in in_points],
                            dtype=np.float32)

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # to those that occur "above" the point we are querying
    intersections = [len([i for i in tree.rayIntersection(
        ray) if i.p[2] >= in_points[k][2]]) for k, ray in enumerate(rayPointList)]

    # Count intersections and return True for odd counts
    # [i % 2 != 0 for i in intersections]
    isin[~out] = np.remainder(list(intersections), 2) != 0
    return isin


def _in_volume_convex(points, volume, remote_instance=None, approximate=False,
                      ignore_axis=[]):
    """ Uses scipy to test if points are within a given CATMAID volume.
    The idea is to test if adding the point to the pointcloud changes the
    convex hull -> if yes, that point is outside the convex hull.
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(volume, str):
        volume = fetch.get_volume(volume, remote_instance)

    verts = volume.vertices

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
