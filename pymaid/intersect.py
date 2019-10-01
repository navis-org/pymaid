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
    logger.warning("Optional dependency 'pyoctree' not found. Falling back "
                   "to Scipy for intersection calculations. This may return "
                   "incorrect results for concave volumes. You might want to "
                   "consider installing pyoctree via: pip3 install pyoctree")

__all__ = sorted(['in_volume', 'intersection_matrix'])


def in_volume(x, volume, inplace=False, mode='IN', prevent_fragments=False,
              method='FAST', remote_instance=None):
    """Test if points/neurons are within a given CATMAID volume.

    Important
    ---------
    This function requires `pyoctree <https://github.com/mhogg/pyoctree>`_
    which is only an optional dependency of pymaid. If pyoctree is not
    installed, we will fall back to using scipy ConvexHull instead. This is
    slower and may give wrong positives for concave meshes!

    Parameters
    ----------
    x :                 list of tuples | numpy.array | pandas.DataFrame | CatmaidNeuron | CatmaidNeuronList

                        - list/numpy.array is treated as list of x/y/z
                          coordinates. Needs to be shape (N,3): e.g.
                          ``[[x1, y1, z1], [x2, y2, z2], ..]``
                        - ``pandas.DataFrame`` needs to have ``x, y, z``
                          columns

    volume :            str | pymaid.Volume | list or dict of either
                        :class:`pymaid.Volume` or name of a CATMAID volume to
                        test. Multiple volumes can be given as list
                        (``[volume1, volume2, ...]``) or dict
                        (``{'label1': volume1, ...}``) of either str or
                        :class:`pymaid.Volume`.
    inplace :           bool, optional
                        If False, a copy of the original DataFrames/Neuron is
                        returned. Does only apply to CatmaidNeuron or
                        CatmaidNeuronList objects. Does apply if multiple
                        volumes are provided.
    mode :              'IN' | 'OUT', optional
                        If 'IN', parts of the neuron that are within the volume
                        are kept.
    prevent_fragments : bool, optional
                        Only relevant if input is CatmaidNeuron/List: if True,
                        will add nodes required to keep neuron from
                        fragmenting.
    method :            'FAST' | 'SAFE', optional
                        Method used for raycasting. "FAST" will cast only a
                        single ray to check for intersections. If you
                        experience problems, set method to "SAFE" to use
                        multiple rays (slower).
    remote_instance :   CATMAID instance, optional
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
    ...                       remote_instance=remote_instance)
    >>> # Calc intersections with each of the above glomeruli
    >>> res = pymaid.in_volume(n, gloms, remote_instance=remote_instance)
    >>> # Extract cable
    >>> cable = {v: res[v].cable_length for v in res}
    >>> # Plot graph
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame(list( cable.values() ),
    ...                   index=list( cable.keys() )
    ...                   )
    >>> df.boxplot()
    >>> plt.show()

    """
    remote_instance = utils._eval_remote_instance(remote_instance,
                                                  raise_error=False)

    # If we are given multiple volumes
    if isinstance(volume, (list, dict, np.ndarray)):
        # Force into dict
        if not isinstance(volume, dict):
            # Make sure all pymaid.Volumes can be uniquely indexed
            vnames = set([v.name for v in volume if isinstance(v, core.Volume)])
            dupli = [v for v in set(vnames) if vnames.count(v) > 1]
            if dupli:
                raise ValueError('Duplicate Volume names detected: {}. Volume.'
                                 'name must be unique.'.format(','.join(dupli)))

            temp = {v: v for v in volume if isinstance(v, str)}
            temp.update({v.name: v for v in volume if isinstance(v, core.Volume)})
            volume = temp

        data = dict()
        for v in config.tqdm(volume, desc='Volumes', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            data[v] = in_volume(x, volume[v], remote_instance=remote_instance,
                                inplace=False, mode=mode, method=method)
        return data

    if isinstance(volume, str):
        volume = fetch.get_volume(volume, remote_instance=remote_instance)

    # Make copy if necessary
    if isinstance(x, (core.CatmaidNeuronList, core.CatmaidNeuron)):
        if inplace is False:
            x = x.copy()

    if isinstance(x, pd.DataFrame):
        points = x[['x', 'y', 'z']].values
    elif isinstance(x, core.CatmaidNeuron):
        in_v = in_volume(x.nodes[['x', 'y', 'z']].values, volume,
                         method=method)

        # If mode is OUT, invert selection
        if mode == 'OUT':
            in_v = ~np.array(in_v)

        x = graph_utils.subset_neuron(x, x.nodes[in_v].treenode_id.values,
                                      inplace=True,
                                      prevent_fragments=prevent_fragments)

        if inplace is False:
            return x
        return
    elif isinstance(x, core.CatmaidNeuronList):
        for n in x:
            _ = in_volume(n, volume, inplace=True, mode=mode, method=method,
                          prevent_fragments=prevent_fragments)

        if inplace is False:
            return x
        return
    else:
        points = x

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Points must be array of shape (N,3).')

    if pyoctree:
        return _in_volume_ray(points, volume,
                              multi_ray=method.upper() == 'SAFE')
    else:
        logger.warning(
            'Package pyoctree not found. Falling back to ConvexHull.')
        return _in_volume_convex(points, volume, approximate=False)


def _in_volume_ray(points, volume, multi_ray=False):
    """Use pyoctree's raycsasting to test if points are within volume."""
    tree = getattr(volume, 'pyoctree', None)

    if not tree:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume.vertices, dtype=float, order='C'),
                                 np.array(volume.faces, dtype=np.int32, order='C')
                                 )
        volume.pyoctree = tree

    # Get min max of volume
    mx = np.array(volume.vertices).max(axis=0)
    mn = np.array(volume.vertices).min(axis=0)

    # Get points outside of bounding box
    bbox_out = (points > mx).any(axis=1) | (points < mn).any(axis=1)
    isin = ~bbox_out
    in_points = points[isin]

    # Perform ray intersection on points inside bounding box
    rayPointList = np.array([[[p[0], mn[1], mn[2]], p] for p in in_points],
                            dtype=np.float32)

    # Get intersections and extract coordinates of intersection
    intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]

    # In a few odd cases we can get the multiple intersections at the exact
    # same coordinate (something funny with the faces).
    unique_int = [np.unique(np.round(i), axis=0) if np.any(i) else i for i in intersections]

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # to those that occur "above" the point we are querying
    unilat_int = [i[i[:, 2] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 2])]

    # Count intersections
    int_count = [i.shape[0] for i in unilat_int]

    # Get odd (= in volume) numbers of intersections
    is_odd = np.remainder(int_count, 2) != 0

    # If we want to play it safe, run the above again with two additional rays
    # and find a majority decision.
    if multi_ray:
        # Run ray from left back
        rayPointList = np.array([[[mn[0], p[1], mn[2]], p] for p in in_points],
                                dtype=np.float32)
        intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]
        unique_int = [np.unique(i, axis=0) if np.any(i) else i for i in intersections]
        unilat_int = [i[i[:, 0] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 0])]
        int_count = [i.shape[0] for i in unilat_int]
        is_odd2 = np.remainder(int_count, 2) != 0

        # Run ray from lower left
        rayPointList = np.array([[[mn[0], mn[1], p[2]], p] for p in in_points],
                                dtype=np.float32)
        intersections = [np.array([i.p for i in tree.rayIntersection(ray)]) for ray in rayPointList]
        unique_int = [np.unique(i, axis=0) if np.any(i) else i for i in intersections]
        unilat_int = [i[i[:, 1] >= p] if np.any(i) else i for i, p in zip(unique_int, in_points[:, 1])]
        int_count = [i.shape[0] for i in unilat_int]
        is_odd3 = np.remainder(int_count, 2) != 0

        # Find majority consensus
        is_odd = is_odd.astype(int) + is_odd2.astype(int) + is_odd3.astype(int)
        is_odd = is_odd >= 2

    isin[isin] = is_odd
    return isin


def _in_volume_convex(points, volume, remote_instance=None, approximate=False,
                      ignore_axis=[]):
    """ Uses scipy to test if points are within a given CATMAID volume.
    The idea is to test if adding the point to the pointcloud changes the
    convex hull -> if yes, that point is outside the convex hull.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(volume, str):
        volume = fetch.get_volume(volume, remote_instance=remote_instance)

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


def intersection_matrix(x, volumes, attr=None, method='FAST',
                        remote_instance=None):
    """Compute intersection matrix between a set of neurons and a set of
    volumes.

    Parameters
    ----------
    x :               pymaid.CatmaidNeuronList | pymaid.CatmaidNeuron
                      Neurons to intersect.
    volume :          list or dict of pymaid.Volume
    attr :            str | None, optional
                      Attribute to return for intersected neurons (e.g.
                      'cable_length'). If None, will return CatmaidNeuron.
    method :          'FAST' | 'SAFE', optional
                      See :func:`pymaid.in_volume`.
    remote_instance : CATMAID instance, optional
                      Pass if ``volume`` is a volume name.

    Returns
    -------
    pandas DataFrame

    """
    if isinstance(x, core.CatmaidNeuron):
        x = core.CatmaidNeuronList(x)

    if not isinstance(x, core.CatmaidNeuronList):
        raise TypeError('x must be CatmaidNeuron/List, not "{}"'.format(type(x)))

    if isinstance(volumes, list):
        volumes = {v.name: v for v in volumes}

    if not isinstance(volumes, (list, dict)):
        raise TypeError('volumes must be given as list or dict, not "{}"'.format(type(volumes)))

    for v in volumes.values():
        if not isinstance(v, core.Volume):
            raise TypeError('Wrong data type found in volumes: "{}"'.format(type(v)))

    data = in_volume(x, volumes, inplace=False, mode='IN', method=method,
                     remote_instance=remote_instance)

    if not attr:
        df = pd.DataFrame([[n for n in data[v]] for v in data],
                          index=list(data.keys()),
                          columns=x.skeleton_id)
    else:
        df = pd.DataFrame([[getattr(n, attr) for n in data[v]] for v in data],
                          index=list(data.keys()),
                          columns=x.skeleton_id)

    return df
