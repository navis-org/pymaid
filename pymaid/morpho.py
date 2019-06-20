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


""" This module contains functions to analyse and manipulate neuron morphology.
"""

import math
import itertools

import pandas as pd
import numpy as np
import scipy.spatial.distance
import networkx as nx

from . import fetch, core, graph_utils, graph, utils, config, resample

# Set up logging
logger = config.logger

__all__ = sorted(['calc_cable', 'strahler_index', 'prune_by_strahler',
                  'stitch_neurons', 'arbor_confidence', 'split_axon_dendrite',
                  'bending_flow', 'flow_centrality', 'segregation_index',
                  'to_dotproduct', 'average_neurons', 'tortuosity',
                  'remove_tagged_branches', 'despike_neuron', 'guess_radius',
                  'smooth_neuron', 'time_machine', 'heal_fragmented_neuron',
                  'break_fragments', 'heal_fragmented_neuron'])


def arbor_confidence(x, confidences=(1, 0.9, 0.6, 0.4, 0.2), inplace=True):
    """ Calculates confidence for each treenode.

    Calculates confidence for each treenode by walking from root to leafs
    starting with a confidence of 1. Each time a low confidence edge is
    encountered the downstream confidence is reduced (see ``confidences``).

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) to calculate confidence for.
    confidences :       list of five floats, optional
                        Values by which the confidence of the downstream
                        branche is reduced upon encounter of a 5/4/3/2/1-
                        confidence edges.
    inplace :           bool, optional
                        If False, a copy of the neuron is returned.

    Returns
    -------
    Adds ``arbor_confidence`` column in ``neuron.nodes``.

    """

    def walk_to_leafs(this_node, this_confidence=1):
        pbar.update(1)
        while True:
            this_confidence *= confidences[5 - nodes.loc[this_node].confidence]
            nodes.loc[this_node, 'arbor_confidence'] = this_confidence

            if len(loc[this_node]) > 1:
                for c in loc[this_node]:
                    walk_to_leafs(c, this_confidence)
                break
            elif len(loc[this_node]) == 0:
                break

            this_node = loc[this_node][0]

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise TypeError('Unable to process data of type %s' % str(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            res = [arbor_confidence(
                n, confidence=confidence, inplace=inplace) for n in x]
        else:
            return core.CatmaidNeuronList([arbor_confidence(n, confidence=confidence, inplace=inplace) for n in x])

    if not inplace:
        x = x.copy()

    loc = graph_utils.generate_list_of_childs(x)

    x.nodes['arbor_confidence'] = [None] * x.nodes.shape[0]

    nodes = x.nodes.set_index('treenode_id')
    nodes.loc[x.root, 'arbor_confidence'] = 1

    with config.tqdm(total=len(x.segments), desc='Calc confidence', disable=config.pbar_hide, leave=config.pbar_leave) as pbar:
        for r in x.root:
            for c in loc[r]:
                walk_to_leafs(c)

    x.nodes['arbor_confidence'] = nodes['arbor_confidence'].values

    if not inplace:
        return x


def _calc_dist(v1, v2):
    return math.sqrt(sum(((a - b)**2 for a, b in zip(v1, v2))))


def _parent_dist(x, root_dist=None):
    """ Adds ``parent_dist`` [nm] column to the treenode table.

    Parameters
    ----------
    x :         CatmaidNeuron | treenode table
    root_dist : int | None
                ``parent_dist`` for the root's row. Set to ``None``, to leave
                at ``NaN`` or e.g. to ``0`` to set to 0.

    Returns
    -------
    Nothing
    """

    if isinstance(x, core.CatmaidNeuron):
        nodes = x.nodes
    elif isinstance(x, pd.DataFrame):
        nodes = x
    else:
        raise TypeError('Need CatmaidNeuron or DataFrame, got "{}"'.format(type(x)))

    # Calculate distance to parent for each node
    wo_root = nodes[~nodes.parent_id.isnull()]
    tn_coords = wo_root[['x', 'y', 'z']].values

    # Ready treenode table to be indexes by treenode_id
    this_tn = nodes.set_index('treenode_id')
    parent_coords = this_tn.loc[wo_root.parent_id.values,
                                ['x', 'y', 'z']].values

    # Calculate distances between nodes and their parents
    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    nodes['parent_dist'] = root_dist
    nodes.loc[~nodes.parent_id.isnull(), 'parent_dist'] = w

    return


def calc_cable(skdata, remote_instance=None, return_skdata=False):
    """ Calculates cable length in micrometer (um).

    Parameters
    ----------
    skdata :            int | str | CatmaidNeuron | CatmaidNeuronList
                        If skeleton ID (str or in), 3D skeleton data will be
                        pulled from CATMAID server.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.
    return_skdata :     bool, optional
                        If True: instead of the final cable length, a dataframe
                        containing the distance to each treenode's parent.

    Returns
    -------
    float
                Cable in micrometers [um]. If ``return_skdata==False``.
    CatmaidNeuron
                Neuron object with a new column in ``x.nodes``:
                ``'parent_dist'``. If ``return_skdata==True``.

    See Also
    --------
    ``pymaid.CatmaidNeuron.cable_length``
                Use this attribute to get the cable length of given neuron.
                Also works with ``CatmaidNeuronList``.
    ``pymaid.smooth_neuron``
                Use this function to smooth the neuron before calculating
                cable.

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(skdata, int) or isinstance(skdata, str):
        skdata = fetch.get_neuron([skdata], remote_instance).loc[0]

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.loc[0]
        elif not return_skdata:
            return sum([calc_cable(skdata.loc[i]) for i in range(skdata.shape[0])])
        else:
            return core.CatmaidNeuronList([calc_cable(skdata.loc[i],
                                                      return_skdata=return_skdata)
                                           for i in range(skdata.shape[0])])
    else:
        raise Exception('Unable to interpret data of type', type(skdata))

    # Catch single-node neurons
    if df.nodes.shape[0] == 1:
        if return_skdata:
            df.nodes['parent_dist'] = 0
            return df
        else:
            return 0

    # Calculate distance to parent for each node
    nodes = df.nodes[~df.nodes.parent_id.isnull()]
    tn_coords = nodes[['x', 'y', 'z']].values

    # Ready treenode table to be indexes by treenode_id
    this_tn = df.nodes.set_index('treenode_id')
    parent_coords = this_tn.loc[nodes.parent_id.values,
                                ['x', 'y', 'z']].values

    # Calculate distances between nodes and their parents
    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    if return_skdata:
        df.nodes.loc[~df.nodes.parent_id.isnull(), 'parent_dist'] = w / 1000
        return df

    # #Remove nan value (at parent node) and return sum of all distances
    return np.sum(w[np.logical_not(np.isnan(w))]) / 1000


def to_dotproduct(x):
    """ Converts a neuron's neurites into dotproducts.

    Dotproducts consist of a point and a vector. This works by (1) finding the
    center between child->parent treenodes and (2) getting the vector between
    them. Also returns the length of the vector.

    Parameters
    ----------
    x :         CatmaidNeuron
                Single neuron

    Returns
    -------
    pandas.DataFrame
            DataFrame in which each row represents a segments between two
            treenodes::

                point  vector  vec_length
             1
             2
             3

    Examples
    --------
    >>> x = pymaid.get_neurons(16)
    >>> dps = pymaid.to_dotproduct(x)
    >>> # Get array of all locations
    >>> locs = numpy.vstack(dps.point.values)

    See Also
    --------
    pymaid.CatmaidNeuron.dps

    """

    if isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError('Please pass only single CatmaidNeurons')

    if not isinstance(x, core.CatmaidNeuron):
        raise ValueError('Can only process CatmaidNeurons')

    # First, get a list of child -> parent locs (exclude root node!)
    tn_locs = x.nodes[~x.nodes.parent_id.isnull()][['x', 'y', 'z']].values
    pn_locs = x.nodes.set_index('treenode_id').loc[x.nodes[~x.nodes.parent_id.isnull(
    )].parent_id][['x', 'y', 'z']].values

    # Get centers between each pair of locs
    centers = tn_locs + (pn_locs - tn_locs) / 2

    # Get vector between points
    vec = pn_locs - tn_locs

    dps = pd.DataFrame([[c, v] for c, v in zip(centers, vec)],
                       columns=['point', 'vector'])

    # Add length of vector (for convenience)
    dps['vec_length'] = (dps.vector ** 2).apply(sum).apply(math.sqrt)

    return dps


def strahler_index(x, inplace=True, method='standard', fix_not_a_branch=False,
                   min_twig_size=None):
    """ Calculates Strahler Index (SI).

    Starts with SI of 1 at each leaf and walks to root. At forks with different
    incoming SIs, the highest index is continued. At forks with the same
    incoming SI, highest index + 1 is continued.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        E.g. from  ``pymaid.get_neuron()``.
    inplace :           bool, optional
                        If False, a copy of original skdata is returned.
    method :            'standard' | 'greedy', optional
                        Method used to calculate strahler indices: 'standard'
                        will use the method described above; 'greedy' will
                        always increase the index at converging branches
                        whether these branches have the same index or not.
                        This is useful e.g. if you want to cut the neuron at
                        the first branch point.
    fix_not_a_branch :  bool, optional
                        If True, terminal branches whose FIRST nodes are
                        tagged with "not a branch" will not contribute to
                        Strahler index calculations and instead be assigned
                        the SI of their parent branch.
    min_twig_size :     int, optional
                        If provided, will ignore terminal (!) twigs with
                        fewer nodes. Instead, they will be assigned the SI of
                        their parent branch.

    Returns
    -------
    if ``inplace=False``
                        Returns nothing but adds new column ``strahler_index``
                        to neuron.nodes.
    if ``inplace=True``
                        Returns copy of original neuron with new column
                        ``strahler_index``.

    """

    if isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            res = []
            for n in config.tqdm(x):
                res.append(strahler_index(n, inplace=inplace, method=method))

            if not inplace:
                return core.CatmaidNeuronList(res)
            else:
                return

    if not inplace:
        x = x.copy()

    # Find branch, root and end nodes
    if 'type' not in x.nodes:
        classify_nodes(x)

    end_nodes = x.nodes[x.nodes.type == 'end'].treenode_id.values
    branch_nodes = x.nodes[x.nodes.type == 'branch'].treenode_id.values
    root = x.nodes[x.nodes.type == 'root'].treenode_id.values

    end_nodes = set(end_nodes)
    branch_nodes = set(branch_nodes)
    root = set(root)

    nab_branch = []
    if fix_not_a_branch and 'not a branch' in x.tags:
        nab_branch += x.tags['not a branch']
    if min_twig_size:
        nab_branch += [seg[0] for seg in x.small_segments if seg[0]
                       in end_nodes and len(seg) < min_twig_size]

    # Generate dicts for childs and parents
    list_of_childs = graph_utils.generate_list_of_childs(x)

    # Reindex according to treenode_id
    this_tn = x.nodes.set_index('treenode_id')

    # Do NOT name anything strahler_index - this overwrites the function!
    SI = {}

    starting_points = end_nodes

    nodes_processed = []

    while starting_points:
        logger.debug('New starting point. Remaining: '
                     '{}'.format(len(starting_points)))
        new_starting_points = []
        starting_points_done = []

        for i, this_node in enumerate(starting_points):
            logger.debug('%i of %i ' % (i, len(starting_points)))

            # Calculate index for this branch
            previous_indices = []
            for child in list_of_childs[this_node]:
                if SI.get(child, None):
                    previous_indices.append(SI[child])

            # If this is a not-a-branch branch
            if this_node in nab_branch:
                this_branch_index = None
            # If this is an end node: start at 1
            elif len(previous_indices) == 0:
                this_branch_index = 1
            # If this is a slab: assign SI of predecessor
            elif len(previous_indices) == 1:
                this_branch_index = previous_indices[0]
            # If this is a branch point at which similar indices collide: +1
            elif previous_indices.count(max(previous_indices)) >= 2 or method == 'greedy':
                this_branch_index = max(previous_indices) + 1
            # If just a branch point: continue max SI
            else:
                this_branch_index = max(previous_indices)

            nodes_processed.append(this_node)
            starting_points_done.append(this_node)

            # Now walk down this spine
            # Find parent
            spine = [this_node]

            #parent_node = list_of_parents [ this_node ]
            parent_node = this_tn.loc[this_node, 'parent_id']

            while parent_node not in branch_nodes and parent_node is not None:
                this_node = parent_node
                parent_node = None

                spine.append(this_node)
                nodes_processed.append(this_node)

                # Find next parent
                try:
                    parent_node = this_tn.loc[this_node, 'parent_id']
                except BaseException:
                    # Will fail if at root (no parent)
                    break

            SI.update({n: this_branch_index for n in spine})

            # The last this_node is either a branch node or the root
            # If a branch point: check, if all its childs have already been
            # processed
            if parent_node is not None:
                node_ready = True
                for child in list_of_childs[parent_node]:
                    if child not in nodes_processed:
                        node_ready = False

                if node_ready is True and parent_node is not None:
                    new_starting_points.append(parent_node)

        # Remove those starting_points that were successfully processed in this
        # run before the next iteration
        for node in starting_points_done:
            starting_points.remove(node)

        # Add new starting points
        starting_points = starting_points | set(new_starting_points)

    # Disconnected single nodes (e.g. after pruning) will end up w/o an entry
    # --> we will give them an SI of 1
    x.nodes['strahler_index'] = [SI.get(n, 1)
                                 for n in x.nodes.treenode_id.values]

    # Fix not-a-branch branches
    if fix_not_a_branch and 'not a branch' in x.tags:
        this_tn = x.nodes.set_index('treenode_id')
        # Go over all terminal branches with the tag
        for tn in x.nodes[(x.nodes.type == 'end') & (x.nodes.treenode_id.isin(x.tags['not a branch']))].treenode_id.values:
            # Get this terminal segment
            this_seg = [s for s in x.small_segments if s[0] == tn][0]
            # Get strahler index of parent branch
            new_SI = this_tn.loc[this_seg[-1]].strahler_index
            # Set these nodes strahler index to that of the last branch point
            x.nodes.loc[x.nodes.treenode_id.isin(
                this_seg), 'strahler_index'] = new_SI

    if not inplace:
        return x


def prune_by_strahler(x, to_prune, reroot_soma=True, inplace=False,
                      force_strahler_update=False, relocate_connectors=False):
    """ Prune neuron based on `Strahler order
    <https://en.wikipedia.org/wiki/Strahler_number>`_.

    Parameters
    ----------
    x :             CatmaidNeuron | CatmaidNeuronList
    to_prune :      int | list | range | slice
                    Strahler indices (SI) to prune. For example:

                    1. ``to_prune=1`` removes all leaf branches
                    2. ``to_prune=[1, 2]`` removes SI 1 and 2
                    3. ``to_prune=range(1, 4)`` removes SI 1, 2 and 3
                    4. ``to_prune=slice(0, -1)`` removes everything but the
                       highest SI
                    5. ``to_prune=slice(-1, None)`` removes only the highest
                       SI

    reroot_soma :   bool, optional
                    If True, neuron will be rerooted to its soma.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.
    relocate_connectors : bool, optional
                          If True, connectors on removed treenodes will be
                          reconnected to the closest still existing treenode.
                          Works only in child->parent direction.

    Returns
    -------
    CatmaidNeuron/List
                    Pruned neuron(s).

    """

    if isinstance(x, core.CatmaidNeuron):
        neuron = x
    elif isinstance(x, core.CatmaidNeuronList):
        temp = [prune_by_strahler(
            n, to_prune=to_prune, inplace=inplace) for n in x]
        if not inplace:
            return core.CatmaidNeuronList(temp, x._remote_instance)
        else:
            return

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    if reroot_soma and neuron.soma:
        neuron.reroot(neuron.soma, inplace=True)

    if 'strahler_index' not in neuron.nodes or force_strahler_update:
        strahler_index(neuron)

    # Prepare indices
    if isinstance(to_prune, int) and to_prune < 0:
        to_prune = range(1, neuron.nodes.strahler_index.max() + (to_prune + 1))

    if isinstance(to_prune, int):
        if to_prune < 1:
            raise ValueError('SI to prune must be positive. Please see help'
                             'for additional options.')
        to_prune = [to_prune]
    elif isinstance(to_prune, range):
        to_prune = list(to_prune)
    elif isinstance(to_prune, slice):
        SI_range = range(1, neuron.nodes.strahler_index.max() + 1)
        to_prune = list(SI_range)[to_prune]

    # Prepare parent dict if needed later
    if relocate_connectors:
        parent_dict = {
            tn.treenode_id: tn.parent_id for tn in neuron.nodes.itertuples()}

    neuron.nodes = neuron.nodes[
        ~neuron.nodes.strahler_index.isin(to_prune)].reset_index(drop=True)

    if not relocate_connectors:
        neuron.connectors = neuron.connectors[neuron.connectors.treenode_id.isin(
            neuron.nodes.treenode_id.values)].reset_index(drop=True)
    else:
        remaining_tns = set(neuron.nodes.treenode_id.values)
        for cn in neuron.connectors[~neuron.connectors.treenode_id.isin(neuron.nodes.treenode_id.values)].itertuples():
            this_tn = parent_dict[cn.treenode_id]
            while True:
                if this_tn in remaining_tns:
                    break
                this_tn = parent_dict[this_tn]
            neuron.connectors.loc[cn.Index, 'treenode_id'] = this_tn

    # Reset indices of node and connector tables (important for igraph!)
    neuron.nodes.reset_index(inplace=True, drop=True)
    neuron.connectors.reset_index(inplace=True, drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more
    # than 1 root node -> we have to fix the nodes that lost their parents
    neuron.nodes.loc[~neuron.nodes.parent_id.isin(
        neuron.nodes.treenode_id.values), 'parent_id'] = None

    # Remove temporary attributes
    neuron._clear_temp_attr()

    if not inplace:
        return neuron
    else:
        return


def split_axon_dendrite(x, method='bending', primary_neurite=True,
                        reroot_soma=True, return_point=False):
    """ Split a neuron into axon, dendrite and primary neurite.

    The result is highly dependent on the method and on your neuron's
    morphology and works best for "typical" neurons, i.e. those where the
    primary neurite branches into axon and dendrites.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) to split into axon, dendrite (and primary
                        neurite).
    method :            'centrifugal' | 'centripetal' | 'sum' | 'bending', optional
                        Type of flow centrality to use to split the neuron.
                        There are four flavors: the first three refer to
                        :func:`~pymaid.flow_centrality`, the last
                        refers to :func:`~pymaid.bending_flow`.

                        Will try using stored centrality, if possible.
    primary_neurite :   bool, optional
                        If True and the split point is at a branch point, will
                        try splittig into axon, dendrite and primary neurite.
                        Works only with ``method=bending``!
    reroot_soma :       bool, optional
                        If True, will make sure neuron is rooted to soma if at
                        all possible.
    return_point :      bool, optional
                        If True, will only return treenode ID of the node at
                        which to split the neuron.

    Returns
    -------
    CatmaidNeuronList
                        Axon, dendrite and primary neurite.

    Examples
    --------
    >>> x = pymaid.get_neuron(123456)
    >>> split = pymaid.split_axon_dendrite(x, method='centrifugal',
    ...                                    reroot_soma=True)
    >>> split
    <class 'pymaid.CatmaidNeuronList'> of 3 neurons
                          neuron_name skeleton_id  n_nodes  n_connectors
    0  neuron 123457_primary_neurite          16      148             0
    1             neuron 123457_axon          16     9682          1766
    2         neuron 123457_dendrite          16     2892           113
    >>> # For convenience, split_axon_dendrite assigns colors to the resulting
    >>> # fragments: axon = red, dendrites = blue, primary neurite = green
    >>> split.plot3d(color=split.color)

    """

    if isinstance(x, core.CatmaidNeuronList) and len(x) == 1:
        x = x[0]
    elif isinstance(x, core.CatmaidNeuronList):
        nl = []
        for n in config.tqdm(x, desc='Splitting', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            nl.append(split_axon_dendrite(n,
                                          method=method,
                                          primary_neurite=primary_neurite,
                                          reroot_soma=reroot_soma,
                                          return_point=return_point))
        return core.CatmaidNeuronList([n for l in nl for n in l])

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process CatmaidNeuron, '
                        'got "{}"'.format(type(x)))

    if method not in ['centrifugal', 'centripetal', 'sum', 'bending']:
        raise ValueError('Unknown parameter for mode: {0}'.format(method))

    if primary_neurite and method != 'bending':
        logger.warning('Primary neurite splits only works well with '
                       'method "bending"')

    if x.soma and x.soma not in x.root and reroot_soma:
        x.reroot(x.soma)

    # Calculate flow centrality if necessary
    last_method = getattr(x, 'centrality_method', None)

    if last_method != method:
        if method == 'bending':
            _ = bending_flow(x)
        elif method in ['centripetal', 'centrifugal', 'sum']:
            _ = flow_centrality(x, mode=method)
        else:
            raise ValueError('Unknown method "{}"'.format(method))

    # Make copy, so that we don't screw things up
    x = x.copy()

    # Now get the node point with the highest flow centrality.
    cut = x.nodes[x.nodes.flow_centrality ==
                  x.nodes.flow_centrality.max()].treenode_id.values

    # If there is more than one point we need to get one closest to the soma
    # (root)
    if len(cut) > 1:
        cut = sorted(cut, key=lambda y: graph_utils.dist_between(
            x.graph, y, x.root[0]))[0]
    else:
        cut = cut[0]

    if return_point:
        return cut

    # If cut node is a branch point, we will try cutting off main neurite
    if x.graph.degree(cut) > 2 and primary_neurite:
        # First make sure that there are no other branch points with flow
        # between this one and the soma
        path_to_root = nx.shortest_path(x.graph, cut, x.root[0])

        # Get flow centrality along the path
        flows = x.nodes.set_index('treenode_id').loc[path_to_root]

        # Subset to those that are branches (exclude mere synapses)
        flows = flows[flows.type == 'branch']

        # Find the first branch point from the soma with no flow
        # (fillna is important!)
        last_with_flow = np.where(flows.flow_centrality.fillna(0).values > 0)[0][-1]

        if method != 'bending' and last_with_flow < (flows.shape[0]-1):
            last_with_flow += 1

        to_cut = flows.iloc[last_with_flow].name

        # Cut off primary neurite
        rest, primary_neurite = graph_utils.cut_neuron(x, to_cut)

        # The new cut node has to be a child of the original cut node
        cut = next(x.graph.predecessors(cut))

        # Change name and color
        primary_neurite.neuron_name = x.neuron_name + '_primary_neurite'
        primary_neurite.color = (0, 255, 0)
        primary_neurite.type = 'primary_neurite'
    else:
        rest = x
        primary_neurite = None

    # Next, cut the rest into axon and dendrite
    a, b = graph_utils.cut_neuron(rest, cut)

    # Figure out which one is which by comparing fraction of in- to outputs
    a_inout = a.n_postsynapses/a.n_presynapses if a.n_presynapses else float('inf')
    b_inout = b.n_postsynapses/b.n_presynapses if b.n_presynapses else float('inf')
    if a_inout > b_inout:
        dendrite, axon = a, b
    else:
        dendrite, axon = b, a

    axon.neuron_name = x.neuron_name + '_axon'
    dendrite.neuron_name = x.neuron_name + '_dendrite'

    axon.type = 'axon'
    dendrite.type = 'dendrite'

    # Change colors
    axon.color = (255, 0, 0)
    dendrite.color = (0, 0, 255)

    if primary_neurite:
        return core.CatmaidNeuronList([primary_neurite, axon, dendrite])
    else:
        return core.CatmaidNeuronList([axon, dendrite])


def segregation_index(x, centrality_method='centrifugal'):
    """ Calculates segregation index (SI).

    The segregation index as established by Schneider-Mizell et al. (eLife,
    2016) is a measure for how polarized a neuron is. SI of 1 indicates total
    segregation of inputs and outputs into dendrites and axon, respectively.
    SI of 0 indicates homogeneous distribution.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron to calculate segregation index (SI). If a
                        NeuronList is provided, will assume that it contains
                        fragments (e.g. from axon/ dendrite splits) of a
                        single neuron.
    centrality_method : 'centrifugal' | 'centripetal' | 'sum' | 'bending'
                        Type of flow centrality to use to split into axon +
                        dendrite of ``x`` is only a single neuron.
                        There are four flavors:
                            - for the first three, see :func:`~pymaid.flow_centrality`
                            - for `bending`, see :func:`~pymaid.bending_flow`

                        Will try using stored centrality, if possible.

    Notes
    -----
    From Schneider-Mizell et al. (2016): "Note that even a modest amount of
    mixture (e.g. axo-axonic inputs) corresponds to values near H = 0.5–0.6
    (Figure 7—figure supplement 1). We consider an unsegregated neuron
    (H ¡ 0.05) to be purely dendritic due to their anatomical similarity with
    the dendritic domains of those segregated neurons that have dendritic
    outputs."

    Returns
    -------
    H :                 float
                        Segregation Index (SI).

    """

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, '
                         'not {0}'.format(type(x)))

    if isinstance(x, core.CatmaidNeuronList) and x.shape[0] == 1:
        x = x[0]

    if not isinstance(x, core.CatmaidNeuronList):
        # Get the branch point with highest flow centrality
        split_point = split_axon_dendrite(
            x, reroot_soma=True, return_point=True)

        # Now make a virtual split (downsampled neuron to speed things up)
        temp = x.copy()
        temp.downsample(10000)

        # Get one of its children
        child = temp.nodes[temp.nodes.parent_id == split_point].treenode_id.values[0]

        # This will leave the proximal split with the primary neurite but
        # since that should not have synapses, we don't care at this point.
        x = core.CatmaidNeuronList(list(graph_utils.cut_neuron(temp, child)))

    # Calculate entropy for each fragment
    entropy = []
    for n in x:
        p = n.n_postsynapses / n.n_connectors

        if 0 < p < 1:
            S = - (p * math.log(p) + (1 - p) * math.log(1 - p))
        else:
            S = 0

        entropy.append(S)

    # Calc entropy between fragments
    S = 1 / sum(x.n_connectors) * \
        sum([e * x[i].n_connectors for i, e in enumerate(entropy)])

    # Normalize to entropy in whole neuron
    p_norm = sum(x.n_postsynapses) / sum(x.n_connectors)
    if 0 < p_norm < 1:
        S_norm = - (p_norm * math.log(p_norm) +
                    (1 - p_norm) * math.log(1 - p_norm))
        H = 1 - S / S_norm
    else:
        S_norm = 0
        H = 0

    return H


def bending_flow(x, polypre=False):
    """ Variation of the algorithm for calculating synapse flow from
    Schneider-Mizell et al. (eLife, 2016).

    The way this implementation works is by iterating over each branch point
    and counting the number of pre->post synapse paths that "flow" from one
    child branch to the other(s).

    Parameters
    ----------
    x :         CatmaidNeuron | CatmaidNeuronList
                Neuron(s) to calculate bending flow for
    polypre :   bool, optional
                Whether to consider the number of presynapses as a multiple of
                the numbers of connections each makes. Attention: this works
                only if all synapses have been properly annotated.

    Notes
    -----
    This is algorithm appears to be more reliable than synapse flow
    centrality for identifying the main branch point for neurons that have
    only partially annotated synapses.

    See Also
    --------
    :func:`~pymaid.flow_centrality`
            Calculate synapse flow centrality after Schneider-Mizell et al
    :func:`~pymaid.segregation_score`
            Uses flow centrality to calculate segregation score (polarity)
    :func:`~pymaid.split_axon_dendrite`
            Split the neuron into axon, dendrite and primary neurite.

    Returns
    -------
    Adds a new column ``'flow_centrality'`` to ``x.nodes``. Branch points only!

    """

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, '
                         'not {0}'.format(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        return [bending_flow(n, mode=mode, polypre=polypre, ) for n in x]

    if x.soma and x.soma not in x.root:
        logger.warning(
            'Neuron {0} is not rooted to its soma!'.format(x.skeleton_id))

    # We will be processing a super downsampled version of the neuron to speed
    # up calculations
    current_level = logger.level
    logger.setLevel('ERROR')
    y = x.copy()
    y.downsample(1000000)
    logger.setLevel(current_level)

    if polypre:
        # Get details for all presynapses
        cn_details = fetch.get_connector_details(
            y.connectors[y.connectors.relation == 0])

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[y.connectors.relation == 0].treenode_id.values
    post_node_ids = y.connectors[y.connectors.relation == 1].treenode_id.values

    # Get list of branch_points
    bp_node_ids = y.nodes[y.nodes.type == 'branch'].treenode_id.values.tolist()
    # Add root if it is also a branch point
    for r in y.root:
        if y.graph.degree(r) > 1:
            bp_node_ids += [r]

    # Get list of childs of each branch point
    bp_childs = {t: [e[0] for e in y.graph.in_edges(t)] for t in bp_node_ids}
    childs = [tn for l in bp_childs.values() for tn in l]

    # Get number of pre/postsynapses distal to each branch's childs
    distal_pre = graph_utils.distal_to(y, pre_node_ids, childs)
    distal_post = graph_utils.distal_to(y, post_node_ids, childs)

    # Multiply columns (presynapses) by the number of postsynaptically
    # connected nodes
    if polypre:
        # Map vertex ID to number of postsynaptic nodes (avoid 0)
        distal_pre *= [max(1, len(cn_details[cn_details.presynaptic_to_node ==
                                             n].postsynaptic_to_node.sum())) for n in distal_pre.columns]

    # Sum up axis - now each row represents the number of pre/postsynapses
    # distal to that node
    distal_pre = distal_pre.T.sum(axis=1)
    distal_post = distal_post.T.sum(axis=1)

    # Now go over all branch points and check flow between branches
    # (centrifugal) vs flow from branches to root (centripetal)
    flow = {bp: 0 for bp in bp_childs}
    for bp in bp_childs:
        # We will use left/right to label the different branches here
        # (even if there is more than two)
        for left, right in itertools.permutations(bp_childs[bp], r=2):
            flow[bp] += distal_post.loc[left] * distal_pre.loc[right]

    # Set flow centrality to None for all nodes
    x.nodes['flow_centrality'] = None

    # Change index to treenode_id
    x.nodes.set_index('treenode_id', inplace=True)

    # Add flow (make sure we use igraph of y to get node ids!)
    x.nodes.loc[flow.keys(), 'flow_centrality'] = list(flow.values())

    # Add little info on method used for flow centrality
    x.centrality_method = 'bending'

    x.nodes.reset_index(inplace=True)

    return


def flow_centrality(x, mode='centrifugal', polypre=False):
    """ Calculates synapse flow centrality (SFC).

    From Schneider-Mizell et al. (2016): "We use flow centrality for
    four purposes. First, to split an arbor into axon and dendrite at the
    maximum centrifugal SFC, which is a preliminary step for computing the
    segregation index, for expressing all kinds of connectivity edges (e.g.
    axo-axonic, dendro-dendritic) in the wiring diagram, or for rendering the
    arbor in 3d with differently colored regions. Second, to quantitatively
    estimate the cable distance between the axon terminals and dendritic arbor
    by measuring the amount of cable with the maximum centrifugal SFC value.
    Third, to measure the cable length of the main dendritic shafts using
    centripetal SFC, which applies only to insect neurons with at least one
    output syn- apse in their dendritic arbor. And fourth, to weigh the color
    of each skeleton node in a 3d view, providing a characteristic signature of
    the arbor that enables subjective evaluation of its identity."

    Losely based on Alex Bate's implemention in `catnat
    <https://github.com/alexanderbates/catnat>`_.

    Catmaid uses the equivalent of ``mode='sum'`` and ``polypre=True``.

    Parameters
    ----------
    x :         CatmaidNeuron | CatmaidNeuronList
                Neuron(s) to calculate flow centrality for
    mode :      'centrifugal' | 'centripetal' | 'sum', optional
                Type of flow centrality to calculate. There are three flavors::
                (1) centrifugal, counts paths from proximal inputs to distal outputs
                (2) centripetal, counts paths from distal inputs to proximal outputs
                (3) the sum of both
    polypre :   bool, optional
                Whether to consider the number of presynapses as a multiple of
                the numbers of connections each makes. Attention: this works
                only if all synapses have been properly annotated (i.e. all
                postsynaptic sites).

    See Also
    --------
    :func:`~pymaid.bending_flow`
            Variation of flow centrality: calculates bending flow.
    :func:`~pymaid.segregation_index`
            Calculates segregation score (polarity) of a neuron
    :func:`~pymaid.flow_centrality_split`
            Tries splitting a neuron into axon, dendrite and primary neurite.


    Returns
    -------
    Adds a new column 'flow_centrality' to ``x.nodes``. Only processes
    branch- and synapse-holding nodes.

    """

    if mode not in ['centrifugal', 'centripetal', 'sum']:
        raise ValueError('Unknown parameter for mode: {0}'.format(mode))

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, '
                         'not {0}'.format(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        return [flow_centrality(n, mode=mode, polypre=polypre) for n in x]

    if x.soma and x.soma not in x.root:
        logger.warning(
            'Neuron {0} is not rooted to its soma!'.format(x.skeleton_id))

    # We will be processing a super downsampled version of the neuron to
    # speed up calculations
    current_level = logger.level
    current_state = config.pbar_hide
    logger.setLevel('ERROR')
    config.pbar_hide = True
    y = resample.downsample_neuron(x, float('inf'),
                                   inplace=False, preserve_cn_treenodes=True)
    logger.setLevel(current_level)
    config.pbar_hide = current_state

    if polypre:
        # Get details for all presynapses
        cn_details = fetch.get_connector_details(
            y.connectors[y.connectors.relation == 0])

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[y.connectors.relation ==
                                0].treenode_id.unique()
    post_node_ids = y.connectors[y.connectors.relation ==
                                 1].treenode_id.unique()
    total_pre = len(pre_node_ids)
    total_post = len(post_node_ids)

    # Get list of points to calculate flow centrality for:
    # branches and nodes with synapses
    calc_node_ids = y.nodes[(y.nodes.type == 'branch') | (
        y.nodes.treenode_id.isin(y.connectors.treenode_id))].treenode_id.values

    # Get number of pre/postsynapses distal to each branch's childs
    distal_pre = graph_utils.distal_to(y, pre_node_ids, calc_node_ids)
    distal_post = graph_utils.distal_to(y, post_node_ids, calc_node_ids)

    # Multiply columns (presynapses) by the number of postsynaptically
    # connected nodes
    if polypre:
        # Map vertex ID to number of postsynaptic nodes (avoid 0)
        distal_pre *= [max(1, len(cn_details[cn_details.presynaptic_to_node ==
                                             n].postsynaptic_to_node.sum())) for n in distal_pre.columns]
        # Also change total_pre as accordingly
        total_pre = sum([max(1, len(row))
                         for row in cn_details.postsynaptic_to_node.values])

    # Sum up axis - now each row represents the number of pre/postsynapses
    # that are distal to that node
    distal_pre = distal_pre.T.sum(axis=1)
    distal_post = distal_post.T.sum(axis=1)

    if mode != 'centripetal':
        # Centrifugal is the flow from all non-distal postsynapses to all
        # distal presynapses
        centrifugal = {
            n: (total_post - distal_post[n]) * distal_pre[n] for n in calc_node_ids}

    if mode != 'centrifugal':
        # Centripetal is the flow from all distal postsynapses to all
        # non-distal presynapses
        centripetal = {
            n: distal_post[n] * (total_post - distal_pre[n]) for n in calc_node_ids}

    # Now map this onto our neuron
    if mode == 'centrifugal':
        x.nodes['flow_centrality'] = x.nodes.treenode_id.map(centrifugal)
    elif mode == 'centripetal':
        x.nodes['flow_centrality'] = x.nodes.treenode_id.map(centripetal)
    elif mode == 'sum':
        combined = {n : centrifugal[n] + centripetal[n] for n in centrifugal}
        x.nodes['flow_centrality'] = x.nodes.treenode_id.map(combined)

    # Add info on method/mode used for flow centrality
    x.centrality_method = mode

    return


def stitch_neurons(*x, method='LEAFS', master='SOMA', tn_to_stitch=None):
    """ Stitch multiple neurons together.

    Uses minimum spanning tree to determine a way to connect all fragments
    while minimizing length (eucledian distance) of the new edges. Nodes
    that have been stitched will be get a "stitched" tag.

    Important
    ---------
    If duplicate node IDs are found across the fragments to stitch they will
    be remapped to new, unique values!

    Parameters
    ----------
    x :                 CatmaidNeurons | CatmaidNeuronList | list of either
                        Neurons to stitch (see examples).
    method :            'LEAFS' | 'ALL' | 'NONE', optional
                        Set stitching method:
                            (1) 'LEAFS': Only leaf (including root) nodes will
                                be allowed to make new edges.
                            (2) 'ALL': All treenodes are considered.
                            (3) 'NONE': Node and connector tables will simply
                                be combined without generating any new edges.
                                The resulting neuron will have multiple roots.
    master :            'SOMA' | 'LARGEST' | 'FIRST', optional
                        Sets the master neuron:
                            (1) 'SOMA': The largest fragment with a soma
                                becomes the master neuron. If no neuron with
                                soma, will pick the largest.
                            (2) 'LARGEST': The largest fragment becomes the
                                master neuron.
                            (3) 'FIRST': The first fragment provided becomes
                                the master neuron.
    tn_to_stitch :      List of treenode IDs, optional
                        If provided, these treenodes will be preferentially
                        used to stitch neurons together. Overrides methods
                        ``'ALL'`` or ``'LEAFS'``.

    Returns
    -------
    core.CatmaidNeuron
                        Stitched neuron.

    Examples
    --------
    Stitching using a neuron list:

    >>> nl = pymaid.get_neuron('annotation:glomerulus DA1 right')
    >>> stitched = pymaid.stitch_neurons(nl, method='NONE')

    Stitching using individual neurons:

    >>> a = pymaid.get_neuron(16)
    >>> b = pymaid.get_neuron(2863104)
    >>> stitched = pymaid.stitch_neurons(a, b, method='NONE')
    >>> # Or alternatively:
    >>> stitched = pymaid.stitch_neurons([a, b], method='NONE')

    """
    method = str(method).upper()
    master = str(master).upper()

    if method not in ['LEAFS', 'ALL', 'NONE']:
        raise ValueError('Unknown method: %s' % str(method))

    if master not in ['SOMA', 'LARGEST', 'FIRST']:
        raise ValueError('Unknown master: %s' % str(master))

    # Compile list of individual neurons
    x = utils._unpack_neurons(x)

    # Use copies of the original neurons!
    x = core.CatmaidNeuronList(x).copy()

    if len(x) < 2:
        logger.warning('Need at least 2 neurons to stitch, '
                       'found %i' % len(x))
        return x[0]

    # First find master
    if master == 'SOMA':
        has_soma = [n for n in x if not isinstance(n.soma, type(None))]
        if len(has_soma) > 0:
            master = has_soma[0]
        else:
            master = sorted(x.neurons,
                            key=lambda x: x.cable_length,
                            reverse=True)[0]
    elif master == 'LARGEST':
        master = sorted(x.neurons,
                        key=lambda x: x.cable_length,
                        reverse=True)[0]
    else:
        # Simply pick the first neuron
        master = x[0]

    # Check if we need to make any node IDs unique
    if x.nodes.duplicated(subset='treenode_id').sum() > 0:
        seen_tn = set(master.nodes.treenode_id)
        for n in [n for n in x if n != master]:
            this_tn = set(n.nodes.treenode_id)

            # Get duplicate node IDs
            non_unique = seen_tn & this_tn

            # Add this neuron's existing nodes to seen
            seen_tn = seen_tn | this_tn
            if non_unique:
                # Generate new, unique node IDs
                new_tn = np.arange(0, len(non_unique)) + max(seen_tn) + 1

                # Generate new map
                new_map = dict(zip(non_unique, new_tn))

                # Remap node IDs - if no new value, keep the old
                n.nodes.treenode_id = n.nodes.treenode_id.map(lambda x: new_map.get(x, x))
                n.connectors.treenode_id = n.connectors.treenode_id.map(lambda x: new_map.get(x, x))
                n.tags = {new_map.get(k, k): v for k, v in n.tags.items()}

                # Remapping parent IDs requires the root to be temporarily set
                # to -1. Otherwise the node IDs will become floats
                new_map[None] = -1
                n.nodes.parent_id = n.nodes.parent_id.map(lambda x: new_map.get(x, x)).astype(object)
                n.nodes.loc[n.nodes.parent_id == -1, 'parent_id'] = None

                # Add new nodes to seen
                seen_tn = seen_tn | set(new_tn)

                # Make sure the graph is updated
                n._clear_temp_attr()

    # If method is none, we can just merge the data tables
    if method == 'NONE' or method is None:
        master.nodes = pd.concat([n.nodes for n in x],
                                 ignore_index=True)
        master.connectors = pd.concat([n.connectors for n in x],
                                      ignore_index=True)
        master.tags = {}
        for n in x:
            master.tags.update(n.tags)

        # Reset temporary attributes of our final neuron
        master._clear_temp_attr()

        return master

    # Fix potential problems with tn_to_stitch
    if not isinstance(tn_to_stitch, type(None)):
        if not isinstance(tn_to_stitch, (list, np.ndarray)):
            tn_to_stitch = [tn_to_stitch]

        # Make sure we're working with integers
        tn_to_stitch = [int(tn) for tn in tn_to_stitch]

    # Generate a union of all graphs
    g = nx.union_all([n.graph for n in x]).to_undirected()

    # Set existing edges to zero weight to make sure they remain when
    # calculating the minimum spanning tree
    nx.set_edge_attributes(g, 0, 'weight')

    # If two nodes occupy the same position (e.g. if fragments are the
    # result of cutting), they will have a distance of 0. Hence, we won't be
    # able to simply filter by distance
    nx.set_edge_attributes(g, False, 'new')

    # Now iterate over every possible combination of fragments
    for a, b in itertools.combinations(x, 2):
        # Collect relevant treenodes
        if not isinstance(tn_to_stitch, type(None)):
            tnA = a.nodes.loc[a.nodes.treenode_id.isin(tn_to_stitch)]
            tnB = b.nodes.loc[a.nodes.treenode_id.isin(tn_to_stitch)]
        else:
            tnA = pd.DataFrame([])
            tnB = pd.DataFrame([])

        if tnA.empty:
            if method == 'LEAFS':
                tnA = a.nodes.loc[a.nodes['type'].isin(['end', 'root'])]
            else:
                tnA = a.nodes
        if tnB.empty:
            if method == 'LEAFS':
                tnB = b.nodes.loc[b.nodes['type'].isin(['end', 'root'])]
            else:
                tnB = b.nodes

        # Get distance between treenodes in A and B
        d = scipy.spatial.distance.cdist(tnA[['x', 'y', 'z']].values,
                                         tnB[['x', 'y', 'z']].values,
                                         metric='euclidean')

        # List of edges
        tn_comb = itertools.product(tnA.treenode_id.values, tnB.treenode_id.values)

        # Add edges to union graph
        g.add_edges_from([(a, b, {'weight': w, 'new': True}) for (a, b), w in zip(tn_comb, d.ravel())])

    # Get minimum spanning tree
    edges = nx.minimum_spanning_edges(g)

    # Edges that need adding are those with weight > 0
    to_add = [e for e in edges if e[2]['new']]

    # Keep track of original master root
    master_root = master.root[0]

    # Generate one big neuron
    master.nodes = x.nodes
    master.connectors = x.connectors
    for n in x:
        master.tags.update(n.tags)
    master._clear_temp_attr()

    for e in to_add:
        # Reroot to one of the nodes in the edge
        master.reroot(e[0], inplace=True)

        # Connect the nodes
        master.nodes.loc[master.nodes.treenode_id == e[0], 'parent_id'] = e[1]

        # Add node tags
        master.tags['stitched'] = master.tags.get('stitched', []) + list(e[0:2])

        # We need to regenerate the graph
        master._clear_temp_attr()

    # Reroot to original root
    master.reroot(master_root, inplace=True)

    return master


def average_neurons(x, limit=10, base_neuron=None):
    """ Computes an average from a list of neurons.

    This is a very simple implementation which may give odd results if used
    on complex neurons. Works fine on e.g. backbones or tracts.

    Parameters
    ----------
    x :             CatmaidNeuronList
                    Neurons to be averaged.
    limit :         int, optional
                    Max distance for nearest neighbour search. In microns.
    base_neuron :   skeleton_ID | CatmaidNeuron, optional
                    Neuron to use as template for averaging. If not provided,
                    the first neuron in the list is used as template!

    Returns
    -------
    CatmaidNeuron

    Examples
    --------
    >>> # Get a bunch of neurons
    >>> da1 = pymaid.get_neurons('annotation:glomerulus DA1 right')
    >>> # Prune down to longest neurite
    >>> da1.reroot(da1.soma)
    >>> da1_pr = da1.prune_by_longest_neurite(inplace=False)
    >>> # Make average
    >>> da1_avg = pymaid.average_neurons(da1_pr)
    >>> # Plot
    >>> da1.plot3d()
    >>> da1_avg.plot3d()

    """

    if not isinstance(x, core.CatmaidNeuronList):
        raise TypeError('Need CatmaidNeuronList, got "{0}"'.format(type(x)))

    if len(x) < 2:
        raise ValueError('Need at least 2 neurons to average!')

    # Generate KDTrees for each neuron
    for n in x:
        n.tree = graph.neuron2KDTree(n, tree_type='c', data='treenodes')

    # Set base for average: we will use this neurons treenodes to query
    # the KDTrees
    if isinstance(base_neuron, core.CatmaidNeuron):
        base_neuron = base_neuron.copy()
    elif isinstance(base_neuron, int):
        base_neuron = x.skid[base_neuron].copy
    elif isinstance(base_neuron, type(None)):
        base_neuron = x[0].copy()
    else:
        raise ValueError('Unable to interpret base_neuron of '
                         'type "{0}"'.format(type(base_neuron)))

    base_nodes = base_neuron.nodes[['x', 'y', 'z']].values
    other_neurons = x[1:]

    # Make sure these stay 2-dimensional arrays -> will add a colum for each
    # "other" neuron
    base_x = base_nodes[:, 0:1]
    base_y = base_nodes[:, 1:2]
    base_z = base_nodes[:, 2:3]

    # For each "other" neuron, collect nearest neighbour coordinates
    for n in other_neurons:
        nn_dist, nn_ix = n.tree.query(
            base_nodes, k=1, distance_upper_bound=limit * 1000)

        # Translate indices into coordinates
        # First, make empty array
        this_coords = np.zeros((len(nn_dist), 3))
        # Set coords without a nearest neighbour within distances to "None"
        this_coords[nn_dist == float('inf')] = None
        # Fill in coords of nearest neighbours
        this_coords[nn_dist != float(
            'inf')] = n.tree.data[nn_ix[nn_dist != float('inf')]]
        # Add coords to base coords
        base_x = np.append(base_x, this_coords[:, 0:1], axis=1)
        base_y = np.append(base_y, this_coords[:, 1:2], axis=1)
        base_z = np.append(base_z, this_coords[:, 2:3], axis=1)

    # Calculate means
    mean_x = np.mean(base_x, axis=1)
    mean_y = np.mean(base_y, axis=1)
    mean_z = np.mean(base_z, axis=1)

    # If any of the base coords has NO nearest neighbour within limit
    # whatsoever, the average of that row will be "NaN" -> in this case we
    # will fall back to the base coordinate
    mean_x[np.isnan(mean_x)] = base_nodes[np.isnan(mean_x), 0]
    mean_y[np.isnan(mean_y)] = base_nodes[np.isnan(mean_y), 1]
    mean_z[np.isnan(mean_z)] = base_nodes[np.isnan(mean_z), 2]

    # Change coordinates accordingly
    base_neuron.nodes.loc[:, 'x'] = mean_x
    base_neuron.nodes.loc[:, 'y'] = mean_y
    base_neuron.nodes.loc[:, 'z'] = mean_z

    return base_neuron


def tortuosity(x, seg_length=10, skip_remainder=False):
    """ Calculates tortuosity for a neurons.

    See Stepanyants et al., Neuron (2004) for detailed explanation. Briefly,
    tortuosity index `T` is defined as the ratio of the branch segment length
    `L` (``seg_length``) to the eucledian distance `R` between its ends.

    Note
    ----
    If you want to make sure that segments are as close to length `L` as
    possible, consider resampling the neuron using :func:`pymaid.resample`.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
    seg_length :        int | float | list, optional
                        Target segment length(s) L in microns [um]. Will try
                        resampling neuron to this resolution. Please note that
                        the final segment length is restricted by the neuron's
                        original resolution.
    skip_remainder :    bool, optional
                        Segments can turn out to be smaller than desired if a
                        branch point or end point is hit before `seg_length`
                        is reached. If ``skip_remainder`` is True, these will
                        be ignored.

    Returns
    -------
    tortuosity :        float | np.array | pandas.DataFrame
                        If x is CatmaidNeuronList, will return DataFrame.
                        If x is single CatmaidNeuron, will return either a
                        single float (if single seg_length is queried) or a
                        DataFrame (if multiple seg_lengths are queried).

    """

    # TODO:
    # - try as angles between dotproduct vectors
    #

    if isinstance(x, core.CatmaidNeuronList):
        if not isinstance(seg_length, (list, np.ndarray, tuple)):
            seg_length = [seg_length]
        df = pd.DataFrame([tortuosity(n, seg_length) for n in config.tqdm(x, desc='Tortuosity', disable=config.pbar_hide, leave=config.pbar_leave)],
                          index=x.skeleton_id, columns=seg_length).T
        df.index.name = 'seg_length'
        return df

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Need CatmaidNeuron, got {0}'.format(type(x)))

    if isinstance(seg_length, (list, np.ndarray)):
        return [tortuosity(x, l) for l in seg_length]

    if seg_length <= 0:
        raise ValueError('Segment length must be >0.')

    # We will collect coordinates and do distance calculations later
    start_tn = []
    end_tn = []
    L = []

    # Go over all segments
    for seg in x.small_segments:
        # Collect distances between treenodes (in microns)
        dist = np.array([x.graph.edges[(c, p)]['weight']
                         for c, p in zip(seg[:-1], seg[1:])]) / 1000
        # Walk the segment, collect stretches of length `seg_length`
        cut_ix = [0]
        for i, tn in enumerate(seg):
            if sum(dist[cut_ix[-1]:i]) > seg_length:
                cut_ix.append(i)

        # If the last node is not a cut node
        if cut_ix[-1] < i and not skip_remainder:
            cut_ix.append(i)

        # Translate into treenode IDs
        if len(cut_ix) > 1:
            L += [sum(dist[s:e]) for s, e in zip(cut_ix[:-1], cut_ix[1:])]
            start_tn += [seg[n] for n in cut_ix[:-1]]
            end_tn += [seg[n] for n in cut_ix[1:]]

    # Now calculate euclidean distances
    tn_table = x.nodes.set_index('treenode_id')
    start_co = tn_table.loc[start_tn, ['x', 'y', 'z']].values
    end_co = tn_table.loc[end_tn, ['x', 'y', 'z']].values
    R = np.linalg.norm(start_co - end_co, axis=1) / 1000

    # Get tortousity
    T = np.array(L) / R

    return T.mean()


def remove_tagged_branches(x, tag, how='segment', preserve_connectors=False,
                           inplace=False):
    """ Removes branches from neuron(s) that have been tagged with a given
    treenode tag (e.g. ``not a branch``).

    Parameters
    ----------
    x :                   CatmaidNeuron | CatmaidNeuronList
                          Neuron(s) to be processed.
    tag :                 str
                          Treeode tag to use.
    how :                 'segment' | 'distal' | 'proximal', optional
                          Method of removal:
                            1. ``segment`` removes entire segment
                            2. ``distal``/``proximal`` removes everything
                               distal/proximal to tagged node(s), including
                               that node.
    preserve_connectors : bool, optional
                          If True, connectors that got disconnected during
                          branch removal will be reattached to the closest
                          surviving downstream node.
    inplace :             bool, optional
                          If False, a copy of the neuron is returned.

    Returns
    -------
    CatmaidNeuron/List
                           Pruned neuron(s). Only if ``inplace=False``.

    Examples
    --------
    1. Remove not-a-branch terminal branches

    >>> x = pymaid.get_neuron(16)
    >>> x_prun = pymaid.remove_tagged_branches(x,
    ...                                        'not a branch',
    ...                                        how='segment',
    ...                                        preserve_connectors=True)

    2. Prune neuron to microtubule-containing backbone

    >>> x_prun = pymaid.remove_tagged_branches(x,
    ...                                        'microtubule ends',
    ...                                        how='distal',
    ...                                        preserve_connectors=False)


    """

    def _find_next_remaining_parent(tn):
        """ Helper function that walks from a treenode to the neurons root and
        returns the first parent that will not be removed.
        """
        this_nodes = x.nodes.set_index('treenode_id')
        while True:
            this_parent = this_nodes.loc[tn, 'parent_id']
            if this_parent not in to_remove:
                return tn
            tn = this_parent

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Removing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            remove_tagged_branches(n, tag,
                                   how=how,
                                   preserve_connectors=preserve_connectors,
                                   inplace=True)

        if not inplace:
            return x
    elif not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process CatmaidNeuron or CatmaidNeuronList, '
                        'not "{0}"'.format(type(x)))

    # Check if method is valid
    VALID_METHODS = ['segment', 'distal', 'proximal']
    if how not in VALID_METHODS:
        raise ValueError('Invalid value for "how": '
                         '{0}. Valid methods are: {1}'.format(how,
                                                              ', '.join(VALID_METHODS)))

    # Skip if tag not present
    if tag not in x.tags:
        logger.info(
            'No "{0}" tag found on neuron #{1}... skipped'.format(tag, x.skeleton_id))
        if not inplace:
            return x
        return

    if not inplace:
        x = x.copy()

    tagged_nodes = set(x.tags[tag])

    if how == 'segment':
        # Find segments that have a tagged node
        tagged_segs = [s for s in x.small_segments if set(s) & tagged_nodes]

        # Sanity check: are any of these segments non-terminals?
        non_term = [s for s in tagged_segs if x.graph.degree(s[0]) > 1]
        if non_term:
            logger.warning(
                'Pruning {0} non-terminal segment(s)'.format(len(non_term)))

        # Get nodes to be removed (excluding the last node -> branch )
        to_remove = set([t for s in tagged_segs for t in s[:-1]])

        # Rewire connectors before we subset
        if preserve_connectors:
            # Get connectors that will be disconnected
            lost_cn = x.connectors[x.connectors.treenode_id.isin(to_remove)]

            # Map to a remaining treenode
            # IMPORTANT: we do currently not account for the possibility that
            # we might be removing the root segment
            new_tn = [_find_next_remaining_parent(tn) for tn in lost_cn.treenode_id.values]
            x.connectors.loc[x.connectors.treenode_id.isin(to_remove), 'treenode_id'] = new_tn

        # Subset to remaining nodes - skip the last node in each segment
        graph_utils.subset_neuron(x,
                                  subset=x.nodes[~x.nodes.treenode_id.isin(
                                      to_remove)].treenode_id.values,
                                  keep_disc_cn=preserve_connectors,
                                  inplace=True)

        if not inplace:
            return x
        return

    elif how in ['distal', 'proximal']:
        # Keep pruning until no more treenodes with our tag are left
        while tag in x.tags:
            # Find nodes distal to this tagged node (includes the tagged node)
            dist_graph = nx.bfs_tree(x.graph, x.tags[tag][0], reverse=True)

            if how == 'distal':
                to_remove = list(dist_graph.nodes)
            elif how == 'proximal':
                # Invert dist_graph
                to_remove = x.nodes[~x.nodes.treenode_id.isin(
                    dist_graph.nodes)].treenode_id.values
                # Make sure the tagged treenode is there too
                to_remove += [x.tags[tag][0]]

            to_remove = set(to_remove)

            # Rewire connectors before we subset
            if preserve_connectors:
                # Get connectors that will be disconnected
                lost_cn = x.connectors[x.connectors.treenode_id.isin(
                    to_remove)]

                # Map to a remaining treenode
                # IMPORTANT: we do currently not account for the possibility
                # that we might be removing the root segment
                new_tn = [_find_next_remaining_parent(tn) for tn in lost_cn.treenode_id.values]
                x.connectors.loc[x.connectors.treenode_id.isin(to_remove), 'treenode_id'] = new_tn

            # Subset to remaining nodes
            graph_utils.subset_neuron(x,
                                      subset=x.nodes[~x.nodes.treenode_id.isin(
                                          to_remove)].treenode_id.values,
                                      keep_disc_cn=preserve_connectors,
                                      inplace=True)

        if not inplace:
            return x
        return


def despike_neuron(x, sigma=5, max_spike_length=1, inplace=False,
                   reverse=False):
    """ Removes spikes in neuron traces (e.g. from jumps in image data).

    For each treenode A, the euclidean distance to its next successor (parent)
    B and the second next successor is computed. If
    :math:`\\frac{dist(A,B)}{dist(A,C)}>sigma`. node B is considered a spike
    and realigned between A and C.

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) to be processed.
    sigma :             float | int, optional
                        Threshold for spike detection. Smaller sigma = more
                        aggressive spike detection.
    max_spike_length :  int, optional
                        Determines how long (# of nodes) a spike can be.
    inplace :           bool, optional
                        If False, a copy of the neuron is returned.
    reverse :           bool, optional
                        If True, will **also** walk the segments from proximal
                        to distal. Use this to catch spikes on e.g. terminal
                        nodes.

    Returns
    -------
    CatmaidNeuron/List
                Despiked neuron(s). Only if ``inplace=False``.

    """

    # TODO:
    # - flattening all segments first before Spike detection should speed up
    #   quite a lot
    # -> as intermediate step: assign all new positions at once

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Despiking', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            despike_neuron(n, sigma=sigma, inplace=True)

        if not inplace:
            return x
        return
    elif not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process CatmaidNeuron or CatmaidNeuronList, '
                        'not "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    # Index treenodes table by treenode ID
    this_treenodes = x.nodes.set_index('treenode_id')

    segs_to_walk = x.segments

    if reverse:
        segs_to_walk += x.segments[::-1]

    # For each spike length do -> do this in reverse to correct the long
    # spikes first
    for l in list(range(1, max_spike_length + 1))[::-1]:
        # Go over all segments
        for seg in x.segments:
            # Get nodes A, B and C of this segment
            this_A = this_treenodes.loc[seg[:-l - 1]]
            this_B = this_treenodes.loc[seg[l:-1]]
            this_C = this_treenodes.loc[seg[l + 1:]]

            # Get coordinates
            A = this_A[['x', 'y', 'z']].values
            B = this_B[['x', 'y', 'z']].values
            C = this_C[['x', 'y', 'z']].values

            # Calculate euclidian distances A->B and A->C
            dist_AB = np.linalg.norm(A - B, axis=1)
            dist_AC = np.linalg.norm(A - C, axis=1)

            # Get the spikes
            spikes_ix = np.where((dist_AB / dist_AC) > sigma)[0]
            spikes = this_B.iloc[spikes_ix]

            if not spikes.empty:
                # Interpolate new position(s) between A and C
                new_positions = A[spikes_ix] + (C[spikes_ix] - A[spikes_ix]) / 2

                this_treenodes.loc[spikes.index, ['x', 'y', 'z']] = new_positions

    # Reassign treenode table
    x.nodes = this_treenodes.reset_index(drop=False)

    # The weights in the graph have changed, we need to update that
    x._clear_temp_attr(exclude=['segments', 'small_segments',
                                'classify_nodes'])

    if not inplace:
        return x


def guess_radius(x, method='linear', limit=None, smooth=True, inplace=False):
    """ Tries guessing radii for all treenodes.

    Uses distance between connectors and treenodes and interpolate for all
    treenodes. Fills in ``radius`` column in treenode table.

    Parameters
    ----------
    x :             CatmaidNeuron | CatmaidNeuronList
                    Neuron(s) to be processed.
    method :        str, optional
                    Method to be used to interpolate unknown radii. See
                    ``pandas.DataFrame.interpolate`` for details.
    limit :         int, optional
                    Maximum number of consecutive missing radii to fill.
                    Must be greater than 0.
    smooth :        bool | int, optional
                    If True, will smooth radii after interpolation using a
                    rolling window. If ``int``, will use to define size of
                    window.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    CatmaidNeuron/List
                    If ``inplace=False``.

    """

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Guessing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            guess_radius(n, method=method, limit=limit, smooth=smooth,
                         inplace=True)

        if not inplace:
            return x
        return

    elif not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process CatmaidNeuron or CatmaidNeuronList, '
                        'not "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    # Set default rolling window size
    if isinstance(smooth, bool) and smooth:
        smooth = 5

    # We will be using the index as distance to interpolate. For this we have
    # to change method 'linear' to 'index'
    method = 'index' if method == 'linear' else method

    # Collect connectors and calc distances
    cn = x.connectors.copy()

    # Prepare nodes (add parent_dist for later, set index)
    _parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('treenode_id')

    # For each connector (pre and post), get the X/Y distance to its treenode
    cn_locs = cn[['x', 'y']].values
    tn_locs = nodes.loc[cn.treenode_id.values,
                        ['x', 'y']].values
    dist = np.sqrt(np.sum((tn_locs - cn_locs) ** 2, axis=1).astype(int))
    cn['dist'] = dist

    # Get max distance per treenode (in case of multiple connectors per
    # treenode)
    cn_grouped = cn.groupby('treenode_id').max()

    # Set undefined radii to None
    nodes.loc[nodes.radius <= 0, 'radius'] = None

    # Assign radii to treenodes
    nodes.loc[cn_grouped.index, 'radius'] = cn_grouped.dist.values

    # Go over each segment and interpolate radii
    for s in config.tqdm(x.segments, desc='Interp.', disable=config.pbar_hide,
                         leave=config.pbar_leave):

        # Get this segments radii and parent dist
        this_radii = nodes.loc[s, ['radius', 'parent_dist']]
        this_radii['parent_dist_cum'] = this_radii.parent_dist.cumsum()

        # Set cumulative distance as index and drop parent_dist
        this_radii = this_radii.set_index('parent_dist_cum',
                                          drop=True).drop('parent_dist',
                                                          axis=1)

        # Interpolate missing radii
        interp = this_radii.interpolate(method=method, limit_direction='both',
                                        limit=limit)

        if smooth:
            interp = interp.rolling(smooth,
                                    min_periods=1).max()

        nodes.loc[s, 'radius'] = interp.values

    # Set non-interpolated radii back to -1
    nodes.loc[nodes.radius.isnull(), 'radius'] = -1

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False)

    if not inplace:
        return x


def smooth_neuron(x, window=5, inplace=False):
    """ Smooth neuron using rolling windows.

    Parameters
    ----------
    x :             CatmaidNeuron | CatmaidNeuronList
                    Neuron(s) to be processed.
    window :        int, optional
                    Size of the rolling window in number of nodes.
    inplace :       bool, optional
                    If False, will use and return copy of original neuron(s).

    Returns
    -------
    CatmaidNeuron/List
                    Smoothed neuron(s). If ``inplace=False``.

    """

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()

        for n in config.tqdm(x, desc='Smoothing', disable=config.pbar_hide,
                             leave=config.pbar_leave):
            smooth_neuron(n, window=window, inplace=True)

        if not inplace:
            return x
        return

    elif not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process CatmaidNeuron or CatmaidNeuronList, '
                        'not "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    # Prepare nodes (add parent_dist for later, set index)
    _parent_dist(x, root_dist=0)
    nodes = x.nodes.set_index('treenode_id')

    # Go over each segment and interpolate radii
    for s in config.tqdm(x.segments, desc='Smoothing',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave):

        # Get this segments radii and parent dist
        this_co = nodes.loc[s, ['x', 'y', 'z', 'parent_dist']]
        this_co['parent_dist_cum'] = this_co.parent_dist.cumsum()

        # Set cumulative distance as index and drop parent_dist
        this_co = this_co.set_index('parent_dist_cum',
                                    drop=True).drop('parent_dist', axis=1)

        interp = this_co.rolling(window, min_periods=1).mean()

        nodes.loc[s, ['x', 'y', 'z']] = interp.values

    # Reassign nodes
    x.nodes = nodes.reset_index(drop=False)

    x._clear_temp_attr()

    if not inplace:
        return x


def time_machine(x, target, inplace=False, remote_instance=None):
    """ Reverses time and make neurons young again!

    Prunes a neuron back to it's state at a given date. Here is what we can
    reverse:

    1. Creation and deletion of nodes
    2. Creation and deletion of connectors (and links)
    3. Movement of nodes and connectors
    4. Cuts and merges
    5. Addition of node tags & annotations (even deleted ones)

    Unfortunately time travel has not yet been perfected. We are oblivious to:

    1. Removal of tags/annotations: i.e. we know when e.g. a tag was added
       and that it was subsequently removed at some point but not when.

    Parameters
    ----------
    x :                 skeleton ID(s) | CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) to rejuvenate.
    target :            str| tuple | datetime | pandas.Timestamp
                        Date or date + time to time-travel to.
    inplace :           bool, optional
                        If True, will perform time travel on and return a copy
                        of original.
    remote_instance :   CatmaidInstance, optional

    Returns
    -------
    CatmaidNeuron/List
                        A younger version of the neuron(s).

    Examples
    --------
    >>> n = pymaid.get_neuron(16)
    >>> previous_n = pymaid.time_machine(n, '2016-1-1')
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, core.CatmaidNeuronList):
        return core.CatmaidNeuronList([time_machine(n, target,
                                                    inplace=inplace,
                                                    remote_instance=remote_instance)
                                  for n in config.tqdm(x,
                                                       'Rejuvenating',
                                                       disable=config.pbar_hide,
                                                       leave=config.pbar_leave)])

    if not isinstance(x, core.CatmaidNeuron):
        x = fetch.get_neuron(x)

    if not inplace:
        x = x.copy()

    if not isinstance(target, pd.Timestamp):
        target = pd.Timestamp(target)

    # Need to localize all timestamps
    target = target.tz_localize('UTC')

    if target > pd.Timestamp.now().tz_localize('UTC'):
        raise ValueError("This is not Back to the Future II: for forward time "
                         "travel, you'll have to trace yourself.")

    # First get the entire history of the neuron
    url = remote_instance._get_compact_details_url(x.skeleton_id,
                                                   with_history=True,
                                                   with_merge_history=True,
                                                   with_connectors=True,
                                                   with_tags=True,
                                                   with_annotations=True)
    data = remote_instance.fetch(url)

    # Turn stuff into DataFrames for easier sifting/sorting
    nodes = pd.DataFrame(data[0], columns=['treenode_id', 'parent_id',
                                           'user_id', 'x', 'y', 'z', 'radius',
                                           'confidence', 'creation_timestamp',
                                           'modified_timestamp', 'ordering_by'])
    nodes.loc[:, 'parent_id'] = nodes.parent_id.values.astype(object)
    nodes.loc[~nodes.parent_id.isnull(), 'parent_id'] = nodes.loc[~nodes.parent_id.isnull(), 'parent_id'].map(int)
    nodes.loc[nodes.parent_id.isnull(), 'parent_id'] = None

    connectors = pd.DataFrame(data[1], columns=['treenode_id', 'connector_id',
                                                'relation', 'x', 'y', 'z',
                                                'creation_timestamp',
                                                'modified_timestamp'])
    # This is a dictionary with {'tag': [[treenode_id, date_tagged], ...]}
    tags = data[2]
    annotations = pd.DataFrame(data[4], columns=['annotation_id',
                                                 'annotated_timestamp'])
    an_list = fetch.get_annotation_list(remote_instance=remote_instance)
    annotations['annotation'] = annotations.annotation_id.map(an_list.set_index('annotation_id').annotation.to_dict())

    # Convert stuff to timestamps
    for ts in ['creation_timestamp', 'modified_timestamp']:
        nodes[ts] = nodes[ts].map(pd.Timestamp)
        connectors[ts] = connectors[ts].map(pd.Timestamp)
    annotations['annotated_timestamp'] = annotations['annotated_timestamp'].map(pd.Timestamp)
    tags = {t: [[e[0], pd.Timestamp(e[1])] for e in tags[t]] for t in tags}

    # General rules:
    # 1. creation_timestamp and modified timestamp represent a validity
    #    intervals.
    # 2. Nodes where creation_timestamp is older than modified_timestamp,
    #    represent the existing, most up-to-date versions.
    # 3. Nodes with creation_timestamp younger than modified_timestamp, and
    #    with NO future version of themselves, got cut off/deleted at
    #    modification time.
    # 4. Useful little detail: nodes/connectors are ordered by new -> old

    # First change the modified_timestamp for nodes that still exist
    # (see rule 2) to right now
    nodes.loc[nodes.creation_timestamp > nodes.modified_timestamp, 'modified_timestamp'] = pd.Timestamp.now().tz_localize('UTC')
    connectors.loc[connectors.creation_timestamp > connectors.modified_timestamp, 'modified_timestamp'] = pd.Timestamp.now().tz_localize('UTC')

    # Remove nodes without a window (these seems to be temporary states)
    nodes = nodes[nodes.creation_timestamp != nodes.modified_timestamp]
    connectors = connectors[connectors.creation_timestamp != connectors.modified_timestamp]

    # Second subset to versions of the nodes that existed at given time
    before_nodes = nodes[(nodes.creation_timestamp <= target) & (nodes.modified_timestamp >= target)]
    before_connectors = connectors[(connectors.creation_timestamp <= target) & (connectors.modified_timestamp >= target)]

    # Now fix tags and annotations
    before_annotations = annotations[annotations.annotated_timestamp <= target]
    before_tags = {t: [e[0] for e in tags[t] if e[1] <= target] for t in tags}
    before_tags = {t: before_tags[t] for t in before_tags if before_tags[t]}

    x.nodes = before_nodes
    x.connectors = before_connectors
    x.annotations = before_annotations
    x.tags = before_tags

    # We might end up with multiple disconnected pieces - I don't yet know why
    x.nodes.loc[~x.nodes.parent_id.isin(x.nodes.treenode_id), 'parent_id'] = None

    # If there is more than one root, we have to remove the disconnected
    # pieces and keep only the "oldest branch".
    # The theory for doing this is: if a node shows up as "root" and the very
    # next step is that it is a child to another node, we should consider
    # it a not-yet connected branch that needs to be removed.
    roots = x.nodes[x.nodes.parent_id.isnull()].treenode_id.tolist()
    if len(roots) > 1:
        after_nodes = nodes[nodes.modified_timestamp > target]
        for r in roots:
            # Find the next version of this node
            nv = after_nodes[(after_nodes.treenode_id == r)]
            # If this node is not a root anymore in its next iteration, it's
            # not the "real" one
            if not nv.empty and nv.iloc[-1].parent_id is not None:
                roots.remove(r)

        # Get disconnected components
        g = graph.neuron2nx(x)
        subgraphs = [l for l in nx.connected_components(nx.to_undirected(g))]

        # If we have a winner root, keep the bit that it is part of
        if len(roots) == 1:
            keep = [l for l in subgraphs if roots[0] in l][0]
        # If we have multiple winners (unlikely) or none (e.g. if the "real"
        # root got rerooted too) go for the biggest branch
        else:
            keep = sorted(subgraphs, key=lambda x: len(x), reverse=True)[0]

        x.nodes = x.nodes[x.nodes.treenode_id.isin(keep)].copy()

    # Remove connectors where the treenode does not even exist yet
    x.connectors = x.connectors[x.connectors.treenode_id.isin(x.nodes.treenode_id)]

    # Take care of connectors where the treenode might exist but was not yet linked
    links = fetch.get_connector_links(x.skeleton_id)
    localize = lambda x: pd.Timestamp.tz_localize(x, 'UTC')
    links.creation_time = links.creation_time.map(localize)
    links = links[links.creation_time <= target]
    links['connector_id'] = links.connector_id.astype(int)
    links['treenode_id'] = links.treenode_id.astype(int)

    # Get connector ID -> treenode combinations
    l = links[['connector_id', 'treenode_id']].T.apply(tuple)
    c = x.connectors[['connector_id', 'treenode_id']].T.apply(tuple)

    # Keep only those where connector->treenode connection is present
    x.connectors = x.connectors[c.isin(l)]

    x._clear_temp_attr()

    if not inplace:
        return x

    return


def break_fragments(x):
    """ Break neuron into continuous fragments.

    Neurons can consists of several disconnected fragments. This function
    breaks neuron(s) into disconnected components.

    Parameters
    ----------
    x :         CatmaidNeuron
                Fragmented neuron.

    Returns
    -------
    CatmaidNeuronList

    See Also
    --------
    :func:`pymaid.heal_fragmented_neuron`
                Use to heal fragmentation instead of breaking it up.

    """
    if isinstance(x, core.CatmaidNeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron/List, got "{}"'.format(type(x)))


    # Don't do anything if not actually fragmented
    if x.n_skeletons > 1:
        # Get connected components
        comp = list(nx.connected_components(x.graph.to_undirected()))
        # Sort so that the first component is the largest
        comp = sorted(comp, key=len, reverse=True)

        return core.CatmaidNeuronList([graph_utils.subset_neuron(x,
                                                                 list(ss),
                                                                 inplace=False)
                                                               for ss in comp])
    else:
        return core.CatmaidNeuronList(x.copy())


def heal_fragmented_neuron(x, min_size=0, method='LEAFS', inplace=False):
    """ Heal fragmented neuron(s).

    Tries to heal a fragmented neuron (i.e. a neuron with multiple roots)
    using a minimum spanning tree.

    Parameters
    ----------
    x :         CatmaidNeuron/List
                Fragmented neuron(s).
    min_size :  int, optional
                Minimum size in nodes for fragments to be reattached.
    method :    'LEAFS' | 'ALL', optional
                Method used to heal fragments:
                        (1) 'LEAFS': Only leaf (including root) nodes will
                            be used to heal gaps.
                        (2) 'ALL': All treenodes can be used to reconnect
                            fragments.
    inplace :   bool, optional
                If False, will perform healing on and return a copy.

    Returns
    -------
    None
                If ``inplace=True``
    CatmaidNeuron/List
                If ``inplace=False``


    See Also
    --------
    :func:`pymaid.stitch_neurons`
                Function used by ``heal_fragmented_neuron`` to stitch
                fragments.
    :func:`pymaid.break_fragments`
                Use to break a fragmented neuron into disconnected pieces.
    """

    method = str(method).upper()

    if method not in ['LEAFS', 'ALL']:
        raise ValueError('Unknown method "{}"'.format(method))

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()
        healed = [heal_fragmented_neuron(n, min_size=min_size, method=method,
                                         inplace=True)
                                for n in config.tqdm(x,
                                                     desc='Healing',
                                                     disable=config.pbar_hide,
                                                     leave=config.pbar_leave)]
        if not inplace:
            return x
        return

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Expected CatmaidNeuron/List, got "{}"'.format(type(x)))

    # Don't do anything if not actually fragmented
    if x.n_skeletons > 1:
        frags = break_fragments(x)
        healed = stitch_neurons(*[f for f in frags if f.n_nodes > min_size],
                                method=method)
        if not inplace:
            return healed
        else:
            x.nodes = healed.nodes #update nodes
            x.tags = healed.tags #update tags
            x._clear_temp_attr()
    elif not inplace:
        return x

