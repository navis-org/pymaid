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

""" This module contains functions to analyse and manipulate neuron morphology.
"""

import itertools
import navis

import pandas as pd
import numpy as np
import networkx as nx

from navis import graph_utils, graph
from . import fetch, core, utils, config

# Set up logging
logger = config.logger

__all__ = sorted(['arbor_confidence', 'remove_tagged_branches', 'time_machine',
                  'union_neurons', 'prune_by_length'])


def arbor_confidence(x, confidences=(1, 0.9, 0.6, 0.4, 0.2), inplace=True):
    """Calculate along-the-arbor confidence for each treenode.

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
            _ = [arbor_confidence(n,
                                  confidences=confidences,
                                  inplace=inplace) for n in x]
        else:
            return core.CatmaidNeuronList([arbor_confidence(n,
                                                            confidences=confidences,
                                                            inplace=inplace) for n in x])

    if not inplace:
        x = x.copy()

    loc = graph_utils.generate_list_of_childs(x)

    x.nodes['arbor_confidence'] = [None] * x.nodes.shape[0]

    nodes = x.nodes.set_index('node_id')
    nodes.loc[x.root, 'arbor_confidence'] = 1

    with config.tqdm(total=len(x.segments), desc='Calc confidence', disable=config.pbar_hide, leave=config.pbar_leave) as pbar:
        for r in x.root:
            for c in loc[r]:
                walk_to_leafs(c)

    x.nodes['arbor_confidence'] = nodes['arbor_confidence'].values

    if not inplace:
        return x


def union_neurons(*x, limit=1, base_neuron=None, track=False, non_overlap='raise'):
    """Generate the union of a set of neurons.

    This implementation works by iteratively merging nodes in neuron A and B
    that are closer than given threshold. This requires neurons to have a
    certain amount of overlap.

    Parameters
    ----------
    *x :            CatmaidNeuron/List
                    Neurons to be merged.
    limit :         int, optional
                    Max distance [microns] for nearest neighbour search.
    base_neuron :   skeleton_ID | CatmaidNeuron, optional
                    Neuron to use as template for union. Node IDs of this
                    neuron will survive. If not provided, the first neuron
                    in the list is used as template!
    track :         bool, optional
                    If True, will add new columns to node/connector table of
                    union neuron to keep track of original node IDs and origin:
                    `node_id_before`, `parent_id_before`, `origin_skeleton`
    non_overlap :   "raise" | "stitch" | "skip", optional
                    Determines how to deal with non-overlapping fragments. If
                    "raise" will raise an exception. If "stitch" will try
                    stitching the fragments using a minimum spanning tree
                    (see :func:`pymaid.stitch_neurons`).


    Returns
    -------
    core.CatmaidNeuron
                    Union of all input neurons.

    See Also
    --------
    :func:`~pymaid.stitch_neurons`
                    If you want to stitch neurons that do not overlap.

    Examples
    --------
    >>> # Get a single neuron
    >>> n = pymaid.get_neuron(16)
    >>> # Prune to its longest neurite
    >>> backbone = n.prune_by_longest_neurite(inplace=False)
    >>> # Remove longest neurite and keep only fine branches
    >>> branches = n.prune_by_longest_neurite(n=slice(1, None), inplace=False)
    >>> # For this exercise we have to make sure skeleton IDs are unique
    >>> branches.skeleton_id = 17
    >>> # Now put both back together using union
    >>> union = pymaid.union_neurons(backbone, branches, limit=2)

    """
    allowed = ['raise', 'stitch', 'skip']
    if non_overlap.lower() not in allowed:
        msg = 'Unexpected value for non_overlap "{}". Please use either:'
        msg = msg.format(non_overlap, '"{}", '.join(allowed))
        raise ValueError(msg)

    # Unpack neurons in *args
    x = utils._unpack_neurons(x)

    # Make sure we're working on copies and don't change originals
    x = [n.copy() for n in x]

    # This is just check on the off-chance that skeleton IDs are not unique
    # (e.g. if neurons come from different projects) -> this is relevant because
    # we identify the master ("base_neuron") via it's skeleton ID
    skids = [n.skeleton_id for n in x]
    if len(skids) > len(np.unique(skids)):
        raise ValueError('Duplicate skeleton IDs found. Try manually assigning '
                         'unique skeleton IDs.')

    if any([not isinstance(n, core.CatmaidNeuron) for n in x]):
        raise TypeError('Input must only be CatmaidNeurons/List')

    if len(x) < 2:
        raise ValueError('Need at least 2 neurons to make a union!')

    # Make sure base_neuron is a skeleton ID
    if isinstance(base_neuron, core.CatmaidNeuron):
        base_skid = base_neuron.skeleton_id
    elif not isinstance(base_neuron, type(None)):
        if base_neuron not in skids:
            raise ValueError('Base neuron skeleton ID "{}" not in NeuronList'.format(base_neuron))
        base_skid = base_neuron
    else:
        base_skid = x[0].skeleton_id

    # Convert distance threshold from microns to nanometres
    limit *= 1000

    # Keep track of old IDs
    if track:
        for n in x:
            # Original skeleton of each node
            n.nodes['origin_skeletons'] = n.skeleton_id
            # Original skeleton of each connector
            n.connectors['origin_skeletons'] = n.skeleton_id
            # Old parent if this node gets rewired
            n.nodes['old_parent'] = n.nodes.parent_id.values

    # Now make unions
    all_clps_nodes = {}
    while len(x) > 1:
        # First we need to find a pair of overlapping neurons
        comb = itertools.combinations(x, 2)
        ol = False
        for c in comb:
            # If combination contains base_skid, make sure it's the master
            if c[0].skeleton_id == base_skid:
                master, minion = c[0], c[1]
            else:
                master, minion = c[1], c[0]

            # Generate KDTree for master neuron
            tree = graph.neuron2KDTree(master, tree_type='c', data='treenodes')

            # For each node in master get the nearest neighbor in minion
            coords = minion.nodes[['x', 'y', 'z']].values
            nn_dist, nn_ix = tree.query(coords, k=1, distance_upper_bound=limit)

            # Use this combination if overlap found
            if any(nn_dist <= limit):
                ol = True
                break

        # If no overlap between remaining fragments
        if not ol:
            miss = [n.skeleton_id for n in x if n.skeleton_id != base_skid]
            msg = "{} fragments do not overlap: {}.".format(len(x) - 1,
                                                            ", ".join(miss))
            # Raise ...
            if non_overlap.lower() == 'raise':
                raise ValueError(msg + " Try increasing the `limit` parameter")
            # ... or stitch up neurons using mst and break the loop...
            elif non_overlap.lower() == 'stitch':
                logger.warning(msg + " Stitching.")
                x = navis.stitch_skeletons(x, method='LEAFS', master=base_skid)
                x = core.CatmaidNeuronList(x)
                break
            # ... or just skip remaining fragments
            else:
                logger.warning(msg + " Skipping.")
                x = [n for n in x if n.skeleton_id == base_skid]
                break

        # Now collapse minion nodes that are within distance limits into master
        to_clps = minion.nodes.loc[nn_dist <= limit, 'node_id'].values
        clps_into = master.nodes.loc[nn_ix[nn_dist <= limit], 'node_id'].values

        # Generate a map: minion node -> master node to collapse into
        clps_map = dict(zip(to_clps, clps_into))

        # Track the collapsed node into the master
        if track:
            for n1, n2 in zip(to_clps, clps_into):
                all_clps_nodes[n2] = all_clps_nodes.get(n2, []) + [n1]

        # Reroot minion to one of the nodes that will be collapsed
        graph_utils.reroot_neuron(minion, to_clps[0], inplace=True)

        # Collapse nodes by first dropping all collapsed nodes
        minion.nodes = minion.nodes.loc[~minion.nodes.node_id.isin(to_clps)]
        # Make independent of original table to prevent warnings
        minion.nodes = minion.nodes.copy()

        # Reconnect children of the collapsed nodes to their new parents
        to_rewire = minion.nodes.parent_id.isin(to_clps)

        # Track old parents before rewiring
        if track:
            minion.nodes['old_parent'] = None
            minion.nodes.loc[to_rewire, 'old_parent'] = minion.nodes.loc[to_rewire, 'parent_id']

        # Now rewire
        new_parents = minion.nodes.loc[to_rewire, 'parent_id'].map(clps_map)
        minion.nodes.loc[to_rewire, 'parent_id'] = new_parents

        # Merge minion's node table into master
        master.nodes = pd.concat([master.nodes, minion.nodes],
                                 axis=0,
                                 sort=True,
                                 ignore_index=True)

        # Now some clean up! First up: node tags
        # Make sure tags in minion are mapped onto their new IDs
        tags = {k: [clps_map.get(n, n) for n in v] for k, v in minion.tags.items()}
        # Combine master and minion tags
        master.tags = {k: v + tags.get(k, []) for k, v in master.tags.items()}
        master.tags.update({k: v for k, v in tags.items() if k not in master.tags})

        # Last but not least: merge connector tables
        new_tn = minion.connectors.loc[minion.connectors.node_id.isin(to_clps),
                                       'node_id'].map(clps_map)

        if track:
            minion.connectors['old_treenode'] = None
            minion.connectors.loc[minion.connectors.node_id.isin(to_clps),
                                  'old_treenode'] = minion.connectors.loc[minion.connectors.node_id.isin(to_clps),
                                                                          'node_id']

        minion.connectors.loc[minion.connectors.node_id.isin(to_clps),
                              'node_id'] = new_tn
        master.connectors = pd.concat([master.connectors, minion.connectors],
                                      axis=0,
                                      sort=True,
                                      ignore_index=True)

        # Reset master's attributes (graph, node types, etc)
        master._clear_temp_attr()

        # Almost done. Just need to pop minion from "x"
        x = [n for n in x if n.skeleton_id != minion.skeleton_id]

    union = x[0]

    # Keep track of old IDs
    if track:
        # List of nodes merged into this node
        union.nodes['treenodes_merged'] = union.nodes.node_id.map(all_clps_nodes)

    # Return the last survivor
    return union


def remove_tagged_branches(x, tag, how='segment', preserve_connectors=False,
                           inplace=False):
    """Removes branches with a given treenode tag (e.g. ``not a branch``).

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
        """Helper function that walks from a treenode to the neurons root and
        returns the first parent that will not be removed.
        """
        this_nodes = x.nodes.set_index('node_id')
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
            lost_cn = x.connectors[x.connectors.node_id.isin(to_remove)]

            # Map to a remaining treenode
            # IMPORTANT: we do currently not account for the possibility that
            # we might be removing the root segment
            new_tn = [_find_next_remaining_parent(tn) for tn in lost_cn.node_id.values]
            x.connectors.loc[x.connectors.node_id.isin(to_remove), 'node_id'] = new_tn

        # Subset to remaining nodes - skip the last node in each segment
        navis.subset_neuron(x,
                            subset=x.nodes[~x.nodes.node_id.isin(
                                           to_remove)].node_id.values,
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
                to_remove = x.nodes[~x.nodes.node_id.isin(dist_graph.nodes)].node_id.values
                # Make sure the tagged treenode is there too
                to_remove += [x.tags[tag][0]]

            to_remove = set(to_remove)

            # Rewire connectors before we subset
            if preserve_connectors:
                # Get connectors that will be disconnected
                lost_cn = x.connectors[x.connectors.node_id.isin(
                    to_remove)]

                # Map to a remaining treenode
                # IMPORTANT: we do currently not account for the possibility
                # that we might be removing the root segment
                new_tn = [_find_next_remaining_parent(tn) for tn in lost_cn.node_id.values]
                x.connectors.loc[x.connectors.node_id.isin(to_remove), 'node_id'] = new_tn

            # Subset to remaining nodes
            navis.subset_neuron(x,
                                subset=x.nodes[~x.nodes.node_id.isin(
                                    to_remove)].node_id.values,
                                keep_disc_cn=preserve_connectors,
                                inplace=True)

        if not inplace:
            return x
        return


def time_machine(x, target, inplace=False, remote_instance=None):
    """Reverses time and make neurons young again!

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
    target :            str | datetime-like | pandas.Timestamp
                        Date or date + time to time-travel to. Must be
                        parsable by ``pandas.TimeStamp``. Format for string
                        is YEAR-MONTH-DAY.
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
    >>> previous_n = pymaid.time_machine(n, '2016-12-1')

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, core.CatmaidNeuronList):
        return core.CatmaidNeuronList([time_machine(n, target,
                                                    inplace=inplace,
                                                    remote_instance=remote_instance)
                                       for n in config.tqdm(x,
                                                            'Traveling time',
                                                            disable=config.pbar_hide,
                                                            leave=config.pbar_leave)])

    if not isinstance(x, core.CatmaidNeuron):
        x = fetch.get_neuron(x, remote_instance=remote_instance)

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
    nodes = pd.DataFrame(data[0], columns=['node_id', 'parent_id',
                                           'user_id', 'x', 'y', 'z', 'radius',
                                           'confidence', 'creation_timestamp',
                                           'modified_timestamp', 'ordering_by'])
    nodes.loc[:, 'parent_id'] = nodes.parent_id.values.astype(object)
    nodes.loc[~nodes.parent_id.isnull(), 'parent_id'] = nodes.loc[~nodes.parent_id.isnull(), 'parent_id'].map(int)
    nodes.loc[nodes.parent_id.isnull(), 'parent_id'] = None

    connectors = pd.DataFrame(data[1], columns=['node_id', 'connector_id',
                                                'relation', 'x', 'y', 'z',
                                                'creation_timestamp',
                                                'modified_timestamp'])
    # This is a dictionary with {'tag': [[node_id, date_tagged], ...]}
    tags = data[2]
    annotations = pd.DataFrame(data[4], columns=['annotation_id',
                                                 'annotated_timestamp'])
    an_list = fetch.get_annotation_list(remote_instance=remote_instance)
    annotations['annotation'] = annotations.annotation_id.map(an_list.set_index('id').name.to_dict())

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

    x.nodes = before_nodes.copy()
    x.connectors = before_connectors.copy()
    x.annotations = before_annotations.copy()
    x.tags = before_tags

    # We might end up with multiple disconnected pieces - I don't yet know why
    x.nodes.loc[~x.nodes.parent_id.isin(x.nodes.node_id), 'parent_id'] = None

    # If there is more than one root, we have to remove the disconnected
    # pieces and keep only the "oldest branch".
    # The theory for doing this is: if a node shows up as "root" and the very
    # next step is that it is a child to another node, we should consider
    # it a not-yet connected branch that needs to be removed.
    roots = x.nodes[x.nodes.parent_id.isnull()].node_id.tolist()
    if len(roots) > 1:
        after_nodes = nodes[nodes.modified_timestamp > target]
        for r in roots:
            # Find the next version of this node
            nv = after_nodes[(after_nodes.node_id == r)]
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

        x.nodes = x.nodes[x.nodes.node_id.isin(keep)].copy()

    # Remove connectors where the treenode does not even exist yet
    x.connectors = x.connectors[x.connectors.node_id.isin(x.nodes.node_id)]

    # Take care of connectors where the treenode might exist but was not yet linked
    links = fetch.get_connector_links(x.skeleton_id)
    #localize = lambda x: pd.Timestamp.tz_localize(x, 'UTC')
    #links['creation_time'] = links.creation_time.map(localize)
    links = links[links.creation_time <= target]
    links['connector_id'] = links.connector_id.astype(int)
    links['node_id'] = links.node_id.astype(int)

    # Get connector ID -> treenode combinations
    l = links[['connector_id', 'node_id']].T.apply(tuple)
    c = x.connectors[['connector_id', 'node_id']].T.apply(tuple)

    # Keep only those where connector->treenode connection is present
    x.connectors = x.connectors[c.isin(l)]

    x._clear_temp_attr()

    if not inplace:
        return x

    return


def prune_by_length(x, min_length=0, max_length=float('inf'), inplace=False):
    """Remove segments of given length.

    This uses :func:`pymaid.graph_utils._generate_segments` to generate
    segments that maximize segment lengths.

    Parameters
    ----------
    x :             CatmaidNeuron/List
    min_length :    int | float
                    Twigs shorter than this length [um] will be pruned.
    max_length :    int | float
                    Segments longer than this length [um] will be pruned.
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.

    Returns
    -------
    CatmaidNeuron/List
                    Pruned neuron(s).

    See Also
    --------
    :func:`pymaid.longest_neurite`
                    If you want to keep/remove just the N longest neurites
                    instead of using a length cut-off.
    :func:`pymaid.prune_twigs`
                    Use if you are looking to remove only terminal branches of
                    a given size.

    Examples
    --------
    >>> import pymaid
    >>> n = pymaid.get_neurons(16)
    >>> # Remove neurites longer than 100mirons
    >>> n_pr = pymaid._prune_by_length(n,
    ...                                min_length=0,
    ...                                max_length=100,
    ...                                inplace=False)
    >>> n.n_nodes > n_pr.n_nodes
    True

    """
    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            x = x.copy()

        [prune_by_length(n,
                         min_length=min_length,
                         max_length=max_length,
                         inplace=True) for n in config.tqdm(x, desc='Pruning',
                                                            disable=config.pbar_hide,
                                                            leave=config.pbar_leave)]

        if not inplace:
            return x
        else:
            return None
    elif isinstance(x, core.CatmaidNeuron):
        neuron = x
    else:
        raise TypeError('Expected CatmaidNeuron/List, got {}'.format(type(x)))

    # Make a copy if necessary before making any changes
    if not inplace:
        neuron = neuron.copy()

    # Convert units to nanometres
    min_length *= 1000
    max_length *= 1000

    # Find terminal segments
    segs = graph_utils._generate_segments(neuron, weight='weight')
    segs = np.array(segs)

    # Get segment lengths
    seg_lengths = np.array([graph_utils.segment_length(neuron, s) for s in segs])

    # Find out which to delete
    segs_to_delete = segs[(seg_lengths < min_length) | (seg_lengths > max_length)]

    if segs_to_delete.any():
        # Unravel the into list of node IDs -> skip the last parent
        nodes_to_delete = [n for s in segs_to_delete for n in s[:-1]]

        # Subset neuron
        nodes_to_keep = neuron.nodes[~neuron.nodes.node_id.isin(nodes_to_delete)].node_id.values
        navis.subset_neuron(neuron, nodes_to_keep, inplace=True)

    if not inplace:
        return neuron
    else:
        return None
