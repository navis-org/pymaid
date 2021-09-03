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

""" This module contains functions to analyse connectivity.
"""
import math

from itertools import combinations

import pandas as pd
import numpy as np
import scipy.spatial
import scipy.stats

from navis import graph_utils, intersect

from . import fetch, core, utils, config

# Set up logging
logger = config.logger

__all__ = sorted(['filter_connectivity', 'cable_overlap',
                  'predict_connectivity', 'adjacency_matrix', 'group_matrix',
                  'adjacency_from_connectors', 'cn_table_from_connectors',
                  'connection_density', 'sparseness', 'shared_partners'])


def filter_connectivity(x, restrict_to, remote_instance=None):
    """Filter connectivity data by volume or skeleton data.

    Use this e.g. to restrict connectivity to edges within a given volume or
    to certain compartments of neurons.

    Important
    ---------
    Duplicate skeleton IDs (e.g. two fragments from the same neuron) will be
    collapsed back into a single neuron! Use only a single fragment per neuron.
    For multiple fragments/neuron see :func:`~pymaid.adjacency_from_connectors`.
    Order of columns/rows may change during filtering.


    Parameters
    ----------
    x :                 Connectivity object
                        Currently accepts either:
                         (1) Connectivity table from :func:`~pymaid.get_partners`
                         (2) Adjacency matrix from :func:`~pymaid.adjacency_matrix`
    restrict_to :       str | pymaid.Volume | CatmaidNeuronList
                        Volume or neurons to restrict connectivity to. Strings
                        will be interpreted as volumes.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.

    Returns
    -------
    Restricted connectivity data

    See Also
    --------
    :func:`~pymaid.adjacency_from_connectors`
            Use this function if you have multiple fragments per neuron.

    """
    if not isinstance(restrict_to, (str, core.Volume,
                                    core.CatmaidNeuron,
                                    core.CatmaidNeuronList)):
        raise TypeError('Unable to restrict connectivity '
                        'to type {}'.format(type(restrict_to)))

    if isinstance(restrict_to, str):
        restrict_to = fetch.get_volume(
            restrict_to, remote_instance=remote_instance)

    if not isinstance(x, pd.DataFrame):
        raise TypeError('Input must be pandas DataFrame, got '
                        ' "{}"'.format(type(x)))

    datatype = getattr(x, 'datatype', None)

    # If no datatype attribute, try guessing
    if isinstance(datatype, type(None)):
        if 'relation' in x.columns:
            datatype = 'connectivity_table'
        else:
            datatype = 'adjacency_matrix'

    if datatype == 'connectivity_table':
        neurons = [c for c in x.columns if str(c).isnumeric()]

        """
        # Keep track of existing edges
        old_upstream = np.array([ [ (source, n) for n in neurons ] for source in x[x.relation=='upstream'].skeleton_id ])
        old_upstream = old_upstream[ x[x.relation=='upstream'][neurons].values > 0 ]

        old_downstream = np.array([ [ (n, target) for n in neurons ] for target in x[x.relation=='downstream'].skeleton_id ])
        old_downstream = old_downstream[ x[x.relation=='downstream'][neurons].values > 0 ]

        old_edges = np.concatenate([old_upstream, old_downstream], axis=0).astype(int)
        """

        # First get connector between neurons on the table
        if not x[x.relation == 'upstream'].empty:
            upstream = fetch.get_connectors_between(x[x.relation == 'upstream'].skeleton_id,
                                                    neurons,
                                                    directional=True,
                                                    remote_instance=remote_instance)
            # Now filter connectors
            if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
                upstream = upstream[upstream.connector_id.isin(
                    restrict_to.connectors.connector_id.values)]
            elif isinstance(restrict_to, core.Volume):
                cn_locs = np.vstack(upstream.connector_loc.values)
                upstream = upstream[intersect.in_volume(cn_locs, restrict_to)]
        else:
            upstream = None

        if not x[x.relation == 'downstream'].empty:
            downstream = fetch.get_connectors_between(neurons,
                                                      x[x.relation == 'downstream'].skeleton_id,
                                                      directional=True,
                                                      remote_instance=remote_instance)
            # Now filter connectors
            if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
                downstream = downstream[downstream.connector_id.isin(
                    restrict_to.connectors.connector_id.values)]
            elif isinstance(restrict_to, core.Volume):
                cn_locs = np.vstack(downstream.connector_loc.values)
                downstream = downstream[intersect.in_volume(
                    cn_locs, restrict_to)]
        else:
            downstream = None

        if not isinstance(downstream, type(None)) and not isinstance(upstream, type(None)):
            cn_data = pd.concat([upstream, downstream], axis=0)
        elif isinstance(downstream, type(None)):
            cn_data = upstream
        else:
            cn_data = downstream

    elif datatype == 'adjacency_matrix':
        if getattr(x, 'is_grouped', False):
            raise TypeError('Adjacency matrix appears to be grouped. Unable '
                            'to process that.')

        cn_data = fetch.get_connectors_between(x.index.values,
                                               x.columns.values,
                                               directional=True,
                                               remote_instance=remote_instance)

        # Now filter connectors
        if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
            cn_data = cn_data[cn_data.connector_id.isin(
                restrict_to.connectors.connector_id.values)]
        elif isinstance(restrict_to, core.Volume):
            cn_locs = np.vstack(cn_data.connector_loc.values)
            cn_data = cn_data[intersect.in_volume(cn_locs, restrict_to)]
    else:
        raise TypeError('Unknown connectivity data type "{}".'.format(datatype)
                        + ' See help(filter_connectivity) for details.')

    if cn_data.empty:
        logger.warning('No connectivity left after filtering')
        # return

    # Reconstruct connectivity data:
    # Collect edges
    edges = cn_data[['source_neuron', 'target_neuron']].values

    if edges.shape[0] > 0:
        # Turn individual edges into synaptic connections
        unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
        unique_skids = np.unique(edges).astype(str)
        unique_edges = unique_edges.astype(str)
    else:
        unique_edges = []
        counts = []
        unique_skids = []

    # Create empty adj_mat
    adj_mat = pd.DataFrame(np.zeros((len(unique_skids), len(unique_skids))),
                           columns=unique_skids, index=unique_skids)

    # Fill in values
    for i, e in enumerate(config.tqdm(unique_edges,
                                      disable=config.pbar_hide,
                                      desc='Regenerating',
                                      leave=config.pbar_leave)):
        # using df.at here speeds things up tremendously!
        adj_mat.at[str(e[0]), str(e[1])] = counts[i]

    if datatype == 'adjacency_matrix':
        return adj_mat.reindex(index=x.index.astype(str),
                               columns=x.columns.astype(str)).fillna(0)

    # Generate connectivity table by subsetting adjacency matrix to our
    # neurons of interest
    us_neurons = [n for n in neurons if n in adj_mat.columns]
    all_upstream = adj_mat[adj_mat[us_neurons].sum(axis=1) > 0][us_neurons]
    all_upstream['skeleton_id'] = all_upstream.index
    all_upstream['relation'] = 'upstream'

    ds_neurons = [n for n in neurons if n in adj_mat.T.columns]
    all_downstream = adj_mat.T[adj_mat.T[ds_neurons].sum(
        axis=1) > 0][ds_neurons]
    all_downstream['skeleton_id'] = all_downstream.index
    all_downstream['relation'] = 'downstream'

    # Merge tables
    df = pd.concat([all_upstream, all_downstream], axis=0, ignore_index=True)
    remaining_neurons = [n for n in neurons if n in df.columns]

    # Remove neurons that were not in the original data - under certain
    # circumstances, neurons can sneak back in
    df = df[df.skeleton_id.isin(x.skeleton_id)]

    # Use original connectivity table to populate data
    aux = x.set_index('skeleton_id')[['neuron_name', 'num_nodes']].to_dict()
    df['num_nodes'] = [aux['num_nodes'][s] for s in df.skeleton_id.values]
    df['neuron_name'] = [aux['neuron_name'][s] for s in df.skeleton_id.values]
    df['total'] = df[remaining_neurons].sum(axis=1)

    # Reorder columns
    df = df[['neuron_name', 'skeleton_id', 'num_nodes',
             'relation', 'total'] + remaining_neurons]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)
    df.type = 'connectivity_table'
    df.reset_index(drop=True, inplace=True)

    return df


def cable_overlap(a, b, dist=2, method='min'):
    """Calculate the amount of cable of neuron A within distance of neuron B.

    DEPCRECATED! PLEASE USE `navis.cable_overlap` INSTEAD!

    Parameters
    ----------
    a,b :       CatmaidNeuron | CatmaidNeuronList
                Neuron(s) for which to compute cable within distance.
    dist :      int, optional
                Maximum distance in microns [um].
    method :    'min' | 'max' | 'avg'
                Method by which to calculate the overlapping cable between
                two cables. Assuming that neurons A and B have 300 and 150
                um of cable within given distances, respectively:
                    1. 'min' returns 150
                    2. 'max' returns 300
                    3. 'avg' returns 225

    Returns
    -------
    pandas.DataFrame
            Matrix in which neurons A are rows, neurons B are columns. Cable
            within distance is given in microns::

                        skidB1   skidB2  skidB3  ...
                skidA1    5        1        0
                skidA2    10       20       5
                skidA3    4        3        15
                ...

    """
    raise DeprecationWarning('This function has been moved to `navis`. Please '
                             'use `navis.cable_overlap` as drop-in replacement.'
                             ' Note that the navis function will return '
                             'in the same units as the neuron - i.e. typically '
                             'nanometers for CatmaidNeurons.')

def predict_connectivity(source, target, method='possible_contacts',
                         remote_instance=None, **kwargs):
    """Calculate potential synapses from source onto target neurons.

    Based on a concept by `Alexander Bates <https://github.com/alexanderbates/catnat>`_.

    Parameters
    ----------
    source,target : CatmaidNeuron | CatmaidNeuronList
                    Neuron(s) for which to compute potential connectivity.
                    This is unidirectional: source -> target.
    method :        'possible_contacts'
                    Method to use for calculations. See Notes.
    **kwargs
                    1. For method 'possible_contacts':
                        - ``dist`` to set distance between connectors and
                          nodes manually.
                        - ``n_irq`` to set number of interquartile ranges of
                          harmonic mean. Default = 2.

    Notes
    -----
    Method ``possible_contacts``:
        1. Calculating harmonic mean of distances ``d`` (connector->node)
           at which onnections between neurons A and neurons B occur.
        2. For all presynapses of neurons A, check if they are within
           ``n_irq`` (default=2) interquartile range  of ``d`` of a
           neuron B node.

    Neurons without cable or presynapses will be assigned a predicted
    connectivity of 0.


    Returns
    -------
    pandas.DataFrame
            Matrix holding possible synaptic contacts. Sources are rows,
            targets are columns::

                         target1  target2  target3  ...
                source1    5        1        0
                source2    10       20       5
                source3    4        3        15
                ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not remote_instance:
        try:
            remote_instance = source._remote_instance
        except BaseException:
            pass

    for _ in [source, target]:
        if not isinstance(_, (core.CatmaidNeuron, core.CatmaidNeuronList)):
            raise TypeError('Need CatmaidNeuron/List, got '
                            '"{}"'.format(type(_)))

    if isinstance(source, core.CatmaidNeuron):
        source = core.CatmaidNeuronList(source)

    if isinstance(target, core.CatmaidNeuron):
        target = core.CatmaidNeuronList(target)

    allowed_methods = ['possible_contacts']
    if method not in allowed_methods:
        raise ValueError('Unknown method "{0}". Allowed methods: "{0}"'.format(
            method, ','.join(allowed_methods)))

    matrix = pd.DataFrame(np.zeros((source.shape[0], target.shape[0])),
                          index=source.skeleton_id,
                          columns=target.skeleton_id)

    if kwargs.get('dist', None):
        dist_threshold = kwargs.get('dist')
    else:
        # First let's calculate at what distance synapses are being made
        cn_between = fetch.get_connectors_between(source, target,
                                                  remote_instance=remote_instance)
        if cn_between.shape[0] > 0:
            cn_locs = np.vstack(cn_between.connector_loc.values)
            tn_locs = np.vstack(cn_between.node2_loc.values)

            distances = np.sqrt(np.sum((cn_locs - tn_locs) ** 2, axis=1))

            logger.info('Average connector->node distances: '
                        '{:.2f} +/- {:.2f} nm'.format(distances.mean(),
                                                      distances.std()))
        else:
            logger.warning('No existing connectors to calculate average'
                           'connector->node distance found. Falling'
                           'back to default of 1um. Use <dist> argument'
                           'to set manually.')
            distances = [1000]

        # Calculate distances threshold
        n_irq = kwargs.get('n_irq', 2)
        # We use the median because some very large connector->treenode
        # distances can massively skew the average
        dist_threshold = scipy.stats.hmean(distances) + n_irq * scipy.stats.iqr(distances)

    with config.tqdm(total=len(target), desc='Predicting',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        for t in target:
            # If no nodes, predict 0 connectivity and skip
            if t.nodes.empty:
                matrix.loc[t.skeleton_id, source.skeleton_id] = 0
                continue

            # Create cKDTree for target
            tree = scipy.spatial.cKDTree(
                t.nodes[['x', 'y', 'z']].values, leafsize=10)
            for s in source:
                # If not synapses, predict 0 connectivity and skip
                if s.presynapses.empty:
                    matrix.at[s.skeleton_id, t.skeleton_id] = 0
                    continue

                # Query against presynapses
                dist, ix = tree.query(s.presynapses[['x', 'y', 'z']].values,
                                      k=1,
                                      distance_upper_bound=dist_threshold,
                                      workers=-1
                                      )

                # Calculate possible contacts
                possible_contacts = sum(dist != float('inf'))

                matrix.at[s.skeleton_id, t.skeleton_id] = possible_contacts

            pbar.update(1)

    return matrix.astype(int)


def cn_table_from_connectors(x, remote_instance=None):
    """Generate connectivity table from neurons' connectors.

    This function creates the connectivity table from scratch using just the
    neurons' connectors. This function is able to deal with non-unique
    skeleton IDs (most other functions won't). Use it e.g. when you
    split neurons into multiple fragments. *The order of the input
    CatmaidNeuronList is preserved!*

    Parameters
    ----------
    x :                 CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) for which to generate connectivity table.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons::

           neuron_name  skeleton_id   relation    total  skid1  skid2 ...
         0   name1         skid1      upstream    n_syn  n_syn  ...
         1   name2         skid2     downstream   n_syn  n_syn  ..
         2   name3         skid3      usptream    n_syn  n_syn  .
         ... ...

        ``relation`` can be ``'upstream'`` (incoming), ``'downstream'``
        (outgoing), ``'attachment'`` or ``'gapjunction'`` (gap junction).

    See Also
    --------
    :func:`~pymaid.get_partners`
            If you are working with "intact" neurons. Much faster!
    :func:`~pymaid.filter_connectivity`
            Use this function if you have only a single fragment per neuron
            (e.g. just the axon). Also way faster.

    Examples
    --------
    >>> # Fetch some neurons
    >>> x = pymaid.get_neuron('annotation:PD2a1/b1')
    >>> # Split into axon / dendrites
    >>> x.reroot(x.soma)
    >>> split = pymaid.split_axon_dendrite(x)
    >>> # Regenerate cn_table
    >>> cn_table = pymaid.cn_table_from_connectors(split)
    >>> # Skeleton IDs are non-unique but column order = input order:
    >>> # in this example the first occurrence is axon, the second dendrites
    >>> cn_table.head()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise TypeError('Need CatmaidNeuron/List, got "{}"'.format(type(x)))

    if isinstance(x, core.CatmaidNeuron):
        x = core.CatmaidNeuronList(x)

    # Get connector details for all neurons
    all_cn = x.connectors.connector_id.values
    cn_details = fetch.get_connector_details(all_cn, remote_instance=remote_instance)

    # Remove connectors for which there are either no pre- or no postsynaptic
    # neurons
    cn_details = cn_details[cn_details.postsynaptic_to.apply(len) != 0]
    cn_details = cn_details[~cn_details.presynaptic_to.isnull()]

    # We need to map treenode ID to skeleton ID in cases where there are more
    # links (postsynaptic_to_node) than targets (postsynaptic_to)
    multi_links = cn_details[cn_details.postsynaptic_to.apply(len) < cn_details.postsynaptic_to_node.apply(len)]
    if not multi_links.empty:
        tn_to_fetch = [tn for l in multi_links.postsynaptic_to_node for tn in l]
        tn_to_skid = fetch.get_skid_from_node(tn_to_fetch, remote_instance=remote_instance)
    else:
        tn_to_skid = {}

    # Collect all pre and postsynaptic neurons
    all_pre = cn_details[~cn_details.presynaptic_to.isin(x.skeleton_id.astype(int))]
    all_post = cn_details[cn_details.presynaptic_to.isin(x.skeleton_id.astype(int))]

    all_partners = np.append(all_pre.presynaptic_to.values,
                             [n for l in all_post.postsynaptic_to.values for n in l])

    us_dict = {}
    ds_dict = {}
    # Go over over all neurons and process connectivity
    for i, n in enumerate(config.tqdm(x, desc='Processing',
                          disable=config.pbar_hide, leave=config.pbar_leave)):

        # First prepare upstream partners:
        # Get all treenodes
        this_tn = set(n.nodes.node_id.values)
        # Prepare upstream partners
        this_us = all_pre[all_pre.connector_id.isin(n.connectors.connector_id.values)].copy()
        # Get the number of all links per connector
        this_us['n_links'] = [len(this_tn & set(r.postsynaptic_to_node))
                                           for r in this_us.itertuples()]
        # Group by input and store as dict. Attention: we NEED to index by
        # neuron as skeleton IDs might not be unique!
        us_dict[n] = this_us.groupby('presynaptic_to').n_links.sum().to_dict()
        this_us = this_us.groupby('presynaptic_to').n_links.sum()

        # Now prepare downstream partners:
        # Get all downstream connectors
        this_ds = all_post[all_post.presynaptic_to == int(n.skeleton_id)]
        # Prepare dict
        ds_dict[n] = {p: 0 for p in all_partners}
        # Easy cases first (single link to target per connector)
        is_single = this_ds.postsynaptic_to.apply(len) >= this_ds.postsynaptic_to_node.apply(len)
        for r in this_ds[is_single].itertuples():
            for s in r.postsynaptic_to:
                ds_dict[n][s] += 1
        # Now hard cases - will have to look up skeleton ID via treenode ID
        for r in this_ds[~is_single].itertuples():
            for s in r.postsynaptic_to_node:
                ds_dict[n][tn_to_skid[s]] += 1

    # Now that we have all data, let's generate the table
    us_table = pd.DataFrame.from_dict(us_dict)
    ds_table = pd.DataFrame.from_dict(ds_dict)

    # Make sure we keep the order of the original neuronlist
    us_table = us_table[[n for n in x]]
    us_table.columns=[n.skeleton_id for n in us_table.columns]
    ds_table = ds_table[[n for n in x]]
    ds_table.columns=[n.skeleton_id for n in ds_table.columns]

    ds_table['relation'] = 'downstream'
    us_table['relation'] = 'upstream'

    # Generate table
    cn_table = pd.concat([us_table, ds_table], axis=0)

    # Replace NaN with 0
    cn_table = cn_table.fillna(0)

    # Make skeleton ID a column
    cn_table = cn_table.reset_index(drop=False)
    cn_table.columns = ['skeleton_id'] + list(cn_table.columns[1:])

    # Add names
    names = fetch.get_names(cn_table.skeleton_id.values,
                            remote_instance=remote_instance)
    cn_table['neuron_name'] = [names[str(s)] for s in cn_table.skeleton_id.values]
    cn_table['total'] = cn_table[x.skeleton_id].sum(axis=1)

    # Drop rows with 0 synapses (e.g. if neuron is only up- but not downstream)
    cn_table = cn_table[cn_table.total > 0]

    # Sort by number of synapses
    cn_table = cn_table.sort_values(['relation', 'total'], ascending=False).reset_index()

    # Sort columnes
    cn_table = cn_table[['neuron_name', 'skeleton_id', 'relation', 'total'] + list(set(x.skeleton_id))]

    return cn_table


def adjacency_from_connectors(source, target=None, remote_instance=None):
    """Regenerate adjacency matrices from neurons' connectors.

    Notes
    -----
    This function creates an adjacency matrix from scratch using just the
    neurons' connectors. This function is able to deal with non-unique
    skeleton IDs (most other functions are not). Use it e.g. when you
    split neurons into multiple fragments.

    Parameters
    ----------
    source,target :     skeleton IDs | CatmaidNeuron | CatmaidNeuronList
                        Neuron(s) for which to generate adjacency matrix.
                        If ``target==None``, will use ``target=source``.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.

    Returns
    -------
    pandas.DataFrame
            Matrix holding possible synaptic contacts. Sources are rows,
            targets are columns. Labels are skeleton IDs. Order is preserved::

                        target1  target2  target3  ...
                source1    5        1        0
                source2    10       20       5
                source3    4        3        15
                ...

    See Also
    --------
    :func:`~pymaid.adjacency_matrix`
            If you are working with "intact" neurons. Much faster!
    :func:`~pymaid.filter_connectivity`
            Use this function if you have only a single fragment per neuron
            (e.g. just the axon). Also way faster.

    Examples
    --------
    >>> # Fetch some neurons
    >>> x = pymaid.get_neuron('annotation:PD2a1/b1')
    >>> # Split into axon / dendrites
    >>> x.reroot(x.soma)
    >>> split = pymaid.split_axon_dendrite(x)
    >>> # Regenerate all-by-all adjacency matrix
    >>> adj = pymaid.adjacency_from_connectors(split)
    >>> # Skeleton IDs are non-unique but column/row order = input order:
    >>> # in this example, the first occurrence is axon, the second dendrites
    >>> adj.head()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(source, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        skids = utils.eval_skids(source, remote_instance=remote_instance)
        source = fetch.get_neuron(skids, remote_instance=remote_instance)

    if isinstance(target, type(None)):
        target = source
    elif not isinstance(target, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        skids = utils.eval_skids(target, remote_instance=remote_instance)
        target = fetch.get_neuron(skids, remote_instance=remote_instance)

    if isinstance(source, core.CatmaidNeuron):
        source = core.CatmaidNeuronList(source)

    if isinstance(target, core.CatmaidNeuron):
        target = core.CatmaidNeuronList(target)

    # Generate empty adjacency matrix
    adj = np.zeros((len(source), len(target)))

    # Get connector details for all neurons
    all_cn = list(set(np.append(source.connectors.connector_id.values,
                                target.connectors.connector_id.values)))
    cn_details = fetch.get_connector_details(all_cn,
                                             remote_instance=remote_instance)

    # Now go over all source neurons and process connections
    for i, s in enumerate(config.tqdm(source, desc='Processing',
                          disable=config.pbar_hide, leave=config.pbar_leave)):

        # Get all connectors presynaptic for this source
        this_cn = cn_details[(cn_details.presynaptic_to == int(s.skeleton_id)) &
                             (cn_details.connector_id.isin(s.connectors.connector_id))
                             ]

        # Go over all target neurons
        for k, t in enumerate(target):
            t_tn = set(t.nodes.node_id.values)
            t_post = t.postsynapses.connector_id.values

            # Extract number of connections from source to this target
            this_t = this_cn[this_cn.connector_id.isin(t_post)]

            # Now figure out how many links are between this connector and
            # the target
            n_links = sum([len(t_tn & set(r.postsynaptic_to_node))
                           for r in this_t.itertuples()])

            adj[i][k] = n_links

    return pd.DataFrame(adj,
                        index=source.skeleton_id,
                        columns=target.skeleton_id)


def _edges_from_connectors(a, b=None, remote_instance=None):
    """Generate list of edges between two sets of neurons from their
    connector data.

    Attention: this is UNIDIRECTIONAL (a->b)!

    Parameters
    ----------
    a,b :       CatmaidNeuron | CatmaidNeuronList | skeleton IDs
                Either a or b HAS to be a neuron object.
                If ``b=None``, will use ``b=a``.

    """
    if not isinstance(a, (core.CatmaidNeuronList, core.CatmaidNeuron)) and \
       not isinstance(b, (core.CatmaidNeuronList, core.CatmaidNeuron)):
        raise ValueError('Either neuron a or b has to be a neuron object.')

    if isinstance(b, type(None)):
        b = a

    cn_between = fetch.get_connectors_between(
        a, b, remote_instance=remote_instance)

    if isinstance(a, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        cn_a = a.connectors.connector_id.values
        cn_between = cn_between[cn_between.connector_id.isin(cn_a)]

    if isinstance(b, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        cn_b = b.connectors.connector_id.values
        cn_between = cn_between[cn_between.connector_id.isin(cn_b)]

    # Count source -> target connectors
    edges = cn_between.groupby(['source_neuron', 'target_neuron']).count()

    # Melt into edge list
    edges = edges.reset_index().iloc[:, :3]
    edges.columns = ['source_skid', 'target_skid', 'weight']

    return edges


def adjacency_matrix(sources, targets=None, source_grp={}, target_grp={},
                     fractions=False, syn_threshold=None, syn_cutoff=None,
                     use_connectors=False, volume_filter=None, remote_instance=None):
    """Generate adjacency matrix between source and target neurons.

    Directional: sources = rows, targets = columns.

    Parameters
    ----------
    sources
                        Source neurons as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    targets
                        Optional. Target neurons as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. ``'annotation:PN right'`
                        4. CatmaidNeuron or CatmaidNeuronList object

                        If not provided, ``source neurons = target neurons``.
    fractions :         bool, optional
                        If True, will return connectivity as fraction of total
                        number of postsynaptic links to target neuron.
    syn_cutoff :        int, optional
                        If set, will cut off connections ABOVE given value.
    syn_threshold :     int, optional
                        If set, will ignore connections with LESS synapses.
    source_grp :        dict, optional
                        Use to collapse sources into groups. Can be either:
                          1. ``{group1: [neuron1, neuron2, ... ], ..}``
                          2. ``{neuron1: group1, neuron2 : group2, ..}``

                        ``syn_cutoff`` and ``syn_threshold`` are applied
                        BEFORE grouping!
    target_grp :        dict, optional
                        See ``source_grp`` for possible formats.
    use_connectors :    bool, optional
                        If True AND ``s`` or ``t`` are ``CatmaidNeuron/List``,
                        restrict adjacency matrix to their connectors. Use
                        if e.g. you are using pruned neurons.
    volume_filter :     Volume | list of Volumes, optional
                        Volume(s) to restrict connections to. Can be a
                        pymaid.Volume, the name of a CATMAID volume or a
                        list thereof.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.

    Returns
    -------
    matrix :          pandas.Dataframe

    See Also
    --------
    :func:`~pymaid.group_matrix`
                More fine-grained control over matrix grouping.
    :func:`~pymaid.adjacency_from_connectors`
                Use this function if you are working with multiple fragments
                per neuron.

    Examples
    --------
    Generate and plot a adjacency matrix:

    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> neurons = pymaid.get_neurons('annotation:test')
    >>> mat = pymaid.adjacency_matrix(neurons)
    >>> g = sns.heatmap(adj_mat, square=True)
    >>> g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=7)
    >>> g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=7)
    >>> plt.show()

    Cut neurons into axon dendrites and compare their connectivity:

    >>> # Get a set of neurons
    >>> nl = pymaid.get_neurons('annnotation:type_16_candidates')
    >>> # Split into axon dendrite by using a tag
    >>> nl.reroot(nl.soma)
    >>> nl_axon = nl.prune_proximal_to('axon', inplace=False)
    >>> nl_dend = nl.prune_distal_to('axon', inplace=False)
    >>> # Get a list of the downstream partners
    >>> cn_table = pymaid.get_partners(nl)
    >>> ds_partners = cn_table[cn_table.relation == 'downstream']
    >>> # Take the top 10 downstream partners
    >>> top_ds = ds_partners.iloc[:10].skeleton_id.values
    >>> # Generate separate adjacency matrices for axon and dendrites
    >>> adj_axon = pymaid.adjacency_matrix(nl_axon, top_ds,
    ...                                    use_connectors=True)
    >>> adj_dend = pymaid.adjacency_matrix(nl_dend, top_ds,
    ...                                    use_connectors=True)
    >>> # Rename rows and merge dataframes
    >>> adj_axon.index += '_axon'
    >>> adj_dend.index += '_dendrite'
    >>> adj_merged = pd.concat([adj_axon, adj_dend], axis=0)
    >>> # Plot heatmap using seaborn
    >>> ax = sns.heatmap(adj_merged)
    >>> plt.show()

    Restrict adjacency matrix to a given volume:

    >>> neurons = pymaid.get_neurons('annotation:glomerulus DA1')
    >>> lh = pymaid.get_volume('LH_R')
    >>> adj = pymaid.adjacency_matrix(neurons, volume_filter=lh)

    Get adjacency matrix with fraction of inputs instead of total
    synapse count:

    >>> neurons = pymaid.get_neurons('annotation:glomerulus DA1')
    >>> adj = pymaid.adjacency_matrix(neurons, fractions=True)

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    source_skids = utils.eval_skids(sources, remote_instance=remote_instance)

    if isinstance(targets, type(None)):
        targets = sources
        target_skids = source_skids
    else:
        target_skids = utils.eval_skids(targets, remote_instance=remote_instance)

    # Make sure neurons are  integers
    source_skids = [int(n) for n in source_skids]
    target_skids = [int(n) for n in target_skids]

    # Make sure skeleton IDs are unique
    source_skids = sorted(set(source_skids), key=source_skids.index)
    target_skids = sorted(set(target_skids), key=target_skids.index)

    # Get the adjacency matrix
    url = remote_instance._get_connectivity_matrix_url()
    post = {}
    post.update({'rows[{}]'.format(i): s for i, s in enumerate(source_skids)})
    post.update({'columns[{}]'.format(i): s for i, s in enumerate(target_skids)})
    post['with_locations'] = bool(volume_filter) or use_connectors

    # Data will be in format::
    # {'source_skid': {'target_skid': {'count': int,
    #                                  'locations': {connector_id: {'pos': [x, y, z],
    #                                                               'count: int'}}}}}
    data = remote_instance.fetch(url, post=post)

    # Check which connectors to keep
    if use_connectors or bool(volume_filter):
        # Extract connector IDs and their locations
        cn_loc = {cn: t['locations'][cn]['pos'] for s in data.values() for t in s.values() for cn in t['locations']}
        to_keep = set(cn_loc.keys())

        # Remove connectors that aren't part of the neurons anymore
        if use_connectors:
            if isinstance(sources, (core.CatmaidNeuron, core.CatmaidNeuronList)):
                to_keep = to_keep & set(sources.connectors.connector_id.values.astype(str))

            if isinstance(targets, (core.CatmaidNeuron, core.CatmaidNeuronList)):
                to_keep = to_keep & set(targets.connectors.connector_id.values.astype(str))

        # Turn into list so we can check for in_volume
        to_keep = np.array(list(to_keep))
        if bool(volume_filter):
            volume_filter = utils._make_iterable(volume_filter)
            for vol in volume_filter:
                if not isinstance(vol, core.Volume):
                    vol = fetch.get_volume(vol, remote_instance=remote_instance)
                # Get positions of remaining connectors
                pos = np.array([cn_loc[cn] for cn in to_keep])
                in_vol = intersect.in_volume(pos, vol)
                to_keep = to_keep[in_vol]

        # Now recount depending on left-over connectors
        to_keep = set(to_keep)  # this speeds up the "in" query by ALOT
        for s in data:
            for t in data[s]:
                # Sum up count for connectors that we have kept
                new_count = sum([val['count'] for cn, val in data[s][t]['locations'].items() if cn in to_keep])
                data[s][t]['count'] = new_count

        # Change to records
        records = {s: {t: val['count'] for t, val in data[s].items()} for s in data}
    else:
        records = data

    # Parse data into adjacency matrix
    matrix = pd.DataFrame.from_records(records).fillna(0).T

    # Make sure Skids are integers
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    # Filter and sort to actual sources and targets
    matrix = matrix.reindex(index=source_skids, columns=target_skids, fill_value=0)

    # Apply cutoff and threshold
    matrix = matrix.clip(upper=syn_cutoff)
    if syn_threshold:
        matrix[matrix < syn_threshold] = 0

    # Convert to fractions
    if fractions:
        cn_counts = fetch.get_connectivity_counts(target_skids,
                                                  source_relations=['postsynaptic_to'],
                                                  target_relations=['presynaptic_to'],
                                                  remote_instance=remote_instance)
        cn_counts = cn_counts['connectivity']

        div = [list(cn_counts.get(s).values())[0] for s in matrix.columns.astype(str)]

        matrix = matrix / div

    matrix.datatype = 'adjacency_matrix'

    if source_grp or target_grp:
        matrix = group_matrix(matrix,
                              source_grp,
                              target_grp,
                              drop_ungrouped=False)

    # Make pretty
    matrix.columns.name = 'targets'
    matrix.index.name = 'sources'

    return matrix


def group_matrix(mat, row_groups={}, col_groups={}, drop_ungrouped=False,
                 method='SUM', remote_instance=None):
    """Groups adjacency matrix into neuron groups.

    Parameters
    ----------
    mat :               pandas.DataFrame | numpy.array
                        Matrix to group.
    row_groups :        dict, optional
                        Row groups to be formed. Can be either:
                          1. ``{group1: [neuron1, neuron2, ...], ...}``
                          2. ``{neuron1: group1, neuron2:group2, ...}``
                        If grouping numpy arrays, use indices!
    col_groups :        dict, optional
                        Col groups. See ``row_groups`` for details.
    drop_ungrouped :    bool, optional
                        If ungrouped, neurons that are not part of a
                        row/col_group are dropped from the matrix.
    method :            'AVERAGE' | 'MAX' | 'MIN' | 'SUM', optional
                        Method by which values are collapsed into groups.
    remote_instance :   CatmaidInstance, optional
                        If not passed, will try using globally defined.


    Returns
    -------
    pandas.DataFrame

    """
    remote_instance = utils._eval_remote_instance(remote_instance,
                                                  raise_error=False)

    PERMISSIBLE_METHODS = ['AVERAGE', 'MIN', 'MAX', 'SUM']
    if method not in PERMISSIBLE_METHODS:
        raise ValueError('Unknown method "{0}". Please use either {1}'.format(
            method, ','.join(PERMISSIBLE_METHODS)))

    if not row_groups and not col_groups:
        logger.warning('No column/row groups provided - skipping.')
        return mat

    # Convert numpy array to DataFrame
    if isinstance(mat, np.ndarray):
        mat = pd.DataFrame(mat)
    # Make copy of original DataFrame
    elif isinstance(mat, pd.DataFrame):
        mat = mat.copy()
    else:
        raise TypeError('Can only work with numpy arrays or pandas '
                        'DataFrames, got "{}"'.format(type(mat)))

    # Convert to neuron->group format if necessary
    if col_groups and utils._is_iterable(list(col_groups.values())[0]):
        col_groups = {n: g for g in col_groups for n in utils.eval_skids(col_groups[g], remote_instance=remote_instance)}
    if row_groups and utils._is_iterable(list(row_groups.values())[0]):
        row_groups = {n: g for g in row_groups for n in utils.eval_skids(row_groups[g], remote_instance=remote_instance)}

    # Make sure everything is string
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    col_groups = {str(k): str(v) for k, v in col_groups.items()}
    row_groups = {str(k): str(v) for k, v in row_groups.items()}

    if row_groups:
        # Drop non-grouped values if applicable
        if drop_ungrouped:
            mat = mat.loc[mat.index.isin(row_groups.keys())]

        # Add temporary grouping column
        mat['row_groups'] = [row_groups.get(s, s) for s in mat.index]

        if method == 'AVERAGE':
            mat = mat.groupby('row_groups').mean()
        elif method == 'MAX':
            mat = mat.groupby('row_groups').max()
        elif method == 'MIN':
            mat = mat.groupby('row_groups').min()
        elif method == 'SUM':
            mat = mat.groupby('row_groups').sum()

    if col_groups:
        # Transpose for grouping
        mat = mat.T

        # Drop non-grouped values if applicable
        if drop_ungrouped:
            mat = mat.loc[mat.index.isin(col_groups.keys())]

        # Add temporary grouping column
        mat['col_groups'] = [col_groups.get(s, s) for s in mat.index]

        if method == 'AVERAGE':
            mat = mat.groupby('col_groups').mean()
        elif method == 'MAX':
            mat = mat.groupby('col_groups').max()
        elif method == 'MIN':
            mat = mat.groupby('col_groups').min()
        elif method == 'SUM':
            mat = mat.groupby('col_groups').sum()

        # Transpose back
        mat = mat.T

    # Preserve datatype
    mat.datatype = 'adjacency_matrix'
    # Add flag that this matrix has been grouped
    mat.is_grouped = True

    return mat


def connection_density(s, t, method='MEDIAN', normalize='DENSITY',
                       remote_instance=None):
    """Calculate connection density.

    Given source neuron(s) ``s`` and a target neuron ``t``, calculate the
    local density of connections as function of the geodesic distance between
    the postsynaptic contacts on target neuron ``t``.

    The general idea here is that spread out contacts might be more
    effective in depolarizing dendrites of the postsynaptic neuron than highly
    localized ones. See `Gouwens and Wilson, Journal of Neuroscience (2009)
    <https://www.ncbi.nlm.nih.gov/pubmed/19439602>`_ for example.

    Parameters
    ----------
    s,t :               skeleton ID | CatmaidNeuron | CatmaidNeuronList
                        Source and target neuron, respectively. Multiple
                        sources are allowed. Target must be single neuron.
                        If ``t`` is a CatmaidNeuron, will use connectors and
                        total cable to this neuron. Use to subset density
                        calculations to e.g. the dendrites.
    method :            'SUM' | 'AVERAGE' | 'MEDIAN', optional
                        Arithmetic method used to collapse pairwise geodesic
                        distances over all synaptic contacts on ``t`` from
                        ``s`` into connection density ``D``.
    normalize :         'DENSITY' | 'CABLE' | False
                        Normalization method:

                        - ``DENSITY``: normalize by synapse density
                          over all postsynapses of the target
                        - ``CABLE``: normalize by total cable length
                          of the target
                        - ``False``: no normalization

    remote_instance :   CatmaidInstance, optional

    Returns
    -------
    connection density : float
                         Will return ``None`` if no connections or if only a
                         single connection between source and target.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    source_skid = utils.eval_skids(s, remote_instance=remote_instance)
    target_skid = utils.eval_skids(t, remote_instance=remote_instance)

    if len(target_skid) != 1:
        raise ValueError('Must provide a single target neuron.')

    if normalize not in [False, 'DENSITY', 'CABLE']:
        raise ValueError('Unknown normalization method "{}"'.format(normalize))

    # Get connectors between source and target and extract postsynaptic
    # treenode IDs
    cn_between = fetch.get_connectors_between(source_skid, target_skid,
                                              directional=True,
                                              remote_instance=remote_instance)
    post_tn = cn_between.node2_id.values

    # Make sure we have neuron to work with
    if not isinstance(t, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        t = fetch.get_neuron(t, remote_instance=remote_instance)
    # If t already is a neuron, subset connectors to those that actually exists
    else:
        post_tn = np.intersect1d(post_tn, t.nodes.node_id.values)

    if isinstance(t, core.CatmaidNeuronList):
        t = t[0]

    # If no connections, return 0
    if not any(post_tn):
        return None
    # If there is only one connection, we won't be able to calculate a density
    elif post_tn.shape[0] == 1:
        return None

    # If we are normalizing to density, include all synapses in geodesic
    # calculations
    if normalize == 'DENSITY':
        to_calc = t.postsynapses.node_id.values
    else:
        to_calc = post_tn

    # Get geodesic distances
    m = graph_utils.geodesic_matrix(t,
                                    to_calc,
                                    directed=False,
                                    weight='weight')

    # Get distances from matrix and convert to microns
    dist = np.array([m.loc[i, j] for i, j in combinations(post_tn, 2)]) / 1000

    # Prepare normalization
    if normalize is None:
        norm = [1]
    elif normalize == 'DENSITY':
        all_post = t.postsynapses.node_id.values
        norm = np.array([m.loc[i, j] for i, j in combinations(all_post, 2)]) / 1000
    elif normalize == 'CABLE':
        norm = [t.cable_length]

    # Combine distances and turn distance into density (1/dist)
    if method == 'SUM':
        dist = np.sum(norm) / np.sum(dist)
    elif method == 'AVERAGE':
        dist = np.average(norm) / np.average(dist)
    elif method == 'MEDIAN':
        dist = np.median(norm) / np.median(dist)
    else:
        raise ValueError('Unknown method "{}"'.format(method))

    # It's possible that all connections are onto the same treenode in which
    # case average/sum/median distance would be 0 -> we will return this as
    # None
    if dist == 0:
        return None

    return dist


def sparseness(x, which='LTS'):
    """Calculate sparseness.

    Sparseness comes in two flavors:

    **Lifetime kurtosis (LTK)** quantifies the widths of tuning curves
    (according to Muench & Galizia, 2016):

    .. math::

        S = \\Bigg\\{ \\frac{1}{N} \\sum^N_{i=1} \\Big[ \\frac{r_i - \\overline{r}}{\\sigma_r} \\Big] ^4  \\Bigg\\} - 3

    where :math:`N` is the number of observations, :math:`r_i` the value of
    observation :math:`i`, and :math:`\\overline{r}` and
    :math:`\\sigma_r` the mean and the standard deviation of the observations'
    values, respectively. LTK is assuming a normal, or at least symmetric
    distribution.

    **Lifetime sparseness (LTS)** quantifies selectivity
    (Bhandawat et al., 2007):

    .. math::

        S = \\frac{1}{1-1/N} \\Bigg[1- \\frac{\\big(\\sum^N_{j=1} r_j / N\\big)^2}{\\sum^N_{j=1} r_j^2 / N} \\Bigg]

    where :math:`N` is the number of observations, and :math:`r_j` is the
    value of an observation.

    Notes
    -----
    ``NaN`` values will be ignored. You can use that to e.g. ignore zero
    values in a large connectivity matrix by changing these values to ``NaN``
    before passing it to ``pymaid.sparseness``.


    Parameters
    ----------
    x :         DataFrame | array-like
                (N, M) dataset with N (rows) observations for M (columns)
                neurons. One-dimensional data will be converted to two
                dimensions (N rows, 1 column).
    which :     "LTS" | "LTK"
                Determines whether lifetime sparseness (LTS) or lifetime
                kurtosis (LTK) is returned.

    Returns
    -------
    sparseness
                ``pandas.Series`` if input was pandas DataFrame, else
                ``numpy.array``.

    Examples
    --------
    Calculate sparseness of olfactory inputs to group of neurons:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> # Generate adjacency matrix
    >>> adj = pymaid.adjacency_matrix(s='annotation:WTPN2017_excitatory_uPN_right',
    ...                               t='annotation:ASB LHN')
    >>> # Calculate lifetime sparseness
    >>> S = pymaid.sparseness(adj, which='LTS')
    >>> # Plot distribution
    >>> ax = S.plot.hist(bins=np.arange(0, 1, .1))
    >>> ax.set_xlabel('LTS')
    >>> plt.show()

    """
    if not isinstance(x, (pd.DataFrame, np.ndarray)):
        x = np.array(x)

    # Make sure we are working with 2 dimensional data
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x = x.reshape(x.shape[0], 1)

    N = np.sum(~np.isnan(x), axis=0)

    if which == 'LTK':
        return np.nansum(((x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)) ** 4, axis=0) / N - 3
    elif which == 'LTS':
        return 1 / (1 - (1 / N)) * (1 - np.nansum(x / N, axis=0) ** 2 / np.nansum(x ** 2 / N, axis=0))
    else:
        raise ValueError('Parameter "which" must be either "LTS" or "LTK"')


def shared_partners(a, b, upstream=True, downstream=True, syn_threshold=None,
                    restrict_to=None, remote_instance=None):
    """Return shared partners between neuron(s) A and B.

    Parameters
    ----------
    a,b :                   CatmaidNeuron/List
                            Neurons to search shared partners for.
    upstream, downstream:   bool, int, optional
                            Set to True/False to restrict direction.
    syn_threshold :         int, optional
                            Synapse threshold. Edges to both neurons A and B
                            must be above threshold!
    restrict_to :           str | pymaid.Volume, optional
                            Volume to restrict connectivity to.
    remote_instance :       CatmaidInstance, optional
                            If not passed, will try using globally defined.

    Returns
    -------
    pandas.DataFrame
                            Pandas DataFrame with shared partners and edges::

                                neuron_name  skeleton_id  relation edges_a  edges_b
                             0
                             1
                             2

    """
    if isinstance(a, core.CatmaidNeuron):
        a = core.CatmaidNeuronList(a)
    elif not isinstance(a, core.CatmaidNeuronList):
        raise TypeError('Expected CatmaidNeuron/List, got {}'.format(type(a)))

    if isinstance(b, core.CatmaidNeuron):
        b = core.CatmaidNeuronList(b)
    elif not isinstance(b, core.CatmaidNeuronList):
        raise TypeError('Expected CatmaidNeuron/List, got {}'.format(type(b)))

    if not isinstance(syn_threshold, (float, int)):
        syn_threshold = 1
    elif syn_threshold <= 0:
        raise ValueError('Synapse threshold must not be <= 0.')

    remote_instance = utils._eval_remote_instance(remote_instance)

    cn = fetch.get_partners(a + b, remote_instance=remote_instance)

    if not upstream:
        cn = cn[cn.relation != 'upstream']

    if not downstream:
        cn = cn[cn.relation != 'downstream']

    if restrict_to:
        cn = filter_connectivity(cn, restrict_to=restrict_to,
                                 remote_instance=remote_instance)

    # Collapse into A + B by direction
    cn['edges_a'] = 0
    cn['edges_b'] = 0
    for rel in cn.relation.unique():
        cn.loc[cn.relation == rel, 'edges_a'] = cn.loc[cn.relation == rel, a.skeleton_id].sum(axis=1)
        cn.loc[cn.relation == rel, 'edges_b'] = cn.loc[cn.relation == rel, b.skeleton_id].sum(axis=1)

    # Remove connections where either a or b are sub-threshold
    cn = cn[(cn['edges_a'] >= syn_threshold) & (cn['edges_b'] >= syn_threshold)]

    return cn[['neuron_name', 'skeleton_id', 'relation', 'edges_a', 'edges_b']].reset_index()
