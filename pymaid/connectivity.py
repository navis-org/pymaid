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


""" This module contains functions to analyse connectivity.
"""

import pandas as pd
import numpy as np
import scipy
import scipy.spatial

from pymaid import fetch, core, intersect, utils, config

# Set up logging
logger = config.logger

__all__ = sorted(['filter_connectivity', 'cable_overlap',
                  'predict_connectivity', 'adjacency_matrix', 'group_matrix',
                  'adjacency_from_connectors', 'cn_table_from_connectors'])


def filter_connectivity(x, restrict_to, remote_instance=None):
    """ Filters connectivity data by volume or skeleton data.

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
    remote_instance :   CATMAID instance, optional
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
        raise TypeError(
            'Unable to restrict connectivity to type'.format(type(restrict_to)))

    if isinstance(restrict_to, str):
        restrict_to = fetch.get_volume(
            restrict_to, remote_instance=remote_instance)

    datatype = getattr(x, 'datatype', None)

    if datatype not in ['connectivity_table', 'adjacency_matrix']:
        raise TypeError(
            'Unknown connectivity data. See help(filter_connectivity) for details.')

    if datatype == 'connectivity_table':
        neurons = [c for c in x.columns if c not in [
            'neuron_name', 'skeleton_id', 'num_nodes', 'relation', 'total']]

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
                                                      x[x.relation ==
                                                          'downstream'].skeleton_id,
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
            raise TypeError(
                'Adjacency matrix appears to be grouped. Unable to process that.')

        cn_data = fetch.get_connectors_between(x.index.tolist(),
                                               x.columns.tolist(),
                                               directional=True,
                                               remote_instance=remote_instance)

        # Now filter connectors
        if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
            cn_data = cn_data[cn_data.connector_id.isin(
                restrict_to.connectors.connector_id.values)]
        elif isinstance(restrict_to, core.Volume):
            cn_locs = np.vstack(cn_data.connector_loc.values)
            cn_data = cn_data[intersect.in_volume(cn_locs, restrict_to)]

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
    for i, e in enumerate(config.tqdm(unique_edges, disable=config.pbar_hide, desc='Regenerating', leave=config.pbar_leave)):
        # using df.at here speeds things up tremendously!
        adj_mat.at[str(e[0]), str(e[1])] = counts[i]

    if datatype == 'adjacency_matrix':
        # Make a copy of original adjaceny matrix
        x = x.copy()
        x.datatype = 'adjacency_matrix'

        # Set everything to 0
        x[:] = 0

        # Update from filtered connectivity
        x.update(adj_mat)

        return x

    # Generate connectivity table by subsetting adjacency matrix to our neurons of interest
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
    df['num_nodes'] = [aux['num_nodes'][s] for s in df.skeleton_id.tolist()]
    df['neuron_name'] = [aux['neuron_name'][s]
                         for s in df.skeleton_id.tolist()]
    df['total'] = df[remaining_neurons].sum(axis=1)

    # Reorder columns
    df = df[['neuron_name', 'skeleton_id', 'num_nodes',
             'relation', 'total'] + remaining_neurons]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)
    df.type = 'connectivity_table'
    df.reset_index(drop=True, inplace=True)

    return df


def cable_overlap(a, b, dist=2, method='min'):
    """ Calculates the amount of cable of neuron A within distance of neuron B.

    Uses dotproduct representation of a neuron!

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
            within distance is given in microns.

            >>> df
                        skidB1   skidB2  skidB3 ...
                skidA1    5        1        0
                skidA2    10       20       5
                skidA3    4        3        15
                ...

    """

    # Convert distance to nm
    dist *= 1000

    if not isinstance(a, (core.CatmaidNeuron, core.CatmaidNeuronList)) or not isinstance(b, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise TypeError('Need to pass CatmaidNeurons')

    if isinstance(a, core.CatmaidNeuron):
        a = core.CatmaidNeuronList(a)

    if isinstance(b, core.CatmaidNeuron):
        b = core.CatmaidNeuronList(b)

    allowed_methods = ['min', 'max', 'avg']
    if method not in allowed_methods:
        raise ValueError('Unknown method "{0}". Allowed methods: "{0}"'.format(
            method, ','.join(allowed_methods)))

    matrix = pd.DataFrame(
        np.zeros((a.shape[0], b.shape[0])), index=a.skeleton_id, columns=b.skeleton_id)

    with config.tqdm(total=len(a), desc='Calc. overlap', disable=config.pbar_hide, leave=config.pbar_leave) as pbar:
        # Keep track of KDtrees
        trees = {}
        for nA in a:
            # Get cKDTree for nA
            tA = trees.get(nA.skeleton_id, None)
            if not tA:
                trees[nA.skeleton_id] = tA = scipy.spatial.cKDTree(
                    np.vstack(nA.dps.point), leafsize=10)

            for nB in b:
                # Get cKDTree for nB
                tB = trees.get(nB.skeleton_id, None)
                if not tB:
                    trees[nB.skeleton_id] = tB = scipy.spatial.cKDTree(
                        np.vstack(nB.dps.point), leafsize=10)

                # Query nB -> nA
                distA, ixA = tA.query(np.vstack(nB.dps.point),
                                      k=1,
                                      distance_upper_bound=dist,
                                      n_jobs=-1
                                      )
                # Query nA -> nB
                distB, ixB = tB.query(np.vstack(nA.dps.point),
                                      k=1,
                                      distance_upper_bound=dist,
                                      n_jobs=-1
                                      )

                nA_in_dist = nA.dps.loc[ixA[distA != float('inf')]]
                nB_in_dist = nB.dps.loc[ixB[distB != float('inf')]]

                if nA_in_dist.empty:
                    overlap = 0
                elif method == 'avg':
                    overlap = (nA_in_dist.vec_length.sum() +
                               nB_in_dist.vec_length.sum()) / 2
                elif method == 'max':
                    overlap = max(nA_in_dist.vec_length.sum(),
                                  nB_in_dist.vec_length.sum())
                elif method == 'min':
                    overlap = min(nA_in_dist.vec_length.sum(),
                                  nB_in_dist.vec_length.sum())

                matrix.at[nA.skeleton_id, nB.skeleton_id] = overlap

            pbar.update(1)

    # Convert to um
    matrix /= 1000

    return matrix


def predict_connectivity(source, target, method='possible_contacts',
                         remote_instance=None, **kwargs):
    """ Calculates potential synapses from source onto target neurons.

    Based on a concept by Alex Bates.

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
                          treenodes manually.
                        - ``stdev`` to set number of standard-deviations of
                          average distance. Default = 2.

    Notes
    -----
    Method ``possible_contacts``:
        1. Calculating mean distance ``d`` (connector->treenode) at which
           connections between neurons A and neurons B occur.
        2. For all presynapses of neurons A, check if they are within `stdev`
           (default=2) standard deviations of ``d`` of a neurons B treenode.


    Returns
    -------
    pandas.DataFrame
            Matrix holding possible synaptic contacts. Sources are rows,
            targets are columns.

            >>> df
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
        except:
            pass

    for _ in [source, target]:
        if not isinstance(_, (core.CatmaidNeuron, core.CatmaidNeuronList)):
            raise TypeError('Need CatmaidNeuron/List, got "{}"'.format(type(_)))

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

    # First let's calculate at what distance synapses are being made
    cn_between = fetch.get_connectors_between(source, target,
                                              remote_instance=remote_instance)

    if kwargs.get('dist', None):
        distances = kwargs.get('dist')
    elif cn_between.shape[0] > 0:
        logger.warning('No ')
        cn_locs = np.vstack(cn_between.connector_loc.values)
        tn_locs = np.vstack(cn_between.treenode2_loc.values)

        distances = np.sqrt(np.sum((cn_locs - tn_locs) ** 2, axis=1))

        logger.info(
            'Average connector->treenode distances: {:.2f} +/- {:.2f} nm'.format(distances.mean(), distances.std()))
    else:
        logger.warning('No existing connectors to calculate average \
                        connector->treenode distance found. Falling \
                        back to default of 1um. Use <stdev> argument\
                        to set manually.')
        distances = 1000

    # Calculate distances threshold
    n_std = kwargs.get('n_std', 2)
    dist_threshold = np.mean(distances) + n_std * np.std(distances)

    with config.tqdm(total=len(target), desc='Predicting', disable=config.pbar_hide,
              leave=config.pbar_leave) as pbar:
        for t in target:
            # Create cKDTree for target
            tree = scipy.spatial.cKDTree(
                t.nodes[['x', 'y', 'z']].values, leafsize=10)
            for s in source:
                # Query against presynapses
                dist, ix = tree.query(s.presynapses[['x', 'y', 'z']].values,
                                      k=1,
                                      distance_upper_bound=dist_threshold,
                                      n_jobs=-1
                                      )

                # Calculate possible contacts
                possible_contacts = sum(dist != float('inf'))

                matrix.at[s.skeleton_id, t.skeleton_id] = possible_contacts

            pbar.update(1)

    return matrix.astype(int)


def cn_table_from_connectors(x, remote_instance=None):
    """ Generate connectivity table from neurons' connectors.

    This function creates the connectivity table from scratch using just the
    neurons' connectors. This function is able to deal with non-unique
    skeleton IDs (most other functions won't). Use it e.g. when you
    split neurons into multiple fragments. *The order of the input
    CatmaidNeuronList is preserved!*

    Parameters
    ----------
    x :         CatmaidNeuron | CatmaidNeuronList
                Neuron(s) for which to generate connectivity table.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons:

        >>> df
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
    >>> # in this example, the first occurrence is axon, the second dendrites
    >>> cn_table.head()

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise TypeError('Need CatmaidNeuron/List, got "{}"'.format(type(x)))

    if isinstance(x, core.CatmaidNeuron):
        x = core.CatmaidNeuronList(x)

    # Get connector details for all neurons
    all_cn = x.connectors.connector_id.values
    cn_details = fetch.get_connector_details(all_cn)

    # Remove connectors for which there are either no pre- or no postsynaptic
    # neurons
    cn_details = cn_details[cn_details.postsynaptic_to.apply(len) != 0]
    cn_details = cn_details[~cn_details.presynaptic_to.isnull()]

    # We need to map treenode ID to skeleton ID in cases where there are more
    # links (postsynaptic_to_node) than targets (postsynaptic_to)
    multi_links = cn_details[cn_details.postsynaptic_to.apply(len) < cn_details.postsynaptic_to_node.apply(len)]
    if not multi_links.empty:
        tn_to_fetch = [tn for l in multi_links.postsynaptic_to_node for tn in l]
        tn_to_skid = fetch.get_skid_from_treenode(tn_to_fetch,
                                                  remote_instance=remote_instance)
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
        this_tn = set(n.nodes.treenode_id.values)
        # Prepare upstream partners
        this_us = all_pre[all_pre.connector_id.isin(n.connectors.connector_id.values)]
        # Get the number of all links per connector
        this_us['n_links'] = [len(this_tn & set(r.postsynaptic_to_node))
                                           for r in this_us.itertuples()]
        # Group by input and store as dict
        us_dict[n] = this_us.groupby('presynaptic_to').n_links.sum().to_dict()
        this_us = this_us.groupby('presynaptic_to').n_links.sum()

        # Now prepare downstream partners:
        # Get all downstream connectors
        this_ds = all_post[all_post.presynaptic_to==int(n.skeleton_id)]
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
    cn_table = pd.concat([us_table,ds_table], axis=0)

    # Replace NaN with 0
    cn_table = cn_table.fillna(0)

    # Make skeleton ID a column
    cn_table = cn_table.reset_index(drop=False)
    cn_table.columns = ['skeleton_id'] + list(cn_table.columns[1:])

    # Add names
    names = fetch.get_names(cn_table.skeleton_id.values)
    cn_table['neuron_name'] = [names[str(s)] for s in cn_table.skeleton_id.values]
    cn_table['total'] = cn_table[x.skeleton_id].sum(axis=1)

    # Drop rows with 0 synapses (e.g. if neuron is only up- but not downstream)
    cn_table = cn_table[cn_table.total > 0]

    # Sort by number of synapses
    cn_table = cn_table.sort_values(['relation', 'total'], ascending=False)

    # Sort columnes
    cn_table = cn_table[['neuron_name', 'skeleton_id', 'relation', 'total'] + list(set(x.skeleton_id))]

    return cn_table


def adjacency_from_connectors(source, target=None, remote_instance=None):
    """ Regenerates adjacency matrices from neurons' connectors.

    Notes
    -----
    This function creates an adjacency matrix from scratch using just the
    neurons' connectors. This function is able to deal with non-unique
    skeleton IDs (most other functions are not). Use it e.g. when you
    split neurons into multiple fragments.

    Parameters
    ----------
    source,target : skeleton IDs | CatmaidNeuron | CatmaidNeuronList
                    Neuron(s) for which to generate adjacency matrix.
                    If ``target==None``, will use ``target=source``.

    Returns
    -------
    pandas.DataFrame
            Matrix holding possible synaptic contacts. Sources are rows,
            targets are columns. Labels are skeleton IDs. Order is preserved.

            >>> df
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
        skids = utils.eval_skids(source)
        source = fetch.get_neuron(skids, remote_instance=remote_instance)

    if isinstance(target, type(None)):
        target = source
    elif not isinstance(target, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        skids = utils.eval_skids(target)
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
    cn_details = fetch.get_connector_details(all_cn)

    # Now go over all source neurons and process connections
    for i, s in enumerate(config.tqdm(source, desc='Processing',
                          disable=config.pbar_hide, leave=config.pbar_leave)):

        # Get all connectors presynaptic for this source
        this_cn = cn_details[(cn_details.presynaptic_to == int(s.skeleton_id)) &
                             (cn_details.connector_id.isin(s.connectors.connector_id))
                             ]

        # Go over all target neurons
        for k, t in enumerate(target):
            t_tn = set(t.nodes.treenode_id.values)
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
    """ Generates list of edges between two sets of neurons from their
    connector data.

    Attention: this is UNIDIRECTIONAL (a->b)!

    Parameters
    ----------
    a, b :      CatmaidNeuron | CatmaidNeuronList | skeleton IDs
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


def adjacency_matrix(s, t=None, remote_instance=None, source_grp={},
                     target_grp={}, syn_threshold=None, syn_cutoff=None,
                     use_connectors=False):
    """ Generate adjacency matrix between sets of neurons.

    Directional: sources = rows, targets = columns.

    Parameters
    ----------
    s
                        Source neurons as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    t
                        Optional. Target neurons as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. ``'annotation:PN right'`
                        4. CatmaidNeuron or CatmaidNeuronList object

                        If not provided, ``source neurons = target neurons``.
    remote_instance :   CATMAID instance, optional
    syn_cutoff :        int, optional
                        If set, will cut off connections above given value.
    syn_threshold :     int, optional
                        If set, will ignore connections with less synapses.
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
                        if e.g. you are using pruned neurons. **Important**:
                        This does not work if you have multiple fragments per
                        neuron!

    Returns
    -------
    matrix :          ``pandas.Dataframe``

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
    >>> import pymaid
    >>> rm = pymaid.CatmaidInstance(url, user, pw,, token)
    >>> neurons = pymaid.get_neurons('annotation:test')
    >>> mat = pymaid.adjacency_matrix( neurons )
    >>> g = sns.heatmap(adj_mat, square=True)
    >>> g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 7)
    >>> g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 7)
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
    >>> ds_partners = cn_table[ cn_table.relation == 'downstream' ]
    >>> # Take the top 10 downstream partners
    >>> top_ds = ds_partners.iloc[:10].skeleton_id.values
    >>> # Generate separate adjacency matrices for axon and dendrites
    >>> adj_axon = pymaid.adjacency_matrix(nl_axon, top_ds, use_connectors=True )
    >>> adj_dend = pymaid.adjacency_matrix(nl_dend, top_ds, use_connectors=True )
    >>> # Rename rows and merge dataframes
    >>> adj_axon.index += '_axon'
    >>> adj_dend.index += '_dendrite'
    >>> adj_merged = pd.concat([adj_axon, adj_dend], axis=0)
    >>> # Plot heatmap using seaborn
    >>> ax = sns.heatmap(adj_merged)
    >>> plt.show()

    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    if t is None:
        t = s

    neuronsA = utils.eval_skids(s, remote_instance=remote_instance)
    neuronsB = utils.eval_skids(t, remote_instance=remote_instance)

    # Make sure neurons are  integers
    neurons = list(set([int(n) for n in (neuronsA + neuronsB)]))
    neuronsA = [int(n) for n in neuronsA]
    neuronsB = [int(n) for n in neuronsB]

    # Make sure neurons are unique
    neuronsA = sorted(set(neuronsA), key=neuronsA.index)
    neuronsB = sorted(set(neuronsB), key=neuronsB.index)

    logger.info('Retrieving and filtering connectivity...')

    if use_connectors and (isinstance(s, (core.CatmaidNeuron, core.CatmaidNeuronList))
                           or isinstance(t, (core.CatmaidNeuron, core.CatmaidNeuronList))):
        edges = _edges_from_connectors(
            s, t, remote_instance=remote_instance)
    else:
        edges = fetch.get_edges(neurons, remote_instance=remote_instance)

    # Turn into a adjacency matrix
    matrix = edges.pivot(values='weight',
                         columns='target_skid',
                         index='source_skid').fillna(0)

    # Filter to actual sources and targets
    matrix = matrix.reindex(neuronsA, columns=neuronsB, fill_value=0)

    # Apply cutoff and threshold
    matrix = matrix.clip(upper=syn_cutoff)

    if syn_threshold:
        matrix[matrix < syn_threshold] = 0

    matrix.datatype = 'adjacency_matrix'

    if source_grp or target_grp:
        matrix = group_matrix(matrix,
                              source_grp,
                              target_grp,
                              drop_ungrouped=False)

    logger.info('Finished!')

    return matrix


def group_matrix(mat, row_groups={}, col_groups={}, drop_ungrouped=False,
                 method='SUM'):
    """ Groups adjacency matrix into neuron groups.

    Parameters
    ----------
    mat :               pandas.DataFrame | numpy.array
                        Matrix to group.
    row_groups :        dict, optional
                        Row groups to be formed. Can be either:
                          1. ``{group1 : [neuron1, neuron2, ...], ...}``
                          2. ``{neuron1 : group1, neuron2: group2, ...}``
                        If grouping numpy arrays, use indices!
    col_groups :        dict, optional
                        Col groups. See ``row_groups`` for details.
    drop_ungrouped :    bool, optional
                        If ungrouped, neurons that are not part of a
                        row/col_group are dropped from the matrix.
    method :            'AVERAGE' | 'MAX' | 'MIN' | 'SUM', optional
                        Method by which values are collapsed into groups.

    Returns
    -------
    pandas.DataFrame
    """

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
        raise TypeError(
            'Can only work with numpy arrays or pandas DataFrames, got "{}"'.format(type(mat)))

    # Convert to neuron->group format if necessary
    if col_groups and utils._is_iterable(list(col_groups.values())[0]):
        col_groups = {n: g for g in col_groups for n in col_groups[g]}
    if row_groups and utils._is_iterable(list(row_groups.values())[0]):
        row_groups = {n: g for g in row_groups for n in row_groups[g]}

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
