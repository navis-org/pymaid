# A collection of tools to remotely access a CATMAID server via its API
#
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

"""This module contains functions to request data from Catmaid server.

Examples
--------
>>> import pymaid
>>> # HTTP_USER AND HTTP_PASSWORD are only necessary if your server requires a
>>> # http authentification
>>> myInstance = pymaid.CatmaidInstance('www.your.catmaid-server.org',
...                                     'api_token',
...                                     'http_user', # Omit if not required
...                                     'http_password')
>>> # Get skeletal data for two neurons
>>> neuron_list = pymaid.get_neuron(['12345', '67890'], myInstance)
>>> neuron_list[0]
type              <class 'pymaid.CatmaidNeuron'>
neuron_name                       Example neuron name
skeleton_id                                     12345
n_nodes                                          9924
n_connectors                                      437
n_branch_nodes                                    207
n_end_nodes                                       214
cable_length                                  1479.81
review_status                                      NA
annotations                                     False
igraph                                          False
tags                                             True
dtype: object

"""

import datetime
import re
import urllib
import webbrowser

import navis as ns
import numpy as np
import networkx as nx
import pandas as pd

from . import core, utils, config, cache
from navis import in_volume


__all__ = sorted(['get_annotation_details', 'get_annotation_id',
                  'get_annotation_list', 'get_annotations', 'get_arbor',
                  'get_connector_details', 'get_connectors',
                  'get_connector_tags',
                  'get_contributor_statistics', 'get_edges', 'get_history',
                  'get_logs', 'get_names', 'get_neuron',
                  'get_neurons', 'get_neurons_in_bbox',
                  'get_neurons_in_volume', 'get_node_tags', 'get_node_details',
                  'get_nodes_in_volume', 'get_partners',
                  'get_partners_in_volume', 'get_paths', 'get_review',
                  'get_review_details', 'get_skids_by_annotation',
                  'get_skids_by_name', 'get_node_info',
                  'get_node_table', 'get_user_annotations',
                  'get_user_list', 'get_volume', 'has_soma', 'neuron_exists',
                  'get_segments',
                  'get_connectors_between', 'url_to_coordinates',
                  'get_label_list', 'find_neurons',
                  'get_skid_from_node', 'get_transactions',
                  'get_connector_links',
                  'get_nth_partners', 'find_nodes',
                  'get_node_location', 'get_annotated',
                  'get_neuron_id',
                  'get_connectors_in_bbox',
                  'get_cable_lengths',
                  'get_connectivity_counts',
                  'get_import_info',
                  'get_origin', 'get_skids_by_origin',
                  'get_sampler', 'get_sampler_domains', 'get_sampler_counts',
                  'get_skeleton_change'])

# Set up logging
logger = config.logger


@cache.undo_on_error
def get_neuron(x, with_connectors=True, with_tags=True, with_history=False,
               with_merge_history=False, with_abutting=False, return_df=False,
               fetch_kwargs={}, init_kwargs={}, raise_missing=True,
               remote_instance=None):
    """Retrieve 3D skeleton data as CatmaidNeuron/List.

    Parameters
    ----------
    x
                        Can be either:

                        1. list of skeleton ID(s), int or str
                        2. list of neuron name(s), str, exact match
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    with_connectors :   bool, optional
                        If True, will include connector data.
                        Note: the CATMAID API endpoint does currently not
                        support retrieving abutting connectors this way.
                        Please use ``with_abutting=True`` to include
                        abutting connectors.
    with_tags :         bool, optional
                        If True, will include node tags.
    with_history:       bool, optional
                        If True, the returned node data will contain
                        creation date and last modified for each
                        node.

                        ATTENTION: if ``with_history=True``, nodes/connectors
                        that have been moved since their creation will have
                        multiple entries reflecting their changes in position!
                        Each state has the date it was modified as creation
                        date and the next state's date as last modified. The
                        most up to date state has the original creation date
                        as last modified.
                        The creator_id is always the original creator though.
    with_abutting:      bool, optional
                        If True, will retrieve abutting connectors.
                        For some reason they are not part of compact-json, so
                        they have to be retrieved via a separate API endpoint
                        -> will show up as connector type 3!
    return_df :         bool, optional
                        If True, a ``pandas.DataFrame`` instead of
                        ``CatmaidNeuron``/``CatmaidNeuronList`` is returned.
    fetch_kwargs :      dict, optional
                        Above BOOLEAN parameters can also be passed as dict.
                        This is then used in CatmaidNeuron objects to
                        override implicitly set parameters!
    init_kwargs :       dict, optional
                        Keyword arguments passed when initializing
                        ``CatmaidNeuron``/``CatmaidNeuronList``.
    raise_missing :     bool, optional
                        If True and any of the queried neurons can not be
                        found, raise an exception. Else just log a warning.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    :class:`~pymaid.CatmaidNeuron`
                        For single neurons.
    :class:`~pymaid.CatmaidNeuronList`
                        For a list of neurons.
    pandas.DataFrame
                        If ``return_df=True``

    Notes
    -----
    The returned objects contain for each neuron::

        neuron_name :           str
        skeleton_id :           str
        nodes / connectors :    pandas.DataFrames containing node/connector
                                ID, coordinates, parent nodes, etc.
        tags :                  dict containing the node tags:
                                ``{'tag': [node_id, node_id, ...]}``

    Dataframe column titles for ``nodes`` and ``connectors`` should be
    self-explanatory with the exception of ``relation`` in connector table.
    This columns describes the connection ("relation") from the neuron's
    node TO the connector::

        connectors['relation']

                    0 = "presynaptic_to" -> this is a presynapse for this neuron
                    1 = "postsynaptic_to" -> this is a postsynapse for this neuron
                    2 = "gapjunction_with"
                    3 = "abutting" (not returned by default)
                    -1 = other (hypothetical as CATMAID does only return the above)

    Examples
    --------
    >>> # Get a single neuron by skeleton id
    >>> n = pymaid.get_neuron(16)
    >>> # Get a bunch of neurons by annotation
    >>> n = pymaid.get_neuron('annotation:glomerulus DA1')

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    # Update from kwargs if available
    with_tags = fetch_kwargs.get('with_tags', with_tags)
    with_connectors = fetch_kwargs.get('with_connectors', with_connectors)
    with_history = fetch_kwargs.get('with_history', with_history)
    with_merge_history = fetch_kwargs.get('with_merge_history', with_merge_history)
    with_abutting = fetch_kwargs.get('with_abutting', with_abutting)
    return_df = fetch_kwargs.get('return_df', return_df)

    # Generate URLs to retrieve
    urls = [remote_instance._get_compact_details_url(s,
                                                     with_history=str(with_history).lower(),
                                                     with_tags=str(with_tags).lower(),
                                                     with_connectors=str(with_connectors).lower(),
                                                     with_merge_history=str(with_merge_history).lower()) for s in x]

    skdata = remote_instance.fetch(urls, desc='Fetch neurons')

    # Retrieve abutting
    if with_abutting:
        urls = [remote_instance._get_connector_links_url(**{'skeleton_ids[0]': str(s),
                                                            'relation_type': 'abutting'}) for s in x]

        cn_data = remote_instance.fetch(urls, desc='Fetch abutting cn')

        # Add abutting to other connectors in skdata with type == 3
        for i, cn in enumerate(cn_data):
            if not with_history:
                skdata[i][1] += [[c[7], c[1], 3, c[2], c[3], c[4]]
                                 for c in cn['links']]
            else:
                skdata[i][1] += [[c[7], c[1], 3, c[2], c[3], c[4], c[8], None]
                                 for c in cn['links']]

    # Get neuron names
    names = get_names(x, remote_instance=remote_instance)

    # Parse column names
    node_cols = ['node_id', 'parent_id', 'creator_id', 'x', 'y', 'z',
                 'radius', 'confidence']
    cn_cols = ['node_id', 'connector_id', 'type', 'x', 'y', 'z']
    if with_history:
        node_cols += ['last_modified', 'creation_date', 'still_on_skeleton']
        cn_cols += ['last_modified', 'creation_date']

    missing = [s for s, d in zip(x, skdata) if not d[0]]
    if missing:
        msg = 'The following skeleton ID(s) could not be found: {}'.format(', '.join(missing))
        if raise_missing:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    def make_node_table(x):
        """Generate node table (incl. correct data types)."""
        df = pd.DataFrame(x,  columns=node_cols).fillna(-1)  # do not remove fillna
        return df.astype(config.node_dtypes)

    def make_cn_table(x):
        """Generate connector table (incl. correct data types)."""
        df = pd.DataFrame(x,  columns=cn_cols)
        return df.astype(config.connector_dtypes)

    # Generate DataFrame containing all neurons
    df = pd.DataFrame([[names[str(x[i])],  # neuron name
                        str(x[i]),  # skeleton ID
                        make_node_table(n[0]),  # nodes
                        make_cn_table(n[1]),  # connectors
                        n[2]  # tags as dictionary
                        ] for i, n in enumerate(skdata) if n[0]],
                      columns=['neuron_name', 'skeleton_id',
                               'nodes', 'connectors', 'tags'])

    if return_df:
        return df

    nl = core.CatmaidNeuronList(df, remote_instance=remote_instance, **init_kwargs)

    return nl[0] if len(nl) == 1 and len(x) == 1 else nl


# This is for legacy reasons -> will remove eventually
get_neurons = get_neuron


@cache.undo_on_error
def get_arbor(x, node_flag=1, connector_flag=1, tag_flag=1, remote_instance=None):
    """Retrieve skeleton data for a list of skeleton ids.

    Similar to :func:`pymaid.get_neuron` but the connector data includes
    the whole chain::

        node1 -> (link_confidence) -> connector -> (link_confidence) -> node2

    This means that connectors can shop up multiple times (i.e. if they have
    multiple postsynaptic targets). Does include connector ``x, y, z``
    coordinates!

    Parameters
    ----------
    x
                        Neurons to retrieve. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    connector_flag :    0 | 1, optional
                        Set if connector data should be retrieved.
    tag_flag :          0 | 1, optional
                        Set if tags should be retrieved.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.


    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron::

              neuron_name   skeleton_id   nodes      connectors   tags
            0    str           str     DataFrame     DataFrame   dict
            1
            2

    Notes
    -----
    - nodes and connectors are pandas.DataFrames themselves
    - tags is a dict: ``{'tag': [node_id, node_id, ...]}``

    Dataframe (df) column titles should be self explanatory with these exception:

    - ``df['relation_1']`` describes node1 to/from connector
    - ``df['relation_2']`` describes node2 to/from connector
    - ``relation`` can be: ``0`` (presynaptic), ``1`` (postsynaptic), ``2`` (gap junction)

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    skdata = []

    for s in config.tqdm(x, desc='Retrieving arbors', disable=config.pbar_hide,
                         leave=config.pbar_leave):
        # Create URL for retrieving example skeleton from server
        remote_compact_arbor_url = remote_instance._get_compact_arbor_url(
            s, node_flag, connector_flag, tag_flag)

        # Retrieve node_data for example skeleton
        arbor_data = remote_instance.fetch(remote_compact_arbor_url)

        skdata.append(arbor_data)

        logger.debug('%s retrieved' % str(s))

    names = get_names(x, remote_instance)

    df = pd.DataFrame([[
        names[str(x[i])],
        str(x[i]),
        pd.DataFrame(n[0], columns=['node_id', 'parent_id', 'creator_id',
                                    'x', 'y', 'z', 'radius', 'confidence']),
        pd.DataFrame(n[1], columns=['node_1', 'link_confidence',
                                    'connector_id', 'link_confidence',
                                    'node_2', 'other_skeleton_id',
                                    'relation_1', 'relation_2']),
        n[2]]
        for i, n in enumerate(skdata)
    ],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    )
    return df


@cache.undo_on_error
def get_partners_in_volume(x, volume, syn_threshold=None, min_size=2,
                           remote_instance=None):
    """Retrieve the synaptic/gap junction partners within a CATMAID Volume.

    Important
    ---------
    Connectivity (total number of connections) returned is restricted to
    that volume.

    Parameters
    ----------
    x
                        Neurons to check. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    volume :            str | list of str | navis.Volume
                        Name of the CATMAID volume to test OR volume dict with
                        {'vertices':[],'faces':[]} as returned by e.g.
                        :func:`~pymaid.get_volume()`.
    syn_threshold :     int, optional
                        Synapse threshold. This threshold is applied to the
                        TOTAL number of synapses across all neurons!
    min_size :          int, optional
                        Minimum node count of partner
                        (default = 2 -> hide single-node partner).
    remote_instance :   CatmaidInstance
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons::

          neuron_name  skeleton_id  num_nodes    relation     skid1  skid2 ...
         1  name1         skid1    node_count1   upstream     n_syn  n_syn ..
         2  name2         skid2    node_count2  downstream    n_syn  n_syn .
         3  name3         skid3    node_count3  gapjunction   n_syn  n_syn .

        - Relation can be: upstream (incoming), downstream (outgoing) of the
          neurons of interest or gap junction
        - partners can show up multiple times if they are e.g. pre- AND
          postsynaptic
        - the number of connections between two partners is restricted to the
          volume

    See Also
    --------
    :func:`~pymaid.get_neurons_in_volume`
            Get all neurons within given volume.
    :func:`~pymaid.filter_connectivity`
            Filter connectivity table or adjacency matrix by volume(s) or to
            parts of neuron(s).

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    # First, get list of connectors
    cn_data = get_connectors(x, remote_instance=remote_instance)

    # Find out which connectors are in the volume of interest
    if isinstance(volume, str):
        volume = get_volume(volume, remote_instance=remote_instance)
    elif isinstance(volume, (list, np.ndarray)):
        for i in range(len(volume)):
            if isinstance(volume[i], str):
                volume[i] = get_volume(volume[i],
                                       remote_instance=remote_instance)

    iv = in_volume(cn_data[['x', 'y', 'z']], volume)

    # Get the subset of connectors within the volume
    cn_in_volume = cn_data[iv].copy()

    logger.info('{} unique connectors in volume. Reconstructing connectivity'
                '...'.format(len(cn_in_volume.connector_id.unique())))

    # Get details for connectors in volume
    cn_details = get_connector_details(cn_in_volume.connector_id.unique(),
                                       remote_instance=remote_instance)

    # Filter those connectors that don't have a presynaptic node
    cn_details = cn_details[~cn_details.presynaptic_to.isnull()]

    # Now reconstruct connectivity table from connector details

    # Some connectors may be connected to the same neuron multiple times
    # In those cases there will be more node IDs in "postsynaptic_to_node"
    # than there are skeleton IDs in "postsynaptic_to". Then we need to map
    # node IDs to neurons
    mismatch = cn_details[cn_details.postsynaptic_to.apply(
        len) < cn_details.postsynaptic_to_node.apply(len)]
    match = cn_details[cn_details.postsynaptic_to.apply(
        len) >= cn_details.postsynaptic_to_node.apply(len)]

    if not mismatch.empty:
        logger.info('Retrieving additional details for {0} '
                    'connectors'.format(mismatch.shape[0]))
        tn_to_skid = get_skid_from_node([tn for l in mismatch.postsynaptic_to_node.values for tn in l],
                                        remote_instance=remote_instance)
    else:
        tn_to_skid = []

    # Now collect edges
    edges = [[cn.presynaptic_to, skid]
             for cn in match.itertuples() for skid in cn.postsynaptic_to]
    edges += [[cn.presynaptic_to, tn_to_skid[tn]]
              for cn in mismatch.itertuples() for tn in cn.postsynaptic_to_node]

    # Turn edges into synaptic connections
    unique_edges, counts = np.unique(edges, return_counts=True, axis=0)
    unique_skids = np.unique(edges).astype(str)
    unique_edges = unique_edges.astype(str)

    # Create empty adj_mat
    adj_mat = pd.DataFrame(np.zeros((len(unique_skids), len(unique_skids))),
                           columns=unique_skids, index=unique_skids)

    for i, e in enumerate(config.tqdm(unique_edges,
                                      disable=config.pbar_hide,
                                      desc='Adj. matrix',
                                      leave=config.pbar_leave)):
        # using df.at here speeds things up tremendously!
        adj_mat.loc[str(e[0]), str(e[1])] = counts[i]

    # There is a chance that our original neurons haven't made it through
    # filtering (i.e. they don't have partners in the volume ). We will simply
    # add these rows and columns and set them to 0
    missing = [n for n in x if n not in adj_mat.columns]
    for n in missing:
        adj_mat[n] = 0

    missing = [n for n in x if n not in adj_mat.index]
    for n in missing:
        adj_mat.loc[n] = [0 for i in range(adj_mat.shape[1])]

    # Generate connectivity table
    all_upstream = adj_mat.T[adj_mat.T[x].sum(axis=1) > 0][x]
    all_upstream['skeleton_id'] = all_upstream.index
    all_upstream['relation'] = 'upstream'

    all_downstream = adj_mat[adj_mat[x].sum(axis=1) > 0][x]
    all_downstream['skeleton_id'] = all_downstream.index
    all_downstream['relation'] = 'downstream'

    # Merge tables
    df = pd.concat([all_upstream, all_downstream], axis=0, ignore_index=True)

    # We will use this to get name and size of neurons
    logger.info('Collecting additional info for {0} neurons'.format(
        len(df.skeleton_id.unique())))
    review = get_review(df.skeleton_id.unique(),
                        remote_instance=remote_instance)
    num_nodes = review.set_index('skeleton_id').total_node_count.to_dict()
    names = get_names(df.skeleton_id.unique(), remote_instance=remote_instance)

    df['neuron_name'] = df.skeleton_id.map(names)
    df['num_nodes'] = df.skeleton_id.map(num_nodes)
    df['total'] = df[x].sum(axis=1)

    # Filter for min size
    df = df[df.num_nodes >= min_size]

    # Filter for synapse threshold
    if syn_threshold:
        df = df[df.total >= syn_threshold]

    # Reorder columns
    df = df[['neuron_name', 'skeleton_id', 'num_nodes', 'relation', 'total'] + x]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)

    return df.reset_index(drop=True)


@cache.undo_on_error
def get_nth_partners(x, n_circles=1, min_pre=2, min_post=2,
                     remote_instance=None):
    """Retrieve Nth partners.

    Partners that are directly (``n_circles = 1``) or via N "hops"
    (``n_circles>1``) connected to a set of seed neurons.

    Parameters
    ----------
    x
                        Seed neurons for which to retrieve partners. Can be:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    n_circles :         int, optional
                        Number of circles around your seed neurons.
    min_pre/min_post :  int, optional
                        Synapse threshold. Set to -1 to not get any pre-/post
                        synaptic partners.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame each row represents a partner::

           neuron_name   skeleton_id
         0   name1           123
         1   name2           456
         2   ...             ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)
    x = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_circles_of_hell_url()
    post = {'n_circles': n_circles, 'min_pre': min_pre, 'min_post': min_post}
    post.update({'skeleton_ids[{}]'.format(i): s for i, s in enumerate(x)})

    # Returns list of skids [0] and names dict [1]
    resp = remote_instance.fetch(url, post=post)

    # If no neurons returned, return empty DataFrame
    if resp[1]:
        # Generate DataFrame
        df = pd.DataFrame.from_dict(resp[1], orient='index').reset_index()
        df.columns = ['skeleton_id', 'neuron_name']
    else:
        df = pd.DataFrame([], columns=['skeleton_id', 'neuron_name'])

    return df


@cache.undo_on_error
def get_partners(x, threshold=1, min_size=2, filt=[], min_confidence=1,
                 directions=['incoming', 'outgoing',
                             'gapjunctions', 'attachments'],
                 remote_instance=None):
    """Retrieve partners connected by synapses, gap junctions or attachments.

    Note
    ----
    This function treats multiple fragments with the same skeleton ID
    (e.g. from splits into axon & dendrites) as a single neuron when fetching
    data from the server. For "fragmented" connectivity use
    :func:`~pymaid.cn_table_from_connectors` instead.

    Parameters
    ----------
    x
                        Neurons for which to retrieve partners. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    threshold :         int, optional
                        Minimum # of links (synapses/gap-junctions/etc).
    min_size :          int, optional
                        Minimum node count of partner
                        (default=2 to hide single-node partners).
    filt :              list of str, optional
                        Filters partners for neuron names (must be exact) or
                        skeleton_ids.
    min_confidence :    int | None, optional
                        If set, edges with lower confidence will be ignored.
                        Applied before ``threshold``.
    directions :        'incoming' | 'outgoing' | 'gapjunctions' | 'attachments', optional
                        Use to restrict to either up- or downstream partners.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron and the number of
        synapses with the query neurons::

           neuron_name  skeleton_id    num_nodes    relation    total  skid1  skid2 ...
         0   name1         skid1      node_count1   upstream    n_syn  n_syn  ...
         1   name2         skid2      node_count2  downstream   n_syn  n_syn  ..
         2   name3         skid3      node_count3  gapjunction  n_syn  n_syn  .
         ...

        ``relation`` can be ``'upstream'`` (incoming), ``'downstream'``
        (outgoing), ``'attachment'`` or ``'gapjunction'`` (gap junction).

    Warning
    -------
    By default, will exclude single node partners! Set ``min_size=1`` to return
    ALL partners including placeholder nodes.

    Notes
    -----
    Partners can show up multiple times if they are e.g. pre- AND postsynaptic!

    Examples
    --------
    >>> example_skids = [16, 201, 150, 20]
    >>> cn = pymaid.get_partners(example_skids)
    >>> # Get only upstream partners
    >>> subset = cn[ cn.relation == 'upstream' ]
    >>> # Get partners with more than e.g. 5 synapses across all neurons
    >>> subset2 = cn[ cn[example_skids].sum(axis=1) > 5 ]
    >>> # Combine above conditions (watch parentheses!)
    >>> subset3 = cn[(cn.relation=='upstream') &
    ...              (cn[example_skids].sum(axis=1) > 5)]

    See Also
    --------
    :func:`~pymaid.adjacency_matrix`
                    Use if you need an adjacency matrix instead of a table.
    :func:`~pymaid.get_partners_in_volume`
                    Use if you only want connectivity within a given volume.
    :func:`~pymaid.filter_connectivity`
                   Use to restrict connector table to given part of a neuron
                   or a volume.
    :func:`~cn_table_from_connectors`
                   Returns "fragmented" connectivity. Use e.g. if you are
                   working with multiple fragments from the same neuron.

    """
    if not isinstance(min_confidence, (float, int)) or min_confidence < 0 or min_confidence > 5:
        raise ValueError('min_confidence must be 0-5.')

    # This maps CATMAID JSON relations to more relatable terms (I think)
    relations = {'incoming': 'upstream',
                 'outgoing': 'downstream',
                 'gapjunctions': 'gapjunction',
                 'attachments': 'attachment'}

    # Catch some easy mistakes regarding relations:
    repl = {v: k for k, v in relations.items()}
    directions = [repl.get(d, d) for d in directions]

    wrong_dir = set(directions) - set(relations.keys())
    if wrong_dir:
        raise ValueError('Unknown direction "{}". Please use a combination '
                         'of "{}"'.format(', '.join(wrong_dir),
                                          ', '.join(relations.keys())))

    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)
    x = np.array(x).astype(str)

    remote_connectivity_url = remote_instance._get_connectivity_url()

    connectivity_post = {}
    connectivity_post['boolean_op'] = 'OR'
    connectivity_post['with_nodes'] = False

    for i, skid in enumerate(x):
        tag = 'source_skeleton_ids[{0}]'.format(i)
        connectivity_post[tag] = skid

    logger.info('Fetching connectivity table for {} neurons'.format(len(x)))
    connectivity_data = remote_instance.fetch(remote_connectivity_url,
                                              post=connectivity_post)

    # Delete directions that we don't want
    connectivity_data.update(
        {d: [] for d in connectivity_data if d not in directions})

    # Get neurons' names
    names = get_names([n for d in connectivity_data for n in connectivity_data[
                      d]] + list(x), remote_instance=remote_instance)

    df = pd.DataFrame(columns=['neuron_name', 'skeleton_id',
                               'num_nodes', 'relation'] + list(x))

    # Number of synapses is returned as list of links with 0-5 confidence:
    # {'skid': [0, 1, 2, 3, 4, 5]}
    # This is being collapsed into a single value before returning it.
    for d in relations:
        if d not in connectivity_data:
            continue
        df_temp = pd.DataFrame([[
            names[str(n)],
            str(n),
            int(connectivity_data[d][n]['num_nodes']),
            relations[d]] +
            [sum(connectivity_data[d][n]['skids'].get(s,
                                                      [0, 0, 0, 0, 0])[min_confidence - 1:]) for s in x]
            for i, n in enumerate(connectivity_data[d])
        ],
            columns=['neuron_name', 'skeleton_id', 'num_nodes',
                     'relation'] + [str(s) for s in x],
            dtype=object
        )

        df = pd.concat([df, df_temp], axis=0)

    df['total'] = df[x].sum(axis=1).values

    # Now filter for synapse threshold and size
    df = df[(df.num_nodes >= min_size) & (df.total >= threshold)]

    df.sort_values(['relation', 'total'], inplace=True, ascending=False)

    if filt:
        if not isinstance(filt, (list, np.ndarray)):
            filt = [filt]

        filt = [str(s) for s in filt]

        df = df[df.skeleton_id.isin(filt) | df.neuron_name.isin(filt)]

    df.datatype = 'connectivity_table'

    # Return reindexed concatenated dataframe
    df.reset_index(drop=True, inplace=True)

    logger.info('Done. Found {0} pre-, {1} postsynaptic and {2} gap '
                'junction-connected neurons'.format(
                *[df[df.relation == r].shape[0] for r in ['upstream',
                                                          'downstream',
                                                          'gapjunction']]))

    return df


@cache.undo_on_error
def get_names(x, remote_instance=None):
    """Retrieve neuron names for a list of skeleton ids.

    Parameters
    ----------
    x
                        Neurons for wich to retrieve names. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        Either pass directly to function or define
                        globally as ``remote_instance``.

    Returns
    -------
    dict
                    ``{skid1: 'neuron_name', skid2: 'neuron_name', ...}``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    x = list(set(x))

    remote_get_names_url = remote_instance._get_neuronnames()

    get_names_postdata = {}
    get_names_postdata['self.project_id'] = remote_instance.project_id

    for i in range(len(x)):
        key = 'skids[%i]' % i
        get_names_postdata[key] = x[i]

    names = remote_instance.fetch(remote_get_names_url, post=get_names_postdata)

    logger.debug('Names for {} of {} skeleton IDs retrieved'.format(len(names),
                                                                    len(x)))

    return names


@cache.undo_on_error
def get_node_details(x, chunk_size=10000, convert_ts=True, remote_instance=None):
    """Retrieve detailed info for nodes and/or connectors.

    Parameters
    ----------
    x :                 list | CatmaidNeuron | CatmaidNeuronList
                        List of node ids: can be node or connector IDs!
                        If CatmaidNeuron/List will get both, nodes and
                        connectors!
    chunk_size :        int, optional
                        Querying large number of nodes will result in server
                        errors. We will thus query them in amenable bouts.
    convert_ts :        bool, optional
                        If True, will convert timestamps from strings to
                        datetime objects.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a node::

            node_id  creation_time  creator  edition_time ...
         0
         1
            editor  reviewers  review_times
         0
         1

    """
    if isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        node_ids = np.append(x.nodes.node_id.values,
                             x.connectors.connector_id.values)
    elif not isinstance(x, (list, tuple, np.ndarray)):
        node_ids = [x]
    else:
        node_ids = x

    remote_instance = utils._eval_remote_instance(remote_instance)

    logger.debug('Retrieving details for {} nodes...'.format(len(node_ids)))

    urls = []
    post = []
    for ix in range(0, len(node_ids), chunk_size):
        urls.append(remote_instance._get_node_info_url())
        post.append({'node_ids[{}]'.format(k): tn for k, tn in enumerate(node_ids[ix:ix + chunk_size])})

    # Get responses
    resp = remote_instance.fetch(urls, post=post, desc='Chunks')
    # Merge into a single dictionary
    data = {k: d[k] for d in resp for k in d}

    # Generate dataframe
    data_columns = ['creation_time', 'user', 'edition_time',
                    'editor', 'reviewers', 'review_times']
    df = pd.DataFrame(
        [[e] + [d[k] for k in data_columns] for e, d in data.items()],
        columns=['node_id'] + data_columns,
        dtype=object
    )

    # Rename column 'user' to 'creator'
    df.rename({'user': 'creator'}, axis='columns', inplace=True)

    if convert_ts:
        df['creation_time'] = pd.to_datetime(df.creation_time)
        df['edition_time'] = pd.to_datetime(df.edition_time)
        df['review_times'] = df.review_times.apply(lambda x: [pd.to_datetime(d)
                                                              for d in x])

    return df


@cache.undo_on_error
def get_skid_from_node(node_ids, remote_instance=None):
    """Retrieve skeleton IDs from a list of nodes.

    Parameters
    ----------
    node_ids :          int | list of int
                        Node ID(s) to retrieve skeleton IDs for.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
            ``{node_id: skeleton_ID, ...}``. If node does not exists,
            ``skeleton_ID`` will be ``None``.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    node_ids = utils.eval_node_ids(node_ids, connectors=False, nodes=True)

    if not isinstance(node_ids, (list, np.ndarray)):
        node_ids = [node_ids]

    urls = [remote_instance._get_skid_from_tnid(tn) for tn in node_ids]

    data = remote_instance.fetch(urls, desc='Fetch skids')

    return {node_ids[i]: d.get('skeleton_id', None) for i, d in enumerate(data)}


@cache.undo_on_error
def get_node_table(x, include_details=True, convert_ts=True, remote_instance=None):
    """Retrieve node table(s) for a list of neurons.

    Parameters
    ----------
    x
                        Catmaid Neuron(s) as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    include_details :   bool, optional
                        If True, tags and reviewer are included in the table.
                        For larger lists, it is recommended to set this to
                        False to improve performance.
    convert_ts :        bool, optional
                        If True, will convert edition timestamp to pandas
                        datetime. Set to False to improve performance.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a node::

           skeleton_id  node_id  parent_id  confidence  x  y  z  ...
         0
         1
         2
         ...
           radius  creator  last_edition  reviewers  tag
         0
         1
         2

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    logger.info('Retrieving {} node table(s)...'.format(len(x)))

    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    # Generate URLs to retrieve
    urls = []
    for skid in x:
        remote_nodes_list_url = remote_instance._get_skeleton_nodes_url(skid)
        urls.append(remote_nodes_list_url)

    node_list = remote_instance.fetch(urls, desc='Get tables')

    logger.info('{} nodes retrieved. Creating table..'
                '.'.format(sum([len(nl[0]) for nl in node_list])))

    all_tables = []

    for i, nl in enumerate(config.tqdm(node_list,
                                       desc='Creating table',
                                       leave=config.pbar_leave,
                                       disable=config.pbar_hide)):

        this_df = pd.DataFrame(nl[0],
                               columns=['node_id', 'parent_node_id',
                                        'confidence', 'x', 'y', 'z', 'radius',
                                        'creator', 'last_edited']
                               )

        # Parent IDs can be `None` here - we will set them to -1
        this_df.loc[this_df.parent_node_id.isnull(), 'parent_node_id'] = -1
        this_df['parent_node_id'] = this_df.parent_node_id.astype(int)

        # Keep track of skeleton ID
        this_df['skeleton_id'] = x[i]

        if include_details:
            tag_dict = {}
            for t in nl[2]:
                tag_dict[t[0]] = tag_dict.get(t[0], []) + [t[1]]

            reviewer_dict = {}
            for r in nl[1]:
                reviewer_dict[r[0]] = reviewer_dict.get(r[0], []) + [user_dict.get(r[1])]

            this_df['reviewers'] = this_df.node_id.map(reviewer_dict)
            this_df['tags'] = this_df.node_id.map(tag_dict)

        all_tables.append(this_df)

    # Concatenate all DataFrames
    tn_table = pd.concat(all_tables, axis=0, ignore_index=True)

    # Replace creator_id with their login and make it a categorical
    tn_table['creator'] = tn_table.creator.map(user_dict).astype('category')

    # Replace timestamp with datetime object
    if convert_ts:
        tn_table['last_edited'] = pd.to_datetime(tn_table.last_edited,
                                                 utc=True,
                                                 unit='s')

    return tn_table


@cache.undo_on_error
def get_edges(x, remote_instance=None):
    """Retrieve edges between sets of neurons.

    Synaptic connections only!

    Parameters
    ----------
    x
                        Neurons for which to retrieve edges. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents an edge::

           source     target     weight
         1
         2
         3

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_edges_url = remote_instance._get_edges_url()

    get_edges_postdata = {}
    get_edges_postdata['confidence_threshold'] = '0'

    for i in range(len(x)):
        key = 'skeleton_ids[%i]' % i
        get_edges_postdata[key] = x[i]

    edges = remote_instance.fetch(remote_get_edges_url, post=get_edges_postdata)

    df = pd.DataFrame([[e[0], e[1], sum(e[2])] for e in edges['edges']],
                      columns=['source', 'target', 'weight']
                      )

    return df


@cache.undo_on_error
def get_connectors(x, relation_type=None, tags=None, remote_instance=None):
    """Retrieve connectors based on a set of filters.

    Parameters
    ----------
    x
                        Neurons for which to retrieve connectors. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
                        5. ``None`` if you want all fetch connectors that
                           match other criteria
    relation_type :     'presynaptic_to' | 'postsynaptic_to' | 'gapjunction_with' | 'abutting' | 'attached_to', optional
                        If provided, will filter for these connection types.
    tags :              str | list of str, optional
                        If provided, will filter connectors for tag(s).
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector::

           connector_id  x  y  z  confidence  creator ...
         0
         1
         ...
           editor  creation_time  edition_time
         0
         1
         ...

    Examples
    --------
    Get all connectors for a single neuron:

    >>> cn = pymaid.get_connectors(16)

    Get every connector with a given tag:

    >>> tagged_cn = pymaid.get_connectors(None, tags='FML_sample')

    Get all tagged connectors for a set of neurons:

    >>> tagged_cn2 = pymaid.get_connectors('annotation:glomerulus DA1',
                                           tags='FML_sample')

    See Also
    --------
    :func:`~pymaid.get_connector_details`
        If you need details about the connectivity of a connector
    :func:`~pymaid.get_connectors_between`
        If you need to find the connectors between sets of neurons.
    :func:`~pymaid.get_connector_links`
        If you ned details about links for each connector.
    :func:`pymaid.find_nodes`
            Function to get nodes by tags, IDs or skeleton.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, type(None)):
        x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_connectors_url = remote_instance._get_connectors_url()

    postdata = {'with_tags': 'true', 'with_partners': 'true'}

    # Add skeleton IDs filter (if applicable)
    if not isinstance(x, type(None)):
        postdata.update(
            {'skeleton_ids[{0}]'.format(i): s for i, s in enumerate(x)})

    # Add tags filter (if applicable)
    if not isinstance(tags, type(None)):
        if not isinstance(tags, (list, np.ndarray)):
            tags = [tags]
        postdata.update({'tags[{0}]'.format(i): str(t)
                         for i, t in enumerate(tags)})

    # Add relation_type filter (if applicable)
    allowed_relations = ['presynaptic_to', 'postsynaptic_to',
                         'gapjunction_with', 'abutting', 'attached_to']
    if not isinstance(relation_type, type(None)):
        if relation_type not in allowed_relations:
            raise ValueError('Unknown relation type "{0}". Must be in '
                             '{1}'.format(relation_type, allowed_relations))
        postdata.update({'relation_type': relation_type})

    data = remote_instance.fetch(remote_get_connectors_url, post=postdata)

    # creator_id and editor_id will be replaced with logins later
    df = pd.DataFrame(data=data['connectors'],
                      columns=['connector_id', 'x', 'y', 'z', 'confidence',
                               'creator_id', 'editor_id', 'creation_time',
                               'edition_time'])

    # Add tags
    df['tags'] = df.connector_id.astype(str).map(data['tags'])

    # Map hardwire connector type ID to their type name
    # ATTENTION: "attachment" can be part of any connector type
    rel_ids = {r['relation_id']: r for r in config.link_types}

    # Get connector type IDs
    cn_ids = {k: v[0][3] for k, v in data['partners'].items()}

    # Map type ID to relation (also note conversion of connector ID to integer)
    cn_type = {int(k): rel_ids.get(v, {'type': 'unknown'})['type']
               for k, v in cn_ids.items()}

    # Map connector ID to connector type
    df['type'] = df.connector_id.map(cn_type)

    # Add creator login instead of id
    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()
    df['creator'] = df.creator_id.map(user_dict)
    df['editor'] = df.editor_id.map(user_dict)
    df.drop(['creator_id', 'editor_id'], inplace=True, axis=1)

    # Convert timestamps to datetimes
    df['creation_time'] = df['creation_time'].apply(
        datetime.datetime.fromtimestamp)
    df['edition_time'] = df['edition_time'].apply(
        datetime.datetime.fromtimestamp)

    df.datatype = 'connector_table'

    return df


@cache.undo_on_error
def get_connector_links(x, with_tags=False, chunk_size=50,
                        remote_instance=None):
    """Retrieve connectors links for a set of neurons.

    In essence, this will get you all "arrows" that point from a connector to
    your neuron or from your neuron to a connector. It does NOT give you the
    entire battery of connectors for a set of connectors. For that you have
    to use :func:`~pymaid.get_connector_details`.

    Parameters
    ----------
    x :                 int | CatmaidNeuron | CatmaidNeuronList
                        Neurons/Skeleton IDs to retrieve link details for. If
                        CatmaidNeuron/List will respect changes made to
                        original neurons (e.g. pruning)!
    with_tags :         bool, optional
                        If True will also return dictionary of connector tags.
    chunk_size :        int, optional
                        Neurons are split into chunks of this size and then
                        queried sequentially to prevent server from returning
                        an error.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector link::

           skeleton_id  relation  connector_id  x  y  z  confidence ...
         0
         1
         2
         ...
           creator  node_id  creation_time  edition_time
         0
         1
         2

    (links, tags)
            If ``with_tags=True``, will return above DataFrame and tags dict.

    See Also
    --------
    :func:`~pymaid.get_connectors`
        If you just need the connector table (ID, x, y, z, creator, etc).
    :func:`~pymaid.get_connector_details`
        Get the same data but by connector, not by link.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, warn_duplicates=False,
                             remote_instance=remote_instance)

    df_collection = []
    tags = {}

    link_types = [l['relation'] for l in config.link_types]

    with config.tqdm(desc='Fetching links', total=len(skids),
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        for chunk in [skids[i:i + chunk_size] for i in range(0, len(skids), chunk_size)]:
            # Generate URLs
            GET = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(chunk)}
            urls = [remote_instance._get_connector_links_url(relation_type=cn,
                                                             **GET) for cn in link_types]

            # Fetch data
            responses = remote_instance.fetch(urls, disable_pbar=True)

            # Extract tags
            if with_tags:
                for r in responses:
                    tags.update(r['tags'])

            # Generate separate DataFrames
            data = [pd.DataFrame(r['links'],
                                 columns=['skeleton_id', 'connector_id',
                                          'x', 'y', 'z', 'confidence',
                                          'creator', 'node_id',
                                          'creation_time', 'edition_time']
                                 ) for r in responses]

            # Add link type to each DataFrame
            for t, d in zip(link_types, data):
                d['relation'] = t

            # Concatenate DataFrames
            df = pd.concat(data, axis=0)

            # Store
            df_collection.append(df)

            # Update progress bar
            pbar.update(len(chunk))

    # Merge DataFrames
    df = pd.concat(df_collection, axis=0)

    # Cater for cases in which the original neurons have been edited
    if isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        df = df[df.connector_id.isin(x.connectors.connector_id)]

    # Convert to timestamps
    df['creation_time'] = pd.to_datetime(df.creation_time)
    df['edition_time'] = pd.to_datetime(df.edition_time)

    if with_tags:
        return df, tags

    return df


@cache.undo_on_error
def get_connector_details(x, remote_instance=None):
    """Retrieve details on sets of connectors.

    Parameters
    ----------
    x :                 list of connector IDs | CatmaidNeuron | CatmaidNeuronList
                        Connector ID(s) to retrieve details for. If
                        CatmaidNeuron/List, will use their connectors.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector::

           connector_id  presynaptic_to  postsynaptic_to  ...
         0
         1
         2
         ...
           presynaptic_to_node  postsynaptic_to_node
         0
         1
         2

    See Also
    --------
    :func:`~pymaid.get_connectors`
        If you just need the connector table (ID, x, y, z, creator, etc).
    :func:`~pymaid.get_connector_links`
        Get the same data but by link, not by connector.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    connector_ids = utils.eval_node_ids(x, connectors=True, nodes=False)

    connector_ids = list(set(connector_ids))

    remote_get_connectors_url = remote_instance._get_connector_details_url()

    # Depending on DATA_UPLOAD_MAX_NUMBER_FIELDS of your CATMAID server
    # (default = 1000), we have to cut requests into batches smaller than that
    DATA_UPLOAD_MAX_NUMBER_FIELDS = min(50000, len(connector_ids))

    connectors = []
    with config.tqdm(total=len(connector_ids), desc='CN details',
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        for b in range(0, len(connector_ids), DATA_UPLOAD_MAX_NUMBER_FIELDS):
            get_connectors_postdata = {}
            for i, s in enumerate(connector_ids[b:b + DATA_UPLOAD_MAX_NUMBER_FIELDS]):
                key = 'connector_ids[%i]' % i
                get_connectors_postdata[key] = s  # connector_ids[i]

            connectors += remote_instance.fetch(remote_get_connectors_url,
                                                post=get_connectors_postdata)

            pbar.update(DATA_UPLOAD_MAX_NUMBER_FIELDS)

    logger.info('Data for %i of %i unique connector IDs retrieved' % (
        len(connectors), len(set(connector_ids))))

    columns = ['connector_id', 'presynaptic_to', 'postsynaptic_to',
               'presynaptic_to_node', 'postsynaptic_to_node']

    df = pd.DataFrame([[cn[0]] + [cn[1][e] for e in columns[1:]] for cn in connectors],
                      columns=columns,
                      dtype=object
                      )

    return df


@cache.undo_on_error
def get_connector_tags(x, remote_instance=None):
    """Retrieve tags on sets of connectors.

    Parameters
    ----------
    x :                 list of connector IDs | CatmaidNeuron | CatmaidNeuronList
                        Connector ID(s) to retrieve details for. If
                        CatmaidNeuron/List, will use their connectors.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    ---------
    dict
                        Dictionary mapping tags (``str``) to connector IDs
                        (``int``)::

                          {
                           tag1: [connector1_id, connector2_id, ...],
                           tag2: [ ... ], ...
                           }

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    connector_ids = utils.eval_node_ids(x, connectors=True, nodes=False)

    connector_ids = list(set(connector_ids))

    remote_get_node_labels_url = remote_instance._get_node_labels_url()

    post = {'connector_ids': ','.join([str(tn) for tn in connector_ids])}

    resp = remote_instance.fetch(remote_get_node_labels_url, post=post)

    cn_tags = {}
    for cnid in resp:
        cn_tags.update({tag: cn_tags.get(tag, []) + [int(cnid)] for tag in resp[cnid]})

    return cn_tags


@cache.undo_on_error
def get_connectors_between(a, b, directional=True, remote_instance=None):
    """Retrieve connectors between sets of neurons.

    Important
    ---------
    This function does currently *not* return gap junctions between neurons.

    Notes
    -----
    A connector can show up multiple times if it is connecting to more than one
    nodes of the same neuron.

    Parameters
    ----------
    a,b
                        Neurons for which to retrieve connectors. Can be:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    directional :       bool, optional
                        If True, only connectors a -> b are listed,
                        otherwise it is a <-> b.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a connector::

           connector_id  connector_loc  node1_id  source_neuron  ...
         0
         1
         2
         ...
           confidence1  creator1 node1_loc node2_id  target_neuron  ...
         0
         1
         2
         ...
          confidence2  creator2  node2_loc
         0
         1
         2

    See Also
    --------
    :func:`~pymaid.get_edges`
        If you just need the number of synapses between neurons, this is much
        faster.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    a = utils.eval_skids(a, remote_instance=remote_instance)
    b = utils.eval_skids(b, remote_instance=remote_instance)

    if len(a) == 0:
        raise ValueError('No source neurons provided')

    if len(b) == 0:
        raise ValueError('No target neurons provided')

    post = {'relation': 'presynaptic_to'}
    post.update({'skids1[{0}]'.format(i): s for i, s in enumerate(a)})
    post.update({'skids2[{0}]'.format(i): s for i, s in enumerate(b)})

    url = remote_instance._get_connectors_between_url()

    data = remote_instance.fetch(url, post=post)

    if not directional:
        post['relation'] = 'postsynaptic_to'
        data += remote_instance.fetch(url, post=post)

    df = pd.DataFrame(data,
                      columns=['connector_id', 'connector_loc', 'node1_id',
                               'source_neuron', 'confidence1', 'creator1',
                               'node1_loc', 'node2_id',
                               'target_neuron', 'confidence2', 'creator2',
                               'node2_loc'])

    # Get user list and replace IDs with logins
    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()
    df['creator1'] = df['creator1'].map(user_dict)
    df['creator2'] = df['creator2'].map(user_dict)

    return df


@cache.undo_on_error
def get_review(x, remote_instance=None):
    """Retrieve review status for a set of neurons.

    Parameters
    ----------
    x
                        Neurons for which to get review status. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron::

           skeleton_id  neuron_name  total_node_count  nodes_reviewed  ...
         0
         1
         ...
           percent_reviewed
         0
         1
         ...

    See Also
    --------
    :func:`~pymaid.get_review_details`
        Gives you review status for individual nodes of a given neuron.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_reviews_url = remote_instance._get_review_status_url()

    names = {}
    review_status = {}

    CHUNK_SIZE = 1000

    with config.tqdm(total=len(x), disable=config.pbar_hide,
                     desc='Rev. status',
                     leave=config.pbar_leave) as pbar:
        for j in range(0, len(x), CHUNK_SIZE):
            get_review_postdata = {}

            for i in range(j, min(j + CHUNK_SIZE, len(x))):
                key = 'skeleton_ids[%i]' % i
                get_review_postdata[key] = str(x[i])

            names.update(get_names(x[j:j + CHUNK_SIZE],
                                   remote_instance=remote_instance))

            review_status.update(remote_instance.fetch(remote_get_reviews_url,
                                                       post=get_review_postdata))

            pbar.update(CHUNK_SIZE)

    df = pd.DataFrame([[s,
                        names[str(s)],
                        review_status[s][0],
                        review_status[s][1],
                        int(review_status[s][1] / review_status[s][0] * 100)
                        ] for s in review_status],
                      columns=['skeleton_id', 'neuron_name',
                               'total_node_count', 'nodes_reviewed',
                               'percent_reviewed']
                      )

    return df


@cache.undo_on_error
def get_user_annotations(x, remote_instance=None):
    """Retrieve annotations used by given user(s).

    Parameters
    ----------
    x
                        User(s) to get annotation for. Can be either:

                        1. single or list of user IDs
                        2. single or list of user login names
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame (df) in which each row represents a single annotation::

           annotation  annotated_on  times_used  user_id  annotation_id  user_login
         0
         1
         ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    # Get user list
    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    try:
        ids = [int(e) for e in x]
    except BaseException:
        ids = user_list.set_index('login').loc[x, 'id'].values

    # This works with neuron_id NOT skeleton_id
    # neuron_id can be requested via neuron_names
    url_list = list()
    postdata = list()

    iDisplayLength = 500

    for u in ids:
        url_list.append(remote_instance._get_annotation_table_url())
        postdata.append(dict(user_id=int(u),
                             iDisplayLength=iDisplayLength))

    # Get data
    annotations = [e['aaData'] for e in remote_instance.fetch(
                   url_list, post=postdata, desc='Get annot')]

    # Add user login
    for i, u in enumerate(ids):
        for an in annotations[i]:
            an.append(user_dict.get(u, 'Anonymous'))

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'annotated_on', 'times_used',
                               'user_id', 'annotation_id', 'user_login'],
                      dtype=object
                      )

    df['annotated_on'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['annotated_on'].values]

    return df.sort_values('times_used').reset_index(drop=True)


@cache.undo_on_error
def get_annotation_details(x, remote_instance=None):
    """Retrieve annotations for a set of neuron.

    Returns more details than :func:`~pymaid.get_annotations` but is slower.
    Contains timestamps and user IDs (same API as neuron navigator).

    Parameters
    ----------
    x
                        Neurons to get annotation details for. Can be either:

                        1. List of skeleton ID(s) (int or str)
                        2. List of neuron name(s) (str, exact match)
                        3. An annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a single annotation::

           annotation  skeleton_id  time_annotated  user_id  annotation_id  user
         0
         1
         ...

    See Also
    --------
    :func:`~pymaid.get_annotations`
                        Gives you annotations for a list of neurons (faster).

    Examples
    --------
    >>> # Get annotations for a set of neurons
    >>> an = pymaid.get_annotation_details([ 12, 57003 ])
    >>> # Get those for a single neuron
    >>> an[ an.skeleton_id == '57003' ]
    >>> # Get annotations given by set of users
    >>> an[ an.user.isin( ['schlegelp', 'lif'] )]
    >>> # Get most recent annotations
    >>> import datetime
    >>> an[ an.time_annotated > datetime.date(2017, 6, 1) ]

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    # This works with neuron_id NOT skeleton_id
    # neuron_id can be requested via neuron_names
    url_list = list()
    postdata = list()
    neuron_ids = get_neuron_id(skids, remote_instance=remote_instance)

    for s in skids:
        nid = neuron_ids.get(str(s))

        url_list.append(remote_instance._get_annotation_table_url())
        postdata.append(dict(neuron_id=int(nid)))

    # Get data
    annotations = [e['aaData'] for e in remote_instance.fetch(url_list,
                                                              post=postdata,
                                                              desc='Get annot')]

    # Get user list
    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    # Add skeleton ID and user login
    for i, s in enumerate(skids):
        for an in annotations[i]:
            an.insert(1, s)
            an.append(user_dict.get(an[4]))

    # Now flatten the list of lists
    annotations = [an for sublist in annotations for an in sublist]

    # Create dataframe
    df = pd.DataFrame(annotations,
                      columns=['annotation', 'skeleton_id', 'time_annotated',
                               'times_used', 'user_id', 'annotation_id',
                               'user'],
                      dtype=object
                      )

    # Times used appears to not be working (always shows "1") - remove it
    df.drop('times_used', inplace=True, axis=1)

    df['time_annotated'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['time_annotated'].values]

    return df.sort_values('annotation').reset_index(drop=True)


@cache.undo_on_error
def get_annotations(x, remote_instance=None):
    """Retrieve annotations for a list of skeleton ids.

    If a neuron has no annotations, it will not show up in returned dict!

    Notes
    -----
    This API endpoint does not process more than 250 neurons at a time!

    Parameters
    ----------
    x
                        Neurons for which to retrieve annotations. Can be
                        either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        ``{skeleton_id: [annnotation, annotation], ...}``

    See Also
    --------
    :func:`~pymaid.get_annotation_details`
                        Gives you more detailed information about annotations
                        of a set of neuron (includes timestamp and user) but
                        is slower.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    remote_get_annotations_url = remote_instance._get_annotations_for_skid_list()

    get_annotations_postdata = {'metaannotations': 0, 'neuronnames': 0}

    for i in range(len(x)):
        key = 'skeleton_ids[%i]' % i
        get_annotations_postdata[key] = str(x[i])

    annotation_list_temp = remote_instance.fetch(remote_get_annotations_url,
                                                 post=get_annotations_postdata)

    annotation_list = {}

    try:
        for skid in annotation_list_temp['skeletons']:
            annotation_list[skid] = []
            # for entry in annotation_list_temp['skeletons'][skid]:
            for entry in annotation_list_temp['skeletons'][skid]['annotations']:
                annotation_id = entry['id']
                annotation_list[skid].append(
                    annotation_list_temp['annotations'][str(annotation_id)])

        return(annotation_list)
    except BaseException:
        raise Exception(
            'No annotations retrieved. Make sure that the skeleton IDs exist.')


@cache.wipe_and_retry
def get_annotation_id(annotations, allow_partial=False, raise_not_found=True,
                      remote_instance=None):
    """Retrieve the annotation ID for single or list of annotation(s).

    Parameters
    ----------
    annotations :       str | list of str
                        Single annotations or list of multiple annotations.
    allow_partial :     bool, optional
                        If True, will allow partial matches.
    raise_not_found :   bool, optional
                        If True raise Exception if no match for any of the
                        query annotations is found. Else log warning.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        ``{'annotation_name': 'annotation_id', ...}``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    logger.debug('Retrieving list of annotations...')

    remote_annotation_list_url = remote_instance._get_annotation_list()
    an_list = remote_instance.fetch(remote_annotation_list_url)

    # Turn into pandas array
    an_list = pd.DataFrame.from_records(an_list['annotations'])

    annotations = utils._make_iterable(annotations)
    annotation_ids = {}
    for an in annotations:
        # This is just to catch misunderstandings with parsing skeleton IDs
        if an.startswith('annotation:'):
            logger.warning('Removing unexpected "annotation:" prefix.')
            an = an[11:]

        # Strip whitespaces
        an = an.strip()

        # Strip tilde -> consider that people might use e.g. "~/VA6" for NOT
        # VA6
        if an.startswith('~'):
            an = an[1:]

        # '/' indicates regex
        if an.startswith('/'):
            re_str = an[1:]
        # If allow partial just use the raw string
        elif allow_partial:
            re_str = an
        # If exact match, encode this in regex
        else:
            re_str = '^{}$'.format(an)

        # Search for matches
        res = an_list[an_list.name.str.match(re_str)].set_index('name').id.to_dict()
        if not res:
            logger.warning('No annotation found for "{}"'.format(an))
        annotation_ids.update(res)

    if not annotation_ids:
        if raise_not_found:
            raise Exception('No matching annotation(s) found')
        else:
            logger.warning('No matching annotation(s) found')

    return annotation_ids


@cache.undo_on_error
def find_nodes(tags=None, node_ids=None, skeleton_ids=None,
                   remote_instance=None):
    """Get nodes by tag (label), ID or associated skeleton.

    Search intersected (logical AND) across parameters but additive (logical OR)
    within each parameter (see examples).

    Parameters
    ----------
    tags :              str | list of str
                        Use to restrict to nodes with given tags.
    node_ids :      int | list of int
                        Use to restrict to nodes with given IDs.
    skeleton_ids :      str | int | CatmaidNeuron/List, optional
                        Use to restrict to a set of neurons. Can be:

                        1. skeleton ID(s) (int or str)
                        2. neuron name(s) (str)
                        3. annotation(s): e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a node::

           skeleton_id  node_id  parent_id   x  y  z  confidence ...
         0
         1
         2
         ...
           radius  edition_time  creator_id
         0
         1
         2

    See Also
    --------
    :func:`pymaid.get_connectors`
            Function to get connectors by neurons and/or by tags.


    Examples
    --------
    Get all nodes with a given tag

    >>> tagged = pymaid.find_nodes(tags='SCHLEGEL_LH')

    Get all nodes of a set of neurons with either of two tags

    >>> tagged = pymaid.find_nodes(tags=['SCHLEGEL_LH', 'SCHLEGEL_AL'],
                                         skeleton_ids='annotation:glomerulus DA1')

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    url = remote_instance._get_node_table_url()

    if all([isinstance(x, type(None)) for x in [tags, skeleton_ids, node_ids]]):
        answer = ""
        while answer not in ["y", "n"]:
            answer = input("Your search parameters will retrieve ALL "
                           "nodes in the dataset. Proceed? "
                           "[Y/N] ").lower()

            if answer != 'y':
                logger.info('Query cancelled')
                return

    post = {}
    if not isinstance(tags, type(None)):
        tags = utils._make_iterable(tags)
        post.update({'label_names[{}]'.format(i): t for i, t in enumerate(tags)})

    if not isinstance(node_ids, type(None)):
        node_ids = utils._make_iterable(node_ids)
        post.update({'treenode_ids[{}]'.format(i): t for i, t in enumerate(node_ids)})

    if not isinstance(skeleton_ids, type(None)):
        skeleton_ids = utils.eval_skids(skeleton_ids, remote_instance=remote_instance)
        post.update({'skeleton_ids[{}]'.format(i): s for i, s in enumerate(skeleton_ids)})

    # Fetch
    resp = remote_instance.fetch(url, post=post)

    # Format is [[ID, parent ID, x, y, z, confidence, radius, skeleton_id,
    # edition_time, user_id], ...]
    df = pd.DataFrame(resp,
                      columns=['node_id', 'parent_id', 'x', 'y', 'z', 'confidence',
                               'radius', 'skeleton_id', 'edition_time',
                               'creator_id'])

    # Reorder and return
    return df[['skeleton_id', 'node_id', 'parent_id', 'x', 'y', 'z',
               'confidence', 'radius', 'edition_time', 'creator_id']]


@cache.undo_on_error
def has_soma(x, tag='soma', min_rad=500, return_ids=False,
             remote_instance=None):
    """Check if neuron(s) has soma.

    Parameters
    ----------
    x
                        Neurons which to check for a soma. Can be either:

                        1. skeleton ID(s) (int or str)
                        2. neuron name(s) (str)
                        3. annotation(s): e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    tag :               str | None, optional
                        Tag we expect the soma to have. Set to ``None`` if
                        not applicable.
    min_rad :           int, optional
                        Minimum radius of soma.
    return_ids :        bool, optional
                        If True, will return node IDs of soma(s) found
                        instead of simply if a soma has been found.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        If ``return_ids=False``::

                          {skid1: True, skid2: False, ...}

                        If ``return_ids=True``::

                          {skid1: [node_id], skid2: [node_id], ...}

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_node_table_url()
    post = {'label_names[0]': tag}
    post.update({'skeleton_ids[{}]'.format(i): s for i, s in enumerate(x)})

    # Fetch  only nodes that have the soma label
    resp = remote_instance.fetch(url, post=post)

    # Format is [[ID, parent ID, x, y, z, confidence, radius, skeleton_id,
    # edition_time, user_id], ...]
    if return_ids is False:
        by_skid = {int(s): False for s in x}
        for e in resp:
            by_skid[e[7]] = max(by_skid[e[7]], e[6] >= min_rad)
    else:
        by_skid = {int(s): [] for s in x}
        for e in resp:
            if e[6] >= min_rad:
                by_skid[e[7]].append(e[0])

    return by_skid


@cache.undo_on_error
def get_annotated(x, include_sub_annotations=False, raise_not_found=True,
                  allow_partial=False, remote_instance=None):
    """Retrieve entities (neurons + annotations) with given annotation(s).

    This works similar to CATMAID's neuron search widget: multiple annotations
    are intersected! Includes meta-annotations.

    Parameters
    ----------
    x :                       str | list of str
                              (Meta-)annotations(s) to search for. Like
                              CATMAID's search widget, you can use regex to
                              search for names by starting the query with a
                              leading ``/``. Use a leading ``~`` (tilde) to
                              indicate ``NOT`` condition.
    include_sub_annotations : bool, optional
                              If True, will include entities that have
                              annotations meta-annotated with ``x``. Does not
                              work on `NOT` search conditions.
    allow_partial :           bool, optional
                              If True, partially matching annotations are
                              searched too.
    raise_not_found :         bool, optional
                              If True raise Exception if no match for any of the
                              query annotations is found. Else log warning.
    remote_instance :         CatmaidInstance, optional
                              If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents an entity::

           id  name  skeleton_ids type
         0
         1
         2
         ...

    See Also
    --------
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    pos, neg = utils._eval_conditions(x)

    post = {'with_annotations': False}
    if pos:
        pos_ids = get_annotation_id(pos, allow_partial=allow_partial,
                                    raise_not_found=raise_not_found,
                                    remote_instance=remote_instance)

        post.update({'annotated_with[{}]'.format(i): n for i, n in enumerate(pos_ids.values())})
        if include_sub_annotations:
            post.update({'sub_annotated_with[{}]'.format(i): n for i, n in enumerate(pos_ids.values())})
    if neg:
        neg_ids = get_annotation_id(neg, allow_partial=allow_partial,
                                    raise_not_found=raise_not_found,
                                    remote_instance=remote_instance)

        post.update({'not_annotated_with[{}]'.format(i): n for i, n in enumerate(neg_ids.values())})

    logger.info('Searching for: {}'.format(','.join([str(s) for s in pos_ids])))
    if neg:
        logger.info('..... and NOT: {}'.format(','.join([str(s) for s in neg_ids])))

    urls = remote_instance._get_annotated_url()

    resp = remote_instance.fetch(urls, post=post, desc='Fetching')

    return pd.DataFrame(resp['entities'])


@cache.undo_on_error
def get_skids_by_name(names, allow_partial=True, raise_not_found=True,
                      remote_instance=None):
    """Retrieve the all neurons with matching name.

    Parameters
    ----------
    names :             str | list of str
                        Name(s) to search for. Like CATMAID's search widget,
                        you can use regex to search for names by starting
                        the query with a leading ``/``.
    allow_partial :     bool, optional
                        If True, partial matches are returned too.
    raise_not_found :   bool, optional
                        If True, will raise an exception of no matches for
                        given name(s) are found. Else will return empty
                        DataFrame.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron::

           name   skeleton_id
         0
         1
         2
         ...

    See Also
    --------
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    # Only look for unique names
    names = list(set(utils._make_iterable(names, force_type=str)))

    # Prepare names for regex search on the backend
    post = []
    for n in names:
        post.append({'name': n,
                     'with_annotations': False,
                     'name_exact': True})
        # If we allow partial matches or are using regex, set exact_name to False
        if allow_partial or n.startswith('/'):
            post[-1]['name_exact'] = False

    urls = [remote_instance._get_annotated_url() for n in post]
    responses = remote_instance.fetch(urls, post=post, desc='Fetching names')

    neurons = [n for res in responses for n in res['entities'] if n['type'] == 'neuron']

    df = pd.DataFrame([[n['name'], n['skeleton_ids'][0]] for n in neurons],
                      columns=['name', 'skeleton_id'])

    if df.empty and raise_not_found:
        raise Exception('No matching name(s) found')

    return df.sort_values(['name']).drop_duplicates().reset_index(drop=True)


@cache.undo_on_error
def get_skids_by_annotation(annotations, allow_partial=False, intersect=False,
                            raise_not_found=True, remote_instance=None):
    """Retrieve the neurons annotated with given annotation(s).

    Parameters
    ----------
    annotations :           str | list
                            Single annotation or list of multiple annotations.
                            Using a tilde (~) as prefix is interpreted as NOT.
    allow_partial :         bool, optional
                            If True, allow partial match of annotation.
    intersect :             bool, optional
                            If True, neurons must have ALL provided
                            annotations.
    raise_not_found :       bool, optional
                            If True raise Exception if no match for any of the
                            query annotations is found. Else log warning.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    list
                            ``[skid1, skid2, skid3, ...]``

    See Also
    --------
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.
    :func:`pymaid.get_annotated`
                            Use to retrieve entities (neurons and annotations).

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    annotations = utils._make_iterable(annotations)
    pos_an = [an for an in annotations if not an.startswith('~')]
    neg_an = [an[1:] for an in annotations if an.startswith('~')]

    # Placeholders in case we don't even ask for pos or neg
    pos_ids = {}
    neg_ids = {}

    if pos_an:
        pos_ids = get_annotation_id(pos_an,
                                    raise_not_found=raise_not_found,
                                    allow_partial=allow_partial,
                                    remote_instance=remote_instance)
    if neg_an:
        neg_ids = get_annotation_id(neg_an,
                                    raise_not_found=raise_not_found,
                                    allow_partial=allow_partial,
                                    remote_instance=remote_instance)

    # Collapse for intersection...
    if intersect:
        annotation_post = [{'annotated_with[{}]'.format(i): v for i, v in enumerate(list(pos_ids.values()))}]
        annotation_post[0].update({'not_annotated_with[{}]'.format(i): v for i, v in enumerate(list(neg_ids.values()))})
    # ... or keep separate for no intersection
    else:
        annotation_post = [{'annotated_with': an} for an in pos_ids.values()]
        annotation_post += [{'not_annotated_with': an} for an in neg_ids.values()]
        # Need to clear empties
        annotation_post = [p for p in annotation_post if p]

    # Query server
    remote_annotated_url = [remote_instance._get_annotated_url() for _ in annotation_post]
    resp = remote_instance.fetch(remote_annotated_url, post=annotation_post)

    # Extract skids from responses
    annotated_skids = [e['skeleton_ids'][0] for r in resp for e in r['entities'] if e['type'] == 'neuron']

    # Remove duplicates
    annotated_skids = list(set(annotated_skids))

    logger.debug('Found {} neurons with matching annotation(s)'.format(len(annotated_skids)))

    return annotated_skids


@cache.undo_on_error
def neuron_exists(x, remote_instance=None):
    """Check if neurons exist in CATMAID.

    Parameters
    ----------
    x
                        Neurons to check if they exist in Catmaid. Can be:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    bool :
                        True if skeleton exists, False if not. If multiple
                        neurons are queried, returns a dict
                        ``{skid1: True, skid2: False, ...}``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    if len(x) > 1:
        return {n: neuron_exists(n, remote_instance=remote_instance) for n in x}
    else:
        x = x[0]

    remote_get_neuron_name = remote_instance._get_single_neuronname_url(x)
    response = remote_instance.fetch(remote_get_neuron_name)

    if 'error' in response:
        return False
    else:
        return True


@cache.undo_on_error
def get_node_info(x, remote_instance=None):
    """Retrieve info for a set of nodes.

    Parameters
    ----------
    x                   CatmaidNeuron | CatmaidNeuronList | list of node IDs
                        Single or list of node IDs. If CatmaidNeuron/List,
                        details for all it's nodes are requested.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
                DataFrame in which each row represents a queried node::

                   node_id  neuron_name  skeleton_id  skeleton_name  neuron_id
                 0
                 1
                 ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    node_ids = utils.eval_node_ids(x, connectors=False, nodes=True)

    urls = [remote_instance._get_single_node_info_url(tn) for tn in node_ids]

    data = remote_instance.fetch(urls, desc='Get info')

    df = pd.DataFrame([[node_ids[i]] + list(n.values()) for i, n in enumerate(data)],
                      columns=['node_id'] + list(data[0].keys())
                      )

    return df


@cache.undo_on_error
def get_node_tags(node_ids, node_type, remote_instance=None):
    """Retrieve tags for a set of nodes OR connectors.

    Parameters
    ----------
    node_ids
                        Single or list of node or connector IDs.
    node_type :         'NODE' | 'CONNECTOR'
                        Set which node type of IDs you have provided as they
                        use different API endpoints!
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                dictionary containing tags for each node:
                ``{'nodeID': ['tag1', 'tag2', ...], 'nodeID' : [...], ...}``

    Examples
    --------
    >>> pymaid.get_node_tags(['6626578', '6633237']
    ...                       'NODE',
    ...                       remote_instance)
    {'6633237': ['ends'], '6626578': ['ends']}

    See Also
    --------
    :func:`pymaid.add_tags`
                        Use to add tags to nodes.
    :func:`pymaid.delete_tags`
                        Use to delete node tags.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(node_ids, (list, np.ndarray)):
        node_ids = [node_ids]

    # Make sure node_ids are strings
    node_ids = [str(n) for n in node_ids]

    url = remote_instance._get_node_labels_url()

    if node_type in ['TREENODE', 'TREENODES', 'NODES', 'NODE']:
        key = 'treenode_ids'
    elif node_type in ['CONNECTOR', 'CONNECTORS']:
        key = 'connector_ids'
    else:
        raise TypeError(f'Unknown node_type parameter: {node_type}')

    POST = {key: ','.join([str(tn) for tn in node_ids])}

    return remote_instance.fetch(url, post=POST)


@cache.undo_on_error
def get_segments(x, remote_instance=None):
    """Retrieve list of segments for a neuron just like the review widget.

    Parameters
    ----------
    x
                        Neurons to retrieve. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    list
                List of node IDs, ordered by length. If multiple neurons
                are requested, returns a dict ``{skid: [], ...}``.

    See Also
    --------
    ``CatmaidNeuron.segments``
    ``CatmaidNeuron.short_segments``
                Use these :class:`pymaid.CatmaidNeuron` attributes to access
                segments generated by pymaid (faster).

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    urls = []
    post_data = []

    for s in x:
        urls.append(remote_instance._get_review_details_url(s))
        # For some reason this needs to fetched as POST (even though actual
        # POST data is not necessary)
        post_data.append({'placeholder': 0})

    rdata = remote_instance.fetch(urls, post=post_data, desc='Get segs')

    if len(x) > 1:
        return {x[i]: [[tn['id'] for tn in arb['sequence']] for arb in rdata[i]] for i in range(len(x))}
    else:
        return [[tn['id'] for tn in arb['sequence']] for arb in rdata[0]]


@cache.undo_on_error
def get_review_details(x, remote_instance=None):
    """Retrieve review status (reviewer + timestamp) by node for given neuron.

    Parameters
    ----------
    x
                        Neurons to get review-details for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
        DataFrame in which each row respresents a node::

            node_id  skeleton_id  reviewer1  reviewer2  reviewer 3
          0    12345       12345123     datetime    NaT      datetime
          1
          ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    node_list = []
    urls = []
    post_data = []

    for s in x:
        urls.append(remote_instance._get_review_details_url(s))
        # For some reason this needs to fetched as POST (even though actual
        # POST data is not necessary)
        post_data.append({'placeholder': 0})

    rdata = remote_instance.fetch(urls,
                                  post=post_data,
                                  desc='Get rev stats')

    for i, neuron in enumerate(rdata):
        # There is a small chance that nodes are counted twice but not
        # tracking node_id speeds up this extraction a LOT
        # node_ids = []
        for arbor in neuron:
            node_list += [(n['id'], x[i], n['rids'])
                          for n in arbor['sequence'] if n['rids']]

    tn_to_skid = {n[0]: n[1] for n in node_list}
    node_dict = {n[0]: {u[0]: datetime.datetime.strptime(
        u[1][:16], '%Y-%m-%dT%H:%M') for u in n[2]} for n in node_list}

    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    df = pd.DataFrame.from_dict(node_dict, orient='index').fillna(np.nan)
    df.columns = df.columns.map(user_dict)
    df['skeleton_id'] = [tn_to_skid[tn] for tn in df.index.values]
    df.index.name = 'node_id'
    df = df.reset_index(drop=False)

    # Make sure we didn't count nodes twice
    df = df[~df.duplicated('node_id')]

    return df


@cache.undo_on_error
def get_logs(operations=[], entries=50, display_start=0, search="",
             remote_instance=None):
    """Retrieve logs (same data as in log widget).

    Parameters
    ----------
    operations :        list of str, optional
                        If empty, all operations will be queried from server
                        possible operations: 'join_skeleton',
                        'change_confidence', 'rename_neuron', 'create_neuron',
                        'create_skeleton', 'remove_neuron', 'split_skeleton',
                        'reroot_skeleton', 'reset_reviews', 'move_skeleton'
    entries :           int, optional
                        Number of entries to retrieve.
    display_start :     int, optional
                        Sets range of entries to return:
                        ``display_start`` to ``display_start + entries``.
    search :            str, optional
                        Use to filter results for e.g. a specific skeleton ID
                        or neuron name.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a single operation::

            user   operation   timestamp   x   y   z   explanation
         0
         1
         ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not operations:
        operations = [-1]
    elif not isinstance(operations, (list, np.ndarray)):
        operations = [operations]

    logs = []
    for op in operations:
        get_logs_postdata = {'sEcho': 6,
                             'iColumns': 7,
                             'iDisplayStart': display_start,
                             'iDisplayLength': entries,
                             'mDataProp_0': 0,
                             'sSearch_0': '',
                             'bRegex_0': False,
                             'bSearchable_0': False,
                             'bSortable_0': True,
                             'mDataProp_1': 1,
                             'sSearch_1': '',
                             'bRegex_1': False,
                             'bSearchable_1': False,
                             'bSortable_1': True,
                             'mDataProp_2': 2,
                             'sSearch_2': '',
                             'bRegex_2': False,
                             'bSearchable_2': False,
                             'bSortable_2': True,
                             'mDataProp_3': 3,
                             'sSearch_3': '',
                             'bRegex_3': False,
                             'bSearchable_3': False,
                             'bSortable_3': False,
                             'mDataProp_4': 4,
                             'sSearch_4': '',
                             'bRegex_4': False,
                             'bSearchable_4': False,
                             'bSortable_4': False,
                             'mDataProp_5': 5,
                             'sSearch_5': '',
                             'bRegex_5': False,
                             'bSearchable_5': False,
                             'bSortable_5': False,
                             'mDataProp_6': 6,
                             'sSearch_6': '',
                             'bRegex_6': False,
                             'bSearchable_6': False,
                             'bSortable_6': False,
                             'sSearch': '',
                             'bRegex': False,
                             'iSortCol_0': 2,
                             'sSortDir_0': 'desc',
                             'iSortingCols': 1,
                             'self.project_id': remote_instance.project_id,
                             'operation_type': op,
                             'search_freetext': search}

        remote_get_logs_url = remote_instance._get_logs_url()
        logs += remote_instance.fetch(remote_get_logs_url,
                                      post=get_logs_postdata)['aaData']

    df = pd.DataFrame(logs,
                      columns=['user', 'operation', 'timestamp',
                               'x', 'y', 'z', 'explanation']
                      )

    df['timestamp'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['timestamp'].values]

    return df


@cache.undo_on_error
def get_contributor_statistics(x, separate=False, max_threads=500,
                               remote_instance=None):
    """Retrieve contributor statistics for given skeleton ids.

    By default, stats are given over all neurons.

    Parameters
    ----------
    x
                        Neurons to get contributor stats for. Can be either:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    separate :          bool, optional
                        If True, stats are given per neuron.
    max_threads :       int, optional
                        Maximum parallel data requests. Overrides
                        ``CatmaidInstance.max_threads``.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Series, if ``separate=False``. DataFrame, if ``separate=True``::

           skeleton_id  node_contributors  multiuser_review_minutes  ..
         1
         2
         3
           post_contributors  construction_minutes  min_review_minutes  ..
         1
         2
         3
           n_postsynapses  n_presynapses  pre_contributors  n_nodes  ..
         1
         2
         3
           review_contributors
         1
         2
         3

    Examples
    --------
    >>> # Plot contributions as pie chart
    >>> import matplotlib.pyplot as plt
    >>> cont = pymaid.get_contributor_statistics("annotation:uPN right")
    >>> plt.subplot(131, aspect=1)
    >>> ax1 = plt.pie(cont.node_contributors.values(),
    ...               labels=cont.node_contributors.keys(),
    ...               autopct='%.0f%%' )
    >>> plt.subplot(132, aspect=1)
    >>> ax2 = plt.pie(cont.pre_contributors.values(),
    ...               labels=cont.pre_contributors.keys(),
    ...               autopct='%.0f%%' )
    >>> plt.subplot(133, aspect=1)
    >>> ax3 = plt.pie(cont.post_contributors.values(),
    ...               labels=cont.post_contributors.keys(),
    ...               autopct='%.0f%%' )
    >>> plt.show()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_skids(x, remote_instance=remote_instance)

    columns = ['skeleton_id', 'n_nodes', 'node_contributors', 'n_presynapses',
               'pre_contributors', 'n_postsynapses', 'post_contributors',
               'review_contributors', 'multiuser_review_minutes',
               'construction_minutes', 'min_review_minutes']

    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    if not separate:
        with config.tqdm(total=len(x), desc='Contr. stats',
                         disable=config.pbar_hide,
                         leave=config.pbar_leave) as pbar:
            stats = []
            for j in range(0, len(x), max_threads):
                pbar.update(j)
                get_statistics_postdata = {}

                for i in range(j, min(len(x), j + max_threads)):
                    key = 'skids[%i]' % i
                    get_statistics_postdata[key] = x[i]

                remote_get_statistics_url = remote_instance._get_contributions_url()
                stats.append(remote_instance.fetch(remote_get_statistics_url,
                                                   post=get_statistics_postdata))

        # Now generate DataFrame
        node_contributors = {user_dict.get(int(u)): sum([st['node_contributors'][u] for st in stats if u in st[
            'node_contributors']]) for st in stats for u in st['node_contributors']}
        pre_contributors = {user_dict.get(int(u)): sum([st['pre_contributors'][u] for st in stats if u in st[
            'pre_contributors']]) for st in stats for u in st['pre_contributors']}
        post_contributors = {user_dict.get(int(u)): sum([st['post_contributors'][u] for st in stats if u in st[
            'post_contributors']]) for st in stats for u in st['post_contributors']}
        review_contributors = {user_dict.get(int(u)): sum([st['review_contributors'][u] for st in stats if u in st[
            'review_contributors']]) for st in stats for u in st['review_contributors']}

        df = pd.Series([
            x,
            sum([st['n_nodes'] for st in stats]),
            node_contributors,
            sum([st['n_pre'] for st in stats]),
            pre_contributors,
            sum([st['n_post'] for st in stats]),
            post_contributors,
            review_contributors,
            sum([st['multiuser_review_minutes'] for st in stats]),
            sum([st['construction_minutes'] for st in stats]),
            sum([st['min_review_minutes'] for st in stats])
        ],
            index=columns,
            dtype=object
        )
    else:
        get_statistics_postdata = [{'skids[0]': s} for s in x]
        remote_get_statistics_url = [
            remote_instance._get_contributions_url() for s in x]

        stats = remote_instance.fetch(remote_get_statistics_url,
                                      post=get_statistics_postdata,
                                      desc='Get contrib.')

        df = pd.DataFrame([[
            s,
            stats[i]['n_nodes'],
            {user_dict.get(int(u)): stats[i]['node_contributors'][u]
                for u in stats[i]['node_contributors']},
            stats[i]['n_pre'],
            {user_dict.get(int(u)): stats[i]['pre_contributors'][u]
                for u in stats[i]['pre_contributors']},
            stats[i]['n_post'],
            {user_dict.get(int(u)): stats[i]['post_contributors'][u]
                for u in stats[i]['post_contributors']},
            {user_dict.get(int(u)): stats[i]['review_contributors'][u]
                for u in stats[i]['review_contributors']},
            stats[i]['multiuser_review_minutes'],
            stats[i]['construction_minutes'],
            stats[i]['min_review_minutes']
        ] for i, s in enumerate(x)],
            columns=columns,
            dtype=object
        )
    return df


@cache.undo_on_error
def get_history(start_date=(datetime.date.today() - datetime.timedelta(days=7)).isoformat(),
                end_date=datetime.date.today().isoformat(), split=True,
                remote_instance=None):
    """Retrieves CATMAID project history.

    If the time window is too large, the connection might time out which will
    result in an error! Make sure ``split=True`` to avoid that.

    Parameters
    ----------
    start_date :        datetime | str | tuple, optional, default=last week
                        dates can be either:
                            - ``datetime.date``
                            - ``datetime.datetime``
                            - str ``'YYYY-MM-DD'``, e.g. ``'2016-03-09'``
                            - tuple ``(YYYY, MM, DD)``, e.g. ``(2016, 3, 9)``
    end_date :          datetime | str | tuple, optional, default=today
                        See start_date.
    split :             bool, optional
                        If True, history will be requested in bouts of 6 months.
                        Useful if you want to look at a very big time window
                        as this can lead to gateway timeout.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pandas.Series
            A pandas.Series with the following entries::

            {
            cable :             DataFrame containing cable created in nm.
                                Rows = users, columns = dates
            connector_links :   DataFrame containing connector links created.
                                Rows = users, columns = dates
            reviewed :          DataFrame containing nodes reviewed.
                                Rows = users, columns = dates
            user_details :      user-list (see pymaid.get_user_list())
            nodes :             DataFrame containing nodes created by user.
            }

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Plot cable created by all users over time
    >>> hist.cable.T.plot()
    >>> plt.show()
    >>> # Collapse users and plot sum of cable over time
    >>> hist.cable.sum(0).plot()
    >>> plt.show()
    >>> # Plot single users cable (index by user login name)
    >>> hist.cable.loc['schlegelp'].T.plot()
    >>> plt.show()
    >>> # Sum up cable created this week by all users
    >>> hist.cable.values.sum()
    >>> # Get number of active (non-zero) users
    >>> active_users = hist.cable.astype(bool).sum(axis=0)

    See Also
    -------
    :func:`~pymaid.get_user_stats`
            Returns a summary of user stats as table.
    :func:`~pymaid.plot_history`
            Quick way to plot history over time.

    """
    def _constructor_helper(data, key, days):
        """ Helper to extract variable from data returned by CATMAID server
        """
        temp = []
        for d in days:
            try:
                temp.append(data[d][key])
            except BaseException:
                temp.append(0)
        return temp

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(start_date, datetime.date):
        start_date = start_date.isoformat()
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.isoformat()[:10]
    elif isinstance(start_date, (tuple, list)):
        start_date = datetime.date(start_date[0], start_date[
                                   1], start_date[2]).isoformat()

    if isinstance(end_date, datetime.date):
        end_date = end_date.isoformat()
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.isoformat()[:10]
    elif isinstance(end_date, (tuple, list)):
        end_date = datetime.date(end_date[0], end_date[
                                 1], end_date[2]).isoformat()

    rounds = []
    if split:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

        logger.info(
            'Retrieving %i days of history in bouts!' % (end - start).days)

        # First make big bouts of roughly 6 months each
        while start < (end - datetime.timedelta(days=6 * 30)):
            rounds.append((start.isoformat(),
                          (start + datetime.timedelta(days=6 * 30)).isoformat()))
            start += datetime.timedelta(days=6 * 30)

        # Append the last bit
        if start < end:
            rounds.append((start.isoformat(), end.isoformat()))
    else:
        rounds = [(start_date, end_date)]

    data = []
    for r in config.tqdm(rounds, desc='Retrieving history',
                         disable=config.pbar_hide, leave=config.pbar_leave):
        get_history_GET_data = {'pid': remote_instance.project_id,
                                'start_date': r[0],
                                'end_date': r[1]
                                }

        remote_get_history_url = remote_instance._get_history_url()

        remote_get_history_url += f'?{urllib.parse.urlencode(get_history_GET_data)}'

        logger.debug(f'Retrieving user history from {r[0]} to {r[1]}.')

        data.append(remote_instance.fetch(remote_get_history_url))

    # Now merge data into a single dict
    stats = dict(data[0])
    for d in data:
        stats['days'] += [e for e in d['days'] if e not in stats['days']]
        stats['daysformatted'] += [e for e in d['daysformatted']
                                   if e not in stats['daysformatted']]

        for u in d['stats_table']:
            stats['stats_table'][u].update(d['stats_table'][u])

    user_list = get_user_list(remote_instance=remote_instance).set_index('id')
    user_list.index = user_list.index.astype(str)
    user_dict = user_list.login.to_dict()

    df = pd.Series([
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_cable_length', stats['days']) for u in stats['stats_table']],
                     index=[user_dict.get(u, f'Anonymous{i}') for i, u in enumerate(stats['stats_table'].keys())],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_treenodes', stats['days']) for u in stats['stats_table']],
                     index=[user_dict.get(u, f'Anonymous{i}') for i, u in enumerate(stats['stats_table'].keys())],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_connectors', stats['days']) for u in stats['stats_table']],
                     index=[user_dict.get(u, f'Anonymous{i}') for i, u in enumerate(stats['stats_table'].keys())],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        pd.DataFrame([_constructor_helper(stats['stats_table'][u], 'new_reviewed_nodes', stats['days']) for u in stats['stats_table']],
                     index=[user_dict.get(u, f'Anonymous{i}') for i, u in enumerate(stats['stats_table'].keys())],
                     columns=pd.to_datetime([datetime.datetime.strptime(d, '%Y%m%d').date() for d in stats['days']])),
        user_list.reset_index(drop=True)
    ],
        index=['cable', 'nodes', 'connector_links',
               'reviewed', 'user_details']
    )

    return df


@cache.undo_on_error
def get_nodes_in_volume(*x,  coord_format='NM', resolution=(4, 4, 50),
                        remote_instance=None):
    """Retrieve nodes and connectors in given bounding box.

    Please note that there is a cap on the number of nodes returned that is
    hard wired into the CATMAID server's settings.

    Parameters
    ----------
    *x
                            Coordinates defining the bounding box. Can be
                            either:

                             - 1d list of coordinates: left, right, top, bottom, z1, z2
                             - 2d list of coordinates: [[left, right], [top, bottom], [z1, z2]]
                             - pymaid.Volume

                            Can be given in nm or pixels.
    coord_format :          str, optional
                            Define whether provided coordinates are in
                            nanometer ('NM') or in pixels/slices ('PIXEL').
    resolution :            tuple of floats, optional
                            x/y/z resolution in nm [default = (4, 4, 50)]
                            Used to transform to nm if limits are given in
                            pixels.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    nodes :         pandas.DataFrame
                    DataFrame in which each row is a node::

                    node_id  parent_id  x  y  z  confidence  radius  skeleton_id  edition_time  user_id
                 0
                 1
                 2

    connectors :    pandas.DataFrame
                    DataFrame in which each row is a connector::

                    connector_id x y z confidence edition_time user_id partners
                 0
                 1
                 2

                    ``partners`` are lists of::

                    [node_id, relation_id, link_confidence, link_edition_time, link_id]

    truncated :     bool
                    If True, lists are truncated due to node limit reached.
    relation_map :  dict
                    Map for ``relation_id`` in connector's ``partner`` column.

    Examples
    --------
    Get (truncated) lists of nodes and connectors in the bounding box of the AL:

    >>> al = pymaid.get_volume('AL_R')
    >>> nodes, connectors, truncated, relation_map = pymaid.get_nodes_in_volume(al)
    >>> truncated
    True

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x[0], ns.Volume):
        x = x[0].bbox

    # Flatten the list of coordinates
    coords = np.array(x).flatten()

    if coords.shape[0] != 6:
        raise ValueError('Must provide 6 coordinates (left, right, top, '
                         'bottom, z1, z1), got {}'.format(coords.shape[0]))

    # Extract coords
    left, right, top, bottom, z1, z2 = coords

    # Set resolution to 1:1 if coordinates are already in nm
    if coord_format == 'NM':
        resolution = (1, 1, 1)

    remote_nodes_list = remote_instance._get_node_list_url()

    node_list_postdata = {
        'left': left * resolution[0],
        'right': right * resolution[0],
        'top': top * resolution[1],
        'bottom': bottom * resolution[1],
        'z1': z1 * resolution[2],
        'z2': z2 * resolution[2],
        # Atnid seems to be related to fetching the active node too
        # -> will be ignored if atnid = -1
        'atnid': -1,
        'labels': False,
        # 'limit': 3500,  # this doesn't do anything -> hard wired into server settings

    }

    node_data = remote_instance.fetch(remote_nodes_list,
                                      post=node_list_postdata)

    tn = pd.DataFrame(node_data[0],
                      columns=['node_id', 'parent_id',
                               'x', 'y', 'z', 'confidence',
                               'radius', 'skeleton_id',
                               'edition_time', 'user_id'])
    # Fix parent ID
    tn.loc[tn.parent_id.isnull(), 'parent_id'] = -1
    tn['parent_id'] = tn.parent_id.astype(int).astype(object)
    tn.loc[tn.parent_id < 0, 'parent_id'] = None

    tn['edition_time'] = pd.to_datetime(tn.edition_time, unit='s', utc=True)

    cn = pd.DataFrame(node_data[1],
                      columns=['connector_id', 'x', 'y', 'z',
                               'confidence', 'edition_time',
                               'user_id', 'partners'])
    cn['edition_time'] = pd.to_datetime(cn.edition_time, unit='s', utc=True)

    node_limit_reached = node_data[3]

    relation_map = node_data[4]

    return tn, cn, node_limit_reached, relation_map


@cache.undo_on_error
def find_neurons(names=None, annotations=None, volumes=None, users=None,
                 from_date=None, to_date=None, reviewed_by=None, skids=None,
                 intersect=False, partial_match=False, only_soma=False,
                 min_size=1, minimum_cont=None, remote_instance=None):
    """Find neurons matching given search criteria.

    Warning
    -------
    Depending on the parameters, this can take quite a while! Also: by default,
    will return single-node neurons! Use the ``min_size`` parameter to change
    that behaviour.

    Parameters
    ----------
    names :             str | list of str
                        Neuron name(s) to search for.
    annotations :       str | list of str
                        Annotation(s) to search for.
    volumes :           str | navis.Volume | list of either
                        CATMAID volume(s) to look into. This uses
                        :func:`~pymaid.get_neurons_in_volumes` and will look
                        for neurons within the **bounding box** of given
                        volume(s).
    users :             int | str | list of either, optional
                        User ID(s) (int) or login(s) (str).
    reviewed_by :       int | str | list of either, optional
                        User ID(s) (int) or login(s) (str) of reviewer.
    from_date :         datetime | list of integers, optional
                        Format: [year, month, day]. Return neurons created
                        after this date. This works ONLY if also querying by
                        ``users`` or ``reviewed_by``!
    to_date :           datetime | list of integers, optional
                        Format: [year, month, day]. Return neurons created
                        before this date. This works ONLY if also querying by
                        ``users`` or ``reviewed_by``!
    skids :             list of skids, optional
                        Can be a list of skids, a CatmaidNeuronList or pandas
                        DataFrame with "skeleton_id" column.
    intersect :         bool, optional
                        If multiple search criteria are provided, neurons have
                        to meet all of them in order to be returned. This
                        is first applied WITHIN search criteria (works for
                        multiple ``annotations``, ``volumes``, ``users`` and
                        ``reviewed_by``) and then ACROSS critera!
    partial_match :     bool, optional
                        If True, partial matches for *names* AND *annotations*
                        are allowed.
    minimum_cont :      int, optional
                        If looking for specific ``users``: minimum contribution
                        (in nodes) to a neuron in order for it to be counted.
                        Only applicable if ``users`` is provided. If multiple
                        users are provided contribution is calculated across
                        all users. Minimum contribution does NOT take start
                        and end dates into account! This is applied AFTER
                        intersecting!
    min_size :          int, optional
                        Minimum size (in nodes) for neurons to be returned.
                        The lower this value, the longer it will take to
                        filter.
    only_soma :         bool, optional
                        If True, only neurons with a soma are returned.
    remote_instance :   CatmaidInstance
                        If not passed directly, will try using globally
                        defined CatmaidInstance.
    Returns
    -------
    :class:`~pymaid.CatmaidNeuronList`

    Examples
    --------
    >>> # Simple request for neurons with given annotations
    >>> to_find = ['glomerulus DA1', 'glomerulus DL4']
    >>> skids = pymaid.find_neurons(annotations=to_find)
    >>> # Get only neurons that have both annotations
    >>> skids = pymaid.find_neurons(annotations=to_find, intersect=True)
    >>> # Get all neurons with more than 1000 nodes
    >>> skids = pymaid.find_neurons(min_size=1000)
    >>> # Get all neurons that have been traced recently by given user
    >>> skids = pymaid.find_neurons(users='schlegelp',
    ...                             from_date=[2017, 10, 1])
    >>> # Get all neurons traced by a given user within a certain volume
    >>> skids = pymaid.find_neurons(users='schlegelp',
    ...                             minimum_cont=1000,
    ...                             volumes='LH_R')

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    # Fist, we have to prepare a whole lot of parameters
    if users:
        users = utils.eval_user_ids(users, remote_instance=remote_instance)
    if reviewed_by:
        reviewed_by = utils.eval_user_ids(
            reviewed_by, remote_instance=remote_instance)

    if annotations and not isinstance(annotations, (list, np.ndarray)):
        annotations = [annotations]
    if names and not isinstance(names, (list, np.ndarray)):
        names = [names]
    if volumes and not isinstance(volumes, (list, np.ndarray)):
        volumes = [volumes]

    # Bring dates into the correct format
    if from_date and not to_date:
        today = datetime.date.today()
        to_date = (today.year, today.month, today.day)
    elif to_date and not from_date:
        from_date = (1900, 1, 1)

    if isinstance(from_date, datetime.date):
        from_date = [from_date.year, from_date.month, from_date.day]

    if isinstance(to_date, datetime.date):
        to_date = [to_date.year, to_date.month, to_date.day]

    # Warn if from/to_date are used without also querying by user or reviewer
    if from_date and not (users or reviewed_by):
        logger.warning('Start/End dates can only be used for queries against '
                       '<users> or <reviewed_by>')

    # Now go over all parameters and get sets of skids
    sets_of_skids = []

    if not isinstance(skids, type(None)):
        skids = utils.eval_skids(skids, remote_instance=remote_instance)
        sets_of_skids.append(set(skids))

    # Get skids by name
    if names:
        urls = [remote_instance._get_annotated_url() for n in names]
        post_data = [{'name': str(n),
                      'with_annotations': False,
                      'name_exact': not partial_match}
                     for n in names]

        results = remote_instance.fetch(urls,
                                        post=post_data,
                                        desc='Get names')

        this_name = []
        for i, r in enumerate(results):
            for e in r['entities']:
                if partial_match and e['type'] == 'neuron' and names[i].lower() in e['name'].lower():
                    this_name.append(e['skeleton_ids'][0])
                if not partial_match and e['type'] == 'neuron' and e['name'] == names[i]:
                    this_name.append(e['skeleton_ids'][0])

        sets_of_skids.append(set(this_name))

    # Get skids by annotation
    if annotations:
        annotation_ids = get_annotation_id(annotations,
                                           allow_partial=partial_match,
                                           remote_instance=remote_instance)

        if not annotation_ids:
            raise Exception('No matching annotation(s) found!')

        if partial_match is True:
            logger.debug('Found {0} id(s) (partial matches '
                         'included)'.format(len(annotation_ids)))
        else:
            logger.debug('Found id(s): %s | Unable to retrieve: %i' % (
                str(annotation_ids), len(annotations) - len(annotation_ids)))

        urls = [remote_instance._get_annotated_url() for an in annotation_ids]
        post_data = [{'annotated_with': str(an), 'with_annotations': 'false'}
                     for an in annotation_ids.values()]
        results = remote_instance.fetch(urls,
                                        post=post_data,
                                        desc='Get annot')

        annotated = [set([e['skeleton_ids'][0] for e in res['entities'] if e['type'] == 'neuron']) for res in results]

        # Intersect within search criteria if applicable
        if intersect:
            sets_of_skids.append(set.intersection(*annotated))
        else:
            sets_of_skids.append(set.union(*annotated))

    # Get skids by user
    if users:
        urls = [remote_instance._get_list_skeletons_url() for u in users]
        GET_data = [{'nodecount_gt': min_size - 1,
                     'created_by': u} for u in users]

        if from_date and to_date:
            dates = {'from': ''.join(['{0:02d}'.format(d) for d in from_date]),
                     'to': ''.join(['{0:02d}'.format(d) for d in to_date])}
            GET_data = [{**d, **dates} for d in GET_data]
        urls = [u + '?%s' % urllib.parse.urlencode(g) for u, g in zip(urls, GET_data)]

        results = remote_instance.fetch(urls, desc='Get users')

        # Intersect within search criteria if applicable
        if intersect:
            sets_of_skids.append(set.intersection(*[set(res) for res in results]))
        else:
            sets_of_skids.append(set.union(*[set(res) for res in results]))

    # Get skids by reviewer
    if reviewed_by:
        urls = [remote_instance._get_list_skeletons_url() for u in reviewed_by]
        GET_data = [{'nodecount_gt': min_size - 1,
                     'reviewed_by': u} for u in reviewed_by]

        if from_date and to_date:
            dates = {'from': ''.join(['{0:02d}'.format(d) for d in from_date]),
                     'to': ''.join(['{0:02d}'.format(d) for d in to_date])}
            GET_data = [{**d, **dates} for d in GET_data]
        urls = [u + '?%s' % urllib.parse.urlencode(g) for u, g in zip(urls, GET_data)]

        results = remote_instance.fetch(urls, desc='Get reviewers')

        # Intersect within search criteria if applicable
        if intersect:
            sets_of_skids.append(set.intersection(*[set(res) for res in results]))
        else:
            sets_of_skids.append(set.union(*[set(res) for res in results]))

    # Get by volume
    if volumes:
        temp = []
        for v in config.tqdm(volumes, desc='Get by vols',
                             disable=config.pbar_hide,
                             leave=config.pbar_leave):
            if not isinstance(v, ns.Volume):
                vol = get_volume(v, remote_instance)
            else:
                vol = v

            temp.append(set(get_neurons_in_bbox(vol.bbox,
                                                remote_instance=remote_instance)))

        # Intersect within search criteria if applicable
        if intersect:
            sets_of_skids.append(set.intersection(*temp))
        else:
            sets_of_skids.append(set.union(*temp))

    # Get neurons by size if only min_size and no other no parameters were
    # provided
    if False not in [isinstance(param, type(None)) for param in [names,
                                                                 annotations,
                                                                 volumes,
                                                                 users,
                                                                 reviewed_by,
                                                                 skids]]:
        # Make sure people don't accidentally request ALL neurons in the
        # dataset
        if min_size <= 1:
            answer = ""
            while answer not in ["y", "n"]:
                answer = input("Your search parameters will retrieve ALL "
                               "neurons in the dataset. Proceed? "
                               "[Y/N] ").lower()

            if answer != 'y':
                logger.info('Query cancelled')
                return

        logger.info(
            'Get all neurons with >= {0} nodes'.format(min_size))
        get_skeleton_list_GET_data = {'nodecount_gt': min_size - 1}
        remote_get_list_url = remote_instance._get_list_skeletons_url()
        remote_get_list_url += '?%s' % urllib.parse.urlencode(
            get_skeleton_list_GET_data)
        these_neurons = set(remote_instance.fetch(remote_get_list_url))

        sets_of_skids.append(these_neurons)

    # Now intersect/merge ACROSS search criteria
    if intersect:
        logger.info('Intersecting by search parameters')
        skids = list(set.intersection(*sets_of_skids))
    else:
        skids = list(set.union(*sets_of_skids))

    # Filtering by size was already done for users and reviewed_by and dates
    # If we queried by annotations, names or volumes we need to do this
    # explicitly here
    if min_size > 1 and (volumes or annotations or names):
        logger.info('Filtering neurons for size')

        get_skeleton_list_GET_data = {'nodecount_gt': min_size - 1}
        remote_get_list_url = remote_instance._get_list_skeletons_url()
        remote_get_list_url += '?%s' % urllib.parse.urlencode(
            get_skeleton_list_GET_data)
        neurons_by_size = set(remote_instance.fetch(remote_get_list_url))

        skids = set.intersection(set(skids), neurons_by_size)

    nl = core.CatmaidNeuronList(list(skids), remote_instance=remote_instance)

    if only_soma:
        hs = has_soma(nl, return_ids=False, remote_instance=remote_instance)
        nl = core.CatmaidNeuronList([n for n in nl if hs[int(n.skeleton_id)]])

    if users and minimum_cont:
        nl.get_skeletons(skip_existing=True)
        nl = core.CatmaidNeuronList([n for n in nl if n.nodes[n.nodes.creator_id.isin(users)].shape[0] >= minimum_cont],
                                    remote_instance=remote_instance)

    if nl.empty:
        logger.warning(
            'No neurons matching the search parameters were found')
    else:
        logger.info(f'Found {len(nl)} neurons matching the search parameters')
        nl.get_names()

    return nl


@cache.undo_on_error
def get_neurons_in_volume(volumes, min_nodes=2, min_cable=1, intersect=False,
                          only_soma=False, remote_instance=None):
    """Retrieves neurons with processes within CATMAID volumes.

    This function uses the **BOUNDING BOX** around volume as proxy and queries
    for neurons that are within that volume. See examples on how to work
    around this.

    Warning
    -------
    Depending on the number of nodes in that volume, this can take quite a
    while! Also: by default, will NOT return single-node neurons - use the
    ``min_nodes`` parameter to change that behaviour.

    Parameters
    ----------
    volumes :               str | navis.Volume | list of either
                            Single or list of CATMAID volumes.
    min_nodes :             int, optional
                            Minimum node count for a neuron within given
                            volume(s).
    min_cable :             int, optional
                            Minimum cable length [nm] for a neuron within
                            given volume(s).
    intersect :             bool, optional
                            If multiple volumes are provided, this parameter
                            determines if neurons have to be in all of the
                            volumes or just a single.
    only_soma :             bool, optional
                            If True, only neurons with a soma will be returned.
    remote_instance :       CatmaidInstance
                            If not passed directly, will try using global.

    Returns
    -------
    list
                            ``[skeleton_id, skeleton_id, ...]``

    See Also
    --------
    :func:`~pymaid.get_partners_in_volume`
                            Get only partners that make connections within a
                            given volume.
    :func:`pymaid.find_neurons`
                            Use to retrieve neurons by combining various
                            search criteria. For example names, reviewers,
                            annotations, etc.

    Examples
    --------
    >>> # Get a volume
    >>> lh = pymaid.get_volume('LH_R')
    >>> # Get neurons within the bounding box of a volume
    >>> skids = pymaid.get_neurons_in_volume(lh, min_nodes=10)
    >>> # Retrieve 3D skeletons of these neurons
    >>> lh_neurons = pymaid.get_neurons(skids)
    >>> # Prune by volume
    >>> lh_pruned = lh_neurons.copy()
    >>> lh_pruned.prune_by_volume(lh)
    >>> # Filter neurons with more than 100um of cable in the volume
    >>> n = lh_neurons[lh_pruned.cable_length > 100]

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if not isinstance(volumes, (list, np.ndarray)):
        volumes = [volumes]

    for i, v in enumerate(volumes):
        if not isinstance(v, ns.Volume):
            volumes[i] = get_volume(v)

    neurons = []

    for v in volumes:
        logger.info('Retrieving neurons in volume {0}'.format(v.name))
        temp = get_neurons_in_bbox(v, min_nodes=min_nodes,
                                   min_cable=min_cable,
                                   remote_instance=remote_instance)

        if not intersect:
            neurons += list(temp)
        else:
            neurons += [temp]

    if intersect:
        # Filter for neurons that show up in all neuropils
        neurons = [n for l in neurons for n in l if False not in [n in v for v in neurons]]

    # Need to do this in case we have several volumes
    neurons = list(set(neurons))

    if only_soma:
        soma = has_soma(neurons, remote_instance=remote_instance)
        neurons = [n for n in neurons if soma[n] is True]

    logger.info('Done. {0} unique neurons found in volume(s) '
                '{1}'.format(len(neurons),
                             ','.join([v.name for v in volumes])))

    return neurons


@cache.undo_on_error
def get_neurons_in_bbox(bbox, unit='NM', min_nodes=1, min_cable=1,
                        remote_instance=None, **kwargs):
    """Retrieve neurons with processes within a defined box volume.

    Parameters
    ----------
    bbox :                  list-like | dict | navis.Volume
                            Coordinates of the bounding box. Can be either:

                              1. List/np.array: ``[[left, right], [top, bottom], [z1, z2]]``
                              2. Dictionary ``{'left': int|float, 'right': ..., ...}``
    unit :                  'NM' | 'PIXEL'
                            Unit of your coordinates. Attention:
                            'PIXEL' will also assume that Z1/Z2 is in slices.
                            By default, xyz resolution of 4x4x40nm per pixel
                            is assumed. Pass e.g. ``res=[8, 8, 40]`` as keyword
                            argument to override this.
    min_nodes :             int, optional
                            Minimum node count for a neuron within given
                            bounding box.
    min_cable :             int, optional
                            Minimum cable length [nm] for a neuron within
                            given bounding box.
    remote_instance :       CatmaidInstance
                            If not passed directly, will try using global.

    Returns
    -------
    list
                            ``[skeleton_id, skeleton_id, ...]``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(bbox, ns.Volume):
        bbox = bbox.bbox

    if isinstance(bbox, dict):
        bbox = np.array([[bbox['left'], bbox['right']],
                         [bbox['top'], bbox['bottom']],
                         [bbox['z1'], bbox['z2']]
                         ])

    bbox = np.asarray(bbox)

    if bbox.shape == (2, 3):
        bbox = bbox.T

    if unit == 'PIXEL':
        res = np.asarray(kwargs.get('res', [4, 4, 40])).reshape(3, 1)
        bbox *= res

    url = remote_instance._get_skeletons_in_bbox(minx=min(bbox[0]),
                                                 maxx=max(bbox[0]),
                                                 miny=min(bbox[1]),
                                                 maxy=max(bbox[1]),
                                                 minz=min(bbox[2]),
                                                 maxz=max(bbox[2]),
                                                 min_nodes=min_nodes,
                                                 min_cable=min_cable)

    return remote_instance.fetch(url)


@cache.undo_on_error
def get_user_list(remote_instance=None):
    """Get list of users.

    Parameters
    ----------
    remote_instance :   CatmaidInstance
                        If not passed directly, will try using global.

    Returns
    ------
    pandas.DataFrame
        DataFrame in which each row represents a user::

            id  login  full_name  first_name  last_name  color
         0
         1
         ...

    Examples
    --------
    >>> user_list = pymaid.get_user_list()
    >>> # To search for e.g. user ID 22
    >>> user_list.set_index('id', inplace=True)
    >>> user_list.loc[22]
    id                                  22
    login                      mustermannm
    full_name          Michaela Mustermann
    first_name                     Michael
    last_name                   Mustermann
    color         [0.91389, 0.877853, 1.0]
    >>> user_list.reset_index(inplace=True)
    >>> # To convert into a classic dict
    >>> d = user_list.set_index('id').T.to_dict()
    >>> d[22]['first_name']
    ... Michaela

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    user_list = remote_instance.fetch(remote_instance._get_user_list_url())

    # It appears that for public CATMAID instances (like VFB) where the users
    # are masked, the user-list endpoint can return just a single dictionary
    # instead of a list of dicts.
    if isinstance(user_list, dict):
        user_list = [user_list]

    # The user list can contain different entries
    # Here we define alternative field names
    columns = [('id', 'userid'),
               ('login', 'username'),
               ('full_name', 'long_name'),
               ('first_name', ),
               ('last_name', ),
               ('color', )]

    data = []
    for user in user_list:
        row = []
        for col in columns:
            value = None
            for key in col:
                if key in user:
                    value = user[key]
                    break
            row.append(value)
        data.append(row)

    df = pd.DataFrame(data, columns=[c[0] for c in columns])

    df.sort_values(['login', 'id'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


@cache.undo_on_error
def get_paths(sources, targets, n_hops=2, min_synapses=1, return_graph=False,
              remove_isolated=False, remote_instance=None):
    """Fetch paths between two sets of neurons.

    Parameters
    ----------
    sources
                        Source neurons.
    targets
                        Target neurons. ``sources`` and ``targets`` can be:

                        1. list of skeleton ID(s) (int or str)
                        2. list of neuron name(s) (str, exact match)
                        3. an annotation as e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object

    n_hops :            int | list | range, optional
                        Number of hops allowed between sources and
                        targets. Direct connection would be 1 hop.

                        1. int, e.g. ``n_hops=3`` will return paths with
                        EXACTLY 3 hops
                        2. list, e.g. ``n_hops=[2,4]`` will return all
                        paths with 2 and 4 hops
                        3. range, e.g. ``n_hops=range(2,4)`` will be converted
                        to a list and return paths with 2 and 3 hops.
    min_synapses :      int, optional
                        Minimum number of synpases between source and target.
    return_graph :      bool, optional
                        If True, will return NetworkX Graph (see below).
    remove_isolated :   bool, optional
                        Remove isolated nodes from NetworkX Graph. Only
                        relevant if ``return_graph=True``.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    paths :     list
                List of skeleton IDs that constitute paths from
                sources to targets::

                    [[source1, ..., target1], [source2, ..., target2], ...]

    networkx.DiGraph
                Only if ``return_graph=True``. Graph contains all neurons that
                connect sources and targets. **Important**: Does only contain
                edges that connect sources and targets via max ``n_hops``!
                Other edges have been removed.

    Examples
    --------
    >>> # This assumes that you have already set up a CatmaidInstance
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> g, paths = pymaid.get_paths(['annotation:glomerulus DA1'],
    ...                             ['2333007'])
    >>> g
    <networkx.classes.digraph.DiGraph at 0x127d12390>
    >>> paths
    [['57381', '4376732', '2333007'], ['57323', '630823', '2333007'], ...
    >>> nx.draw(g)
    >>> plt.show()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    sources = utils.eval_skids(sources, remote_instance=remote_instance)
    targets = utils.eval_skids(targets, remote_instance=remote_instance)

    targets = utils._make_iterable(targets).astype(int)
    sources = utils._make_iterable(sources).astype(int)

    if isinstance(n_hops, (int, np.int)):
        n_hops = [n_hops]

    if not utils._is_iterable(n_hops):
        raise TypeError('Expected `n_hops` to be iterable or integer, got '
                        f'"{type(n_hops)}"')

    if min(n_hops) <= 0:
        raise ValueError('n_hops must not be <= 0')

    # We need to query to endpoints:
    # First get the neurons involved
    response = []
    url = remote_instance._get_graph_dps_url()
    for h in range(1, max(n_hops) + 1):
        if h == 1:
            response += list(sources) + list(targets)
            continue

        post_data = {
            'n_hops': h,
            'min_synapses': min_synapses
        }

        for i, s in enumerate(sources):
            post_data['sources[%i]' % i] = s

        for i, t in enumerate(targets):
            post_data['targets[%i]' % i] = t

        # Response is just a set of skeleton IDs
        response += remote_instance.fetch(url, post=post_data)

    # Get unique edges
    skids = np.unique(np.asarray(response).astype(int))

    # Now get edges between those neurons
    edges = get_edges(skids, remote_instance=remote_instance)

    # Turn neurons into an NetworkX graph
    g = ns.network2nx(edges, threshold=min_synapses)

    # Get all paths between sources and targets
    all_paths = [p for s in sources for t in targets for p in
                 nx.all_simple_paths(g, s, t,
                                     cutoff=max(n_hops)) if len(p) - 1 in n_hops]

    if not return_graph:
        return all_paths

    # Turn into edges
    edges_to_keep = set([e for l in all_paths for e in nx.utils.pairwise(l)])

    # Remove edges
    g.remove_edges_from([e for e in g.edges if e not in edges_to_keep])

    if remove_isolated:
        # Remove isolated nodes
        g.remove_nodes_from(list(nx.isolates(g)))

    return all_paths, g


@cache.undo_on_error
def get_volume(volume_name=None, color=(120, 120, 120, .6), combine_vols=False,
               remote_instance=None):
    """Retrieves volume (mesh).

    Parameters
    ----------
    volume_name :       int | str | list of str or int
                        Name(s) (as ``str``) or ID (as ``int``) of the volume
                        to import. Names must be EXACT!
                        If ``volume_name=None``, will return list of all
                        available CATMAID volumes. If list of volume names,
                        will return a dictionary ``{name: Volume, ... }``
    color :             tuple, optional
                        (R, G, B, alpha) values used by :func:`~pymaid.plot3d`.
    combine_vols :      bool, optional
                        If True and multiple volumes are requested, the will
                        be combined into a single volume.
    remote_instance :   CATMAIDInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    navis.Volume
            If ``volume_name`` is list of volumes, returns a dictionary of
            Volumes: ``{name1: Volume1, name2: Volume2, ...}``

    Examples
    --------
    >>> import pymaid
    >>> rm = CatmaidInstance('server_url', 'api_token', 'http_user', 'http_pw')
    >>> # Retrieve volume
    >>> vol = pymaid.get_volume('LH_R')
    >>> # Plot volume
    >>> vol.plot3d()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(volume_name, type(None)):
        logger.info('Retrieving list of available volumes.')
    elif not isinstance(volume_name, (int, str, list, np.ndarray)):
        raise TypeError('Volume name must be id (int), str or list of either, not {}.'.format(type(volume_name)))

    volume_names = utils._make_iterable(volume_name)

    # First, get volume IDs
    get_volumes_url = remote_instance._get_volumes()
    response = remote_instance.fetch(get_volumes_url)

    all_vols = pd.DataFrame(response['data'], columns=response['columns'])

    if isinstance(volume_name, type(None)):
        return all_vols

    req_vols = all_vols[(all_vols.name.isin(volume_names)) |
                        (all_vols.id.isin(volume_names))]
    volume_ids = req_vols.id.values

    if len(volume_ids) < len(volume_names):
        not_found = set(volume_names).difference(set(all_vols.name) |
                                                 set(all_vols.id))
        raise Exception(
            'No volume(s) found for: {}'.format(','.join(not_found)))

    url_list = [remote_instance._get_volume_details(v) for v in volume_ids]

    # Get data
    responses = remote_instance.fetch(url_list, desc='Volumes')

    # Generate volume(s) from responses
    volumes = {}
    for r in responses:
        mesh_str = r['mesh']
        mesh_name = r['name']
        mesh_id = r['id']

        mesh_type = re.search('<(.*?) ', mesh_str).group(1)

        # Now reverse engineer the mesh
        if mesh_type == 'IndexedTriangleSet':
            t = re.search("index='(.*?)'", mesh_str).group(1).split(' ')
            faces = [(int(t[i]), int(t[i + 1]), int(t[i + 2]))
                     for i in range(0, len(t) - 2, 3)]

            v = re.search("point='(.*?)'", mesh_str).group(1).split(' ')
            vertices = [(float(v[i]), float(v[i + 1]), float(v[i + 2]))
                        for i in range(0, len(v) - 2, 3)]

        elif mesh_type == 'IndexedFaceSet':
            # For this type, each face is indexed and an index of -1 indicates
            # the end of this face set
            t = re.search("coordIndex='(.*?)'", mesh_str).group(1).split(' ')
            faces = []
            this_face = []
            for f in t:
                if int(f) != -1:
                    this_face.append(int(f))
                else:
                    faces.append(this_face)
                    this_face = []

            # Make sure the last face is also appended
            faces.append(this_face)

            v = re.search("point='(.*?)'", mesh_str).group(1).split(' ')
            vertices = [(float(v[i]), float(v[i + 1]), float(v[i + 2]))
                        for i in range(0, len(v) - 2, 3)]

        else:
            logger.error("Unknown volume type: %s" % mesh_type)
            raise Exception("Unknown volume type: %s" % mesh_type)

        # In this format vertices are not unique - i.e. a given vertex defined
        # by its x/y/z position shows up as many times as it participates in
        # a face.
        # Fortunately, navis.Volume being a subclass of trimesh.Trimesh takes
        # care of the deduplication
        v = ns.Volume(name=mesh_name,
                      volume_id=mesh_id,
                      vertices=vertices,
                      faces=faces,
                      color=color)

        volumes[mesh_name] = v

    # Return just the volume if a single one was requested
    if len(volumes) == 1:
        return list(volumes.values())[0]

    return volumes


@cache.undo_on_error
def get_annotation_list(remote_instance=None):
    """Get a list of all annotations in the project.

    Parameters
    ----------
    remote_instance : CatmaidInstance, optional
                      If not passed directly, will try using global.

    Returns
    -------
    pandas DataFrame
            DataFrame in which each row represents an annotation::

                name   id   users
             0
             1
             ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    an = remote_instance.fetch(remote_instance._get_annotation_list())[
        'annotations']

    df = pd.DataFrame.from_dict(an)

    return df


def url_to_coordinates(coords, stack_id, active_skeleton_id=None,
                       active_node_id=None, zoom=0, tool='tracingtool',
                       open_browser=False, remote_instance=None):
    """Generate URL to a location.

    Parameters
    ----------
    coords :                list | np.ndarray | pandas.DataFrame
                            ``x``, ``y``, ``z`` coordinates.
    stack_id :              int | list/array of ints
                            ID of the image stack you want to link to.
                            Depending on your setup this parameter might be
                            overriden by local user settings.
    active_skeleton_id :    int | list/array of ints, optional
                            Skeleton ID of the neuron that should be selected.
    active_node_id :        int | list/array of ints, optional
                            Node/Connector ID of the node that should be
                            active.
    zoom :                  int, optional
    tool :                  str, optional
    open_browser :          bool, optional
                            If True will open *all* generated URLs as new
                            tabs in the standard webbrowser.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    str | list of str
                URL(s) to the coordinates provided.

    Examples
    --------
    >>> # Get URL for a single coordinate
    >>> url = pymaid.url_to_coordinates([1000, 1000, 1000], stack_id=5)
    >>> # Get URLs for all low-confidence nodes of a neuron
    >>> n = pymaid.get_neuron(27295)
    >>> low_c = n.nodes.loc[n.nodes.confidence < 5]
    >>> urls = pymaid.url_to_coordinates(low_c[['x', 'y', 'z']].values,
    ...                                  stack_id=5,
    ...                                  active_node_id=low_c.node_id.values)

    """
    def gen_url(c, stid, nid, sid):
        """ This function generates the actual urls
        """
        GET_data = {'pid': remote_instance.project_id,
                    'xp': int(c[0]),
                    'yp': int(c[1]),
                    'zp': int(c[2]),
                    'tool': tool,
                    'sid0': stid,
                    's0': zoom
                    }

        if sid:
            GET_data['active_skeleton_id'] = sid
        if nid:
            GET_data['active_node_id'] = nid

        return(remote_instance.make_url('?%s' % urllib.parse.urlencode(GET_data)))

    def list_helper(x):
        """ Helper function to turn variables into lists matching length of coordinates
        """
        if not isinstance(x, (list, np.ndarray)):
            return [x] * len(coords)
        elif len(x) != len(coords):
            raise ValueError('Parameters must be the same shape as coords.')
        else:
            return x

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(coords, (pd.DataFrame, pd.Series)):
        try:
            coords = coords[['x', 'y', 'z']].values
        except BaseException:
            raise ValueError(
                'Pandas DataFrames must have "x","y" and "z" columns.')
    elif isinstance(coords, list):
        coords = np.array(coords)

    if isinstance(coords, np.ndarray) and coords.ndim > 1:
        stack_id = list_helper(stack_id)
        active_skeleton_id = list_helper(active_skeleton_id)
        active_node_id = list_helper(active_node_id)

        urls = [gen_url(c, stid, nid, sid) for c, stid, nid, sid in zip(coords, stack_id, active_node_id, active_skeleton_id)]

        if open_browser:
            for u in urls:
                webbrowser.open_new_tab(u)

        return urls
    else:
        url = gen_url(coords, stack_id, active_node_id, active_skeleton_id)

        if open_browser:
            webbrowser.open_new_tab(url)

        return url


@cache.undo_on_error
def get_node_location(x, sort=True, remote_instance=None):
    """Retrieves location for a set of nodes or connectors.

    Parameters
    ----------
    x :                 int | list of int
                        Node ID(s).
    sort :              bool, optional
                        If True, will sort returned DataFrame to be in the same
                        order as input data.
    remote_instance :   CatmaidInstance, optional
                        If not provided, will search for globally defined
                        remote instance.

    Returns
    -------
    pandas.DataFrame
            DataFrame in which each row represents a node::

                node_id  x  y  z
             0
             1
             ...

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    x = utils.eval_node_ids(x, connectors=True, nodes=True)

    url = remote_instance._get_node_location_url()
    post = {'node_ids[{}]'.format(i): n for i, n in enumerate(x)}

    data = remote_instance.fetch(url, post=post)
    df = pd.DataFrame(data, columns=['node_id', 'x', 'y', 'z'])

    if sort:
        df = df.set_index('node_id').loc[x].reset_index(drop=False)

    return df


@cache.undo_on_error
def get_label_list(remote_instance=None):
    """Retrieves all labels (node tags only) in a project.

    Parameters
    ----------
    remote_instance :   CatmaidInstance, optional
                        If not provided, will search for globally defined
                        remote instance.

    Returns
    -------
    pandas.DataFrame
            DataFrame in which each row represents a label::

                label_id  tag  skeleton_id  node_id
             0
             1
             ...

    Examples
    --------
    >>> # Get all labels
    >>> labels = pymaid.get_label_list()
    >>> # Get all nodes with a given tag
    >>> nodes = labels[labels.tag == 'my_label'].node_id
    >>> # Get neuron that have at least a single node with a given tag
    >>> neurons = labels[ labels.tag == 'my_label' ].skeleton_id.unique()

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    labels = remote_instance.fetch(remote_instance._get_label_list_url())

    return pd.DataFrame(labels, columns=['label_id', 'tag', 'skeleton_id',
                                         'node_id'])


@cache.undo_on_error
def get_transactions(range_start=None, range_length=25, remote_instance=None):
    """Retrieve individual transactions with server.

    **This API endpoint is extremely slow!**

    Parameters
    ----------
    range_start :       int, optional
                        Start of table. Transactions are returned in
                        chronological order (most recent transactions first)
    range_length :      int, optional
                        End of table. If None, will return all.
    remote_instance :   CatmaidInstance, optional
                        If not provided, will search for globally defined
                        CatmaidInstance.

    Returns
    -------
    pandas.DataFrame
            DataFrame listing individual transactions::

               change_type      execution_time          label        ...
             0  Backend       2017-12-26 03:37:00     labels.update  ...
             1  Backend       2017-12-26 03:37:00  treenodes.create  ...
             2  Backend       2017-12-26 03:37:00  treenodes.create  ...
             3  Backend       2017-12-26 03:37:00  treenodes.create  ...
             4  Backend       2017-12-26 03:32:00  treenodes.create  ...
               project_id  transaction_id  user_id    user
             0  1            404899166        151     dacksa
             1  1            404899165        151     dacksa
             2  1            404899164        151     dacksa
             3  1            404899163        151     dacksa
             4  1            404899162        151     dacksa

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    remote_transactions_url = remote_instance._get_transactions_url()

    desc = {'range_start': range_start, 'range_length': range_length}
    desc = {k: v for k, v in desc.items() if v is not None}

    remote_transactions_url += '?%s' % urllib.parse.urlencode(desc)

    data = remote_instance.fetch(remote_transactions_url)

    df = pd.DataFrame.from_dict(data['transactions'])

    user_list = get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    df['user'] = df.user_id.map(user_dict)

    df['execution_time'] = [datetime.datetime.strptime(
        d[:16], '%Y-%m-%dT%H:%M') for d in df['execution_time'].values]

    return df


@cache.undo_on_error
def get_neuron_id(x, remote_instance=None):
    """Get neuron ID(s) for given skeleton(s).

    Parameters
    ----------
    x :                 list-like | CatmaidNeuron/List
                        Skeleton IDs for which to get neuron IDs.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        ``{skeleton_id (str): neuron_id (int), ... }``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_neuron_ids_url()
    post = {'model_ids[{}]'.format(i): s for i, s in enumerate(skids)}

    resp = remote_instance.fetch(url, post=post)

    return resp


@cache.undo_on_error
def get_cable_lengths(x, chunk_size=500, remote_instance=None):
    """Get cable lengths directly from Catmaid Server.

    Parameters
    ----------
    x :                 list-like | CatmaidNeuron/List
                        Skeleton IDs for which to get cable lengths.
    chunk_size :        int, optional
                        Retrieves cable in chunks of given size.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
                        ``{skeleton_id (str): cable [nm] (int), ... }``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_neuron_cable_url()

    cable = {}
    for i in config.trange(0, len(skids), int(chunk_size),
                           desc='Fetching chunks'):
        chunk = skids[i: i + chunk_size]
        post = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(chunk)}

        resp = remote_instance.fetch(url, post=post)
        cable.update(resp)

    return cable


@cache.undo_on_error
def get_connectors_in_bbox(bbox, unit='NM', limit=None, restrict_to=False,
                           ret='COORDS', remote_instance=None, **kwargs):
    """Retrieves connectors within given bounding box.

    Parameters
    ----------
    bbox :                  list-like | dict | pymaid.Volume
                            Coordinates of the bounding box. Can be either:

                              1. List/np.array: ``[[left, right], [top, bottom], [z1, z2]]``
                              2. Dictionary ``{'left': int|float, 'right': ..., ...}``
    unit :                  'NM' | 'PIXEL'
                            Unit of your coordinates. Attention:
                            'PIXEL' will also assume that Z1/Z2 is in slices.
                            By default, a X/Y resolution of 3.8nm and a Z
                            resolution of 35nm is assumed. Pass 'xy_res' and
                            'z_res' as ``**kwargs`` to override this.
    limit :                 int, optional
                            Limit the number of connectors returned.
    restrict_to :           list, optional
                            List of skeleton IDs to return connectors for.
    ret :                   'IDS' |'COORDS' | 'LINKS'
                            Connector data to be returned. See below for
                            explanation.
    remote_instance :       CatmaidInstance
                            If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
            If ``ret="COORDS"`` (default): DataFrame in which each row
            represents a connector:

                connector_id  x  y  z
             0
             1
             ..

    list
            If ``ret="IDS"``: list of connector IDs.

    pandas.DataFrame
            If ``ret="LINKS"``: DataFrame in which each row represents a
            connector. Please note that connectors can show up multiple times
            - once for each link.

                connector_id  x  y  z skeleton confidence creator_id ..
             0
             1
             ..

               .. connected_node creation_time edition_time relation_id
             0
             1
             ..

    """
    if ret.upper() not in ['IDS', 'COORDS', 'LINKS']:
        raise ValueError('"ret" must be "IDS", "COORDS" or "LINKS"')

    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(bbox, ns.Volume):
        bbox = bbox.bbox

    if isinstance(bbox, dict):
        bbox = np.array([[bbox['left'], bbox['right']],
                         [bbox['top'], bbox['bottom']],
                         [bbox['z1'], bbox['z2']]
                         ])

    if not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)

    if unit == 'PIXEL':
        bbox[[0, 1], :] = bbox[[0, 1], :] * kwargs.get('xy_res', 4)
        bbox[[2], :] = bbox[[2], :] * kwargs.get('z_res', 40)

    url = remote_instance._get_connector_in_bbox_url()

    post = dict(minx=min(bbox[0]),
                maxx=max(bbox[0]),
                miny=min(bbox[1]),
                maxy=max(bbox[1]),
                minz=min(bbox[2]),
                maxz=max(bbox[2]),
                limit=limit if limit else 0
                )

    if ret.upper() in ['COORDS', 'LINKS']:
        # post['with_links'] = True
        post['with_locations'] = True

    if ret.upper() == 'LINKS':
        post['with_links'] = True

    if restrict_to:
        restrict_to = utils._make_iterable(restrict_to)
        post.update({'skeleton_ids[{}]'.format(i): s for i, s in enumerate(restrict_to)})

    data = remote_instance.fetch(url, post=post)

    if ret.upper() == 'IDS':
        return data

    data = pd.DataFrame(data)

    if ret.upper() == 'COORDS':
        data.columns = ['connector_id', 'x', 'y', 'z']
    else:
        data.columns = ['connector_id', 'x', 'y', 'z', 'skeleton',
                        'confidence', 'creator_id', 'connected_node',
                        'creation_time', 'edition_time', 'relation_id']

    return data


@cache.undo_on_error
def get_connectivity_counts(x, source_relations=['presynaptic_to'],
                            target_relations=['postsynaptic_to'],
                            count_partner_links=True, remote_instance=None):
    """Fetch number of connections of a given type for a set of neurons.

    Parameters
    ----------
    x :                     list-like | CatmaidNeuron/List
                            Skeleton IDs for which to get cable lengths.
    source_relations :      str | list of str, optional
                            A list of pre-connector relations.
    target_relations :      str | list of str, optional
                            A list of post-connector relations. Default
                            settings count the number of outgoing connections
                            for the input neurons.
    count_partner_links :   bool, optional
                            Whether to count partner links or links
                            to a connector.
    remote_instance :       CatmaidInstance
                            If not passed directly, will try using global.

    Examples
    --------
    # Get the count of all outgoing connections (default):

    >>> counts = pymaid.get_connectivity_counts('annotation:glomerulus DA1')

    # Get both incoming and outgoing connections:

    >>> counts = pymaid.get_connectivity_counts('annotation:glomerulus DA1',
    ...                                         source_relations=['presynaptic_to',
    ...                                                           'postsynaptic_to'],
    ...                                         target_relations=['postsynaptic_to',
    ...                                                           'presynaptic_to'])


    Returns
    -------
    dict
                Dictionary with server response.

                    {'connectivity': {skid1: {relation_ID: count},
                                      skid2: {relation_ID: count}},
                     'relations': {relation_ID: relation_name}}

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_connectivity_counts_url()

    source_relations = utils._make_iterable(source_relations)
    target_relations = utils._make_iterable(target_relations)

    post = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(skids)}
    post.update({'source_relations[{}]'.format(i): s for i, s in enumerate(source_relations)})
    post.update({'target_relations[{}]'.format(i): t for i, t in enumerate(target_relations)})

    return remote_instance.fetch(url, post=post)


@cache.undo_on_error
def get_import_info(x, with_nodes=False, chunk_size=500, remote_instance=None):
    """Get count of imported nodes for given neuron(s).

    Parameters
    ----------
    x :                 list-like | CatmaidNeuron/List
                        Skeleton IDs for which to get import info.
    with_nodes :        bool, optional
                        Whether to include IDs of all imported nodes.
    chunk_size :        int, optional
                        Retrieves data in chunks of this size.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_import_info_url()

    info = {}
    for i in config.trange(0, len(skids), int(chunk_size),
                           desc='Fetching info'):
        chunk = skids[i: i + chunk_size]
        post = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(chunk)}
        post['with_treenodes'] = with_nodes

        resp = remote_instance.fetch(url, post=post)
        info.update(resp)

    return info


@cache.undo_on_error
def get_origin(x, chunk_size=500, remote_instance=None):
    """Get origin of given neuron(s).

    Parameters
    ----------
    x :                 list-like | CatmaidNeuron/List
                        Skeleton IDs for which to get their origin.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    dict
            {'data_sources': {'1': {'name': None,
                                    'source_project_id': 1,
                                    'url': 'https://.../tracing/fafb/v14-seg-li-190805.0'}},
             'origins': {'13348203': {'data_source_id': 1, 'source_id': 13348108}}}

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_skeleton_origin_url()

    post = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(skids)}

    resp = remote_instance.fetch(url, post=post)

    return resp


@cache.undo_on_error
def get_skids_by_origin(source_ids, source_url, source_project_id,
                        remote_instance=None):
    """Get skeleton IDs by origin.

    Parameters
    ----------
    source_ids :            list of int
                            Source IDs to search for.
    source_url :            str
                            Source url to search for.
    source_project_id :     int
                            Source project ID to search for.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    dict
                    {'source_id': skeleton_id}

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    source_ids = utils._make_iterable(source_ids)

    url = remote_instance._get_skeleton_by_origin_url()

    post = {'source_ids[{}]'.format(i): s for i, s in enumerate(source_ids)}
    post['source_url'] = source_url
    post['source_project_id'] = source_project_id

    resp = remote_instance.fetch(url, post=post)

    return resp


@cache.undo_on_error
def get_sampler(x=None, remote_instance=None):
    """Get list of reconstruction samplers.

    Parameters
    ----------
    x :                     list-like | CatmaidNeuron/List | None, optional
                            Skeleton IDs for which to get samplers. If ``None``
                            will return all samplers.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
                            DataFrame containing all samplers. Returns empty
                            DataFrame if no samplers.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(x, type(None)):
        url = remote_instance._get_sampler_list_url()
    else:
        skids = utils.eval_skids(x, remote_instance=remote_instance)
        GET = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(skids)}
        url = remote_instance._get_sampler_list_url(**GET)

    resp = remote_instance.fetch(url)

    if not resp:
        return pd.DataFrame([])

    # Turn into DataFrame
    df = pd.DataFrame.from_records(resp)

    # Convert timestamps
    df['creation_time'] = pd.to_datetime(df.creation_time, unit='s', utc=True)
    df['edition_time'] = pd.to_datetime(df.creaedition_timetion_time, unit='s', utc=True)

    return df


@cache.undo_on_error
def get_sampler_domains(sampler, remote_instance=None):
    """Get list of domains for given sampler.

    Parameters
    ----------
    sampler :               int
                            ID of sampler to fetch domains for.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    pandas.DataFrame
                            DataFrame containing domains for given sampler.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    url = remote_instance._get_sampler_domains_url(sampler)

    resp = remote_instance.fetch(url)

    # Turn into DataFrame
    df = pd.DataFrame.from_records(resp)

    # Convert timestamps
    df['creation_time'] = pd.to_datetime(df.creation_time, unit='s', utc=True)
    df['edition_time'] = pd.to_datetime(df.creaedition_timetion_time, unit='s', utc=True)

    return df


@cache.undo_on_error
def get_sampler_counts(x, remote_instance=None):
    """Get number of reconstruction samplers for a set of neurons.

    Parameters
    ----------
    x :                     list-like | CatmaidNeuron/List | None, optional
                            Skeleton IDs for which to get sampler counts.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    dict
                            ``{skeleton_id: count, ...}``

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    url = remote_instance._get_sampler_counts_url()

    post = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(skids)}

    resp = remote_instance.fetch(url, post=post)

    return resp


@cache.undo_on_error
def get_skeleton_change(x, chunk_size=50, remote_instance=None):
    """Get split and merge history of skeletons.

    Parameters
    ----------
    x :                     list-like | CatmaidNeuron/List | None, optional
                            Skeleton IDs for which to get split/merge history.
    chunk_size :            int, optional
                            Change history will be queried in chunks. Reduce
                            the number if you experience problems.
    remote_instance :       CatmaidInstance, optional
                            If not passed directly, will try using global.

    Returns
    -------
    list
                            List of unique skeleton paths in historic order,
                            newest last::

                              [[skid1, skid2, ..., skidN],  ...]

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    change = []
    with config.tqdm(desc='Fetching change', total=len(skids),
                     disable=config.pbar_hide,
                     leave=config.pbar_leave) as pbar:
        chunks = [skids[i:i + chunk_size] for i in range(0, len(skids), chunk_size)]
        for ch in chunks:
            # Generate URLs
            GET = {'skeleton_ids[{}]'.format(i): s for i, s in enumerate(ch)}
            url = remote_instance._get_skeleton_change_url(**GET)

            # Fetch data
            resp = remote_instance.fetch(url, disable_pbar=True)
            change += resp

            # Update progress bar
            pbar.update(len(ch))

    return change
