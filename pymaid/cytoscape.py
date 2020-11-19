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

""" Set of functions to interface with Cytoscape using its CyREST API. This
module requires py2cytoscape (https://github.com/cytoscape/py2cytoscape)
"""

import logging
import time
import datetime

import networkx as nx
import numpy as np
import pandas as pd
from py2cytoscape.data.cyrest_client import CyRestClient

from . import utils, fetch, graph

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def get_client():
    """Initialises connection with Cytoscape and returns client."""
    return CyRestClient()


def get_pymaid_style():
    """Returns our default network style."""

    cy = get_client()

    all_styles = cy.style.get_all()
    s = cy.style.create('pymaid')

    # If the style already existed, return unchanged
    if 'pymaid' in all_styles:
        return s

    # If we created the style from scratch, apply some basic settings
    basic_settings = dict(
        # You can set default values as key-value pairs.

        NODE_FILL_COLOR='#FEC44F',
        NODE_SIZE=18,
        NODE_BORDER_WIDTH=7,
        NODE_BORDER_COLOR='#999999',
        NODE_LABEL_COLOR='#555555',
        NODE_LABEL_FONT_SIZE=14,
        NODE_LABEL_POSITION='S,NW,c,0.00,3.00',

        EDGE_WIDTH=2,
        EDGE_TRANSPARENCY=100,
        EDGE_CURVED=True,
        EDGE_BEND='0.728545744495502,-0.684997151948455,0.6456513365424503',
        EDGE_UNSELECTED_PAINT='#CCCCCC',
        EDGE_STROKE_UNSELECTED_PAINT='#333333',
        EDGE_TARGET_ARROW_SHAPE='DELTA',

        NETWORK_BACKGROUND_PAINT='#FFFFFF',
    )

    s.update_defaults(basic_settings)

    return s


def generate_network(x, layout='fruchterman-rheingold', apply_style=True,
                     clear_session=True):
    """ Loads a CATMAID network into Cytoscape.

    Parameters
    ----------
    x :             networkX Graph | pandas.DataFrame
                    Network to export to Cytoscape. Can be:
                      1. NetworkX Graph e.g. from pymaid.networkx (preferred!)
                      2. Pandas DataFrame. Mandatory columns:
                         'source','target','interaction'
    layout :        str | None, optional
                    Layout to apply. Set to ``None`` to not apply any.
    apply_style :   bool, optional
                    If True will apply a "pymaid" style to the network.
    clear_session : bool, optional
                    If True, will clear session before adding network.

    Returns
    -------
    cytoscape Network
    """

    # Initialise connection with Cytoscape
    cy = get_client()

    if layout not in cy.layout.get_all() + [None]:
        raise ValueError('Unknown layout. Available options: '
                         ', '.join(cy.layout.get_all()))

    # Clear session
    if clear_session:
        cy.session.delete()

    if isinstance(x, nx.Graph):
        n = cy.network.create_from_networkx(x)
    elif isinstance(x, np.ndarray):
        n = cy.network.create_from_ndarray(x)
    elif isinstance(x, pd.DataFrame):
        n = cy.network.create_from_dataframe(x)
    else:
        raise TypeError('Unable to generate network from data of '
                        'type "{0}"'.format(type(x)))

    if layout:
        # Apply basic layout
        cy.layout.apply(name=layout, network=n)

    if apply_style:
        # Get our default style
        s = get_pymaid_style()

        # Add some passthough mappings to the style
        s.create_passthrough_mapping(
            column='neuron_name', vp='NODE_LABEL', col_type='String')
        max_edge_weight = n.get_edge_column('weight').max()
        s.create_continuous_mapping(column='weight', vp='EDGE_WIDTH', col_type='Double',
                                    points=[{'equal': '1.0', 'greater': '1.0', 'lesser': '1.0', 'value': 1.0},
                                            {'equal': max_edge_weight / 3, 'greater': 1.0, 'lesser': max_edge_weight / 3, 'value': max_edge_weight}]
                                    )

        # Apply style
        cy.style.apply(s, n)

    return n


def watch_network(x, sleep=3, n_circles=1, min_pre=2, min_post=2, layout=None,
                  remote_instance=None, verbose=True, group_by=None):
    """ Loads and **continuously updates** a network into Cytoscape.

    Use CTRL-C to stop.

    Parameters
    ----------
    x :                 skeleton IDs | CatmaidNeuron/List
                        Seed neurons to keep track of.
    sleep :             int | None, optional
                        Time in seconds to sleep after each update.
    n_circles :         int, optional
                        Number of circles around seed neurons to include in
                        the network. See also :func:`pymaid.get_nth_partners`.
                        Set to ``None | 0 | False`` to only update
                        seed nodes.
    min_pre/min_post :  int | dict, optional
                        Synapse threshold to apply to ``n_circles``.
                        Set to -1 to not get any pre-/post synaptic partners.
                        Please note: as long as there is a single
                        above-threshold connection, a neuron will be included.
                        This does not remove other, sub-threshold connections.
                        Use dictionary to assign individual thresholds to
                        neurons: e.g. ``min_pre={16: 5, 2333007: 10}``. Neurons
                        in ``x`` that are not in the dictionary will be given
                        a threshold of ``-1`` (no partners).
    layout :            str | None, optional
                        Name of a Cytoscape layout. If provided, will update
                        the network's layout on every change.
    remote_instance :   CatmaidInstance, optional
    verbose :           bool, optional
                        If True, will log changes made to the network.
    group_by :          None | dict, optional
                        Provide a dictionary ``{group_name: [skid1, skid2, ...]}``
                        to collapse sets of nodes into groups.

    Returns
    -------
    Nothing

    Examples
    --------
    A basic example:

    >>> import pymaid
    >>> import pymaid.cytoscape as cytomaid
    >>> rm = pymaid.CatmaidInstance('server_url', 'api_token', 'http_user',
    ...                             'http_password')
    >>> # Don't forget to start Cytoscape!
    >>> cytomaid.watch_network('annotation:glomerulus DA1', min_pre=5,
    ...                         min_post=-1, sleep=5)
    >>> # Use CTRL-C to stop the loop

    More advanced: individual thresholds

    >>> # Get seed neurons
    >>> seeds = pymaid.get_skids_by_annotation('glomerulus DA1')
    >>> # Set individual downstream thresholds
    >>> min_pre = {2863104: 5, 2319457: 10, 57323: 10, 57311: 20}
    >>> # Neurons not mapped in min_pre will default to -1 (no partners)
    >>> cytomaid.watch_network(seeds, min_pre=min_pre,
    ...                        min_post=-1, sleep=5)
    """

    # TODO:
    # - smart parsing of group_by: use eval_skids()

    cy = get_client()

    remote_instance = utils._eval_remote_instance(remote_instance)

    sleep = 0 if not sleep else sleep

    x = utils.eval_skids(x, remote_instance=remote_instance)

    # Prepare individual thresholds:
    if not isinstance(min_pre, dict):
        min_pre = {s: min_pre for s in x}
    if not isinstance(min_post, dict):
        min_post = {s: min_post for s in x}

    # Make sure everything is integer
    min_pre = {int(s): int(v) for s, v in min_pre.items()}
    min_post = {int(s): int(v) for s, v in min_post.items()}

    # Add missing neurons with threshold -1
    min_pre.update({int(s): min_pre.get(int(s), -1) for s in x})
    min_post.update({int(s): min_post.get(int(s), -1) for s in x})

    # Group thresholds to minimize number of queries
    by_neuron = {int(s) : (min_pre[int(s)], min_post[int(s)]) for s in x}
    by_threshold = {v : [s for s, t in by_neuron.items() if v == t] for v in by_neuron.values()}

    # Generate the initial network
    to_add = np.array(x)
    if n_circles:
        for (t_pre, t_post), skids in by_threshold.items():
            # Don't attempt to fetch neither pre nor post
            if t_pre == -1 and t_post == -1:
                continue
            temp = fetch.get_nth_partners(skids, n_circles=n_circles,
                                          min_pre=t_pre, min_post=t_post,
                                          remote_instance=remote_instance).skeleton_id
            to_add = np.concatenate([to_add, temp])

    g = graph.network2nx(to_add.astype(int),
                         group_by=group_by,
                         remote_instance=remote_instance)
    network = generate_network(g, clear_session=True, apply_style=False,
                               layout=layout)

    if layout:
        cy.layout.apply(name=layout, network=network)

    logger.info('Watching network. Use CTRL-C to stop.')
    if remote_instance.caching:
        logger.warning('Caching disabled.')
        remote_instance.caching = False
    utils.set_loggers('WARNING')
    while True:
        # Pull new set of partners
        to_add = x
        if n_circles:
            for (t_pre, t_post), skids in by_threshold.items():
                # Don't attempt to fetch neither pre nor post
                if t_pre == -1 and t_post == -1:
                    continue
                temp = fetch.get_nth_partners(skids, n_circles=n_circles,
                                              min_pre=t_pre, min_post=t_post,
                                              remote_instance=remote_instance).skeleton_id
                to_add = np.concatenate([to_add, temp])

        g = graph.network2nx(to_add.astype(int),
                             group_by=group_by,
                             remote_instance=remote_instance)

        # Add nodes that came in new
        ntable = network.get_node_table()
        nodes_to_add = [s for s in g.nodes if s not in ntable.id.values]
        if nodes_to_add:
            network.add_nodes(nodes_to_add)

        # Update neuron names
        ntable = network.get_node_table()
        names = ntable.set_index('name').neuron_name.to_dict()
        names.update({s: g.nodes[s]['neuron_name'] for s in g.nodes})
        ntable['id'] = ntable.name
        ntable['neuron_name'] = ntable.name.map(names)
        network.update_node_table(ntable, data_key_col='name',
                                  network_key_col='name')

        # Remove nodes that do not exist anymore
        ntable = network.get_node_table()
        nodes_to_remove = ntable[~ntable['id'].isin(g.nodes)]
        if not nodes_to_remove.empty:
            for v in nodes_to_remove.SUID.values:
                network.delete_node(v)

        # Remove edges
        etable = network.get_edge_table()
        edges_removed = 0
        for e in etable.itertuples():
            if (e.source, e.target) not in g.edges:
                edges_removed += 1
                network.delete_edge(e.SUID)

        # Add edges
        etable = network.get_edge_table()
        edges = [(s, t) for s, t in zip(etable.source.values, etable.target.values)]
        skid_to_SUID = ntable.set_index('name').SUID.to_dict()
        edges_to_add = []
        for e in set(g.edges) - set(edges):
            edges_to_add.append({'source': skid_to_SUID[e[0]],
                                 'target': skid_to_SUID[e[1]],
                                 'interaction': None,
                                 'directed': True})
        if edges_to_add:
            network.add_edges(edges_to_add)

        # Fix table and modify weights if applicable
        etable = network.get_edge_table()
        if not etable.loc[etable.source.isnull()].empty:
            etable.loc[etable.source.isnull(), 'source'] = etable.loc[etable.source.isnull(), 'name'].map(lambda x : x[:x.index('(')-1])
            etable.loc[etable.target.isnull(), 'target'] = etable.loc[etable.target.isnull(), 'name'].map(lambda x : x[x.index(')')+2:])
        new_weights = [g.edges[e]['weight'] for e in etable[['source','target']].values]
        weights_modified = [new_w for new_w, old_w in zip(new_weights, etable.weight.values) if new_w != old_w]
        etable['weight'] = new_weights
        # For some reason, there os no official wrapper for this, so we have to get our hands dirty
        network._CyNetwork__update_table('edge', etable,
                                         network_key_col='SUID',
                                         data_key_col='SUID')

        # If changes were made, give some feedback and/or change layout
        if nodes_to_add or not nodes_to_remove.empty or edges_to_add or edges_removed or weights_modified:
            if verbose:
                logger.info('{} - nodes added/removed: {}/{}; edges added/removed/modified {}/{}/{}'.format(datetime.datetime.now(),
                                                           len(nodes_to_add),
                                                           len(nodes_to_remove),
                                                           len(edges_to_add),
                                                           edges_removed,
                                                           len(weights_modified),
                                                           )
                            )

            if layout:
                cy.layout.apply(name=layout, network=network)

        # ZzzZzzzZ
        time.sleep(sleep)
