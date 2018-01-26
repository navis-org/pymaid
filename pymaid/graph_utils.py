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

""" Collection of tools to maniuplate CATMAID neurons using Graph representations.
"""


import sys
import math
import time
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import networkx as nx

from scipy.sparse import csgraph


from pymaid import graph, core

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

__all__ = sorted([ 'classify_nodes','cut_neuron', 'longest_neurite',
            'reroot_neuron', 'distal_to', 'dist_between',
            'generate_list_of_childs','geodesic_matrix'])

def _generate_segments(x, append=True):
    """ Generate linear segments for a given neuron.

    Parameters
    ----------
    x :         {CatmaidNeuron,CatmaidNeuronList}
                May contain multiple neurons.
    append :    bool, optional
                If True slabs will be appended to neuron.

    Returns
    -------
    list
                Segments as list of lists containing treenode ids.
    """

    if isinstance(x, pd.DataFrame) or isinstance(x, core.CatmaidNeuronList):
        return [_generate_segments(x.loc[i], append=append) for i in range(x.shape[0])]
    elif isinstance(x, core.CatmaidNeuron):
        pass
    else:
        module_logger.error('Unexpected datatype: %s' % str(type(skdata)))
        raise ValueError

    g = x.graph

    seeds = x.nodes[ x.nodes.type.isin(['branch','end']) ].treenode_id.values
    stops = x.nodes[ x.nodes.type.isin(['branch','root']) ].treenode_id.values

    seg_list = []
    for s in seeds:
        parent = next( g.successors(s), None )
        seg = [ s, parent ]
        while parent and parent not in stops:
            parent = next( g.successors(parent), None )
            seg.append(parent)
        seg_list.append(seg)

    if append:
        x.seg_list = seg_list

    return seg_list

def classify_nodes(x, inplace=True):
    """ Classifies neuron's treenodes into end nodes, branches, slabs
    or root.

    Parameters
    ----------
    x :         {CatmaidNeuron,CatmaidNeuronList}
                Neuron(s) to classify nodes for.
    inplace :   bool, optional
                If False, nodes will be classified on a copy which is then
                returned.

    Returns
    -------
    skdata
               Only if ``inplace=False``. Added column 'type' to ``skdata.nodes``.

    """

    if not inplace:
        x = x.copy()

    # If more than one neuron
    if isinstance(x, (pd.DataFrame, core.CatmaidNeuronList)):
        for i in trange(x.shape[0], desc='Classifying'):
            classify_nodes(x.ix[i], inplace=True)
    elif isinstance(x, (pd.Series, core.CatmaidNeuron)):
        # Get graph representation of neuron
        g = x.graph
        # Get branch and end nodes based on their degree of connectivity
        ends = [ n for n in g.nodes if g.degree(n) == 1 ]
        branches = [ n for n in g.nodes if g.degree(n) > 2 ]

        x.nodes['type'] = 'slab'
        x.nodes.loc[ x.nodes.treenode_id.isin(ends), 'type' ] = 'end'
        x.nodes.loc[ x.nodes.treenode_id.isin(branches), 'type' ] = 'branch'
        x.nodes.loc[ x.nodes.parent_id.isnull(), 'type' ] = 'root'

    else:
        raise TypeError('Unknown neuron type: %s' % str(type(x)))

    if not inplace:
        return x

def distal_to(x, a=None, b=None):
    """ Checks if nodes A are distal to nodes B.

    Important
    ---------
    Please note that if node A is not distal to node B, this does **not**
    automatically mean it is proximal instead: if nodes are on different
    branches, they are neither distal nor proximal to one another! To test
    for this case run a->b and b->a - if both return ``False``, nodes are on
    different branches.

    Also: if a and b are the same node, this function will return ``True``!

    Parameters
    ----------
    x :     CatmaidNeuron
    a,b :   {single treenode ID, list of treenode IDs, None}, optional
            If no treenode IDs are provided, will consider all treenodes.

    Returns
    -------
    bool
            If a and b are single treenode IDs respectively
    pd.DataFrame
            If a and/or b are lists of treenode IDs. Columns and rows (index)
            represent treenode IDs. Neurons *a* are rows, neurons *b* are
            columns.

    Examples
    --------
    >>> # Get a neuron
    >>> x = pymaid.get_neuron(16)
    >>> # Get a treenode ID from tag
    >>> a = x.tags['TEST_TAG'][0]
    >>> # Check all nodes if they are distal or proximal to that tag
    >>> df = pymaid.distal_to(x, a )
    >>> # Get the IDs of the nodes that are distal
    >>> df[ df[a] ].index.tolist()

    """

    if isinstance(x, core.CatmaidNeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.CatmaidNeuron ):
        raise ValueError('Please pass a SINGLE CatmaidNeuron')

    if not isinstance(a, type(None) ):
        if not isinstance(a, (list, np.ndarray)):
            a = [a]
        # Make sure we're dealing with integers
        a = np.unique(a).astype(int)
    else:
        a = x.nodes.treenode_id.values

    if not isinstance(b, type(None)):
        if not isinstance(b, (list, np.ndarray)):
            b = [b]
        # Make sure we're dealing with integers
        b = np.unique(b).astype(int)
    else:
        b = x.nodes.treenode_id.values

    df = pd.DataFrame( np.zeros( (len(a),len(b)), dtype=bool ),
                       columns=b, index=a )

    # Iterate over all targets
    for nB in tqdm(b, desc='Querying paths', disable = len (b) < 1000):
        # Get all paths TO this target
        paths = nx.shortest_path_length(x.graph, source=None, target=nB)
        # Check if sources are among our targets
        df[nB] = [ nA in paths for nA in a ]

    if df.shape == (1,1):
        return df.values[0][0]
    else:
        # Return boolean
        return df

def geodesic_matrix(x, tn_ids=None, directed=False):
    """ Generates all-by-all geodesic (along-the-arbor) distance matrix for a neuron.

    Parameters
    ----------
    x :         {CatmaidNeuron, CatmaidNeuronList}
                If list, must contain a SINGLE neuron.
    tn_ids :    {list, numpy.ndarray}, optional
                If provided, will compute distances only from this subset.
    directed :  bool, optional
                If True, pairs without a child->parent path will be returned with
                distance = "inf".

    Returns
    -------
    pd.DataFrame
            Geodesic distance matrix. Distances in nanometres.

    See Also
    --------
    :func:`~pymaid.distal_to`
        Check if a node A is distal to node B.
    :func:`~pymaid.dist_between`
        Get point-to-point geodesic distances.

    """

    if isinstance(x, core.CatmaidNeuronList):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError('Cannot process more than a single neuron.')
    elif isinstance(x, core.CatmaidNeuron):
        pass
    else:
        raise ValueError('Unable to process data of type "{0}"'.format(type(x)))

    nodeList = tuple(x.graph.nodes())

    if not isinstance(tn_ids, type(None)):
        tn_ids = set(tn_ids)
        tn_indices = tuple(i for i,node in enumerate(nodeList) if node in tn_ids)
        ix = tn_ids
    else:
        tn_indices = None
        ix = nodeList

    dmat = csgraph.dijkstra(nx.to_scipy_sparse_matrix(x.graph, nodeList),
            directed=directed, indices=tn_indices)

    return pd.DataFrame(dmat, columns=nodeList, index=ix )

def dist_between(x, a, b):
    """ Returns the geodesic distance between treenodes in nanometers.

    Parameters
    ----------
    x :             {CatmaidNeuron, CatmaidNeuronList}
                    Neuron containing the nodes
    a, b :          treenode IDs
                    Treenodes to check.

    Returns
    -------
    int
                    distance in nm

    See Also
    --------
    :func:`~pymaid.distal_to`
        Check if a node A is distal to node B.
    :func:`~pymaid.geodesic_matrix`
        Get all-by-all geodesic distance matrix.

    """

    if isinstance( x, core.CatmaidNeuronList ):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError('Need a single CatmaidNeuron')
    elif isinstance( x, core.CatmaidNeuron ):
        pass
    else:
        raise ValueError('Unable to process data of type {0}'.format(type(x)))

    try:
        _ = int(a)
        _ = int(b)
    except:
        raise ValueError('a, b need to be treenode IDs')


    return int( nx.algorithms.shortest_path_length( g.to_undirected(as_view=True),
                                                    a, b,
                                                    weight='weight') )


def longest_neurite(x, n=1, reroot_to_soma=False, inplace=False):
    """ Returns a neuron consisting of the longest neurite(s) based on
    geodesic distance.

    Parameters
    ----------
    x :                 {CatmaidNeuron,CatmaidNeuronList}
                        May contain multiple neurons.
    n :                 int, optional
                        Number of longest neurites to retain.
    reroot_to_soma :    bool, optional
                        If True, neuron will be rerooted to soma. Soma is the
                        node with >1000 radius.
    inplace :           bool, optional
                        If False, copy of the neuron will be trimmed down to
                        longest neurite and returned.

    Returns
    -------
    pandas.DataFrame/CatmaidNeuron object
                   Contains only node data of the longest neurite
    """

    if isinstance(x, core.CatmaidNeuron):
        x = x
    elif isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x.loc[0]
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % x.shape[0])
            raise Exception
    else:
        raise TypeError('Unable to process data of type "{0}"'.format(type(x)))

    if not inplace:
        x = x.copy()

    if reroot_to_soma and x.soma:
        x.reroot( x.soma )

    tn_to_preserve = []

    for i in range(n):
        if tn_to_preserve:
            # Generate fresh graph
            g = graph.neuron2nx( x )

            # Remove nodes that we have already preserved
            g.remove_nodes_from( tn_to_preserve )
        else:
            g = x.graph

        # Get path
        tn_to_preserve += nx.dag_longest_path(g)

    #Subset neuron
    x.nodes = x.nodes[x.nodes.treenode_id.isin(
        tn_to_preserve)].reset_index(drop=True)
    x.connectors = x.connectors[ x.connectors.treenode_id.isin(
                    tn_to_preserve)].reset_index(drop=True)

    # Reset indices of node and connector tables (important for igraph!)
    x.nodes.reset_index(inplace=True,drop=True)
    x.connectors.reset_index(inplace=True,drop=True)

    x._clear_temp_attr()

    if not inplace:
        return x

def reroot_neuron(x, new_root, inplace=False):
    """ Reroot neuron to new root.

    Parameters
    ----------
    x :        {CatmaidNeuron, CatmaidNeuronList}
               List must contain a SINGLE neuron.
    new_root : {int, str}
               Node ID or tag of the node to reroot to.
    inplace :  bool, optional
               If True the input neuron will be rerooted.

    Returns
    -------
    CatmaidNeuron object
               Containing the rerooted neuron.

    See Also
    --------
    :func:`~pymaid.CatmaidNeuron.reroot`
                Quick access to reroot directly from CatmaidNeuron/List objects

    """

    if new_root == None:
        raise ValueError('New root can not be <None>')

    if isinstance(x, core.CatmaidNeuron):
        df = x
    elif isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            df = x.loc[0]
        else:
            raise Exception(
                '{0} neurons provided. Please provide only a single neuron!'.format(x.shape[0]))
    else:
        raise Exception('Unable to process data of type "{0}"'.format(type(x)))

    # If new root is a tag, rather than a ID, try finding that node
    if isinstance(new_root, str):
        if new_root not in df.tags:
            module_logger.error(
                '#%s: Found no treenodes with tag %s - please double check!' % (str(df.skeleton_id),str(new_root)))
            return
        elif len(df.tags[new_root]) > 1:
            module_logger.error(
                '#%s: Found multiple treenodes with tag %s - please double check!' % (str(df.skeleton_id),str(new_root)))
            return
        else:
            new_root = df.tags[new_root][0]

    if not inplace:
        x = x.copy()

    g = x.graph

    # Walk from new root to old root and remove edges along the way
    parent = next(g.successors(new_root), None)
    if not parent:
        # new_root is already the root
        return
    path = [new_root]
    weights = []
    while parent is not None:
        weights.append( g[path[-1]][parent]['weight'] )
        g.remove_edge(path[-1],parent)
        path.append(parent)
        parent = next(g.successors(parent), None)

    # Invert path and add weights
    new_edges = [ ( path[i+1], path[i], {'weight':weights[i]} ) for i in range(len(path)-1) ]

    # Add inverted path between old and new root
    g.add_edges_from( new_edges )

    # Propagate changes in graph back to treenode table
    x.nodes.set_index('treenode_id', inplace=True)
    x.nodes.loc[ path[1:], 'parent_id' ] = path[:-1]
    x.nodes.reset_index(drop=False, inplace=True)

    # Set new root's parent to None
    x.nodes.parent_id = x.nodes.parent_id.astype(object)
    x.nodes.loc[ x.nodes.treenode_id == new_root, 'parent_id' ] = None

    x._clear_temp_attr(exclude=['graph'])

    if not inplace:
        return x
    else:
        return

def cut_neuron(x, cut_node, g=None):
    """ Split neuron at given point. Returns two new neurons.

    Parameters
    ----------
    x :        {CatmaidNeuron, CatmaidNeuronList}
               Must be a single neuron.
    cut_node : {int, str}
               Node ID or a tag of the node to cut.

    Returns
    -------
    neuron_dist
                Part of the neuron distal to the cut.
    neuron_prox
                Part of the neuron proximal to the cut.

    Examples
    --------
    >>> # Example for multiple cuts
    >>> import pymaid
    >>> remote_instance = pymaid.CatmaidInstance( url, http_user, http_pw, token )
    >>> n = pymaid.get_neuron(skeleton_id)
    >>> # First cut
    >>> nA, nB = cut_neuron2( n, cut_node1 )
    >>> # Second cut
    >>> nD, nE = cut_neuron2( nA, cut_node2 )

    See Also
    --------
    :func:`~pymaid.CatmaidNeuron.prune_distal_to`
    :func:`~pymaid.CatmaidNeuron.prune_proximal_to`

    """

    if isinstance(x, core.CatmaidNeuron):
        pass
    elif isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % x.shape[0])
            raise Exception(
                '%i neurons provided. Please provide only a single neuron!' % x.shape[0])
    else:
        raise TypeError('Unable to process data of type "{0}"'.format(type(x)))

    # If cut_node is a tag (rather than an ID), try finding that node
    if isinstance(cut_node, str):
        if cut_node not in x.tags:
            raise ValueError(
                '#%s: Found no treenode with tag %s - please double check!' % (str(x.skeleton_id),str(cut_node)))

        elif len(x.tags[cut_node]) > 1:
            raise ValueError(
                '#%s: Found multiple treenode with tag %s - please double check!' % (str(x.skeleton_id),str(cut_node)))
        else:
            cut_node = x.tags[cut_node][0]

    # Get subgraphs consisting of nodes proximal/distal to cut node
    prox_graph = nx.bfs_tree( x.graph, cut_node )
    dist_graph = nx.bfs_tree( x.graph, cut_node, reverse=True )

    # bfs_tree does not preserve 'weight' -> need to subset original graph by those nodes
    prox_graph = x.graph.subgraph( prox_graph.nodes )
    dist_graph = x.graph.subgraph( dist_graph.nodes )
    # ATTENTION: prox/dist_graph contain pointers to the original graph
    # -> changes to structure don't but changes to attributes will propagate back

    # Generate new neurons (this is the actual bottleneck of the function: ~70% of time)
    dist = x.copy()
    prox = x.copy()

    # Filter treenodes
    dist.nodes = dist.nodes[ dist.nodes.treenode_id.isin( dist_graph.nodes ) ]
    prox.nodes = prox.nodes[ prox.nodes.treenode_id.isin( prox_graph.nodes ) ]

    # Change new root for dist
    dist.nodes.loc[ dist.nodes.treenode_id == cut_node, 'parent_id' ] = None
    dist.nodes.loc[ dist.nodes.treenode_id == cut_node, 'type' ] = 'root'

    # Change cut node to end node for prox
    prox.nodes.loc[ prox.nodes.treenode_id == cut_node, 'type' ] = 'end'

    # Filter connectors
    dist.connectors = dist.connectors[ dist.connectors.treenode_id.isin( dist_graph.nodes ) ]
    prox.connectors = prox.connectors[ prox.connectors.treenode_id.isin( prox_graph.nodes ) ]

    # Filter tags
    dist.tags = { t : [ tn for tn in dist.tags[t] if tn in dist_graph.nodes ] for t in dist.tags }
    prox.tags = { t : [ tn for tn in prox.tags[t] if tn in prox_graph.nodes ] for t in prox.tags }

    # Remove empty tags
    dist.tags = { t : dist.tags[t] for t in dist.tags if dist.tags[t] }
    prox.tags = { t : prox.tags[t] for t in prox.tags if prox.tags[t] }

    # Reassign graphs
    dist.graph = dist_graph
    prox.graph = prox_graph

    # Clear other temporary attributes
    dist._clear_temp_attr(exclude=['graph','type','classify_nodes'])
    prox._clear_temp_attr(exclude=['graph','type','classify_nodes'])

    return dist, prox

def generate_list_of_childs(x):
    """ Returns list of childs

    Parameters
    ----------
    x :     {CatmaidNeuron, CatmaidNeuronList}
            If List, must contain a SINGLE neuron.

    Returns
    -------
    dict
     ``{ treenode_id : [ child_treenode, child_treenode, ... ] }``

    """

    return { n : [ e[0] for e in x.graph.in_edges(n) ] for n in x.graph.nodes }

def node_label_sorting(x):
    """ Returns treenodes ordered by node label sorting according to Cuntz
    et al., PLoS Computational Biology (2010).

    Parameters
    ----------
    x :         {CatmaidNeuron}

    Returns
    -------
    list
        [ root, treenode_id, treenode_id, ... ]
    """
    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Need CatmaidNeuron, got "{0}"'.format(type(x)))

    if len(x.root) > 1:
        raise ValueError('Unable to process multi-root neurons!')

    # Get relevant branch points
    term = x.nodes[x.nodes.type=='end'].treenode_id.values

    # Get distance from all branch_points
    dist_mat = geodesic_matrix(x, tn_ids = term, directed=True  )
    # Set distance between unreachable points to None
    dist_mat[dist_mat == float('inf')] = None

    # Get starting points and sort by longest path to a terminal
    curr_points = sorted( list( x.simple.graph.predecessors(x.root[0]) ),
                           key= lambda n : dist_mat[n].max(),
                           reverse=True )

    # Walk from root along towards terminals, prioritising longer branches
    nodes_walked = [ ]
    while curr_points:
        nodes_walked.append( curr_points.pop(0) )
        if nodes_walked[-1] in term:
            pass
        else:
            new_points = sorted( list( x.simple.graph.predecessors( nodes_walked[-1] ) ),
                           key= lambda n : dist_mat[n].max(),
                           reverse=True )
            curr_points = new_points + curr_points

    # Translate into segments
    node_list =[ x.root[0] ]
    for n in nodes_walked:
        node_list += [ seg for seg in x.segments if seg[0] == n ][0][:-1]

    return node_list



