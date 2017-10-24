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

import sys
import math
import time
import logging
import pandas as pd
import numpy as np
import scipy
from scipy.spatial import ConvexHull
from tqdm import tqdm, trange

from pymaid import pymaid, igraph_catmaid, core

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

__all__ = [ 'calc_cable','calc_strahler_index','classify_nodes','cut_neuron',
            'downsample_neuron','in_volume','longest_neurite',
            'prune_by_strahler','reroot_neuron','synapse_root_distances',
            'cable_within_distance','stitch_neurons','arbor_confidence']

def generate_list_of_childs(skdata):
    """ Transforms list of nodes into a dictionary { parent: [child1,child2,...]}

    Parameters
    ----------
    skdata :   {CatmaidNeuron,CatmaidNeuronList} 
               Must contain a SINGLE neuron

    Returns
    -------
    dict   
     ``{ treenode_id : [ child_treenode, child_treenode, ... ] }``

    """
    module_logger.debug('Generating list of childs...')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        nodes = skdata.nodes
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            nodes = skdata.ix[0].nodes
        else:
            module_logger.error('Please pass a SINGLE neuron.')
            raise Exception('Please pass a SINGLE neuron.')

    list_of_childs = {n.treenode_id: [] for n in nodes.itertuples()}

    for n in nodes.itertuples():
        try:
            list_of_childs[n.parent_id].append(n.treenode_id)
        except:
            list_of_childs[None] = [None]

    return list_of_childs


def classify_nodes(skdata, inplace=True):
    """ Takes list of nodes and classifies them as end nodes, branches, slabs
    and root

    Parameters
    ----------
    skdata :    {CatmaidNeuron,CatmaidNeuronList} 
                Neuron(s) to classify nodes for.
    inplace :   bool, optional 
                If False, nodes will be classified on a copy which is then 
                returned

    Returns
    -------
    skdata 
               Only if inplace=False. Added columns 'type' and 'has_connectors'
               to skdata.nodes

    """

    module_logger.debug('Looking for end, branch and root points...')

    if not inplace:
        skdata = skdata.copy()

    # If more than one neuron
    if isinstance(skdata, (pd.DataFrame, core.CatmaidNeuronList)):
        for i in trange(skdata.shape[0], 'Classifying'):
            classify_nodes(skdata.ix[i], inplace=True)
    elif isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        list_of_childs = generate_list_of_childs(skdata)
        
        classes = {}
        for n in list_of_childs:
            if len(list_of_childs[n]) == 0:
                classes[n] = 'end'            
            elif len(list_of_childs[n]) > 1:
                classes[n] = 'branch'
            else:
                classes[n] = 'slab'
        
        root = skdata.nodes[skdata.nodes[
            'parent_id'].isnull()].treenode_id.values
        classes.update({n: 'root' for n in root})

        new_column = [classes[n] for n in skdata.nodes.treenode_id.tolist()]
        skdata.nodes['type'] = new_column

        nodes_w_connectors = skdata.connectors.treenode_id.tolist()
        new_column = [
            n in nodes_w_connectors for n in skdata.nodes.treenode_id.tolist()]
        skdata.nodes['has_connectors'] = new_column
    else:
        module_logger.error('Unknown neuron type: %s' % str(type(skdata)))

    if not inplace:
        return skdata


def _generate_slabs(x, append=True):
    """ Generate slabs of a given neuron.

    Parameters
    ----------
    x :         {CatmaidNeuron,CatmaidNeuronList} 
                May contain multiple neurons
    append :    bool, optional 
                If true slabs will be appended to neuron

    Returns
    -------
    list      
                Slabs as list of lists containing treenode ids
    """

    if isinstance(x, pd.DataFrame) or isinstance(x, core.CatmaidNeuronList):
        return [_generate_slabs(x.ix[i], append=append) for i in range(x.shape[0])]
    elif isinstance(x, pd.Series):
        if x.igraph is None:
            x.igraph = igraph_catmaid.neuron2graph(x)
    elif isinstance(x, core.CatmaidNeuron):
        pass
    else:
        module_logger.error('Unexpected datatype: %s' % str(type(skdata)))
        raise ValueError

    # Make a copy of the graph -> we will delete edges later on
    g = x.igraph.copy()

    if 'type' not in x.nodes:
        classify_nodes(x)

    branch_points = x.nodes[x.nodes.type == 'branch'].treenode_id.tolist()
    root_node = x.nodes[x.nodes.type == 'root'].treenode_id.tolist()    

    # Now remove edges which have a branch point or the root node as target
    node_indices = [i for i in range(len(g.vs)) if g.vs[i][
        'node_id'] in branch_points + root_node]
    edges_to_delete = [i for i in range(len(g.es)) if g.es[
        i].target in node_indices]
    g.delete_edges(edges_to_delete)

    # Now chop graph into disconnected components
    components = [g.subgraph(sg) for sg in g.components(mode='WEAK')]

    # Problem here: components appear ordered by treenode_id. We have to generate 
    # subgraphs, sort them and turn vertex IDs into treenode ids. Attention: 
    # order changed when we created the subgraphs!
    slabs = [[sg.vs[i]['node_id'] for i in sg.topological_sorting()]
             for sg in components]

    # Delete root node slab -> otherwise this will be its own slab
    #slabs = [l for l in slabs if len(l) != 1 and l[0] not in root_node]
    slabs = [l for l in slabs if l[0] not in root_node]

    # Now add the parent to the last node of each slab (which should be a
    # branch point or the node)
    list_of_parents = {
        n.treenode_id: n.parent_id for n in x.nodes.itertuples()}
    for s in slabs:
        s.append(list_of_parents[s[-1]])

    if append:
        x.slabs = slabs

    return slabs


def downsample_neuron(skdata, resampling_factor, inplace=False, preserve_cn_treenodes=True, preserve_tag_treenodes=False):
    """ Downsamples neuron(s) by a given factor. Preserves root, leafs, 
    branchpoints by default. Preservation of treenodes with synapses can
    be toggled.

    Parameters
    ----------
    skdata :                 {CatmaidNeuron,CatmaidNeuronList} 
                             Neuron(s) to downsample
    resampling_factor :      int 
                             Factor by which to reduce the node count
    inplace :                bool, optional   
                             If True, will modify original neuron. If False, a 
                             downsampled copy is returned.
    preserve_cn_treenodes :  bool, optional
                             If True, treenodes that have connectors are 
                             preserved.
    preserve_tag_treenodes : bool, optional
                             If True, treenodes with tags are preserved.    

    Returns
    -------
    skdata
                         Downsampled Pandas Dataframe or CatmaidNeuron object
    """

    if isinstance(skdata, pd.DataFrame):
        return pd.DataFrame([downsample_neuron(skdata.ix[i], resampling_factor, inplace=inplace) for i in range(skdata.shape[0])])
    elif isinstance(skdata, core.CatmaidNeuronList):
        return core.CatmaidNeuronList([downsample_neuron(skdata.ix[i], resampling_factor, inplace=inplace) for i in range(skdata.shape[0])])
    elif isinstance(skdata, pd.Series):
        if not inplace:
            df = skdata.copy()
            df.nodes = df.nodes.copy()
            df.connectors = df.connectors.copy()
        else:
            df = skdata
    elif isinstance(skdata, core.CatmaidNeuron):
        if not inplace:
            df = core.CatmaidNeuron(skdata)
        else:
            df = skdata
    else:
        module_logger.error('Unexpected datatype: %s' % str(type(skdata)))
        raise ValueError

    if df.nodes.shape[0] == 0:
        module_logger.warning('Unable to downsample: no nodes in neuron')
        return df

    module_logger.info('Preparing to downsample neuron...')

    list_of_parents = {
        n.treenode_id: n.parent_id for n in df.nodes.itertuples()}

    if 'type' not in df.nodes:
        classify_nodes(df)

    selection = df.nodes.type != 'slab'    

    if preserve_cn_treenodes:
        selection = selection | df.nodes.has_connectors == True    

    if preserve_tag_treenodes:
        with_tags = [ t for l in df.tags.values() for t in l ]
        selection = selection | df.nodes.treenode_id.isin( with_tags )    

    fix_points = df.nodes[ selection ].treenode_id.values    

    # Walk from all fix points to the root - jump N nodes on the way
    new_parents = {}

    module_logger.info('Sampling neuron down by factor of %i' %
                       resampling_factor)
    for en in fix_points:
        this_node = en

        while True:
            stop = False
            np = list_of_parents[this_node]
            if np != None:
                for i in range(resampling_factor):
                    if np in fix_points:
                        new_parents[this_node] = np
                        stop = True
                        break
                    else:
                        np = list_of_parents[np]

                if stop is True:
                    break
                else:
                    new_parents[this_node] = np
                    this_node = np
            else:
                new_parents[this_node] = None
                break    
    new_nodes = df.nodes[
        [tn in new_parents for tn in df.nodes.treenode_id]].copy()    
    new_nodes.loc[:,'parent_id'] = [new_parents[tn]
                           for tn in new_nodes.treenode_id]    

    # We have to temporarily set parent of root node from 1 to an integer
    root_index = new_nodes[new_nodes.parent_id.isnull()].index[0]    
    new_nodes.loc[root_index, 'parent_id'] = 0    
    new_nodes.loc[:,'parent_id'] = new_nodes.parent_id.values.astype(
        int)  # first convert everything to int
    
    new_nodes.loc[:,'parent_id'] = new_nodes.parent_id.values.astype(
        object)  # then back to object so that we can add a 'None'
    
    # Reassign parent_id None to root node
    new_nodes.loc[root_index, 'parent_id'] = None
    
    module_logger.info('Nodes before/after: %i/%i ' %
                       (len(df.nodes), len(new_nodes)))

    df.nodes = new_nodes

    # This is essential -> otherwise e.g. igraph_catmaid.neuron2graph will fail
    df.nodes.reset_index(inplace=True, drop=True)

    if not inplace:
        return df


def longest_neurite(skdata, reroot_to_soma=False, inplace=False):
    """ Returns a neuron consisting only of the longest neurite (based on 
    geodesic distance)

    Parameters
    ----------
    skdata :            {CatmaidNeuron,CatmaidNeuronList} 
                        May contain multiple neurons
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

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0]
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])
            raise Exception

    if not inplace:
        df = df.copy()

    if reroot_to_soma:
        soma = df.nodes[df.nodes.radius > 1000].reset_index()
        if soma.shape[0] != 1:
            module_logger.error(
                'Unable to reroot: None or multiple soma found for neuron %s ' % df.neuron_name)
            raise Exception
        if soma.ix[0].parent_id != None:
            df = reroot_neuron(df, soma.ix[0].treenode_id)

    #First collect leafs and root
    leaf_nodes = df.nodes[df.nodes.type=='end'].treenode_id.tolist()
    root_nodes = df.nodes[df.nodes.type=='root'].treenode_id.tolist()

    #Convert to igraph vertex indices
    leaf_ix = [ v.index for v in df.igraph.vs if v['node_id'] in leaf_nodes ]
    root_ix = [ v.index for v in df.igraph.vs if v['node_id'] in root_nodes ]

    #Now get paths from all tips to the root
    paths = df.igraph.get_shortest_paths( root_ix[0], leaf_ix, mode='ALL' )

    #Translate indices back into treenode ids
    paths_tn = [ [ df.igraph.vs[i]['node_id'] for i in p ] for p in paths ]    

    #Generate DataFrame with all the info
    path_df = pd.DataFrame( [ [p, df.nodes.set_index('treenode_id').ix[p].reset_index() ] for p in paths_tn ],
                            columns=['path','nodes']  )

    #Now calculate cable of each of the paths
    path_df['cable'] = [ calc_cable(path_df.ix[i], return_skdata=False, smoothing=1) for i in range(path_df.shape[0])]    

    tn_to_preverse = path_df.sort_values('cable', ascending=False).reset_index().ix[0].path
    
    df.nodes = df.nodes[df.nodes.treenode_id.isin(
        tn_to_preverse)].reset_index(drop=True)

    if not inplace:
        return df


def reroot_neuron(skdata, new_root, g=None, inplace=False):
    """ Uses igraph to reroot the neuron at given point. 

    Parameters
    ----------
    skdata :   {CatmaidNeuron, CatmaidNeuronList} 
               Must contain a SINGLE neuron
    new_root : {int, str}
               Node ID or tag of the node to reroot to 
    inplace :  bool, optional
               If True the input neuron will be rerooted. Default = False 

    Returns
    -------
    pandas.Series or CatmaidNeuron object
               Containing the rerooted neuron
    """

    if new_root == None:
        raise ValueError('New root can not be <None>')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0]
        else:
            module_logger.error(
                '#%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])
            raise Exception(
                '#%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])
    else:
        module_logger.error(
            'Unable to process data of type %s' % str(type(skdata)))
        raise Exception('Unable to process data of type %s' %
                        str(type(skdata)))

    start_time = time.time()

    # If cut_node is a tag, rather than a ID, try finding that node
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

    if df.nodes.set_index('treenode_id').ix[new_root].parent_id == None:
        module_logger.info('New root == old root! No need to reroot.')
        if not inplace:
            return df
        else:
            return

    if not g:
        if isinstance(df, core.CatmaidNeuron):
            g = df.igraph
        elif instance(df, pd.Series):
            if df.igraph != None:
                # Generate iGraph -> order/indices of vertices are the same as
                # in skdata
                g = igraph_catmaid.neuron2graph(df)
                df.igraph = g
            else:
                g = df.igraph

    if not inplace:
        # Now that we have generated the graph, make sure to make all further
        # work on a copy!
        df = df.copy()
        df = skdata.copy()
        # Make sure to copy nodes/connectors as well (essentially everything
        # that is a DataFrame itself)
        df.nodes = df.nodes.copy()
        df.connectors = df.connectors.copy()
        df.igraph = df.igraph.copy()

    try:
        # Select nodes with the correct ID as cut node
        new_root_index = g.vs.select(node_id=int(new_root))[0].index
    # Should have found only one cut node
    except:
        module_logger.error(
            '#%s: Found no treenodes with ID %s - please double check!' % (str(df.skeleton_id),str(new_root)))            
        return

    if 'type' not in df.nodes:
        classify_nodes(df)

    leaf_nodes = df.nodes[df.nodes.type == 'end'].treenode_id.tolist()

    # We need to make sure that the current root node is not a leaf node itself.
    # If it is, we have to add it too:
    old_root = df.nodes.set_index('parent_id').ix[None].treenode_id
    if df.nodes[df.nodes.parent_id==old_root].shape[0] < 2:
        leaf_nodes.append( old_root )     

    leaf_indices = [v.index for v in g.vs if v['node_id'] in leaf_nodes]

    shortest_paths = g.get_shortest_paths(
        new_root_index, to=leaf_indices, mode='ALL')

    # Convert indices to treenode ids
    new_paths = [[g.vs[i]['node_id'] for i in p] for p in shortest_paths]

    # Remove root node to root node (path length == 1)
    new_paths = [n for n in new_paths if len(n) > 1]

    new_parents = {}
    new_edges = []
    for p in new_paths:
        new_parents.update({p[-i]: p[-i - 1] for i in range(1, len(p))})
    # This is a placeholder! Can't set this yet otherwise .astype(int) will
    # fail
    new_parents.update({new_root: 0})

    df.nodes.set_index('treenode_id', inplace=True)  # Set index to treenode_id
    # index is currently the treenode_id
    df.nodes.parent_id = [new_parents[n] for n in df.nodes.index.tolist()]
    df.nodes.parent_id = df.nodes.parent_id.values.astype(
        int)  # first convert everything to int
    df.nodes.parent_id = df.nodes.parent_id.values.astype(
        object)  # then back to object so that we can add a 'None'
    # Set parent_id to None (previously 0 as placeholder)
    df.nodes.loc[new_root, 'parent_id'] = None
    df.nodes.reset_index(inplace=True)  # Reset index

    # Recalculate node types
    classify_nodes(df)

    # Delete igraph
    #df.igraph = igraph_catmaid.neuron2graph(df)
    del df.igraph

    module_logger.info('%s #%s successfully rerooted (%s s)' % (
        df.neuron_name, df.skeleton_id, round(time.time() - start_time, 1)))

    if not inplace:
        return df
    else:
        return


def cut_neuron(skdata, cut_node, g=None):
    """ Uses igraph to Cut the neuron at given point and returns two new neurons.

    Parameters
    ----------
    skdata :   {CatmaidNeuron, CatmaidNeuronList} 
               Must contain a SINGLE neuron
    cut_node : {int, str}
               Node ID or a tag of the node to cut   

    Returns
    -------
    neuron_dist 
                Part of the neuron distal to the cut
    neuron_prox 
                Part of the neuron proximal to the cut

    Examples
    --------
    >>> # Example for multiple cuts 
    >>> import pymaid    
    >>> remote_instance = pymaid.CatmaidInstance( url, http_user, http_pw, token )
    >>> skeleton_dataframe = pymaid.get_neuron(skeleton_id,remote_instance)   
    >>> # First cut
    >>> nA, nB = cut_neuron2( skeleton_data, cut_node1 )
    >>> # Second cut
    >>> nD, nE = cut_neuron2( nA, cut_node2 )  
    """
    start_time = time.time()

    module_logger.info('Cutting neuron...')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata.copy()
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0].copy()
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])
            raise Exception(
                '%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])

    if g is None:
        g = df.igraph

    if g is None:
        # Generate iGraph -> order/indices of vertices are the same as in
        # skdata
        g = igraph_catmaid.neuron2graph(df)
    else:
        g = g.copy()

    # If cut_node is a tag (rather than an ID), try finding that node
    if isinstance(cut_node, str):
        if cut_node not in df.tags:
            module_logger.error(
                '#%s: Found no treenode with tag %s - please double check!' % (str(df.skeleton_id),str(cut_node)))                
            raise ValueError(
                '#%s: Found no treenode with tag %s - please double check!' % (str(df.skeleton_id),str(cut_node)))

        elif len(df.tags[cut_node]) > 1:
            module_logger.warning(
                '#%s: Found multiple treenode with tag %s - please double check!' % (str(df.skeleton_id),str(cut_node)))
            raise ValueError(
                '#%s: Found multiple treenode with tag %s - please double check!' % (str(df.skeleton_id),str(cut_node)))
        else:
            cut_node = df.tags[cut_node][0]

    module_logger.debug('Cutting neuron...')

    try:
        # Select nodes with the correct ID as cut node
        cut_node_index = g.vs.select(node_id=int(cut_node))[0].index
    # Should have found only one cut node
    except:
        module_logger.error(
            'No treenode with ID %s in graph - please double check!' % str(cut_node))
        raise ValueError(
            'No treenode with ID %s in graph - please double check!' % str(cut_node))

    # Select the cut node's parent
    try:
        parent_node_index = g.es.select(_source=cut_node_index)[0].target
    except:
        module_logger.error(
            'Unable to find parent for cut node. Is cut node = root?')
        #parent_node_index = g.es.select(_target=cut_node_index)[0].source
        raise Exception('Unable to find parent for cut node. Is cut node = root?')

    # Now calculate the min cut
    mc = g.st_mincut(parent_node_index, cut_node_index, capacity=None)

    # mc.partition holds the two partitions with mc.partition[0] holding part
    # with the source and mc.partition[1] the target
    if g.vs.select(mc.partition[0]).select(node_id=int(cut_node)):
        dist_partition = mc.partition[0]
        dist_graph = mc.subgraph(0)
        prox_partition = mc.partition[1]
        prox_graph = mc.subgraph(1)
    else:
        dist_partition = mc.partition[1]
        dist_graph = mc.subgraph(1)
        prox_partition = mc.partition[0]
        prox_graph = mc.subgraph(0)

    # Set parent_id of distal fragment's graph to None
    dist_graph.vs.select(node_id=int(cut_node))[0]['parent_id'] = None

    # Partitions hold the indices -> now we have to translate this into node
    # ids
    dist_partition_ids = g.vs.select(dist_partition)['node_id']
    prox_partition_ids = g.vs.select(prox_partition)['node_id']

    # Set dataframe indices to treenode IDs - will facilitate distributing
    # nodes
    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    neuron_dist = pd.DataFrame([[
        df.neuron_name + '_dist',
        df.skeleton_id,
        df.nodes.ix[dist_partition_ids],
        df.connectors[
            [c.treenode_id in dist_partition_ids for c in df.connectors.itertuples()]].reset_index(),
        df.tags,
        dist_graph
    ]],
        columns=['neuron_name', 'skeleton_id',
                 'nodes', 'connectors', 'tags', 'igraph'],
        dtype=object
    ).ix[0]

    neuron_dist.nodes.loc[cut_node, 'parent_id'] = None

    neuron_prox = pd.DataFrame([[
        df.neuron_name + '_prox',
        df.skeleton_id,
        df.nodes.ix[prox_partition_ids],
        df.connectors[
            [c.treenode_id not in dist_partition_ids for c in df.connectors.itertuples()]].reset_index(),
        df.tags,
        prox_graph
    ]],
        columns=['neuron_name', 'skeleton_id',
                 'nodes', 'connectors', 'tags', 'igraph'],
        dtype=object
    ).ix[0]

    # Reclassify cut node in distal as 'root' and its parent in proximal as
    # 'end'
    if 'type' in df.nodes:
        neuron_dist.nodes.loc[cut_node, 'type'] = 'root'
        neuron_prox.nodes.loc[df.nodes.ix[cut_node].parent_id, 'type'] = 'end'

    # Now reindex dataframes
    neuron_dist.nodes.reset_index(inplace=True)
    neuron_prox.nodes.reset_index(inplace=True)
    df.nodes.reset_index(inplace=True)

    module_logger.debug('Cutting finished in %is' %
                        round(time.time() - start_time))
    module_logger.info('Distal: %i nodes/%i synapses| |Proximal: %i nodes/%i synapses' % (neuron_dist.nodes.shape[
                       0], neuron_dist.connectors.shape[0], neuron_prox.nodes.shape[0], neuron_prox.connectors.shape[0]))

    if isinstance(df, pd.Series):
        return neuron_dist, neuron_prox
    elif isinstance(df, core.CatmaidNeuron):
        n_dist = df.copy()
        n_dist.neuron_name += '_dist'
        #n_dist.nodes = df.nodes.ix[dist_partition_ids].reset_index(drop=True).copy()
        n_dist.nodes = neuron_dist.nodes
        n_dist.connectors = df.connectors[
            [c.treenode_id in dist_partition_ids for c in df.connectors.itertuples()]].reset_index().copy()
        n_dist.igraph = dist_graph.copy()
        n_dist.df = neuron_dist

        n_prox = df.copy()
        n_prox.neuron_name += '_prox'
        #n_prox.nodes = df.nodes.ix[prox_partition_ids].copy()
        n_prox.nodes = neuron_prox.nodes
        n_prox.connectors = df.connectors[
            [c.treenode_id in prox_partition_ids for c in df.connectors.itertuples()]].reset_index().copy()
        n_prox.igraph = prox_graph.copy()
        n_prox.df = neuron_prox

        return n_dist, n_prox


def _cut_neuron(skdata, cut_node):
    """ DEPRECATED! Cuts a neuron at given point and returns two new neurons. 
    Does not use igraph (slower)

    Parameters
    ----------
    skdata :   {CatmaidNeuron, CatmaidNeuronList} 
               Must contain a SINGLE neuron
    cut_node : {int, str}    
               Node ID or a tag of the node to cut

    Returns
    -------
    neuron_dist
               Neuron object distal to the cut
    neuron_prox
               Neuron object proximal to the cut
    """
    start_time = time.time()

    module_logger.info('Preparing to cut neuron...')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata.copy()
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0].copy()
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])
            raise Exception(
                '%i neurons provided. Please provide only a single neuron!' % skdata.shape[0])

    list_of_childs = generate_list_of_childs(skdata)
    list_of_parents = {
        n.treenode_id: n.parent_id for n in df.nodes.itertuples()}

    if 'type' not in df.nodes:
        classify_nodes(df)

    # If cut_node is a tag, rather than a ID, try finding that node
    if type(cut_node) == type(str()):
        if cut_node not in df.tags:
            module_logger.error(
                'Error: Found no treenodes with tag %s - please double check!' % str(cut_node))
            return
        elif len(df.tags[cut_node]) > 1:
            module_logger.error(
                'Error: Found multiple treenodes with tag %s - please double check!' % str(cut_node))
            return
        else:
            cut_node = df.tags[cut_node][0]

    if len(list_of_childs[cut_node]) == 0:
        module_logger.warning('Cannot cut: cut_node is a leaf node!')
        return
    elif list_of_parents[cut_node] == None:
        module_logger.warning('Cannot cut: cut_node is a root node!')
        return
    elif cut_node not in list_of_parents:
        module_logger.warning('Cannot cut: cut_node not found!')
        return

    end_nodes = df.nodes[df.nodes.type == 'end'].treenode_id.values
    branch_nodes = df.nodes[df.nodes.type == 'branch'].treenode_id.values
    root = df.nodes[df.nodes.type == 'root'].treenode_id.values[0]

    # Walk from all end points to the root - if you hit the cut node assign
    # this branch to neuronA otherwise neuronB
    distal_nodes = []
    proximal_nodes = []

    module_logger.info('Cutting neuron...')
    for i, en in enumerate(end_nodes.tolist() + [cut_node]):
        this_node = en
        nodes_walked = [en]
        while True:
            this_node = list_of_parents[this_node]
            nodes_walked.append(this_node)

            # Stop if this node is the cut node
            if this_node == cut_node:
                distal_nodes += nodes_walked
                break
            # Stop if this node is the root node
            elif this_node == root:
                proximal_nodes += nodes_walked
                break
            # Stop if we have seen this branchpoint before
            elif this_node in branch_nodes:
                if this_node in distal_nodes:
                    distal_nodes += nodes_walked
                    break
                elif this_node in proximal_nodes:
                    proximal_nodes += nodes_walked
                    break

    # Set dataframe indices to treenode IDs - will facilitate distributing
    # nodes
    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    distal_nodes = list(set(distal_nodes))
    proximal_nodes = list(set(proximal_nodes))

    neuron_dist = pd.DataFrame([[
        df.neuron_name + '_dist',
        df.skeleton_id,
        df.nodes.ix[distal_nodes],
        df.connectors[
            [c.treenode_id in distal_nodes for c in df.connectors.itertuples()]].reset_index(),
        df.tags
    ]],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    ).ix[0]

    neuron_dist.nodes.ix[cut_node].parent_id = None
    neuron_dist.nodes.ix[cut_node].type = 'root'

    neuron_prox = pd.DataFrame([[
        df.neuron_name + '_prox',
        df.skeleton_id,
        df.nodes.ix[proximal_nodes],
        df.connectors[
            [c.treenode_id not in distal_nodes for c in df.connectors.itertuples()]].reset_index(),
        df.tags
    ]],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    ).ix[0]

    # Reclassify cut node in proximal neuron as end node
    neuron_prox.nodes.ix[cut_node].type = 'end'

    # Now reindex dataframes
    neuron_dist.nodes.reset_index(inplace=True)
    neuron_prox.nodes.reset_index(inplace=True)
    df.nodes.reset_index(inplace=True)

    module_logger.info('Cutting finished in %is' %
                       round(time.time() - start_time))
    module_logger.info('Distal to cut node: %i nodes/%i synapses' %
                       (neuron_dist.nodes.shape[0], neuron_dist.connectors.shape[0]))
    module_logger.info('Proximal to cut node: %i nodes/%i synapses' %
                       (neuron_prox.nodes.shape[0], neuron_prox.connectors.shape[0]))

    return neuron_dist, neuron_prox


def synapse_root_distances(skdata, pre_skid_filter=[], post_skid_filter=[], remote_instance=None):
    """ Calculates geodesic (along the arbor) distance of synapses to root 
    (i.e. soma)

    Parameters
    ----------  
    skdata :            {CatmaidNeuron, CatmaidNeuronList} 
                        Must contain a SINGLE neuron
    pre_skid_filter :   list of int, optional
                        If provided, only synapses from these neurons will be 
                        processed.
    post_skid_filter :  list of int, optional 
                        If provided, only synapses to these neurons will be 
                        processed.
    remote_instance :   CatmaidInstance, optional
                        If not passed directly, will try using global.

    Returns
    -------
    pre_node_distances 
       ``{'connector_id: distance_to_root[nm]'}`` for all presynaptic sites of 
       this neuron
    post_node_distances 
       ``{'connector_id: distance_to_root[nm]'}`` for all postsynaptic sites of 
       this neuron
    """

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            module_logger.error(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            raise Exception(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0]
        else:
            module_logger.error(
                '%i neurons provided. Currently, only a single neuron is supported!' % skdata.shape[0])
            raise Exception(
                '%i neurons provided. Currently, only a single neuron is supported!' % skdata.shape[0])

    # Reindex dataframe to treenode
    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    # Calculate distance to parent for each node
    tn_coords = skdata.nodes[['x', 'y', 'z']].reset_index()
    parent_coords = skdata.nodes.ix[skdata.nodes.parent_id.tolist()][
        ['x', 'y', 'z']].reset_index()
    w = np.sqrt(np.sum(
        (tn_coords[['x', 'y', 'z']] - parent_coords[['x', 'y', 'z']]) ** 2, axis=1)).tolist()

    # Get connector details
    cn_details = pymaid.get_connector_details(
        skdata.connectors.connector_id.tolist(), remote_instance=remote_instance)

    list_of_parents = {n[0]: (n[1], n[3], n[4], n[5]) for n in skdata[0]}

    if pre_skid_filter or post_skid_filter:
        # Filter connectors that are both pre- and postsynaptic to the skid in
        # skid_filter
        filtered_cn = [c for c in cn_details.itertuples() if True in [int(
            f) in c.postsynaptic_to for f in post_skid_filter] and True in [int(f) == c.presynaptic_to for f in pre_skid_filter]]
        module_logger.debug('%i of %i connectors left after filtering' % (
            len(filtered_cn), cn_details.shape[0]))
    else:
        filtered_cn = cn_details

    pre_node_distances = {}
    post_node_distances = {}
    visited_nodes = {}

    module_logger.info('Calculating distances to root')
    for i, cn in enumerate(filtered_cn):

        if i % 10 == 0:
            module_logger.debug('%i of %i' % (i, len(filtered_cn)))

        if cn[1]['presynaptic_to'] == int(skdata.skeleton_id) and cn[1]['presynaptic_to_node'] in list_of_parents:
            dist, visited_nodes = _walk_to_root([(n[0], n[3], n[4], n[5]) for n in skdata[
                                                0] if n[0] == cn[1]['presynaptic_to_node']][0], list_of_parents, visited_nodes)

            pre_node_distances[cn[1]['presynaptic_to_node']] = dist

        if int(skdata.skeleton_id) in cn[1]['postsynaptic_to']:
            for nd in cn[1]['postsynaptic_to_node']:
                if nd in list_of_parents:
                    dist, visited_nodes = _walk_to_root([(n[0], n[3], n[4], n[5]) for n in skdata[
                                                        0] if n[0] == nd][0], list_of_parents, visited_nodes)

                    post_node_distances[nd] = dist

    # Reindex dataframe
    df.nodes.reset_index(inplace=True)

    return pre_node_distances, post_node_distances

def arbor_confidence(x, confidences=(1,0.9,0.6,0.4,0.2), inplace=True):
    """ Calculates confidence for each treenode by walking from root to leafs
    starting with a confidence of 1. Each time a low confidence edge is 
    encountered the downstream confidence is reduced (see value parameter).

    Parameters
    ----------
    x :                 {CatmaidNeuron, CatmaidNeuronList}       
                        Neuron(s) to calculate confidence for.
    confidences :       list of five floats, optional
                        Values by which the confidence of the downstream
                        branche is reduced upon encounter of a 5/4/3/2/1-
                        confidence edges.
    inplace :           bool, optional
                        If False, a copy of the neuron is returned.

    Returns
    -------
    Adds "arbor_confidence" column in neuron.nodes.
    """

    def walk_to_leafs( this_node, this_confidence=1 ):
        pbar.update(1)                
        while True:                    
            this_confidence *= confidences[ 5 - x.nodes.loc[ this_node ].confidence ]
            x.nodes.loc[ this_node,'arbor_confidence'] = this_confidence            

            if len(loc[this_node]) > 1:
                for c in loc[this_node]:
                    walk_to_leafs( c, this_confidence )
                break
            elif len(loc[this_node]) == 0:
                break

            this_node = loc[this_node][0]

    if not isinstance(x, ( core.CatmaidNeuron, core.CatmaidNeuronList )):
        raise TypeError('Unable to process data of type %s' % str(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        if not inplace:
            res = [ arbor_confidence(n, confidence=confidence, inplace=inplace) for n in x ]
        else:
            return core.CatmaidNeuronList( [ arbor_confidence(n, confidence=confidence, inplace=inplace) for n in x ] )

    if not inplace:
        x = x.copy()

    loc = generate_list_of_childs(x)   

    x.nodes['arbor_confidence'] = [None] * x.nodes.shape[0] 

    root = x.root
    x.nodes.set_index('treenode_id', inplace=True, drop=True)
    x.nodes.loc[root,'arbor_confidence'] = 1

    with tqdm(total=len(x.slabs),desc='Calc confidence' ) as pbar:
        for c in loc[root]:
            walk_to_leafs(c)

    x.nodes.reset_index(inplace=True, drop=False)

    if not inplace:
        return x

def _calc_dist(v1, v2):
    return math.sqrt(sum(((a - b)**2 for a, b in zip(v1, v2))))


def calc_cable(skdata, smoothing=1, remote_instance=None, return_skdata=False):
    """ Calculates cable length in micrometer (um) of a given neuron     

    Parameters
    ----------
    skdata :            {int, str, CatmaidNeuron, CatmaidNeuronList}       
                        If skeleton ID (str or in), 3D skeleton data will be 
                        pulled from CATMAID server
    smoothing :         int, optional
                        Use to smooth neuron by downsampling. 
                        Default = 1 (no smoothing)                  
    remote_instance :   CATMAID instance, optional
                        Pass if skdata is a skeleton ID.
    return_skdata :     bool, optional
                        If True: instead of the final cable length, a dataframe 
                        containing the distance to each treenode's parent.                         

    Returns
    -------
    cable_length 
                Cable in micrometers [um]

    skdata      
                If return_skdata = True. Neuron object with 
                ``nodes.parent_dist`` containing the distances to parent
    """

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if isinstance(skdata, int) or isinstance(skdata, str):
        skdata = pymaid.get_neuron([skdata], remote_instance).ix[0]

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0]
        elif not return_skdata:
            return sum([calc_cable(skdata.ix[i]) for i in range(skdata.shape[0])])
        else:
            return core.CatmaidNeuronList([calc_cable(skdata.ix[i], return_skdata=return_skdata) for i in range(skdata.shape[0])])
    else:
        raise Exception('Unable to interpret data of type', type(skdata))

    # Copy node data too
    df.nodes = df.nodes.copy()

    if smoothing > 1:
        df = downsample_neuron(df, smoothing)

    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    # Calculate distance to parent for each node
    tn_coords = df.nodes[['x', 'y', 'z']].reset_index()
    parent_coords = df.nodes.ix[[n for n in df.nodes.parent_id.tolist()]][
        ['x', 'y', 'z']].reset_index()

    # Calculate distances between nodes and their parents
    w = np.sqrt(np.sum(
        (tn_coords[['x', 'y', 'z']] - parent_coords[['x', 'y', 'z']]) ** 2, axis=1))

    df.nodes.reset_index(inplace=True)

    if return_skdata:
        df.nodes['parent_dist'] = [v / 1000 for v in list(w)]
        return df

    # #Remove nan value (at parent node) and return sum of all distances
    return np.sum(w[np.logical_not(np.isnan(w))]) / 1000


def cable_within_distance(a, b, increment=1 ):
    """ Calculates the cable of neuron a that is within distance of neuron b.
    This uses distances between treenodes for simplicity: if treenode X of 
    neuron a is within given distance to any treenode of neuron b, then the
    distance between X and parent of X counts as "within distance".

    Parameters
    ----------
    a :         CatmaidNeuron
                Neuron for which to compute cable within distance.
    b :         CatmaidNeuron
                Neuron which needs to be within given distance.
    increment : int, optional
                Increments of distance in [um] to return cable for.

    Returns
    -------
    pandas.DataFrame
            within_distance[um]   percent_cable   total_cable[um]
                1                    5               12              
                2                    15              36
                ...                  ...            ...

    """

    if not isinstance(a, core.CatmaidNeuron) or not isinstance(b, core.CatmaidNeuron):
        raise TypeError('Need to pass CatmaidNeurons')

    #First, generate an all-by-all distance matrix between all points of a neuron
    dist_mat = scipy.spatial.distance.cdist(  a.nodes[['x','y','z']], 
                                              b.nodes[['x','y','z']] )

    # Convert to um
    dist_mat /= 1000

    #Get closest distances
    closest_dist = dist_mat.min(axis=1)

    # Calculate distance to parent for each node
    a = calc_cable(a, smoothing=1, return_skdata=True)

    total_cable = np.sum(a.nodes.parent_dist.values [ np.logical_not( np.isnan( a.nodes.parent_dist.values ) ) ] ) 

    #Now generate DataFrame
    df = pd.DataFrame( [ [  d, 
                            round(a.nodes[ closest_dist <= d ].parent_dist.sum()/total_cable*100, 2),
                            round(a.nodes[ closest_dist <= d ].parent_dist.sum(),2),
                            ]
                        for d in range(0, math.ceil(max(closest_dist)), increment ) ],
                        columns = ['within_distance[um]','percent_cable','total_cable[um]']
                          )

    return df


def calc_strahler_index(skdata, inplace=True, method='standard'):
    """ Calculates Strahler Index -> starts with index of 1 at each leaf, at 
    forks with varying incoming strahler index, the highest index
    is continued, at forks with the same incoming strahler index, highest 
    index + 1 is continued. Starts with end nodes, then works its way from 
    branch nodes to branch nodes up to root node

    Parameters
    ----------
    skdata :      {CatmaidNeuron, CatmaidNeuronList}       
                  E.g. from  ``pymaid.pymaid.get_neuron()``
    inplace :     bool, optional
                  If False, a copy of original skdata is returned.
    method :      {'standard','greedy'}, optional
                  Method used to calculate strahler indices: 'standard' will 
                  use the method described above; 'greedy' will always 
                  increase the index at converging branches whether these
                  branches have the same index or not. This is useful e.g. if
                  you want to cut the neuron at the first branch point.

    Returns
    -------
    skdata
                  With new column ``skdata.nodes.strahler_index``
    """

    module_logger.info('Calculating Strahler indices...')

    start_time = time.time()

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.ix[0]
        else:
            res = []
            for i in trange(0, skdata.shape[0] ):
                res.append(  calc_strahler_index(skdata.ix[i], inplace=inplace, method=method ) ) 

            if not inplace:
                return core.CatmaidNeuronList( res )
            else:
                return

    if not inplace:
        df = df.copy()

    # Make sure dataframe is not indexed by treenode_id for preparing lists
    df.nodes.reset_index(inplace=True, drop=True)

    # Find branch, root and end nodes
    if 'type' not in df.nodes:
        classify_nodes(df)

    end_nodes = df.nodes[df.nodes.type == 'end'].treenode_id.tolist()
    branch_nodes = df.nodes[df.nodes.type == 'branch'].treenode_id.tolist()
    root = df.nodes[df.nodes.type == 'root'].treenode_id.tolist()

    # Generate dicts for childs and parents
    list_of_childs = generate_list_of_childs(skdata)
    #list_of_parents = { n[0]:n[1] for n in skdata[0] }

    # Reindex according to treenode_id
    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    strahler_index = {n: None for n in list_of_childs if n != None}

    starting_points = end_nodes

    nodes_processed = []

    while starting_points:
        module_logger.debug('New starting point. Remaining: %i' %
                            len(starting_points))
        new_starting_points = []
        starting_points_done = []

        for i, en in enumerate(starting_points):
            this_node = en

            module_logger.debug('%i of %i ' % (i, len(starting_points)))

            # Calculate index for this branch
            previous_indices = []
            for child in list_of_childs[this_node]:
                previous_indices.append(strahler_index[child])

            if len(previous_indices) == 0:
                this_branch_index = 1
            elif len(previous_indices) == 1:
                this_branch_index = previous_indices[0]
            elif previous_indices.count(max(previous_indices)) >= 2 or method == 'greedy':
                this_branch_index = max(previous_indices) + 1
            else:
                this_branch_index = max(previous_indices)

            nodes_processed.append(this_node)
            starting_points_done.append(this_node)

            # Now walk down this spine
            # Find parent
            spine = [this_node]

            #parent_node = list_of_parents [ this_node ]
            parent_node = df.nodes.ix[this_node].parent_id

            while parent_node not in branch_nodes and parent_node != None:
                this_node = parent_node
                parent_node = None

                spine.append(this_node)
                nodes_processed.append(this_node)

                # Find next parent
                try:
                    parent_node = df.nodes.ix[this_node].parent_id
                except:
                    # Will fail if at root (no parent)
                    break

            strahler_index.update({n: this_branch_index for n in spine})

            # The last this_node is either a branch node or the root
            # If a branch point: check, if all its childs have already been
            # processed
            node_ready = True
            for child in list_of_childs[parent_node]:
                if child not in nodes_processed:
                    node_ready = False

            if node_ready is True and parent_node != None:
                new_starting_points.append(parent_node)

        # Remove those starting_points that were successfully processed in this
        # run before the next iteration
        for node in starting_points_done:
            starting_points.remove(node)

        # Add new starting points
        starting_points += new_starting_points

    df.nodes.reset_index(inplace=True)

    df.nodes['strahler_index'] = [strahler_index[n]
                                  for n in df.nodes.treenode_id.tolist()]

    module_logger.debug('Done in %is' % round(time.time() - start_time))    

    if not inplace:
        return df


def prune_by_strahler(x, to_prune=range(1, 2), reroot_soma=True, inplace=False, force_strahler_update=False, relocate_connectors=False):
    """ Prune neuron based on strahler order.

    Parameters
    ----------
    x :             {core.CatmaidNeuron, core.CatmaidNeuronList}
    to_prune :      {int, list, range}, optional
                    Strahler indices to prune. 
                    1. ``to_prune = 1`` removes all leaf branches
                    2. ``to_prune = [1,2]`` removes indices 1 and 2
                    3. ``to_prune = range(1,4)`` removes indices 1, 2 and 3  
                    4. ``to_prune = -1`` removes everything but the highest index       
    reroot_soma :   bool, optional
                    If True, neuron will be rerooted to its soma
    inplace :       bool, optional
                    If False, pruning is performed on copy of original neuron
                    which is then returned.  
    relocate_connectors : bool, optional
                          If True, connectors on removed treenodes will be 
                          reconnected to the closest still existing treenode.
                          Works only in child->parent direction.


    Returns
    -------
    pruned neuron
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
        neuron.reroot(neuron.soma)

    if 'strahler_index' not in neuron.nodes or force_strahler_update:
        calc_strahler_index(neuron)

    # Prepare indices
    if isinstance(to_prune, int) and to_prune < 0:
        to_prune = range(1, neuron.nodes.strahler_index.max() + (to_prune + 1))
    elif isinstance(to_prune, int):
        to_prune = [to_prune]
    elif isinstance(to_prune, range):
        to_prune = list(to_prune)

    # Prepare parent dict if needed later
    if relocate_connectors:
        parent_dict = { tn.treenode_id : tn.parent_id for tn in neuron.nodes.itertuples() }

    neuron.nodes = neuron.nodes[
        ~neuron.nodes.strahler_index.isin(to_prune)].reset_index(drop=True)

    if not relocate_connectors:
        neuron.connectors = neuron.connectors[neuron.connectors.treenode_id.isin(
            neuron.nodes.treenode_id.tolist())].reset_index(drop=True)
    else:
        remaining_tns = neuron.nodes.treenode_id.tolist()
        for cn in neuron.connectors[~neuron.connectors.treenode_id.isin(neuron.nodes.treenode_id.tolist())].itertuples():
            this_tn = parent_dict[ cn.treenode_id ]
            while True:
                if this_tn in remaining_tns:
                    break
                this_tn = parent_dict[ this_tn ]
            neuron.connectors.loc[cn.Index,'treenode_id'] = this_tn

    # Remove temporary attributes
    neuron._clear_temp_attr()    

    if not inplace:
        return neuron
    else:
        return


def _walk_to_root(start_node, list_of_parents, visited_nodes):
    """ Helper function for synapse_root_distances(): 
    Walks to root from start_node and sums up geodesic distances along the way.     

    Parameters
    ----------
    start_node :        (node_id, x,y,z)
    list_of_parents :   {node_id: (parent_id, x,y,z) }
    visited_nodes :     {node_id: distance_to_root}
                        Make sure to not walk the same path twice by keeping 
                        track of visited nodes and their distances to soma

    Returns
    -------
    [1] distance_to_root
    [2] updated visited_nodes
    """
    dist = 0
    distances_traveled = []
    nodes_seen = []
    this_node = start_node

    # Walk to root
    while list_of_parents[this_node[0]][0] != None:
        parent = list_of_parents[this_node[0]]
        if parent[0] not in visited_nodes:
            d = _calc_dist(this_node[1:], parent[1:])
            distances_traveled.append(d)
            nodes_seen.append(this_node[0])
        else:
            d = visited_nodes[parent[0]]
            distances_traveled.append(d)
            nodes_seen.append(this_node[0])
            break

        this_node = parent

    # Update visited_nodes
    visited_nodes.update(
        {n: sum(distances_traveled[i:]) for i, n in enumerate(visited_nodes)})

    return round(sum(distances_traveled)), visited_nodes


def in_volume(x, volume, remote_instance=None, inplace=False, mode='IN'):
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

                      1. List/np array -  ``[ ( x, y , z ), [ ... ] ]``
                      2. DataFrame - needs to have 'x','y','z' columns                      

    volume :          {str, list of str, core.Volume} 
                      Name of the CATMAID volume to test OR core.Volume dict
                      as returned by e.g. :func:`pymaid.pymaid.get_volume()`
    remote_instance : CATMAID instance, optional
                      Pass if volume is a volume name
    inplace :         bool, optional
                      If False, a copy of the original DataFrames/Neuron is 
                      returned. Does only apply to CatmaidNeuron or 
                      CatmaidNeuronList objects. Does apply if multiple 
                      volumes are provided
    mode :            {'IN','OUT'}, optional
                      If 'IN', parts of the neuron that are within the volume
                      are kept.

    Returns
    -------
    CatmaidNeuron
                      If input is CatmaidNeuron or CatmaidNeuronList - will
                      return parts of the neuron (nodes and connectors) that
                      are within the volume
    list of bools
                      If input is list or DataFrame - True if in volume, 
                      False if not
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

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if isinstance(volume, list):
        data = dict()
        for v in tqdm(volume, desc='Volumes', disable=module_logger.getEffectiveLevel()>=40):
            data[v] = in_volume(
                x, v, remote_instance=remote_instance, inplace=False, mode=mode)
        return data

    if isinstance(volume, str):
        volume = pymaid.get_volume(volume, remote_instance)        

    if isinstance(x, pd.DataFrame):
        points = x[['x', 'y', 'z']].as_matrix()
    elif isinstance(x, core.CatmaidNeuron):
        n = x

        if not inplace:
            n = n.copy()
            try:
                del n.igraph
            except:
                pass

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

    from pyoctree import pyoctree

    # Generate rays for points
    mx = np.array(volume['vertices']).max(axis=0)
    mn = np.array(volume['vertices']).min(axis=0)

    # Create octree
    tree = pyoctree.PyOctree(np.array(volume['vertices'], dtype=float),
                             np.array(volume['faces'], dtype=np.int32)
                             )
    
    rayPointList = np.array(
                [[[p[0], p[1], mn[2]], [p[0], p[1], mx[2]]] for p in points], dtype=np.float32)

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # by those that occur "above" the point
    intersections = [len([i for i in tree.rayIntersection(ray) if i.p[
                         2] >= points[k][2]])for k, ray in enumerate(rayPointList)]

    # Count odd intersection
    return [i % 2 != 0 for i in intersections]


def _in_volume_convex(points, volume, remote_instance=None, approximate=False, ignore_axis=[]):
    """ Uses scipy to test if points are within a given CATMAID volume.
    The idea is to test if adding the point to the cloud would change the
    convex hull. 
    """

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if type(volume) == type(str()):
        volume = pymaid.get_volume(volume, remote_instance)

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


def stitch_neurons( *neurons, tn_to_stitch=None, method='ALL'):
    """ Function to stich multiple neurons together. The first neuron provided
    will be the master neuron. Unless treenode_ids are provided, neurons will
    be stitched at the closest distance (see method parameter).

    Parameters
    ----------
    neurons :           CatmaidNeuron/CatmaidNeuronList
                        Neurons to stitch.
    tn_to_stitch :      List of treenode IDs, optional
                        If provided, these treenodes will be preferentially
                        used to stitch neurons together. If there are more 
                        than two possible treenodes for a single stitching
                        operation, the two closest are used.
    method :            {'LEAFS','ALL'}, optional
                        Defines automated stitching mode: if 'LEAFS', only
                        leaf (including root) nodes will be considered for
                        stitching. If 'ALL', all treenodes are considered.

    Returns
    -------
    core.CatmaidNeuron
    """

    if method not in ['LEAFS', 'ALL']:
        raise ValueError('Unknown method: %s' % str(method))

    for n in neurons:
        if not isinstance(n, (core.CatmaidNeuron, core.CatmaidNeuronList) ):
            raise TypeError( 'Unable to stitch non-CatmaidNeuron objects' )
        elif isinstance( n, (core.CatmaidNeuronList,list) ):
            neurons += n.neurons

    #Use copies of the original neurons!
    neurons = [ n.copy() for n in neurons if isinstance(n, core.CatmaidNeuron )]

    if len(neurons) < 2:
        raise ValueError('Need at least 2 neurons to stitch, found %i' % len(neurons))

    module_logger.debug('Stitching %i neurons...' % len(neurons))

    stitched_n = neurons[0]

    if tn_to_stitch and not isinstance(tn_to_stitch, (list, np.ndarray)):
        tn_to_stitch = [ tn_to_stitch ]
        tn_to_stitch = [ str(tn) for tn in tn_to_stitch ]

    for nB in neurons[1:]:
        #First find treenodes to connect
        if tn_to_stitch and set(tn_to_stitch) & set(stitched_n.nodes.treenode_id) and set(tn_to_stitch) & set(nB.nodes.treenode_id):
            treenodesA = stitched_n.nodes.set_index('treenode_id').ix[ tn_to_stitch ].reset_index()
            treenodesB = nB.nodes.set_index('treenode_id').ix[ tn_to_stitch ].reset_index()
        elif method == 'LEAFS':
            treenodesA = stitched_n.nodes[ stitched_n.nodes.type.isin(['end','root']) ].reset_index()
            treenodesB = nB.nodes[ nB.nodes.type.isin(['end','root']) ].reset_index()
        else:
            treenodesA = stitched_n.nodes
            treenodesB = nB.nodes

        #Calculate pairwise distances
        pdist = scipy.spatial.distance.cdist( treenodesA[['x','y','z']].values, 
                                              treenodesB[['x','y','z']].values, 
                                              metric='euclidean' )                

        #Get the closest treenodes
        tnA = treenodesA.ix[ pdist.argmin(axis=0)[0] ].treenode_id
        tnB = treenodesB.ix[ pdist.argmin(axis=1)[0] ].treenode_id

        module_logger.info('Stitching treenodes %s and %s' % ( str(tnA), str(tnB) ))

        #Reroot neuronB onto the node that will be stitched
        nB.reroot( tnB )

        #Change neuronA root node's parent to treenode of neuron B
        nB.nodes.loc[ nB.nodes.parent_id.isnull(), 'parent_id' ] = tnA

        #Add nodes, connectors and tags onto the stitched neuron
        stitched_n.nodes = pd.concat( [ stitched_n.nodes, nB.nodes ], ignore_index=True )
        stitched_n.connectors = pd.concat( [ stitched_n.connectors, nB.connectors ], ignore_index=True )
        stitched_n.tags.update( nB.tags )

    #Reset temporary attributes of our final neuron
    stitched_n._clear_temp_attr()

    return stitched_n

















