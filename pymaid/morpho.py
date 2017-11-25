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
import scipy.interpolate
from tqdm import tqdm, trange
import warnings
import itertools
from igraph import Graph

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

try:
    from pyoctree import pyoctree
except:
    module_logger.warning("Module pyoctree not found. Falling back to scipy's ConvexHull for intersection calculations.")

__all__ = sorted([ 'calc_cable','calc_strahler_index','classify_nodes','cut_neuron',
            'downsample_neuron','in_volume','longest_neurite',
            'prune_by_strahler','reroot_neuron','synapse_root_distances',
            'calc_overlap','stitch_neurons','arbor_confidence',
            'split_axon_dendrite', 'distal_to', 'calc_bending_flow', 'calc_flow_centrality',
            'calc_segregation_index', 'filter_connectivity','dist_between',
            'to_dotproduct','resample_neuron'])

def generate_list_of_childs(skdata):
    """ Transforms list of nodes into a dictionary { parent: [child1,child2,...]}

    Parameters
    ----------
    skdata :   {CatmaidNeuron,CatmaidNeuronList} 
               Must contain a SINGLE neuron.

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
            nodes = skdata.loc[0].nodes
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
    and root.

    Parameters
    ----------
    skdata :    {CatmaidNeuron,CatmaidNeuronList} 
                Neuron(s) to classify nodes for.
    inplace :   bool, optional 
                If False, nodes will be classified on a copy which is then 
                returned.

    Returns
    -------
    skdata 
               Only if ``inplace=False``. Added columns 'type' and 
               'has_connectors' to skdata.nodes.

    """    

    if not inplace:
        skdata = skdata.copy()

    # If more than one neuron
    if isinstance(skdata, (pd.DataFrame, core.CatmaidNeuronList)):        
        for i in trange(skdata.shape[0], desc='Classifying'):
            classify_nodes(skdata.ix[i], inplace=True)
    elif isinstance(skdata, (pd.Series, core.CatmaidNeuron)):
        # Get iGraph representation of neuron
        g = skdata.igraph
        # Get branch and end nodes based on their degree of connectivity 
        ends = g.vs.select(_degree=1)['node_id']        
        branches = g.vs.select(_degree_gt=2)['node_id']
        
        skdata.nodes['type'] = 'slab'
        skdata.nodes.loc[ skdata.nodes.treenode_id.isin(ends), 'type' ] = 'end'
        skdata.nodes.loc[ skdata.nodes.treenode_id.isin(branches), 'type' ] = 'branch'
        skdata.nodes.loc[ skdata.nodes.parent_id.isnull(), 'type' ] = 'root'
        
        skdata.nodes.loc[:,'has_connectors'] = False
        skdata.nodes.loc[ skdata.nodes.treenode_id.isin( skdata.connectors.treenode_id ), 'has_connectors' ] = True
    else:
        raise TypeError('Unknown neuron type: %s' % str(type(skdata)))

    if not inplace:
        return skdata

def _generate_segments(x, append=True):
    """ Generate segments for a given neuron.

    Parameters
    ----------
    x :         {CatmaidNeuron,CatmaidNeuronList} 
                May contain multiple neurons.
    append :    bool, optional 
                If True slabs will be appended to neuron.

    Returns
    -------
    list      
                Slabs as list of lists containing treenode ids.
    """

    if isinstance(x, pd.DataFrame) or isinstance(x, core.CatmaidNeuronList):
        return [_generate_segments(x.loc[i], append=append) for i in range(x.shape[0])]
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

def _resample_neuron_spline(x, resample_to, inplace=False):
    """ Resamples neuron(s) by given resolution. Uses spline interpolation.

    Important
    ---------
    Currently, this function replaces treenodes without mapping e.g. synapses
    or tags back!

    Parameters
    ----------
    x :                 {CatmaidNeuron,CatmaidNeuronList} 
                             Neuron(s) to downsample.
    resampling_factor :      int 
                             Factor by which to reduce the node count.
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
    x
                         Downsampled Pandas Dataframe or CatmaidNeuron object
    """    
    if isinstance(x, core.CatmaidNeuronList):
        results = [ resample_neuron(x.loc[i], resample_to, inplace=inplace) for i in range(x.shape[0]) ]
        if not inplace:
            return core.CatmaidNeuronList( results )    
    elif not isinstance(x, core.CatmaidNeuron):
        module_logger.error('Unexpected datatype: %s' % str(type(x)))
        raise ValueError

    if not inplace:
        x = x.copy()

    # Prepare nodes for subsetting
    nodes = x.nodes.set_index('treenode_id')
    
    # Iterate over segments
    for i,seg in enumerate(tqdm(x.segments, desc='Working on segments')):
        # Get length of this segment
        this_length = x.igraph.shortest_paths( x.igraph.vs.select(node_id=seg[0])[0], 
                                               x.igraph.vs.select(node_id=seg[-1])[0], 
                                               weights='weight')[0][0]
        
        if this_length < resample_to or len(seg) <= 3:
            continue

        # Get new number of points on the spline
        n_nodes = int(this_length / resample_to)        

        # Get all coordinates of all nodes in this segment
        coords = nodes.loc[ seg, ['x','y','z'] ].values

        # Transpose to get in right format
        data = coords.T.astype(float)

        try:
            # Get all knots and info about the interpolated spline
            tck, u = scipy.interpolate.splprep(data)
        except:
            module_logger.warning('Error downsampling segment {0} ({1} nodes, {2} nm) '.format(i, len(seg), int(this_length), n_nodes))
            continue

        # Interpolate to new resolution
        new_coords = scipy.interpolate.splev(np.linspace(0,1,n_nodes), tck)  

        # Change back into x/y/z
        new_coords = np.array( new_coords ).T.round()

        # Now that we have new coordinates, we need to "rewire" the neuron
        # First, add new treenodes (we're starting at the distal node!) and
        # discard every treenode but the first and the last
        
        max_tn_id = x.nodes.treenode_id.max() + 1        
        new_ids = seg[:1] + [ max_tn_id + i  for i in range( len(new_coords) - 2 ) ] + seg[-1:]

        new_nodes = pd.DataFrame( [ [ tn, pn, None , co[0], co[1], co[2], -1, 5, 'slab', False ] for tn, pn,co in zip( new_ids[:-1], new_ids[1:], new_coords[:-1] ) ],
                                  columns=['treenode_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence', 'type', 'has_connectors'] )
        new_nodes.loc[0, 'type'] = x.nodes.set_index('treenode_id').loc[ seg[0], 'type' ]       

        # Remove old treenodes
        x.nodes = x.nodes[ ~x.nodes.treenode_id.isin( seg[:-1] ) ]

        # Append new treenodes
        x.nodes = x.nodes.append( new_nodes, ignore_index=True )

    x._clear_temp_attr()

    if not inplace:
        return x

def resample_neuron(x, resample_to, method='linear', inplace=False):
    """ Resamples neuron(s) to given resolution. Preserves root, leafs, 
    branchpoints. 

    Important
    ---------
    Currently, this function generate new treenodes without mapping synapses
    or tags back onto them!

    Parameters
    ----------
    x :                 {CatmaidNeuron,CatmaidNeuronList} 
                        Neuron(s) to resample.
    resampling_factor : int 
                        Factor by which to reduce the node count.
    method :            str, optional
                        See `scipy.interpolate.interp1d` for possible options.
                        By default, we're using linear interpolation.
    inplace :           bool, optional   
                        If True, will modify original neuron. If False, a 
                        resampled copy is returned. 

    Returns
    -------
    x
                        Downsampled CatmaidNeuron/List object

    See Also
    --------
    :func:`pymaid.downsample_neuron`
                        This function reduces the number of nodes instead of
                        resample to certain resolution. Usefull if you are 
                        just after some simplification e.g. for speeding up 
                        your calculations or you want to preserve more of a 
                        neuron's strucutre.
    """   

    if isinstance(x, core.CatmaidNeuronList):
        results = [ resample_neuron(x.loc[i], resample_to, inplace=inplace) 
                        for i in trange(x.shape[0], desc='Resampl. neurons', disable=module_logger.getEffectiveLevel()>=40 ) ]
        if not inplace:
            return core.CatmaidNeuronList( results )    
    elif not isinstance(x, core.CatmaidNeuron):
        module_logger.error('Unexpected datatype: %s' % str(type(x)))
        raise ValueError

    if not inplace:
        x = x.copy()

    nodes = x.nodes.set_index('treenode_id')

    new_nodes = []
    max_tn_id = x.nodes.treenode_id.max() + 1 

    # Iterate over segments
    for i,seg in enumerate(tqdm(x.segments, desc='Proc. segments', disable=module_logger.getEffectiveLevel()>=40 )):
        coords = nodes.loc[ seg, ['x','y','z'] ].values.astype(float)
        
        # vecs between subsequently measured points
        vecs = np.diff(coords.T)

        # path: cum distance along points (norm from first to ith point)
        path = np.cumsum(np.linalg.norm(vecs, axis=0))
        path = np.insert(path, 0, 0)

        # If path is too short, just use first and last treenode
        if path[-1] < resample_to:
            new_nodes += [[ seg[0], seg[-1], None, coords[0][0], coords[0][1], coords[0][2], -1, 5 ]]
            continue

        # Coords of interpolation
        n_nodes = int(path[-1] / resample_to)
        interp_coords = np.linspace(path[0], path[-1], n_nodes) 

        # Interpolation func for each axis with the path
        sampleX = scipy.interpolate.interp1d(path, coords[:,0], kind=method)
        sampleY = scipy.interpolate.interp1d(path, coords[:,1], kind=method)
        sampleZ = scipy.interpolate.interp1d(path, coords[:,2], kind=method)

        # Sample each dim
        xnew = sampleX(interp_coords)
        ynew = sampleY(interp_coords)
        znew = sampleZ(interp_coords)

        # Generate new coordinates
        new_coords = np.array([xnew, ynew, znew]).T.round()
               
        # Generate new ids
        new_ids = seg[:1] + [ max_tn_id + i  for i in range( len(new_coords) - 2 ) ] + seg[-1:]        

        # Keep track of new nodes
        new_nodes += [ [ tn, pn, None , co[0], co[1], co[2], -1, 5, ] for tn, pn,co in zip( new_ids[:-1], new_ids[1:], new_coords ) ]

        # Increase max index
        max_tn_id += len(new_ids)        

    # Add root node
    root = nodes.loc[ x.root ]
    new_nodes += [ [ x.root, root.parent_id, root.creator_id, root.x, root.y, root.z, root.radius, root.confidence ] ]

    # Generate new nodes dataframe
    new_nodes = pd.DataFrame( data = new_nodes,
                              columns=['treenode_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence'],
                              dtype=object
                               )
   
    # Set nodes
    x.nodes = new_nodes

    # Remove duplicate treenodes (branch points)
    x.nodes = x.nodes[ ~x.nodes.treenode_id.duplicated() ]

    # Clear attributes
    x._clear_temp_attr()

    # Reclassfiy nodes
    classify_nodes(x, inplace=True)

    if not inplace:
        return x


def downsample_neuron(skdata, resampling_factor, inplace=False, preserve_cn_treenodes=True, preserve_tag_treenodes=False):
    """ Downsamples neuron(s) by a given factor. Preserves root, leafs, 
    branchpoints by default. Preservation of treenodes with synapses can
    be toggled.

    Parameters
    ----------
    skdata :                 {CatmaidNeuron,CatmaidNeuronList} 
                             Neuron(s) to downsample.
    resampling_factor :      int 
                             Factor by which to reduce the node count.
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
                             Downsampled Pandas Dataframe or CatmaidNeuron 
                             object.

    See Also
    --------
    :func:`pymaid.downsample_neuron`
                             This function resamples a neuron to given 
                             resolution.
    """

    if isinstance(skdata, pd.DataFrame):
        return pd.DataFrame([downsample_neuron(skdata.loc[i], resampling_factor, inplace=inplace) for i in range(skdata.shape[0])])
    elif isinstance(skdata, core.CatmaidNeuronList):
        return core.CatmaidNeuronList([downsample_neuron(skdata.loc[i], resampling_factor, inplace=inplace) for i in range(skdata.shape[0])])
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

    # Add soma node
    if not isinstance( df.soma, type(None) ) and df.soma not in fix_points:   
        fix_points = np.append( fix_points, df.soma )

    # Walk from all fix points to the root - jump N nodes on the way
    new_parents = {}

    module_logger.info('Sampling neuron down by factor of %i' %
                       resampling_factor)
    for en in fix_points:
        this_node = en

        while True:
            stop = False
            new_p = list_of_parents[this_node]
            if new_p != None:
                for i in range(resampling_factor):
                    if new_p in fix_points:
                        new_parents[this_node] = new_p
                        stop = True
                        break
                    else:
                        new_p = list_of_parents[new_p]

                if stop is True:
                    break
                else:
                    new_parents[this_node] = new_p
                    this_node = new_p
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


def longest_neurite(x, reroot_to_soma=False, inplace=False):
    """ Returns a neuron consisting only of the longest neurite (based on 
    geodesic distance).

    Parameters
    ----------
    x :                 {CatmaidNeuron,CatmaidNeuronList} 
                        May contain multiple neurons.
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

    if isinstance(x, pd.Series) or isinstance(x, core.CatmaidNeuron):
        x = x
    elif isinstance(x, pd.DataFrame) or isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x.loc[0]
        else:
            module_logger.error(
                '%i neurons provided. Please provide only a single neuron!' % x.shape[0])
            raise Exception

    if not inplace:
        x = x.copy()

    if reroot_to_soma and x.soma:
        x.reroot( x.soma )       

    #First collect leafs and root
    leaf_nodes = x.nodes[x.nodes.type=='end'].treenode_id.tolist()
    root_nodes = x.nodes[x.nodes.type=='root'].treenode_id.tolist()

    #Convert to igraph vertex indices
    leaf_ix = [ v.index for v in x.igraph.vs if v['node_id'] in leaf_nodes ]
    root_ix = [ v.index for v in x.igraph.vs if v['node_id'] in root_nodes ]

    #Now get paths from all tips to the root
    paths = x.igraph.get_shortest_paths( root_ix[0], leaf_ix, mode='ALL' )

    #Translate indices back into treenode ids
    paths_tn = [ [ x.igraph.vs[i]['node_id'] for i in p ] for p in paths ]    

    #Generate DataFrame with all the info
    path_df = pd.DataFrame( [ [p, x.nodes.set_index('treenode_id').loc[p].reset_index() ] for p in paths_tn ],
                            columns=['path','nodes']  )

    #Now calculate cable of each of the paths
    path_df['cable'] = [ calc_cable(path_df.loc[i], return_skdata=False, smoothing=1) for i in range(path_df.shape[0])]    

    tn_to_preverse = path_df.sort_values('cable', ascending=False).reset_index().loc[0].path
    
    x.nodes = x.nodes[x.nodes.treenode_id.isin(
        tn_to_preverse)].reset_index(drop=True)

    # Reset indices of node and connector tables (important for igraph!)
    x.nodes.reset_index(inplace=True,drop=True)
    x.connectors.reset_index(inplace=True,drop=True)

    if not inplace:
        return x


def reroot_neuron(skdata, new_root, g=None, inplace=False):
    """ Uses igraph to reroot the neuron at given point. 

    Parameters
    ----------
    skdata :   {CatmaidNeuron, CatmaidNeuronList} 
               Must contain a SINGLE neuron.
    new_root : {int, str}
               Node ID or tag of the node to reroot to.
    inplace :  bool, optional
               If True the input neuron will be rerooted.

    Returns
    -------
    pandas.Series or CatmaidNeuron object
               Containing the rerooted neuron.

    See Also
    --------
    :func:`~pymaid.CatmaidNeuron.reroot`
                Quick access to reroot directly from CatmaidNeuron/List objects
    """

    if new_root == None:
        raise ValueError('New root can not be <None>')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.loc[0]
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

    if df.nodes.set_index('treenode_id').loc[new_root].parent_id == None:
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
    old_root = df.nodes.set_index('parent_id').loc[None, 'treenode_id']
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
    >>> skeleton_dataframe = pymaid.get_neuron(skeleton_id,remote_instance)   
    >>> # First cut
    >>> nA, nB = cut_neuron2( skeleton_data, cut_node1 )
    >>> # Second cut
    >>> nD, nE = cut_neuron2( nA, cut_node2 )  

    See Also
    --------
    :func:`~pymaid.CatmaidNeuron.prune_distal_to`
    :func:`~pymaid.CatmaidNeuron.prune_proximal_to`

    """
    start_time = time.time()

    module_logger.info('Cutting neuron...')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata.copy()
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.loc[0].copy()
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

    # Select the cut node's parent (works only if it is not already at root)
    if cut_node != df.root:
        parent_node_index = g.es.select(_source=cut_node_index)[0].target
    # If we are at root but root is a fork, try using one of its child
    elif df.nodes[ df.nodes.parent_id == df.root ].shape[0] > 1:
        parent_node_index = cut_node_index
        cut_node_index = [ v.index for v in df.igraph.vs if v['parent_id'] == cut_node ][0]
    # If that also fails: throw exception
    else:
        module_logger.error(
            'Unable to find parent for cut node. Is cut node = root?')
        #parent_node_index = g.es.select(_target=cut_node_index)[0].source
        raise Exception('Unable to cut: Is cut node = root?')

    # Now calculate the min cut
    mc = g.st_mincut(parent_node_index, cut_node_index, capacity=None)

    # mc.partition holds the two partitions with mc.partition[0] holding the 
    # part with the source and mc.partition[1] the part with the target
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
        df.nodes.loc[dist_partition_ids],
        df.connectors[
            [c.treenode_id in dist_partition_ids for c in df.connectors.itertuples()]].reset_index(),
        df.tags,
        dist_graph
    ]],
        columns=['neuron_name', 'skeleton_id',
                 'nodes', 'connectors', 'tags', 'igraph'],
        dtype=object
    ).loc[0]

    neuron_dist.nodes.loc[cut_node, 'parent_id'] = None

    neuron_prox = pd.DataFrame([[
        df.neuron_name + '_prox',
        df.skeleton_id,
        df.nodes.loc[prox_partition_ids],
        df.connectors[
            [c.treenode_id not in dist_partition_ids for c in df.connectors.itertuples()]].reset_index(),
        df.tags,
        prox_graph
    ]],
        columns=['neuron_name', 'skeleton_id',
                 'nodes', 'connectors', 'tags', 'igraph'],
        dtype=object
    ).loc[0]

    # Reclassify cut node in distal as 'root' and its parent in proximal as
    # 'end' (unless cut was made at the root node)
    if 'type' in df.nodes:
        neuron_dist.nodes.loc[cut_node, 'type'] = 'root'
        if cut_node != df.nodes[df.nodes.parent_id.isnull()].index[0]:
            neuron_prox.nodes.loc[df.nodes.loc[cut_node,'parent_id'], 'type'] = 'end'
        else:            
            neuron_prox.nodes = pd.concat( [ neuron_prox.nodes, neuron_dist.nodes.loc[ [cut_node] ] ] )

    # Now reindex dataframes
    neuron_dist.nodes.reset_index(inplace=True)
    neuron_prox.nodes.reset_index(inplace=True)
    df.nodes.reset_index(inplace=True)

    module_logger.debug('Cutting finished in {0}s'.format(
                        round(time.time() - start_time)) )
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

        n_prox = df.copy()
        n_prox.neuron_name += '_prox'
        #n_prox.nodes = df.nodes.ix[prox_partition_ids].copy()
        n_prox.nodes = neuron_prox.nodes
        n_prox.connectors = df.connectors[
            [c.treenode_id in prox_partition_ids for c in df.connectors.itertuples()]].reset_index().copy()
        n_prox.igraph = prox_graph.copy()        

        return n_dist, n_prox


def _cut_neuron(skdata, cut_node):
    """ DEPRECATED! Cuts a neuron at given point and returns two new neurons. 
    Does not use igraph (slower).

    Parameters
    ----------
    skdata :   {CatmaidNeuron, CatmaidNeuronList} 
               Must contain a SINGLE neuron.
    cut_node : {int, str}    
               Node ID or a tag of the node to cut.

    Returns
    -------
    neuron_dist
               Neuron object distal to the cut.
    neuron_prox
               Neuron object proximal to the cut.
    """
    start_time = time.time()

    module_logger.info('Preparing to cut neuron...')

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata.copy()
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.loc[0].copy()
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
        df.nodes.loc[distal_nodes],
        df.connectors[
            [c.treenode_id in distal_nodes for c in df.connectors.itertuples()]].reset_index(),
        df.tags
    ]],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    ).loc[0]

    neuron_dist.nodes.loc[cut_node,'parent_id'] = None
    neuron_dist.nodes.loc[cut_node,'type'] = 'root'

    neuron_prox = pd.DataFrame([[
        df.neuron_name + '_prox',
        df.skeleton_id,
        df.nodes.loc[proximal_nodes],
        df.connectors[
            [c.treenode_id not in distal_nodes for c in df.connectors.itertuples()]].reset_index(),
        df.tags
    ]],
        columns=['neuron_name', 'skeleton_id', 'nodes', 'connectors', 'tags'],
        dtype=object
    ).loc[0]

    # Reclassify cut node in proximal neuron as end node
    neuron_prox.nodes.loc[cut_node,'type'] = 'end'

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
    (i.e. soma).

    Parameters
    ----------  
    skdata :            {CatmaidNeuron, CatmaidNeuronList} 
                        Must contain a SINGLE neuron.
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
            df = skdata.loc[0]
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
    parent_coords = skdata.nodes.loc[skdata.nodes.parent_id.tolist(),
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
    Adds ``arbor_confidence`` column in neuron.nodes.
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

    with tqdm(total=len(x.slabs),desc='Calc confidence', disable=module_logger.getEffectiveLevel()>=40 ) as pbar:
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
                        pulled from CATMAID server.
    smoothing :         int, optional
                        Use to smooth neuron by downsampling. 
                        Default = 1 (no smoothing) .                 
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
                ``nodes.parent_dist`` containing the distances to parent.
    """

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if isinstance(skdata, int) or isinstance(skdata, str):
        skdata = pymaid.get_neuron([skdata], remote_instance).loc[0]

    if isinstance(skdata, pd.Series) or isinstance(skdata, core.CatmaidNeuron):
        df = skdata
    elif isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):
        if skdata.shape[0] == 1:
            df = skdata.loc[0]
        elif not return_skdata:
            return sum([calc_cable(skdata.loc[i]) for i in range(skdata.shape[0])])
        else:
            return core.CatmaidNeuronList([calc_cable(skdata.loc[i], return_skdata=return_skdata) for i in range(skdata.shape[0])])
    else:
        raise Exception('Unable to interpret data of type', type(skdata))

    # Copy node data too
    df.nodes = df.nodes.copy()

    # Catch single-node neurons
    if df.nodes.shape[0] == 1:
        if return_skdata:
            df.nodes['parent_dist'] = 0
            return df
        else:
            return 0

    if smoothing > 1:
        df = downsample_neuron(df, smoothing)

    if df.nodes.index.name != 'treenode_id':
        df.nodes.set_index('treenode_id', inplace=True)

    # Calculate distance to parent for each node
    tn_coords = df.nodes[['x', 'y', 'z']].reset_index()
    parent_coords = df.nodes.loc[[n for n in df.nodes.parent_id.tolist()],
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


def to_dotproduct(x):
    """ Converts a neuron's neurites into dotproducts consisting of a point
    and a vector. This works by (1) finding the center between child->parent
    treenodes and (2) getting the vector between them. Also returns the length
    of the vector.

    Parameters
    ----------
    x :         {CatmaidNeuron}
                Single neuron

    Returns
    -------
    pandas.DataFrame
                point  vector  vec_length

    Examples
    --------
    >>> x = pymaid.get_neurons(16)
    >>> dps = pymaid.to_dps(x)
    >>> # Get array of all locations
    >>> locs = numpy.vstack(dps.point.values)

    """

    if isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] == 1:
            x = x[0]
        else:
            raise ValueError('Please pass only single CatmaidNeurons')

    if not isinstance(x, core.CatmaidNeuron ):
        raise ValueError('Can only process CatmaidNeurons')    

    # First, get a list of child -> parent locs (exclude root node!)
    tn_locs = x.nodes[ ~x.nodes.parent_id.isnull() ][['x','y','z']].values
    pn_locs = x.nodes.set_index('treenode_id').loc[ x.nodes[ ~x.nodes.parent_id.isnull() ].parent_id ][['x','y','z']].values
    
    # Get centers between each pair of locs
    centers = tn_locs + ( pn_locs - tn_locs ) / 2

    # Get vector between points
    vec = pn_locs - tn_locs

    dps = pd.DataFrame( [ [ c, v] for c,v in zip(centers,vec) ], columns=['point','vector'] )

    # Add length of vector (for convenience)
    dps['vec_length'] = (dps.vector ** 2).apply(sum).apply(math.sqrt)

    return dps


def calc_overlap(a, b, dist=2, method='avg' ):
    """ Calculates the amount of cable of neurons A within distance of neuron b.
    This uses distances between treenodes for simplicity: if treenode X of 
    neuron a is within given distance to any treenode of neuron b, then the
    distance between X and parent of X counts as "within distance". This uses
    the dotprodut representation of a neuron!

    Parameters
    ----------
    a :         {CatmaidNeuron, CatmaidNeuronList}
                Neuron(s) for which to compute cable within distance.
    b :         {CatmaidNeuron, CatmaidNeuronList}
                Neuron(s) which needs to be within given distance.
    dist :      int, optional
                Distance in microns to return cable for.
    method :    {'min','max','avg'}
                Method by which to calculate the overlapping cable between
                the nearest two segments.

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

    # TODO:   

    if not isinstance(a, (core.CatmaidNeuron, core.CatmaidNeuronList)) or not isinstance(b, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        raise TypeError('Need to pass CatmaidNeurons')

    if isinstance(a, core.CatmaidNeuron):
        a = core.CatmaidNeuronList(a)

    if isinstance(b, core.CatmaidNeuron):
        b = core.CatmaidNeuronList(b)

    allowed_methods = ['min','max','avg']
    if method not in allowed_methods:
        raise ValueError('Unknown method "{0}". Allowed methods: "{0}"'.format(method, ','.join(allowed_methods)))

    matrix = pd.DataFrame( np.zeros(( a.shape[0], b.shape[0] )), index=a.skeleton_id, columns=b.skeleton_id )

    with tqdm(total=len(a)*len(b), desc='Calc. overlap', disable=module_logger.getEffectiveLevel()>=40) as pbar:
        for nA in a:
            for nB in b:
                #First, generate an all-by-all distance matrix between all points of both neurons
                dist_mat = scipy.spatial.distance.cdist( np.vstack( nA.dps.point ), 
                                                         np.vstack( nB.dps.point ) )

                # Convert to um
                dist_mat /= 1000

                #Get closest distances
                closest_dist = dist_mat.min(axis=1)

                # Get indices of closest distances
                closest_ix = dist_mat.argmin(axis=1)

                # Get vectors to match up
                to_add_nA = nA.dps.loc[ closest_dist <= dist ]
                to_add_nB = nB.dps.loc[ closest_ix[ closest_dist <= dist ] ]

                vec_lengths = np.array( [ [l,r] for l,r in zip( to_add_nA.vec_length.values, to_add_nB.vec_length.values ) ] )

                if not vec_lengths.any():
                    overlap = 0
                elif method == 'avg':
                    overlap = vec_lengths.mean(axis=1).sum() 
                elif method == 'max':
                    overlap = vec_lengths.max(axis=1).sum() 
                elif method == 'min':
                    overlap = vec_lengths.min(axis=1).sum() 

                # Convert to um
                overlap /= 1000

                matrix.at[ nA.skeleton_id, nB.skeleton_id ] = overlap

                pbar.update( 1 )

    return matrix


def calc_strahler_index(skdata, inplace=True, method='standard'):
    """ Calculates Strahler Index -> starts with index of 1 at each leaf, at 
    forks with varying incoming strahler index, the highest index
    is continued, at forks with the same incoming strahler index, highest 
    index + 1 is continued. Starts with end nodes, then works its way from 
    branch nodes to branch nodes up to root node

    Parameters
    ----------
    skdata :      {CatmaidNeuron, CatmaidNeuronList}       
                  E.g. from  ``pymaid.get_neuron()``.
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
            df = skdata.loc[0]
        else:
            res = []
            for i in trange(0, skdata.shape[0] ):
                res.append(  calc_strahler_index(skdata.loc[i], inplace=inplace, method=method ) ) 

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
            parent_node = df.nodes.loc[this_node,'parent_id']

            while parent_node not in branch_nodes and parent_node != None:
                this_node = parent_node
                parent_node = None

                spine.append(this_node)
                nodes_processed.append(this_node)

                # Find next parent
                try:
                    parent_node = df.nodes.loc[this_node,'parent_id']
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
                    Strahler indices to prune:

                      (1) ``to_prune=1`` removes all leaf branches
                      (2) ``to_prune=[1,2]`` removes indices 1 and 2
                      (3) ``to_prune=range(1,4)`` removes indices 1, 2 and 3  
                      (4) ``to_prune=s-1`` removes everything but the highest 
                          index       
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

    # Reset indices of node and connector tables (important for igraph!)
    neuron.nodes.reset_index(inplace=True,drop=True)
    neuron.connectors.reset_index(inplace=True,drop=True)

    # Theoretically we can end up with disconnected pieces, i.e. with more than 1 root node 
    # We have to fix the nodes that lost their parents
    neuron.nodes.loc[ ~neuron.nodes.parent_id.isin( neuron.nodes.treenode_id.tolist() ), 'parent_id' ] = None

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


def in_volume(x, volume, inplace=False, mode='IN', remote_instance=None):
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

                      1. List/np array -  ``[ ( x,y,z ), ( ... ) ]``
                      2. DataFrame - needs to have 'x','y','z' columns                      

    volume :          {str, list of str, core.Volume} 
                      Name of the CATMAID volume to test OR core.Volume dict
                      as returned by e.g. :func:`~pymaid.get_volume()`    
    inplace :         bool, optional
                      If False, a copy of the original DataFrames/Neuron is 
                      returned. Does only apply to CatmaidNeuron or 
                      CatmaidNeuronList objects. Does apply if multiple 
                      volumes are provided
    mode :            {'IN','OUT'}, optional
                      If 'IN', parts of the neuron that are within the volume
                      are kept.
    remote_instance : CATMAID instance, optional
                      Pass if volume is a volume name

    Returns
    -------
    CatmaidNeuron
                      If input is CatmaidNeuron or CatmaidNeuronList, will
                      return parts of the neuron (nodes and connectors) that
                      are within the volume
    list of bools
                      If input is list or DataFrame, returns boolean: ``True`` 
                      if in volume, ``False`` if not
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

    if isinstance(volume, (list, dict, np.ndarray)) and not isinstance(volume, core.Volume):
        #Turn into dict 
        if not isinstance(volume, dict):
            volume = { v['name'] : v for v in volume }

        data = dict()
        for v in tqdm(volume, desc='Volumes', disable=module_logger.getEffectiveLevel()>=40):
            data[v] = in_volume(
                x, volume[v], remote_instance=remote_instance, inplace=False, mode=mode)
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

        # Reset indices of node and connector tables (important for igraph!)
        n.nodes.reset_index(inplace=True,drop=True)
        n.connectors.reset_index(inplace=True,drop=True)

        # Theoretically we can end up with disconnected pieces, i.e. with more than 1 root node 
        # We have to fix the nodes that lost their parents
        n.nodes.loc[ ~n.nodes.parent_id.isin( n.nodes.treenode_id.tolist() ), 
                          'parent_id' ] = None

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
    
    if 'pyoctree' in volume:
        # Use store octree if available
        tree = volume['pyoctree']
    else:
        # Create octree from scratch
        tree = pyoctree.PyOctree(np.array(volume['vertices'], dtype=float),
                                 np.array(volume['faces'], dtype=np.int32)
                                 )
        volume['pyoctree'] = tree

    # Generate rays for points
    mx = np.array(volume['vertices']).max(axis=0)
    mn = np.array(volume['vertices']).min(axis=0)
    
    rayPointList = np.array(
                [[[p[0], p[1], mn[2]], [p[0], p[1], mx[2]]] for p in points], dtype=np.float32)

    # Unfortunately rays are bidirectional -> we have to filter intersections
    # by those that occur "above" the point
    intersections = [len([i for i in tree.rayIntersection(ray) if i.p[
                         2] >= points[k][2]])for k, ray in enumerate( tqdm(rayPointList, 
                                                                           desc='Calc. intersections',
                                                                           disable=module_logger.getEffectiveLevel()>=40
                                                                        ))]

    # Count odd intersection
    return [i % 2 != 0 for i in intersections]

def split_axon_dendrite(x, method='centrifugal', primary_neurite=True, reroot_soma=True, return_point=False ):
    """ This function tries to split a neuron into axon, dendrite and primary
    neurite. The result is highly depending on the method and on your 
    neuron's morphology and works best for "typical" neurons, i.e. those where
    the primary neurite branches into axon and dendrites. 
    See :func:`~pymaid.calc_flow_centrality` for details on the flow
    centrality algorithm.

    Parameters
    ----------
    x :                 CatmaidNeuron
                        Neuron to split into axon, dendrite and primary neurite
    method :            {'centrifugal','centripetal','sum', 'bending'}, optional
                        Type of flow centrality to use to split the neuron. 
                        There are four flavors: the first three refer to
                        :func:`~pymaid.calc_flow_centrality`, the last 
                        refers to :func:`~pymaid.calc_bending_flow`.

                        Will try using stored centrality, if possible.
    primary_neurite :   bool, optional
                        If True and the split point is at a branch point, will
                        split into axon, dendrite and primary neurite.
    reroot_soma :       bool, optional
                        If True, will make sure neuron is rooted to soma if at 
                        all possible.
    return_point :      bool, optional
                        If True, will only return treenode ID of the node at which
                        to split the neuron.    

    Returns
    -------
    CatmaidNeuronList
        Contains Axon, Dendrite and primary neurite    

    Examples
    --------
    >>> x = pymaid.get_neuron(123456)
    >>> split = pymaid.split_axon_dendrite(x, method='centrifugal', reroot_soma=True)
    >>> split    
    <class 'pymaid.CatmaidNeuronList'> of 3 neurons 
                          neuron_name skeleton_id  n_nodes  n_connectors  
    0  neuron 123457_primary_neurite          16      148             0   
    1             neuron 123457_axon          16     9682          1766   
    2         neuron 123457_dendrite          16     2892           113 
    >>> # Plot in their respective colors
    >>> for n in split:
    >>>   n.plot3d(color=self.color)     

    """

    if isinstance(x, core.CatmaidNeuronList) and len(x) == 1:
        x = x[0]

    if not isinstance(x, core.CatmaidNeuron):
        raise TypeError('Can only process a single CatmaidNeuron')

    if method not in ['centrifugal','centripetal','sum','bending']:
        raise ValueError('Unknown parameter for mode: {0}'.format(mode))

    if x.soma and x.soma != x.root and reroot_soma:
        x.reroot(x.soma)

    # Calculate flow centrality if necessary
    try:
        last_method = x.centrality_method
    except:
        last_method = None
    
    if last_method != method: 
        if method == 'bending':
            _ = calc_bending_flow(x)
        else:
            _ = calc_flow_centrality(x, mode = method)

    #Make copy, so that we don't screw things up
    x = x.copy()

    module_logger.info('Splitting neuron #{0} by flow centrality'.format(x.skeleton_id))
    
    # Now get the node point with the highest flow centrality.
    cut = x.nodes[ (x.nodes.flow_centrality == x.nodes.flow_centrality.max()) ].treenode_id.tolist()    

    # If there is more than one  point we need to get one closest to the soma (root)    
    cut = sorted(cut, key = lambda y : len( x.igraph.get_shortest_paths( x.igraph.vs.select(node_id=y)[0], 
                                                            x.igraph.vs.select(node_id=x.root)[0] )[0] )
           )[0]

    if return_point:
        return cut

    # If cut node is a branch point, we will try cutting off main neurite
    if x.nodes.set_index('treenode_id').loc[ cut ].type =='branch' and primary_neurite:
        rest, primary_neurite = cut_neuron( x, cut )
        # Change name and color
        primary_neurite.neuron_name = x.neuron_name + '_primary_neurite'
        primary_neurite.color = (0,255,0)
    else:
        rest = x
        primary_neurite = None

    # Next, cut the rest into axon and dendrite
    a, b = cut_neuron( rest, cut )

    # Figure out which one is which by comparing number of presynapses
    if a.n_presynapses < b.n_presynapses:
        dendrite, axon = a, b
    else:
        dendrite, axon = b, a
    
    axon.neuron_name = x.neuron_name + '_axon'
    dendrite.neuron_name = x.neuron_name + '_dendrites'

    #Change colors    
    axon.color = (255,0,0)
    dendrite.color = (0,0,255)

    if primary_neurite:        
        return core.CatmaidNeuronList([ primary_neurite, axon, dendrite ])  
    else:
        return core.CatmaidNeuronList([ axon, dendrite ])  

def calc_segregation_index(x, centrality_method='centrifugal'):
    """ Calculates segregation index (SI) from Schneider-Mizell et al., eLife
    (2016) as metric for how polarized a neuron is. SI of 1 indicates total 
    segregation of inputs and outputs into dendrites and axon, respectively. 
    SI of 0 indicates homogneous distribution.

    Parameters
    ----------
    x :                 {CatmaidNeuron, CatmaidNeuronList}
                        Neuron to calculate segregation index (SI). If a 
                        NeuronList is provided, will assume that this is a 
                        split.
    centrality_method : {'centrifugal','centripetal','sum', 'bending'}, optional
                        Type of flow centrality to use to split the neuron. 
                        There are four flavors: the first three refer to
                        :func:`~pymaid.calc_flow_centrality`, the last 
                        refers to :func:`~pymaid.calc_bending_flow`.

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
                        Segregation Index (SI)
    """

    if not isinstance(x, (core.CatmaidNeuron,core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, not {0}'.format(type(x)))

    if not isinstance(x, core.CatmaidNeuronList):
        # Get the branch point with highest flow centrality
        split_point = split_axon_dendrite(x, reroot_soma=True, return_point=True )    

        # Now make a virtual split (downsampled neuron to speed things up)
        temp = x.copy()
        temp.downsample(10000)

        # Get one of its children
        child = temp.nodes[ temp.nodes.parent_id == split_point ].treenode_id.tolist()[0]

        # This will leave the proximal split with the primary neurite but 
        # since that should not have synapses, we don't care at this point.
        x = core.CatmaidNeuronList( cut_neuron( temp, child ) )

    # Calculate entropy for each fragment
    entropy = []
    for n in x:
        p = n.n_postsynapses / n.n_connectors

        if 0 < p < 1:
            S = - ( p * math.log( p ) + ( 1 - p ) * math.log( 1 - p ) )
        else:
            S = 0

        entropy.append(S)    

    # Calc entropy between fragments
    S = 1 / sum(x.n_connectors) * sum( [  e * x[i].n_connectors for i,e in enumerate(entropy) ] )

    # Normalize to entropy in whole neuron
    p_norm = sum(x.n_postsynapses) / sum(x.n_connectors)
    if 0 < p_norm < 1:
        S_norm = - ( p_norm * math.log( p_norm ) + ( 1 - p_norm ) * math.log( 1 - p_norm ) )
        H = 1 - S / S_norm
    else:
        S_norm = 0
        H = 0  

    return H

def calc_bending_flow(x, polypre=False):
    """ Variation of the algorithm for calculating synapse flow from 
    Schneider-Mizell et al., eLife (2016). 

    The way this implementation works is by iterating over each branch point
    and counting the number of pre->post synapse paths that "flow" from one 
    child branch to the other(s). 

    Parameters
    ----------
    x :         {CatmaidNeuron, CatmaidNeuronList}
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
    :func:`~pymaid.calc_flow_centrality`
            Calculate synapse flow centrality after Schneider-Mizell et al
    :func:`~pymaid.segregation_score`
            Uses flow centrality to calculate segregation score (polarity)
    :func:`~pymaid.split_axon_dendrite`
            Split the neuron into axon, dendrite and primary neurite.

    Returns
    -------        
    Adds a new column 'flow_centrality' to ``x.nodes``. Branch points only!

    """
    module_logger.info('Calculating bending flow centrality for neuron #{0}'.format(x.skeleton_id))

    start_time = time.time()

    if not isinstance(x, (core.CatmaidNeuron,core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, not {0}'.format(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        return [ calc_bending_flow(n, mode=mode, polypre=polypre, ) for n in x ]

    if x.soma and x.root != x.soma:
        module_logger.warning('Neuron {0} is not rooted to its soma!'.format(x.skeleton_id))

    # We will be processing a super downsampled version of the neuron to speed up calculations
    current_level = module_logger.level
    module_logger.setLevel('ERROR')
    y = x.copy()
    y.downsample(1000000)
    module_logger.setLevel(current_level)

    if polypre:
        # Get details for all presynapses
        cn_details = pymaid.get_connector_details( y.connectors[ y.connectors.relation==0 ] )            

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[ y.connectors.relation==0 ].treenode_id.unique()
    post_node_ids = y.connectors[ y.connectors.relation==1 ].treenode_id.unique()
    total_pre = len(pre_node_ids)
    total_post = len(post_node_ids)

    # Turn these into vertex indices of the igraph representation
    # We need to make this somewhat complicated construct in case a treenode
    # has multiple connectors
    pre_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in pre_node_ids ]
    post_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in post_node_ids ]

    # Get list of branch_points
    bp_node_ids = y.nodes[ y.nodes.type == 'branch' ].treenode_id.values.tolist()
    # Add root if it is also a branch point
    if y.igraph.vs.select( node_id = y.root )[0].degree() > 1:
        bp_node_ids.append( y.root )

    # Get indices of the igraph nodes
    bp_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in bp_node_ids ]

    # Get indices of the childs of our branch points and map them
    bp_childs = { t : [ e.source for e in y.igraph.es.select(_target=t) ] for t in bp_vs_ix }
    childs_vs_ix = [ ix for l in bp_childs.values() for ix in l ]

    # Get possible paths from the branch point childs FROM all pre and postynapses
    pre_paths = y.igraph.shortest_paths_dijkstra( childs_vs_ix, pre_vs_ix, mode='IN' )
    post_paths = y.igraph.shortest_paths_dijkstra( childs_vs_ix, post_vs_ix, mode='IN' )    

    # Turn into DataFramex
    distal_pre = pd.DataFrame( np.array(pre_paths) != float('inf'), index=childs_vs_ix, columns=pre_vs_ix )
    distal_post = pd.DataFrame( np.array(post_paths) != float('inf'), index=childs_vs_ix, columns=post_vs_ix )

    # Multiply columns (presynapses) by the number of postsynaptically connected nodes
    if polypre:
        # Map vertex ID to number of postsynaptic nodes (avoid 0)        
        distal_pre *= [ max( 1, len( cn_details[ cn_details.presynaptic_to_node== y.igraph.vs[n]['node_id'] ].postsynaptic_to_node.sum() ) ) for n in distal_pre.columns ]
        # Also change total_pre as accordingly
        total_pre = sum( [ max( 1, len(row) ) for row in cn_details.postsynaptic_to_node.tolist() ] )

    # Sum up axis - now each row 
    distal_pre = distal_pre.sum(axis=1)
    distal_post = distal_post.sum(axis=1)

    # Now go over all branch points and check flow between branches (centrifugal) vs flow from branches to root (centripetal)
    flow = { bp : 0 for bp in bp_childs }    
    for bp in bp_childs:
        # We will use left/right to label the different branches here (even if there is more than two)        
        for left, right in itertools.permutations( bp_childs[bp], r=2 ):            
            flow[bp] += distal_post.loc[ left ] * distal_pre.loc[ right ]            

    # Set flow centrality to None for all nodes
    x.nodes['flow_centrality'] = None

    # Change index to treenode_id
    x.nodes.set_index('treenode_id', inplace=True)

    # Add flow (make sure we use igraph of y to get node ids!)
    x.nodes.loc[ [ y.igraph.vs[i]['node_id'] for i in flow.keys() ], 'flow_centrality' ] = list(flow.values())

    # Add little info on method used for flow centrality
    x.centrality_method = 'bending'

    x.nodes.reset_index(inplace=True)

    module_logger.debug('Total time for bending flow calculation: {0}s'.format( round(time.time() - start_time ) ))

    return 


def calc_flow_centrality(x, mode = 'centrifugal', polypre=False ):
    """ This is an old, slow (and possibly faulty) implementation
    of the algorithm for calculating flow centralities from 
    Schneider-Mizell et al., eLife (2016). Losely based on Alex Bate's 
    implemention in https://github.com/alexanderbates/catnat.

    Parameters
    ----------
    x :         {CatmaidNeuron, CatmaidNeuronList}
                Neuron(s) to calculate flow centrality for
    mode :      {'centrifugal','centripetal','sum'}, optional
                Type of flow centrality to calculate. There are three flavors::
                (1) centrifugal, which counts paths from proximal inputs to distal outputs
                (2) centripetal, which counts paths from distal inputs to proximal outputs
                (3) the sum of both
    polypre :   bool, optional
                Whether to consider the number of presynapses as a multiple of
                the numbers of connections each makes. Attention: this works
                only if all synapses have been properly annotated (i.e. all
                postsynaptic sites).    

    Notes
    -----
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

    Pymaid uses the equivalent of ``mode='sum'`` and ``polypre=True``.

    See Also
    --------
    :func:`~pymaid.calc_bending_flow`
            Variation of flow centrality: calculates bending flow.
    :func:`~pymaid.calc_segregation_index`
            Calculates segregation score (polarity) of a neuron
    :func:`~pymaid.flow_centrality_split`
            Tries splitting a neuron into axon, dendrite and primary neurite.


    Returns
    -------        
    Adds a new column 'flow_centrality' to ``x.nodes``. Ignores non-synapse
    holding slab nodes!

    """

    module_logger.info('Calculating flow centrality for neuron #{0}'.format(x.skeleton_id))

    start_time = time.time()

    if mode not in ['centrifugal','centripetal','sum']:
        raise ValueError('Unknown parameter for mode: {0}'.format(mode))

    if not isinstance(x, (core.CatmaidNeuron,core.CatmaidNeuronList)):
        raise ValueError('Must pass CatmaidNeuron or CatmaidNeuronList, not {0}'.format(type(x)))

    if isinstance(x, core.CatmaidNeuronList):
        return [ calc_flow_centrality(n, mode=mode, polypre=polypre, ) for n in x ]

    if x.soma and x.root != x.soma:
        module_logger.warning('Neuron {0} is not rooted to its soma!'.format(x.skeleton_id))

    # We will be processing a super downsampled version of the neuron to speed up calculations
    current_level = module_logger.level
    module_logger.setLevel('ERROR')
    y = x.copy()
    y.downsample(1000000)
    module_logger.setLevel(current_level)

    if polypre:
        # Get details for all presynapses
        cn_details = pymaid.get_connector_details( y.connectors[ y.connectors.relation==0 ] )            

    # Get list of nodes with pre/postsynapses
    pre_node_ids = y.connectors[ y.connectors.relation==0 ].treenode_id.unique()
    post_node_ids = y.connectors[ y.connectors.relation==1 ].treenode_id.unique()
    total_pre = len(pre_node_ids)
    total_post = len(post_node_ids)

    # Turn these into vertex indices of the igraph representation
    # We need to make this somewhat complicated construct in case a treenode
    # has multiple connectors
    pre_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in pre_node_ids ]
    post_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in post_node_ids ]

    # Get list of points to calculate flow centrality for: 
    # branches and nodes with synapses
    calc_node_ids = y.nodes[ (y.nodes.type == 'branch') | (y.nodes.has_connectors == True ) ].treenode_id.values

    # Get indices of the igraph nodes
    calc_vs_ix = [ [v.index for v in y.igraph.vs if v['node_id'] == tn ][0] for tn in calc_node_ids ]    

    # Get possible paths to our calc_nodes FROM all pre and postynapses    
    pre_paths = y.igraph.shortest_paths_dijkstra( calc_vs_ix, pre_vs_ix, mode='IN' )
    post_paths = y.igraph.shortest_paths_dijkstra( calc_vs_ix, post_vs_ix, mode='IN' )    

    # Turn into binary DataFrame: True == this nodes is distal to, False if not distal
    distal_pre = pd.DataFrame( np.array(pre_paths) != float('inf'), index=calc_node_ids, columns=pre_vs_ix )
    distal_post = pd.DataFrame( np.array(post_paths) != float('inf'), index=calc_node_ids, columns=post_vs_ix )

    # Multiply columns (presynapses) by the number of postsynaptically connected nodes
    if polypre:
        # Map vertex ID to number of postsynaptic nodes (avoid 0)        
        distal_pre *= [ max( 1, len( cn_details[ cn_details.presynaptic_to_node== y.igraph.vs[n]['node_id'] ].postsynaptic_to_node.sum() ) ) for n in distal_pre.columns ]
        # Also change total_pre as accordingly
        total_pre = sum( [ max( 1, len(row) ) for row in cn_details.postsynaptic_to_node.tolist() ] )

    # Sum up axis - now each row represents the number of pre/postsynapses that are distal to that node
    distal_pre = distal_pre.sum(axis=1)
    distal_post = distal_post.sum(axis=1)
        
    # Centrifugal is the flow from all non-distal postsynapses to all distal presynapses
    centrifugal = { n : ( total_post - distal_post[n] ) * distal_pre[n] for n in calc_node_ids }

    # Centripetal is the flow from all distal postsynapses to all non-distal presynapses
    centripetal = { n : distal_post[n] * ( total_post - distal_pre[n]) for n in calc_node_ids }    

    # Set flow centrality to None for all nodes
    x.nodes['flow_centrality'] = None

    # Change index to treenode_id
    x.nodes.set_index('treenode_id', inplace=True)

    # Now map this onto our neuron
    if mode == 'centrifugal':
        res = list( centrifugal.values() )
    elif mode == 'centripetal':
        res = list( centripetal.values() )
    elif mode == 'sum':
        res = np.array( list(centrifugal.values()) ) + np.array( list(centripetal.values()) )    

    # Add results
    x.nodes.loc[ list( centrifugal.keys() ), 'flow_centrality' ] = res

    # Add little info on method/mode used for flow centrality
    x.centrality_method = mode

    x.nodes.reset_index(inplace=True)

    module_logger.debug('Total time for SFC calculation: {0}s'.format( round(time.time() - start_time ) ))

    return 


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


def stitch_neurons( *x, tn_to_stitch=None, method='ALL'):
    """ Function to stich multiple neurons together. The first neuron provided
    will be the master neuron. Unless treenode IDs are provided via 
    ``tn_to_stitch``, neurons will be stitched at the closest point.

    Parameters
    ----------
    x :                 CatmaidNeuron/CatmaidNeuronList
                        Neurons to stitch.
    tn_to_stitch :      List of treenode IDs, optional
                        If provided, these treenodes will be preferentially
                        used to stitch neurons together. If there are more 
                        than two possible treenodes for a single stitching
                        operation, the two closest are used.
    method :            {'LEAFS','ALL','NONE'}, optional
                        Defines automated stitching mode.
                            (1) 'LEAFS': only leaf (including root) nodes will 
                                be considered for stitching
                            (2) 'ALL': all treenodes are considered
                            (3) 'NONE': node and connector tables will simply
                                be combined. Use this if your neurons consists
                                of fragments with multiple roots.

    Returns
    -------
    core.CatmaidNeuron
    """

    if method not in ['LEAFS', 'ALL', 'NONE']:
        raise ValueError('Unknown method: %s' % str(method))

    # Compile list of individual neurons
    neurons = []
    for n in x:        
        if not isinstance(n, (core.CatmaidNeuron, core.CatmaidNeuronList) ):
            raise TypeError( 'Unable to stitch non-CatmaidNeuron objects' )
        elif isinstance(n, core.CatmaidNeuronList):
            neurons += n.neurons
        else:
            neurons.append(n)

    #Use copies of the original neurons!
    neurons = [ n.copy() for n in neurons if isinstance(n, core.CatmaidNeuron )]

    if len(neurons) < 2:
        module_logger.warning('Need at least 2 neurons to stitch, found %i' % len(neurons))
        return neurons[0]

    module_logger.debug('Stitching %i neurons...' % len(neurons))

    stitched_n = neurons[0]

    if tn_to_stitch and not isinstance(tn_to_stitch, (list, np.ndarray)):
        tn_to_stitch = [ tn_to_stitch ]
        tn_to_stitch = [ str(tn) for tn in tn_to_stitch ]

    for nB in neurons[1:]:
        #First find treenodes to connect
        if tn_to_stitch and set(tn_to_stitch) & set(stitched_n.nodes.treenode_id) and set(tn_to_stitch) & set(nB.nodes.treenode_id):
            treenodesA = stitched_n.nodes.set_index('treenode_id').loc[ tn_to_stitch ].reset_index()
            treenodesB = nB.nodes.set_index('treenode_id').loc[ tn_to_stitch ].reset_index()
        elif method == 'LEAFS':
            treenodesA = stitched_n.nodes[ stitched_n.nodes.type.isin(['end','root']) ].reset_index()
            treenodesB = nB.nodes[ nB.nodes.type.isin(['end','root']) ].reset_index()
        else:
            treenodesA = stitched_n.nodes
            treenodesB = nB.nodes

        if method != 'NONE':
            #Calculate pairwise distances
            pdist = scipy.spatial.distance.cdist( treenodesA[['x','y','z']].values, 
                                                  treenodesB[['x','y','z']].values, 
                                                  metric='euclidean' )                

            #Get the closest treenodes
            tnA = treenodesA.loc[ pdist.argmin(axis=0)[0] ].treenode_id
            tnB = treenodesB.loc[ pdist.argmin(axis=1)[0] ].treenode_id

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

def dist_between(x, a, b):
    """ Returns the geodesic distance between two nodes in nanometers.

    Parameters
    ----------
    x :             {CatmaidNeuron, iGraph}
                    Neuron containing the nodes
    a,b :           treenode IDs
                    Treenodes to check.
    
    Returns
    -------
    int
                    distance in nm

    See Also
    --------
    :func:`~pymaid.distal_to`
        Check if a node A is distal to node B
    """

    if isinstance( x, core.CatmaidNeuron ):
        g = x.igraph 
    elif isinstance( x, Graph ):
        g = x
    else:
        raise ValueError('Need either CatmaidNeuron or iGraph object')

    try:
        _ = int(a)
        _ = int(b)
    except:
        raise ValueError('a, b need to be treenode IDs')

    # Find treenodes in Graph
    a = [ n for n in g.vs if n['node_id'] == a ]
    b = [ n for n in g.vs if n['node_id'] == b ]

    if not a or not b:
        raise ValueError('Unable to find treenode ID(s) in graph')    

    return int( g.shortest_paths( a[0], b[0], mode='ALL', weights='weight' )[0][0] )

def distal_to(x, a=None, b=None):
    """ Checks if nodes a are distal to nodes b.

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
            represent treenode IDs.

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

    if a:
        if not isinstance(a, (list, np.ndarray)):
            a = [a]
        # Make sure we're dealing with integers
        a = np.array(a).astype(int)
        # Get iGraph vertex indices for our treenodes
        ix_a = np.array([ v.index for v in x.igraph.vs if v['node_id'] in a ])

        if ix_a.shape != a.shape :
            module_logger.warning('{0} source treenodes were not found'.format( a.shape[0] - ix_a.shape[0] ))
    else:
        ix_a = None
        a = [ v['node_id'] for v in x.igraph.vs ]
    
    if b:
        if not isinstance(b, (list, np.ndarray)):
            b = [b]
        # Make sure we're dealing with integers
        b = np.array(b).astype(int)
        ix_b = np.array([ v.index for v in x.igraph.vs if v['node_id'] in b ])

        if ix_b.shape != b.shape :
            module_logger.warning('{0} source treenodes were not found'.format( b.shape[0] - ix_b.shape[0] ))    
    else:
        ix_b = None
        b = [ v['node_id'] for v in x.igraph.vs ]
    
    # This holds distances
    all_paths = x.igraph.shortest_paths_dijkstra(ix_a, ix_b, mode='IN')

    # Make matrix and transpose
    df = pd.DataFrame( all_paths, columns=b, index=a ).T

    if df.shape == (1,1):
        return df.values[0][0] != float('inf')
    else:
        # Return boolean
        return df != float('inf')

def filter_connectivity( x, restrict_to, remote_instance=None):
    """ Filters connectivity data by volume or skeleton data. Use this e.g. to 
    restrict connectivity to edges within a given volume or to certain 
    compartments of neurons.

    Important
    ---------
    Order of columns/rows may change during filtering.


    Parameter
    ---------
    x :                 Connectivity object
                        Currently accepts either::                        
                         (1) Connectivity table from :func:`~pymaid.get_partners`
                         (2) Adjacency matrix from :func:`~pymaid.adjacency_matrix`
    restrict_to :       {str, pymaid.Volume, pymaid.CatmaidNeuronList}
                        Volume or neurons to restrict connectivity to. Strings
                        will be interpreted as volumes.
    remote_instance :   CATMAID instance, optional
                        If not passed, will try using globally defined.

    Returns
    -------
    Restricted connectivity data

    """     

    if not isinstance( restrict_to, (str, core.Volume, 
                                          core.CatmaidNeuron, 
                                          core.CatmaidNeuronList ) ):
        raise TypeError('Unable to restrict connectivity to type'.format(type(restrict_to)))

    if isinstance(restrict_to, str):
        restrict_to = pymaid.get_volume( restrict_to, remote_instance = remote_instance )

    datatype = getattr(x, 'datatype', None)

    if datatype not in ['connectivity_table','adjacency_matrix']:
        raise TypeError('Unknown connectivity data. See help(filter_connectivity) for details.')
    
    if datatype == 'connectivity_table':
        neurons = [ c for c in x.columns if c not in ['neuron_name', 'skeleton_id', 'num_nodes', 'relation', 'total'] ]

        # First get connector between neurons on the table
        if not x[ x.relation=='upstream' ].empty:    
            upstream = pymaid.get_connectors_between( x[ x.relation=='upstream' ].skeleton_id,
                                                      neurons,
                                                      directional=True,
                                                      remote_instance=remote_instance )
            # Now filter connectors
            if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
                upstream = upstream[ upstream.connector_id.isin( x.connectors ) ]
            elif isinstance( restrict_to, core.Volume ):
                upstream = upstream[ in_volume( upstream.connector_loc.values , restrict_to ) ]
        else:
            upstream = None                             

        if not x[ x.relation=='downstream' ].empty:
            downstream = pymaid.get_connectors_between( neurons,
                                                        x[ x.relation=='downstream'].skeleton_id,
                                                        directional=True,
                                                        remote_instance=remote_instance )      
            # Now filter connectors
            if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):                
                downstream = downstream[ downstream.connector_id.isin( x.connectors ) ]
            elif isinstance( restrict_to, core.Volume ):                
                downstream = downstream[ in_volume( downstream.connector_loc.values , restrict_to ) ]
        else:
            downstream = None

        if not isinstance( downstream, type(None)) and not isinstance( upstream, type(None)):
            cn_data = pd.concat( [upstream, downstream], axis=0 )
        elif isinstance( downstream, type(None)):
            cn_data = upstream
        else:
            cn_data = downstream            

    elif datatype == 'adjacency_matrix':
        cn_data = pymaid.get_connectors_between( x.index.tolist(), 
                                                 x.columns.tolist(),
                                                 directional=True,
                                                 remote_instance=remote_instance )

        # Now filter connectors
        if isinstance(restrict_to, (core.CatmaidNeuron, core.CatmaidNeuronList)):
            cn_data = cn_data[ cn_data.connector_id.isin( x.connectors ) ]            
        elif isinstance( restrict_to, core.CatmaidNeuron ):
            cn_data = cn_data[ in_volume( cn_data.connector_loc.values , restrict_to ) ]

    if cn_data.empty:
        module_logger.warning('No connectivity left after filtering')
        return

    # Reconstruct connectivity data:    
    # Collect edges
    edges = cn_data[['source_neuron','target_neuron']].values

    # Turn individual edges into synaptic connections
    unique_edges, counts = np.unique( edges, return_counts=True, axis=0 )
    unique_skids = np.unique(edges).astype(str)
    unique_edges=unique_edges.astype(str)    

    # Create empty adj_mat
    adj_mat = pd.DataFrame( np.zeros(( len(unique_skids), len(unique_skids) )),
                            columns=unique_skids, index=unique_skids )

    # Fill in values
    for i, e in enumerate(tqdm(unique_edges, disable=module_logger.getEffectiveLevel()>=40, desc='Adj. matrix')):
        # using df.at here speeds things up tremendously!
        adj_mat.at[ str(e[0]), str(e[1]) ] = counts[i]

    if datatype == 'adjacency_matrix':
        adj_mat.datatype = 'adjacency_matrix'
        # Bring into original format
        adj_mat = adj_mat.loc[ x.index, x.columns ]
        # If we dropped any columns/rows because they didn't contain connectivity, we have to fill them now
        adj_mat.fillna(0, inplace=True)
        return adj_mat 

    # Generate connectivity table by subsetting adjacency matrix to our neurons of interest
    all_upstream = adj_mat[ adj_mat[ neurons ].sum(axis=1) > 0 ][ neurons ]
    all_upstream['skeleton_id'] = all_upstream.index
    all_upstream['relation'] = 'upstream'

    all_downstream = adj_mat.T[ adj_mat.T[ neurons ].sum(axis=1) > 0 ][ neurons ]
    all_downstream['skeleton_id'] = all_downstream.index
    all_downstream['relation'] = 'downstream'    

    # Merge tables
    df = pd.concat( [all_upstream,all_downstream], axis=0, ignore_index=True )

    # Use original connectivity table to populate data
    aux = x.set_index('skeleton_id')[['neuron_name','num_nodes']].to_dict()    
    df['num_nodes'] = [ aux['num_nodes'][s] for s in df.skeleton_id.tolist() ]
    df['neuron_name'] = [ aux['neuron_name'][s] for s in df.skeleton_id.tolist() ]    
    df['total'] = df[neurons].sum(axis=1)

    # Reorder columns
    df = df[[ 'neuron_name', 'skeleton_id', 'num_nodes', 'relation','total'] + neurons ]

    df.sort_values(['relation','total'], inplace=True, ascending=False)  
    df.type = 'connectivity_table'
    df.reset_index(drop=True, inplace=True)  

    return df


