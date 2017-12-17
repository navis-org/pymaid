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


""" This module contains functions to manipulate neuron morphology.
"""

import sys
import math
import time
import logging
import pandas as pd
import numpy as np
import scipy
import scipy.interpolate
from tqdm import tqdm, trange

from pymaid import core, graph_utils

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


__all__ = sorted([ 'downsample_neuron','resample_neuron'])

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
        this_length = graph_utils.dist_between( x, seg[0], seg[-1] )        
        
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

        new_nodes = pd.DataFrame( [ [ tn, pn, None , co[0], co[1], co[2], -1, 5, 'slab' ] for tn, pn,co in zip( new_ids[:-1], new_ids[1:], new_coords[:-1] ) ],
                                  columns=['treenode_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence', 'type'] )
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
    branchpoints. Tags, connectors and radii > 0 are mapped onto the closest
    new treenode. Columns "confidence" and "creator" of the treenode table
    are currently discarded.

    Important
    ---------
    This generates an entirely new set of treenode IDs! Those will be unique
    within a neuron, but you may encounter duplicates across neurons.

    Parameters
    ----------
    x :                 {CatmaidNeuron,CatmaidNeuronList} 
                        Neuron(s) to resample.
    resample_to :       int
                        New resolution in nanometer.
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
    locs = nodes[['x','y','z']]

    new_nodes = []
    max_tn_id = x.nodes.treenode_id.max() + 1 

    # Iterate over segments
    for i,seg in enumerate(tqdm(x.segments, desc='Proc. segments', disable=module_logger.getEffectiveLevel()>=40 )):
        coords = locs.loc[ seg ].values.astype(float)
        
        # vecs between subsequently measured points
        vecs = np.diff(coords.T)

        # path: cum distance along points (norm from first to ith point)
        path = np.cumsum(np.linalg.norm(vecs, axis=0))
        path = np.insert(path, 0, 0)

        # If path is too short, just keep the first and last treenode
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

    # Add root node(s)
    root = x.root
    if not isinstance(root, (np.ndarray, list)):
        root = [x.root]
    root = x.nodes.loc[ x.nodes.treenode_id.isin(root) , ['treenode_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence']]
    new_nodes += [ list(r) for r in root.values  ]

    # Generate new nodes dataframe
    new_nodes = pd.DataFrame( data = new_nodes,
                              columns=['treenode_id', 'parent_id', 'creator_id', 'x', 'y', 'z', 'radius', 'confidence'],
                              dtype=object
                               )

    # Convert columns to appropriate dtypes
    dtypes = {'treenode_id':int, 'parent_id':object, 'x':int, 'y':int, 'z':int, 
              'radius':int, 'confidence':int}

    for k, v in dtypes.items():        
        new_nodes[k] = new_nodes[k].astype(v)

    # Remove duplicate treenodes (branch points)
    new_nodes = new_nodes[ ~new_nodes.treenode_id.duplicated() ]

    # Map connectors back:
    # 1. Get position of old synapse-bearing treenodes
    old_tn_position = x.nodes.set_index('treenode_id').loc[ x.connectors.treenode_id, ['x','y','z']].values
    # 2. Get closest neighbours
    distances = scipy.spatial.distance.cdist( old_tn_position, new_nodes[['x','y','z']].values )
    min_ix = np.argmin(distances, axis=1)
    # 3. Map back onto neuron
    x.connectors['treenode_id'] = new_nodes.iloc[ min_ix ].treenode_id.values

    # Map tags back:
    if x.tags:
        # 1. Get position of old tag bearing treenodes
        tag_tn = set( [ tn for l in x.tags.values() for tn in l ] )
        old_tn_position = x.nodes.set_index('treenode_id').loc[ tag_tn, ['x','y','z']].values
        # 2. Get closest neighbours
        distances = scipy.spatial.distance.cdist( old_tn_position, new_nodes[['x','y','z']].values )
        min_ix = np.argmin(distances, axis=1)
        # 3. Create a dictionary
        new_tag_tn = { tn : new_nodes.iloc[ min_ix[i] ].treenode_id for i, tn in enumerate( tag_tn ) }
        # 4. Map tags back
        new_tags = { t : [ new_tag_tn[tn] for tn in x.tags[t] ] for t in x.tags }
        x.tags = new_tags

    # Map nodes with radius > 0 back
    # 1. Get position of old synapse-bearing treenodes
    old_tn_position = x.nodes.loc[ x.nodes.radius > 0, ['x','y','z']].values
    # 2. Get closest neighbours
    distances = scipy.spatial.distance.cdist( old_tn_position, new_nodes[['x','y','z']].values )
    min_ix = np.argmin(distances, axis=1)
    # 3. Map radii onto 
    new_nodes.loc[ min_ix, 'radius'] = x.nodes.loc[ x.nodes.radius > 0, 'radius' ].values
   
    # Set nodes
    x.nodes = new_nodes

    # Clear and regenerate temporary attributes
    x._clear_temp_attr()

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
        graph_utils.classify_nodes(df)

    selection = df.nodes.type != 'slab'    

    if preserve_cn_treenodes:
        selection = selection | df.nodes.treenode_id.isin(df.connectors.treenode_id)

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
    new_nodes = df.nodes[ df.nodes.treenode_id.isin( list(new_parents.keys()) ) ].copy()           
    new_nodes.loc[:,'parent_id'] = [new_parents[tn] for tn in new_nodes.treenode_id]

    # We have to temporarily set parent of root node from 1 to an integer
    root_ix = new_nodes[new_nodes.parent_id.isnull()].index
    new_nodes.loc[root_ix, 'parent_id'] = 0
    # first convert everything to int
    new_nodes.loc[:,'parent_id'] = new_nodes.parent_id.values.astype(int)      
    # then back to object so that we can add a 'None'
    new_nodes.loc[:,'parent_id'] = new_nodes.parent_id.values.astype(object)  
    
    # Reassign parent_id None to root node
    new_nodes.loc[root_ix, 'parent_id'] = None
    
    module_logger.info('Nodes before/after: %i/%i ' %
                       (len(df.nodes), len(new_nodes)))

    df.nodes = new_nodes

    # This is essential -> otherwise e.g. graph.neuron2graph will fail
    df.nodes.reset_index(inplace=True, drop=True)

    if not inplace:
        return df