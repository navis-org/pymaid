"""
    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
    Copyright (C) 2017 Philipp Schlegel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along
"""


try:
   from pymaid import get_3D_skeleton, get_connectors, get_connector_details, get_skids_by_annotation, get_volume
except:
   from pymaid.pymaid import get_3D_skeleton, get_connectors, get_connector_details, get_skids_by_annotation, get_volume
import math
import time
import logging
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

#Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if not module_logger.handlers:
  #Generate stream handler
  sh = logging.StreamHandler()
  sh.setLevel(logging.DEBUG)
  #Create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  sh.setFormatter(formatter)
  module_logger.addHandler(sh)

try:
   from pymaid.igraph_catmaid import igraph_from_skeleton, dist_from_root
except:  
   from igraph_catmaid import igraph_from_skeleton, dist_from_root


def generate_list_of_childs( skdata ):
   """ Transforms list of nodes into a dictionary { parent: [child1,child2,...]}

   Parameter:
   ---------
   skdata :        Pandas dataframe containing a SINGLE neuron 

   Returns:
   -------
   dict :  { treenode_id : [ child_treenode, child_treenode, ... ] }

   """
   module_logger.debug('Generating list of childs...')
   
   try:
      nodes = skdata.ix[0].nodes
   except:
      nodes = skdata.nodes   

   list_of_childs = { n.treenode_id : [] for n in nodes.itertuples() }

   for n in nodes.itertuples():
      try:
         list_of_childs[ n.parent_id ].append( n.treenode_id )
      except:
         list_of_childs[None]=[None]   

   module_logger.debug('Done')

   return list_of_childs

def classify_nodes ( skdata ):
   """ Takes list of nodes and classifies them as end nodes, branches, slabs
   and root
   
   Parameters:
   ----------
   skdata :             Pandas dataframe containing neuron(s)

   Returns
   -------
   skdata :             added columns 'type' and 'has_synapse'
   """

   module_logger.debug('Looking for end, branch and root points...')

   #If more than one neuron
   if type(skdata.skeleton_id) != type(str()):
      for i in skdata.index:
        skdata.ix[i] = classify_nodes( skdata.ix[i] )
   else:
     list_of_childs  = generate_list_of_childs( skdata )
     list_of_parent = { n.treenode_id : n.parent_id for n in skdata.nodes.itertuples() }   

     end_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) == 0 ]   
     slabs = [ n for n in list_of_childs if len(list_of_childs[n]) == 1 ]
     branch_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) > 1 ]
     root = skdata.nodes[ skdata.nodes['parent_id'].isnull() ].treenode_id.values

     classes = { n : 'slab' for n in skdata.nodes.treenode_id.tolist() }
     classes.update( { n : 'end' for n in end_nodes } )
     classes.update( { n : 'branch' for n in branch_nodes } )
     classes.update( { n : 'root' for n in root } )  

     new_column = [ classes[n] for n in skdata.nodes.treenode_id.tolist() ]
     skdata.nodes['type'] = new_column

     nodes_w_synapses = skdata.connectors.treenode_id.values
     new_column = [ n in nodes_w_synapses for n in skdata.nodes.treenode_id.tolist() ]
     skdata.nodes['has_synapses'] = new_column

   return skdata

def downsample_neuron ( skdata, resampling_factor):
   """ Downsamples a neuron by a given factor. Preserves root, leafs, 
   branchpoints and synapse nodes
   
   Parameter
   ---------
   skdata :             Pandas dataframe containing a SINGLE neuron 
   resampling_factor :  Factor by which to reduce the node count

   Returns
   -------
   skdata :             downsampled Pandas Dataframe
   """

   try: 
     skdata.nodes.shape[1]
   except:
     module_logger.warning('Please pass dataframe for a single neuron. Use e.g. df.ix[0]')
     return skdata

   if skdata.nodes.shape[0] == 0: 
     module_logger.warning('Unable to downsample: no nodes in neuron')
     return skdata 

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   else:
      df = skdata.copy()

   module_logger.info('Preparing to downsample neuron...')
   
   list_of_parents = { n.treenode_id : n.parent_id for n in df.nodes.itertuples() }   

   if 'type' not in df.nodes:
      df = classify_nodes( df )
   
   fix_points = df.nodes[ (df.nodes.type != 'slab') | (df.nodes.has_synapses == True) ].treenode_id.values      

   #Walk from all fix points to the root - jump N nodes on the way
   new_parents = {}

   module_logger.info('Sampling neuron down by factor of %i' % resampling_factor)
   for en in fix_points:
      this_node = en

      while True:
         stop = False         
         np = list_of_parents[ this_node ]
         if np != None:
            for i in range( resampling_factor ):         
               if np in fix_points:             
                  new_parents[ this_node ] = np             
                  stop = True                            
                  break
               else:             
                  np = list_of_parents [ np ]

            if stop is True:
               break       
            else:
               new_parents[ this_node ] = np
               this_node = np       
         else:
            new_parents[ this_node ] = None
            break   
   
   new_nodes = df.nodes[[ n.treenode_id in new_parents for n in df.nodes.itertuples() ] ]   
   new_nodes.parent_id = [ new_parents[ n.treenode_id ] for n in new_nodes.itertuples() ]

   module_logger.info('Nodes before/after: %i/%i ' % ( len( df.nodes ), len( new_nodes ) ) ) 

   df.nodes = new_nodes

   return df

def longest_neurite( skdata, root_to_soma = False ):
   """ Returns a neuron consisting only of the longest neurite

   Parameter:
   ---------
   skdata :       Pandas dataframe containing a SINGLE neuron
   root_to_soma : boolean (default = False)
                  If true, neuron will be rerooted to soma.
                  Soma is the node with >1000 radius

   Returns:
   --------
   pandas DataFrame
   """

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()  

   if root_to_soma:   
      soma = df.nodes[ df.nodes.radius > 1000 ].reset_index()      
      if soma.shape[0] != 1:
         module_logger.error('Unable to reroot: No or multiple soma found for neuron %s ' % df.neuron_name ) 
         return
      if soma.ix[0].parent_id != None:
         df = reroot_neuron( skdata, soma.ix[0].treenode_id )
   
   #This here needs to be optimised -> takes so long because it calculates distances between ALL pairs of nodes
   #Instead: calculate only from root to each node?
   df, g = dist_from_root( df, return_graph = True )   

   df.nodes.sort_values('dist_to_root', inplace=True, ascending = False)
   df.nodes.reset_index(inplace=True, drop = True)   

   tip = df.nodes.ix[0].treenode_id
   root = df.nodes[ df.nodes.parent_id.isnull() ].reset_index().ix[0].treenode_id

   tip_index = g.vs.select( node_id=int(tip) )[0].index
   root_index = g.vs.select( node_id=int(root) )[0].index

   shortest_path = g.get_shortest_paths( tip_index ,to = root_index, mode='ALL' )

   tn_to_preverse = [ [ g.vs[i]['node_id'] for i in p ] for p in shortest_path ][0]    

   df.nodes = df.nodes [ df.nodes.treenode_id.isin( tn_to_preverse ) ].reset_index( drop = True)

   return df
     

def reroot_neuron( skdata, new_root, g = None ):
   """ Uses igraph to reroot the neuron at given point. Creating the iGraph is the 
   bottleneck - if you already have it, pass it along to speed things up (see example 
   below)!

   Parameter
   ---------
   skdata :       Pandas dataframe containing a SINGLE neuron
   new_root :     node ID or a tag of the node to reroot to
   g (optional):  iGraph of skdata - if not provided it will be generated

   Returns:
   --------
   pandas DataFrame containing the rerooted neuron
   """

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()  

   #Make sure to copy nodes
   df.nodes = df.nodes.copy()

   #If cut_node is a tag, rather than a ID, try finding that node  
   if type(new_root) == type( str() ):
      if new_root not in df.tags:
         module_logger.error('Error: Found no treenodes with tag %s - please double check!' % str( new_root ) )
         return 
      elif len( df.tags[new_root] ) > 1:
         module_logger.error('Error: Found multiple treenodes with tag %s - please double check!' % str( new_root ) )
         return
      else:
         new_root = df.tags[new_root][0]

   if df.nodes.set_index('treenode_id').ix[ new_root ].parent_id == None:
      module_logger.error('Error: New root is old root!')
      return df

   if not g:
      #Generate iGraph -> order/indices of vertices are the same as in skdata
      g = igraph_from_skeleton(df)

   try:
      #Select nodes with the correct ID as cut node
      new_root_index = g.vs.select( node_id=int(new_root) )[0].index
   #Should have found only one cut node
   except:
      module_logger.error('Error: Found no treenodes with ID %s - please double check!' % str( new_root ) )
      return 

   #get_shortest_paths() returns a list of paths from the new root to every other node
   #format is [ root_node, node1, node2, node3, ... , node 10 ]
   #all we need is the last (treenode_id) and the second last node ( its new parent )   
   shortest_paths = g.get_shortest_paths( new_root_index ,to = None, mode='ALL' )

   new_paths = [ [ g.vs[i]['node_id'] for i in p ] for p in shortest_paths ]   

   df.nodes.set_index('treenode_id', inplace = True )
   
   for p in new_paths:
      if len( p ) > 1:
         df.nodes.ix[ p[-1] ].parent_id = p[-2]
      else:
         df.nodes.ix[ p[-1] ].parent_id = None

   df.nodes.reset_index( inplace=True ) 

   module_logger.info('Info: %s #%s successfully rerooted' % ( df.neuron_name, df.skeleton_id ) )

   return df


def cut_neuron2( skdata, cut_node, g = None ):
   """ Uses igraph to Cut the neuron at given point and returns two new neurons. 
   Creating the iGraph is the bottleneck - if you already have it, pass it along 
   to speed things up (see example below)!

   Parameter
   ---------
   skdata :       Pandas dataframe containing a SINGLE neuron
   cut_node :     node ID or a tag of the node to cut
   g (optional):  iGraph of skdata - if not provided it will be generated

   Returns
   -------
   [1] neuron_dist : distal to the cut
   [2] neuron_prox : proximal to the cut

   Example:
   -------
   #Example for multiple cuts 

   from pymaid.igraph_catmaid import igraph_from_skeleton
   from pymaid.morpho import cut_neuron2
   from pymaid.pymaid import get_3D_skeleton, CatmaidInstance

   remote_instance = CatmaidInstance( url, http_user, http_pw, token )

   skeleton_dataframe = get_3D_skeleton( skeleton_id, remote_instance )

   #Generate igraph object once (time consuming step) and reuse for each cut
   g = igraph_from_skeleton( skeleton_dataframe )

   #First cut
   nA, nB = cut_neuron2( skeleton_data, cut_node1, g = g )

   #Second cut
   nA, nB = cut_neuron2( skeleton_data, cut_node2, g = g )  
   """
   start_time = time.time()  

   module_logger.info('Cutting neuron.' )

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()  

   if g is None:
      #Generate iGraph -> order/indices of vertices are the same as in skdata
      g = igraph_from_skeleton(df)

   #If cut_node is a tag, rather than a ID, try finding that node
   if type(cut_node) == type( str() ):
      if cut_node not in df.tags:
        module_logger.error('Error: Found no treenodes with tag %s - please double check!' % str( cut_node ) )
        return 
      elif len( df.tags[cut_node] ) > 1:
        module_logger.error('Error: Found multiple treenodes with tag %s - please double check!' % str( cut_node ) )
        return
      else:
        cut_node = df.tags[cut_node][0]

   module_logger.debug('Cutting neuron...')

   try:
      #Select nodes with the correct ID as cut node
      cut_node_index = g.vs.select( node_id=int(cut_node) )[0].index
   #Should have found only one cut node
   except:
      module_logger.error('Error: Found no treenodes with ID %s - please double check!' % str( cut_node ) )
      return 

   #Select the cut node's parent 
   parent_node_index = g.es.select( _source = cut_node_index )[0].target

   #Now calculate the min cut
   mc = g.st_mincut( parent_node_index , cut_node_index , capacity=None )

   #mc.partition holds the two partitions with mc.partition[0] holding part with the source and mc.partition[1] the target
   if g.vs.select(mc.partition[0]).select(node_id=int(cut_node)):
      dist_partition = mc.partition[0]
      prox_partition = mc.partition[1]
   else:
      dist_partition = mc.partition[1]
      prox_partition = mc.partition[0]

   #Partitions hold the indices -> now we have to translate this into node ids
   dist_partition_ids = g.vs.select(dist_partition)['node_id']   
   prox_partition_ids = g.vs.select(prox_partition)['node_id']   

   #Set dataframe indices to treenode IDs - will facilitate distributing nodes
   if df.nodes.index.name != 'treenode_id':
      df.nodes.set_index( 'treenode_id' , inplace = True )   

   neuron_dist = pd.DataFrame( [ [
                                  df.neuron_name + '_dist',
                                  df.skeleton_id,                            
                                  df.nodes.ix[ dist_partition_ids ],
                                  df.connectors[ [ c.treenode_id in dist_partition_ids for c in df.connectors.itertuples() ] ].reset_index(),                                  
                                  df.tags 
                              ]], 
                              columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                              dtype=object
                              ).ix[0]   

   neuron_dist.nodes.ix[ cut_node ].parent_id = None   

   neuron_prox = pd.DataFrame( [[ 
                                  df.neuron_name + '_prox',
                                  df.skeleton_id,                            
                                  df.nodes.ix[ prox_partition_ids ],
                                  df.connectors[ [ c.treenode_id not in dist_partition_ids for c in df.connectors.itertuples() ] ].reset_index(),                                  
                                  df.tags
                              ]], 
                              columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                              dtype=object
                              ).ix[0]

   #Reclassify cut node in distal as 'root' and its parent in proximal as 'end'
   if 'type' in df.nodes:
      neuron_prox.nodes.ix[ cut_node ].type = 'end'
      neuron_dist.nodes.ix[ cut_node ].type = 'root'  

   #Now reindex dataframes
   neuron_dist.nodes.reset_index( inplace = True )
   neuron_prox.nodes.reset_index( inplace = True )
   df.nodes.reset_index( inplace = True )

   module_logger.debug('Cutting finished in %is' % round ( time.time() - start_time ) ) 
   module_logger.info('Distal: %i nodes/%i synapses| |Proximal: %i nodes/%i synapses' % ( neuron_dist.nodes.shape[0], neuron_dist.connectors.shape[0],neuron_prox.nodes.shape[0], neuron_prox.connectors.shape[0] ) )   

   return neuron_dist, neuron_prox


def cut_neuron( skdata, cut_node ):
   """ Cuts a neuron at given point and returns two new neurons.   

   Parameter
   ---------
   skdata :       Pandas dataframe containing a SINGLE neuron
   cut_node :     node ID or a tag of the node to cut

   Returns
   -------
   [1] neuron_dist : distal to the cut
   [2] neuron_prox : proximal to the cut

   If you intend doing multiple cuts, consider using cut_neuron2() instead.    
   """
   start_time = time.time()

   module_logger.info('Preparing to cut neuron...' )

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()  

   list_of_childs  = generate_list_of_childs( skdata )   
   list_of_parents = { n.treenode_id : n.parent_id for n in df.nodes.itertuples() }   

   if 'type' not in df.nodes:
      df = classify_nodes( df )

   #If cut_node is a tag, rather than a ID, try finding that node
   if type( cut_node ) == type( str() ):
      if cut_node not in df.tags:
        module_logger.error('Error: Found no treenodes with tag %s - please double check!' % str( cut_node ) )
        return 
      elif len( df.tags[ cut_node ] ) > 1:
        module_logger.error('Error: Found multiple treenodes with tag %s - please double check!' % str( cut_node ) )
        return
      else:
        cut_node = df.tags[ cut_node ][0]

   if len( list_of_childs[ cut_node ] ) == 0:
      module_logger.warning('Cannot cut: cut_node is a leaf node!')
      return
   elif list_of_parents[ cut_node ] == None:
      module_logger.warning('Cannot cut: cut_node is a root node!')
      return
   elif cut_node not in list_of_parents:
      module_logger.warning('Cannot cut: cut_node not found!')
      return

   end_nodes = df.nodes[ df.nodes.type == 'end' ].treenode_id.values
   branch_nodes = df.nodes[ df.nodes.type == 'branch' ].treenode_id.values
   root = df.nodes[ df.nodes.type == 'root' ].treenode_id.values[0]

   #Walk from all end points to the root - if you hit the cut node assign this branch to neuronA otherwise neuronB
   distal_nodes = []
   proximal_nodes = []

   module_logger.info('Cutting neuron...')
   for i, en in enumerate( end_nodes.tolist() + [ cut_node ] ):     
      this_node = en
      nodes_walked = [ en ]      
      while True:                         
         this_node = list_of_parents[ this_node ]        
         nodes_walked.append( this_node )

         #Stop if this node is the cut node
         if this_node == cut_node:           
            distal_nodes += nodes_walked
            break
         #Stop if this node is the root node
         elif this_node == root:          
            proximal_nodes += nodes_walked            
            break
         #Stop if we have seen this branchpoint before   
         elif this_node in branch_nodes:           
            if this_node in distal_nodes:             
               distal_nodes += nodes_walked
               break          
            elif this_node in proximal_nodes:               
               proximal_nodes += nodes_walked
               break

   #Set dataframe indices to treenode IDs - will facilitate distributing nodes
   if df.nodes.index.name != 'treenode_id':
      df.nodes.set_index( 'treenode_id' , inplace = True )   
   
   distal_nodes = list ( set( distal_nodes ) )
   proximal_nodes = list ( set( proximal_nodes ) )

   neuron_dist = pd.DataFrame( [ [
                                  df.neuron_name + '_dist',
                                  df.skeleton_id,                            
                                  df.nodes.ix[ distal_nodes ],
                                  df.connectors[ [ c.treenode_id in distal_nodes for c in df.connectors.itertuples() ] ].reset_index(),                                  
                                  df.tags 
                              ]], 
                              columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                              dtype=object
                              ).ix[0]

   neuron_dist.nodes.ix[ cut_node ].parent_id = None
   neuron_dist.nodes.ix[ cut_node ].type = 'root'   

   neuron_prox = pd.DataFrame( [[ 
                                  df.neuron_name + '_prox',
                                  df.skeleton_id,                            
                                  df.nodes.ix[ proximal_nodes ],
                                  df.connectors[ [ c.treenode_id not in distal_nodes for c in df.connectors.itertuples() ] ].reset_index(),                                  
                                  df.tags
                              ]], 
                              columns = ['neuron_name','skeleton_id','nodes','connectors','tags'],
                              dtype=object
                              ).ix[0]

   #Reclassify cut node in proximal neuron as end node
   neuron_prox.nodes.ix[ cut_node ].type = 'end'

   #Now reindex dataframes
   neuron_dist.nodes.reset_index( inplace = True )
   neuron_prox.nodes.reset_index( inplace = True )
   df.nodes.reset_index( inplace = True )

   module_logger.info('Cutting finished in %is' % round ( time.time() - start_time ) ) 
   module_logger.info('Distal to cut node: %i nodes/%i synapses' % ( neuron_dist.nodes.shape[0], neuron_dist.connectors.shape[0] ) )
   module_logger.info('Proximal to cut node: %i nodes/%i synapses' % ( neuron_prox.nodes.shape[0], neuron_prox.connectors.shape[0]  ) )

   return neuron_dist, neuron_prox

def synapse_root_distances(skdata, remote_instance, pre_skid_filter = [], post_skid_filter = [] ):    
   """ Calculates geodesic (along the arbor) distance of synapses to root 
   (i.e. soma)

   Parameter
   ---------   
   skdata :             Pandas dataframe containing a SINGLE neuron
   pre_skid_filter :    (optional) if provided, only synapses from these 
                        neurons will be processed
   post_skid_filter :   (optional) if provided, only synapses to these neurons 
                        will be processed

   Returns
   -------
   [1] pre_node_distances :   {'connector_id: distance_to_root[nm]'} for all 
                              presynaptic sistes of this neuron
   [2] post_node_distances :  {'connector_id: distance_to_root[nm]'} for all 
                              postsynaptic sites of this neuron
   """

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy() 

   #Reindex dataframe to treenode
   if df.nodes.index.name != 'treenode_id':
      df.nodes.set_index( 'treenode_id' , inplace = True )   

   #Calculate distance to parent for each node
   tn_coords = skdata.nodes[ ['x','y','z' ] ].reset_index()
   parent_coords = skdata.nodes.ix[ skdata.nodes.parent_id.tolist() ][ [ 'x','y','z'] ].reset_index()
   w = np.sqrt( np.sum(( tn_coords[ ['x','y','z' ] ] - parent_coords[ ['x','y','z' ] ] ) **2, axis=1 )).tolist()

   #Get connector details
   cn_details = get_connector_details (  skdata.connectors.connector_id.tolist() , remote_instance = remote_instance)   

   list_of_parents = { n[0]: (n[1], n[3], n[4], n[5] ) for n in skdata[0] }    

   if pre_skid_filter or post_skid_filter:
      #Filter connectors that are both pre- and postsynaptic to the skid in skid_filter 
      filtered_cn = [ c for c in cn_details.itertuples() if True in [ int(f) in c.postsynaptic_to for f in post_skid_filter ] and True in [ int(f) == c.presynaptic_to for f in pre_skid_filter ] ]
      module_logger.debug('%i of %i connectors left after filtering' % ( len( filtered_cn ) , cn_details.shape[0] ) )
   else:
      filtered_cn = cn_details 

   pre_node_distances = {}
   post_node_distances = {}
   visited_nodes = {}

   module_logger.info('Calculating distances to root')
   for i,cn in enumerate(filtered_cn):

      if i % 10 == 0:
         module_logger.debug('%i of %i' % ( i, len(filtered_cn) ) )   

      if cn[1]['presynaptic_to'] == int(skdata.skeleton_id) and cn[1]['presynaptic_to_node'] in list_of_parents:
         dist, visited_nodes = walk_to_root( [ ( n[0], n[3],n[4],n[5] ) for n in skdata[0] if n[0] == cn[1]['presynaptic_to_node'] ][0] , list_of_parents, visited_nodes )

         pre_node_distances[ cn[1]['presynaptic_to_node'] ] = dist

      if int(skdata.skeleton_id) in cn[1]['postsynaptic_to']:
         for nd in cn[1]['postsynaptic_to_node']:
            if nd in list_of_parents:                    
               dist, visited_nodes = walk_to_root( [ ( n[0], n[3],n[4],n[5] ) for n in skdata[0] if n[0] == nd ][0] , list_of_parents, visited_nodes )                

               post_node_distances[ nd ] = dist

   #Reindex dataframe
   df.nodes.reset_index( inplace = True )
         
   return pre_node_distances, post_node_distances

def calc_dist(v1,v2):        
    return math.sqrt(sum(((a-b)**2 for a,b in zip(v1,v2))))

def fix_neuron ( skdata ):
    """ Helper to fix morphology of neurons after some virtual operation 
    (i.e. cutting, merging, etc. ):
    (1) nodes w/o parents - sets their parent_id to None
    (2) connectors w/o parent nodes - removes them

    Parameters:
    ----------
    skdata :          pandas DataFrame containing a SINGLE neuron

    Returns:
    -------
    cleaned-up pandas DataFrame
    """

    #Check for new root node(s) and set their parent_id to None
    skdata.nodes.loc[ ~skdata.nodes.parent_id.isin( skdata.nodes.treenode_id.tolist() ), 'parent_id' ] = None

    #Check and remove disconnected connectors
    skdata.connectors = skdata.connectors[ skdata.connectors.treenode_id.isin( skdata.nodes.treenode_id.tolist() ) ] 

    if skdata.nodes.loc[ ~skdata.nodes.parent_id.isin( skdata.nodes.treenode_id.tolist() ), 'parent_id' ].shape[0] > 1:
      module_logger.warning('Warning: %s #%s - multiple nodes w/o a parent detected!' % ( skdata.neuron_name ,str(skdata.skeleton_id) ) ) 

    return skdata

def calc_cable( skdata , smoothing = 1, remote_instance = None, return_skdata = False ):
   """ Calculates cable length in micro meter (um) of a given neuron     

    Parameters:
    -----------
    skdata :            Either a skeleton ID or Pandas dataframe containing 3d 
                        skeleton data. If skeleton ID, 3d skeleton data will be 
                        pulled from CATMAID server
    smoothing :         int (default = 1)
                        use to smooth neuron by downsampling; 1 = no smoothing                  
    remote_instance :   CATMAID instance (optional)
                        pass if skdata is a skeleton ID, not 3D skeleton data
    return_skdata :     boolean (default = False)
                        If True: instead of the final cable length, a dataframe
                        containing the distance to each treenode's parent

    Returns:
    --------
    cable_length [um]   

    OR (if return_skdata = True)

    pandas DataFrame with df.nodes.parent_dist containing the distances to parent
    """   

   if type(skdata) == type( int() ) or type(skdata) == type( str() ) :
      skdata = get_3D_skeleton( [skdata], remote_instance).ix[0]

   if type( skdata ) == type( pd.DataFrame() ):      
      df = skdata.ix[0].copy()
   elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()

   #Copy node data too
   df.nodes = df.nodes.copy()

   if smoothing > 1:
      df = downsample_neuron( df, smoothing )   

   if df.nodes.index.name != 'treenode_id':
      df.nodes.set_index( 'treenode_id' , inplace = True )

   #Calculate distance to parent for each node
   tn_coords = df.nodes[ ['x','y','z' ] ].reset_index()
   parent_coords = df.nodes.ix[ [ n for n in df.nodes.parent_id.tolist() ] ][ [ 'x','y','z'] ].reset_index()   

   #Calculate distances between nodes and their parents
   w = np.sqrt( np.sum(( tn_coords[ ['x','y','z' ] ] - parent_coords[ ['x','y','z' ] ] ) **2, axis=1 ))

   df.nodes.reset_index( inplace = True )

   if return_skdata:      
      df.nodes['parent_dist'] =  [ v / 1000 for v in list(w) ]
      return df    

   # #Remove nan value (at parent node) and return sum of all distances
   return np.sum( w[ np.logical_not( np.isnan(w) ) ] ) / 1000

def calc_strahler_index( skdata, return_dict = False ):
    """ Calculates Strahler Index -> starts with index of 1 at each leaf, at 
    forks with varying incoming strahler index, the highest index
    is continued, at forks with the same incoming strahler index, highest 
    index + 1 is continued. Starts with end nodes, then works its way from 
    branch nodes to branch nodes up to root node

    Parameters:
    ----------
    skdata :              skeleton data from pymaid.get_3D_skeleton()
    return_dict :         boolean (default = False)
                          If True, a dict is returned instead of the dataframe

    Returns:
    -------
    skdata with new column: skdata.nodes.strahler_index    
    """    

    module_logger.info('Calculating Strahler indices...')

    start_time = time.time()     

    if type( skdata ) == type( pd.DataFrame() ):      
      df = skdata.ix[0].copy()
    elif type( skdata ) == type( pd.Series() ):
      df = skdata.copy()    

    #Make sure dataframe is not indexed by treenode_id for preparing lists 
    df.nodes.reset_index( inplace = True, drop = True )

    #Find branch, root and end nodes
    if 'type' not in df.nodes:
      df = classify_nodes( df )

    end_nodes = df.nodes[ df.nodes.type == 'end' ].treenode_id.tolist()
    branch_nodes = df.nodes[ df.nodes.type == 'branch' ].treenode_id.tolist()
    root = df.nodes[ df.nodes.type == 'root' ].treenode_id.tolist()

    #Generate dicts for childs and parents
    list_of_childs = generate_list_of_childs( skdata )
    #list_of_parents = { n[0]:n[1] for n in skdata[0] } 

    #Reindex according to treenode_id
    if df.nodes.index.name != 'treenode_id':
      df.nodes.set_index( 'treenode_id' , inplace = True )        

    strahler_index = { n : None for n in list_of_childs if n != None }                

    starting_points = end_nodes
    
    nodes_processed = []

    while starting_points:
        module_logger.debug('New starting point. Remaining: %i' % len( starting_points ) )        
        new_starting_points = []
        starting_points_done = []

        for i,en in enumerate(starting_points):            
            this_node = en                     

            module_logger.debug('%i of %i ' % ( i ,len(starting_points) ) )

            #Calculate index for this branch                
            previous_indices = []
            for child in list_of_childs[this_node]:
                previous_indices.append(strahler_index[child])

            if len(previous_indices) == 0:
                this_branch_index = 1
            elif len(previous_indices) == 1:
                this_branch_index = previous_indices[0]
            elif previous_indices.count(max(previous_indices)) >= 2:
                this_branch_index = max(previous_indices) + 1
            else:
                this_branch_index = max(previous_indices)                            

            nodes_processed.append( this_node )
            starting_points_done.append( this_node )

            #Now walk down this spone
            #Find parent
            spine = [ this_node ]

            #parent_node = list_of_parents [ this_node ]              
            parent_node = df.nodes.ix[ this_node ].parent_id

            while parent_node not in branch_nodes and parent_node != None:
                this_node = parent_node
                parent_node = None                
                
                spine.append( this_node )                                 
                nodes_processed.append(this_node)

                #Find next parent
                try:
                  parent_node = df.nodes.ix[ this_node ].parent_id
                except:
                  #Will fail if at root (no parent)
                  break

            strahler_index.update( { n : this_branch_index for n in spine } )                 

            #The last this_node is either a branch node or the root
            #If a branch point: check, if all its childs have already been processed
            node_ready = True 
            for child in list_of_childs[ parent_node ]:
                if child not in nodes_processed:
                    node_ready = False

            if node_ready is True and parent_node != None:
                new_starting_points.append( parent_node )

        #Remove those starting_points that were successfully processed in this run before the next iteration
        for node in starting_points_done:
            starting_points.remove(node)        

        #Add new starting points
        starting_points += new_starting_points

    df.nodes.reset_index( inplace = True )

    df.nodes['strahler_index'] = [ strahler_index[n] for n in df.nodes.treenode_id.tolist() ] 

    module_logger.info('Done in %is' % round(time.time() -  start_time ) ) 

    if return_dict:
      return strahler_index

    return df

def walk_to_root( start_node, list_of_parents, visited_nodes ):
    """ Helper function for synapse_root_distances(): 
    Walks to root from start_node and sums up geodesic distances along the way.     

    Parameters:
    -----------
    start_node :        (node_id, x,y,z)
    list_of_parents :   {node_id: (parent_id, x,y,z) }
    visited_nodes :     {node_id: distance_to_root}
                        Make sure to not walk the same path twice by keeping 
                        track of visited nodes and their distances to soma

    Returns:
    --------
    [1] distance_to_root
    [2] updated visited_nodes
    """
    dist = 0
    distances_traveled = []
    nodes_seen = []
    this_node = start_node    

    #Walk to root
    while list_of_parents[ this_node[0] ][0] != None:
        parent = list_of_parents[ this_node[0] ]
        if parent[0] not in visited_nodes:
            d = calc_dist( this_node[1:], parent[1:] )
            distances_traveled.append( d )
            nodes_seen.append( this_node[0] )
        else:
            d = visited_nodes[ parent[0] ]
            distances_traveled.append ( d )
            nodes_seen.append( this_node[0] )
            break

        this_node = parent

    #Update visited_nodes
    visited_nodes.update( { n: sum( distances_traveled[i:] ) for i,n in enumerate(visited_nodes) } ) 

    return round ( sum( distances_traveled ) ), visited_nodes

def in_volume( points, volume, remote_instance, approximate = False, ignore_axis = [] ):
    """ Uses scipy to test if points are within a given CATMAID volume.
    The idea is to test if adding the point to the cloud would change the
    convex hull. 

    Parameters:
    -----------
    points :            list of points; format [ [ x, y , z ], [  ] ]
                        can be numpy array, pandas df or list
    volume :            name of the CATMAID volume to test or list of vertices
                        as returned by pymaid.get_volume()                        
    remote_instance :   CATMAID instance (optional)
                        pass if skdata is a skeleton ID, not 3D skeleton data
    approximate :       boolean (default = False)
                        if True, bounding box around the volume is used. Will
                        speed up calculations a lot!
    ignore_axis :       list of integers ( default = None )
                        Provide axes that should be ignored. Only works when
                        approximate = True. For example ignore_axis = [0,1]
                        will ignore x and y axis, and include nodes that fit
                        within z axis of the bounding box.

    Returns:
    --------
    list of booleans :  True if in volume, False if not
    """

    if type(volume) == type(str()):
      volume = get_volume ( volume, remote_instance )
      verts = np.array( volume[0] )
    else:
      verts = np.array( volume )

    if not approximate:
      intact_hull = ConvexHull(verts)
      intact_verts = list( intact_hull.vertices )

      if type(points) == type(list()):
        points = pd.DataFrame( points )

      return [ list( ConvexHull( np.append( verts, list( [p] ), axis = 0 ) ).vertices ) == intact_verts for p in points.itertuples( index = False ) ]
    else:
      bbox = [ ( min( [ v[0] for v in verts ] ), max( [ v[0] for v in verts ] )  ),
               ( min( [ v[1] for v in verts ] ), max( [ v[1] for v in verts ] )  ),
               ( min( [ v[2] for v in verts ] ), max( [ v[2] for v in verts ] )  )
              ]

      for a in ignore_axis:
        bbox[a] = ( float('-inf'), float('inf') )

      return [ False not in [  bbox[0][0] < p.x < bbox[0][1], bbox[1][0] < p.y < bbox[1][1], bbox[2][0] < p.z < bbox[2][1], ] for p in points.itertuples( index = False ) ]


if __name__ == '__main__':
   """
   FOR DEBUGGING ONLY
   """
   import sys
   sys.path.append('/Users/philipps/OneDrive/Cloudbox/Python')
   sys.path.append('/Users/philipps/OneDrive/Cloudbox/Python/PyMaid/pymaid')
   from connect_catmaid import connect_adult_em
   rm = connect_adult_em()

   #print(calc_cable ( 21999, smoothing = 3, remote_instance = rm) )

   from pymaid import get_3D_skeleton  

   skdata = get_3D_skeleton( [21999], rm)[0]     

   nA, nB = cut_neuron2( skdata, 132996 )

   #downsample_neuron ( nA , 10 )
