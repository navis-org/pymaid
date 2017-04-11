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
   from pymaid import get_3D_skeleton, get_connectors, get_connector_details, get_skids_by_annotation
except:
   from pymaid.pymaid import get_3D_skeleton, get_connectors, get_connector_details, get_skids_by_annotation
import math
import time
import logging

#Set up logging
module_logger = logging.getLogger('natpy')
module_logger.setLevel(logging.DEBUG)
#Generate stream handler
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
#Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
module_logger.addHandler(sh)

try:
   from pymaid.igraph_catmaid import igraph_from_skeleton
except:  
   from igraph_catmaid import igraph_from_skeleton


def generate_list_of_childs(skdata):
   """ Transforms list of nodes into a dictionary { parent: [child1,child2,...]}
   
   Parameter:
   ---------
   skdata :          CATMAID skeleton for a single neuron

   Returns:
   list_of_childs :  dict()

   """
   module_logger.debug('Generating list of childs...')
   list_of_childs = { n[0] : [] for n in skdata[0] }   

   for n in skdata[0]:
      try:
         list_of_childs[ n[1] ].append( n[0] )
      except:
         list_of_childs[None]=[None]

   module_logger.debug('Done')

   return list_of_childs

def downsample_neuron ( skdata, resampling_factor):
   """ Downsamples a neuron by a given factor. Preserves root, leafs, 
   branchpoints and synapse nodes
   
   Parameter
   ---------
   skdata :             CATMAID skeleton data
   resampling_factor :  Factor by which to reduce the node count

   Returns
   -------
   skdata :             downsample CATMAID skeleton data 
   """

   list_of_childs  = generate_list_of_childs(skdata)
   list_of_parents = { n[0]:n[1] for n in skdata[0] }

   end_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) == 0 ]   
   branch_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) > 1 ]  
   root = [ n[0]  for n in skdata[0] if n[1] == None ][0]
   synapse_nodes = [ n[0] for n in skdata[1] ]

   fix_points = list ( set( end_nodes + branch_nodes + synapse_nodes + [root] ) )

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
               if np in fix_points + [root]:             
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

   new_nodes = [ [ n[0], new_parents[ n[0] ],n[2],n[3],n[4],n[5],n[6],n[7] ] for n in skdata[0] if n[0] in new_parents ]   

   module_logger.info('Node before/after: %i/%i ' % ( len( skdata[0] ), len( new_nodes ) ) ) 

   return [ new_nodes, skdata[1] ]

def cut_neuron2( skdata, cut_node, g = None ):
   """ Uses igraph to Cut the neuron at given point and returns two new neurons. Creating
   the iGraph is the bottleneck - if you already have it, pass it along to speed things up!

   Parameter
   ---------
   skdata :       CATMAID skeleton data (including connector data)
   cut_node :     ID of the node to cut
   g (optional):  iGraph of skdata - if not provided it will be generated

   Returns
   -------
   [1] neuron_dist : distal to the cut
   [2] neuron_prox : proximal to the cut
   """
   start_time = time.time()    

   if g is None:
      #Generate iGraph -> order/indices of vertices are the same as in skdata
      g = igraph_from_skeleton(skdata)

   module_logger.info('Cutting neuron...')
   try:
      #Select nodes with the correct ID as cut node
      cut_node_index = g.vs.select( node_id=int(cut_node) )[0].index
   #Should have found only one cut node
   except:
      module_logger.error('Error: Found %i nodes with that ID - please double check!')
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

   neuron_dist = [ [] , [] ]
   neuron_prox = [ [] , [] ]

   for n in skdata[0]:
      if n[0] in dist_partition_ids:
         neuron_dist[0].append( n )
      else:
         neuron_prox[0].append( n )

   #Change cut_node to be the root node in distal neuron
   neuron_dist[0][ [ neuron_dist[0].index(e) for e in neuron_dist[0] if e[0] == int(cut_node) ][0] ][1] = None
   print('!!!!', neuron_dist[0][ [ neuron_dist[0].index(e) for e in neuron_dist[0] if e[0] == int(cut_node) ][0] ])
   print( [ n for n in neuron_dist[0] if None in n ])

   #Add synapses
   neuron_dist[1] = [ s for s in skdata[1] if s[0] in dist_partition_ids ]
   neuron_prox[1] = [ s for s in skdata[1] if s[0] not in dist_partition_ids ]      

   module_logger.info('Cutting finished in %is' % round ( time.time() - start_time ) ) 
   module_logger.info('Distal to cut node: %i nodes/%i synapses' % (len( neuron_dist[0] ),len( neuron_dist[1] )) )
   module_logger.info('Proximal to cut node: %i nodes/%i synapses' % (len( neuron_prox[0] ),len( neuron_prox[1] )) )

   return neuron_dist, neuron_prox


def cut_neuron( skdata, cut_node ):
   """ Cuts a neuron at given point and returns two new neurons.

   Parameter
   ---------
   skdata :    CATMAID skeleton data
   cut_node :  ID of the node to cut

   Returns
   -------
   [1] neuron_dist : distal to the cut
   [2] neuron_prox : proximal to the cut
   """
   start_time = time.time()

   list_of_childs  = generate_list_of_childs(skdata)
   list_of_parents = { n[0]:n[1] for n in skdata[0] }

   if len( list_of_childs[ cut_node ] ) == 0:
      module_logger.warning('Cannot cut: cut_node is a leaf node!')
      return
   elif list_of_parents[ cut_node ] == None:
      module_logger.warning('Cannot cut: cut_node is a root node!')
      return

   end_nodes = list ( set( [ n for n in list_of_childs if len(list_of_childs[n]) == 0 ] + [ cut_node ] ) ) 
   branch_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) > 1 ]  
   root = [ n[0]  for n in skdata[0] if n[1] == None ][0]

   #Walk from all end points to the root - if you hit the cut node assign this branch to neuronA otherwise neuronB
   distal_nodes = []
   proximal_nodes = []

   module_logger.info('Cutting neuron...')
   for i, en in enumerate(end_nodes):     
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

   #Remove cut_node from distal -> we have to add this manually later on with parent = None
   distal_nodes = list ( set(distal_nodes) )
   distal_nodes.remove( cut_node )
   proximal_nodes = list ( set(proximal_nodes) )

   neuron_dist = [ [ n for n in skdata[0] if n[0] in distal_nodes ] , [ c for c in skdata[1] if c[0] in distal_nodes ] ]   
   neuron_dist[0].append( [ [ n[0], None , n[2], n[3], n[4], n[5], n[6], n[7] ] for n in skdata[0] if n[0] == cut_node ][0] )

   neuron_prox = [ [ n for n in skdata[0] if n[0] in proximal_nodes ] , [ c for c in skdata[1] if c[0] in proximal_nodes ] ]

   module_logger.info('Cutting done after %is' % round ( time.time()- start_time ) )   
   module_logger.info('Distal to cut node: %i nodes/%i synapses' % (len( neuron_dist[0] ),len( neuron_dist[1] )) )
   module_logger.info('Proximal to cut node: %i nodes/%i synapses' % (len( neuron_prox[0] ),len( neuron_prox[1] )) )

   return neuron_dist, neuron_prox

def synapse_root_distances(skid, skdata, remote_instance, pre_skid_filter = [], post_skid_filter = [] ):    
   """ Calculates distance of synapses to root (i.e. soma)

   Parameter
   ---------
   skid :               this neuron's skeleton id
   skdata :             CATMAID skeleton data
   pre_skid_filter :    (optional) if provided, only synapses from these neurons will be processed
   post_skid_filter :   (optional) if provided, only synapses to these neurons will be processed

   Returns
   -------
   [1] pre_node_distances :   {'connector_id: distance_to_root[nm]'} for all presynaptic sistes of this neuron
   [2] post_node_distances :  {'connector_id: distance_to_root[nm]'} for all postsynaptic sites of this neuron
   """

   cn_details = get_connector_details ( [ c[1] for c in skdata[1] ] , remote_instance = remote_instance)   

   list_of_parents = { n[0]: (n[1], n[3], n[4], n[5] ) for n in skdata[0] }    

   if pre_skid_filter or post_skid_filter:
      #Filter connectors that are both pre- and postsynaptic to the skid in skid_filter 
      filtered_cn = [ c for c in cn_details if True in [ int(f) in c[1]['postsynaptic_to'] for f in post_skid_filter ] and True in [ int(f) == c[1]['presynaptic_to'] for f in pre_skid_filter ] ]
      module_logger.debug('%i of %i connectors left after filtering' % ( len( filtered_cn ) ,len( cn_details ) ) )
   else:
      filtered_cn = cn_details 

   pre_node_distances = {}
   post_node_distances = {}
   visited_nodes = {}

   module_logger.info('Calculating distances to root')
   for i,cn in enumerate(filtered_cn):

      if i % 10 == 0:
         module_logger.debug('%i of %i' % ( i, len(filtered_cn) ) )   

      if cn[1]['presynaptic_to'] == int(skid) and cn[1]['presynaptic_to_node'] in list_of_parents:
         dist, visited_nodes = walk_to_root( [ ( n[0], n[3],n[4],n[5] ) for n in skdata[0] if n[0] == cn[1]['presynaptic_to_node'] ][0] , list_of_parents, visited_nodes )

         pre_node_distances[ cn[1]['presynaptic_to_node'] ] = dist

      if int(skid) in cn[1]['postsynaptic_to']:
         for nd in cn[1]['postsynaptic_to_node']:
            if nd in list_of_parents:                    
               dist, visited_nodes = walk_to_root( [ ( n[0], n[3],n[4],n[5] ) for n in skdata[0] if n[0] == nd ][0] , list_of_parents, visited_nodes )                

               post_node_distances[ nd ] = dist
         
   return pre_node_distances, post_node_distances

def calc_dist(v1,v2):        
    return math.sqrt(sum(((a-b)**2 for a,b in zip(v1,v2))))

def calc_cable( skdata , smoothing = 1, remote_instance = None ):
   """ Calculates cable length in micro meter (um) of a given neuron     

    Parameters:
    -----------
    skdata :         either a skeleton ID or 3d skeleton data (optional)
                     if skeleton ID, 3d skeleton data will be pulled from CATMAID server
    smoothing :      int (default = 1)
                     use to smooth neuron by downsampling; 1 = no smoothing                  
    remote_instance :   CATMAID instance (optional)
                        pass if skdata is a skeleton ID, not 3D skeleton data

    Returns:
    --------
    cable_length [um]
    """

   if type(skdata) != type( list() ) :
      skdata = get_3D_skeleton( [skdata], rm)[0]

   if smoothing > 1:
      skdata = downsample_neuron(skdata, smoothing)

   parent_loc = { n[0] : [ p for p in skdata[0] if p[0] == n[1]][0][3:6] for n in skdata[0] if n[1] != None }

   #Now add distance between all child->parents
   return round( sum( [ calc_dist( n[3:6], parent_loc[n[0]] ) for n in skdata[0] if n[1] != None ]  )/1000, 3)

def walk_to_root( start_node, list_of_parents, visited_nodes ):
    """ Walks to root from start_node and sums up geodesic distances along the way.     

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
