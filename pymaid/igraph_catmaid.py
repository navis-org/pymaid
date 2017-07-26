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

""" Collection of tools to turn CATMAID neurons into iGraph objects to 
efficiently calculate distances and cluster synapses.

Examples
--------
>>> from pymaid import CatmaidInstance, get_3D_skeleton
>>> from catmaid_igraph import neuron2graph, cluster_nodes_w_synapses
>>> remote_instance = CatmaidInstance(    'www.your.catmaid-server.org', 
...                                       'user', 
...                                       'password', 
...                                       'token' 
...                                    )
>>> #Example skid
>>> skid = '12345'
>>> #Retrieve 3D skeleton data for neuron of interest
>>> skdata = get_3D_skeleton ( [ example_skid ], 
...                             remote_instance, 
...                              connector_flag = 1, 
...                              tag_flag = 0 )[0]
>>> #Cluster synapses - generates plot and returns clustering for nodes with synapses
>>> syn_linkage = cluster_nodes_w_synapses( skdata, plot_graph = True )
>>> #Find the last two clusters (= the two biggest):
>>> clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')
>>> #Print summary
>>> print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters), 
... len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))

"""

import math, pylab
import numpy as np
import matplotlib.pyplot as plt
from igraph import *
from scipy import cluster, spatial
import logging
import pandas as pd
import time
from pymaid import core, pymaid

#Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)
if not module_logger.handlers:
   #Generate stream handler
   sh = logging.StreamHandler()
   sh.setLevel(logging.INFO)
   #Create formatter and add it to the handlers
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   sh.setFormatter(formatter)
   module_logger.addHandler(sh)

def network2graph( x, remote_instance=None, threshold=1 ):
   """ Generates igraph network object for a set of neurons

   Parameters
   ----------
   x :                 Catmaid Neurons as single or list of either:
                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
   remote_instance :   CATMAID instance; either pass directly to function or 
                       define globally as 'remote_instance'
   threshold :         int
                       Connections weaker than this will be excluded 

   Returns
   ------- 
   iGraph representation of the network 
   """

   if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            print('Please either pass a CATMAID instance or define globally as "remote_instance" ')
            return

   module_logger.info('Generating graph from skeleton data...')

   skids = pymaid.eval_skids ( x , remote_instance = remote_instance )
   indices = { s: skids.index(s) for s in skids }

   try:
      neuron_names = x.neuron_names.tolist()
   except:
      names = pymaid.get_names( skids, remote_instance = remote_instance )
      neuron_names = [ names[ str(n) ] for n in skids ]

   edges = pymaid.get_edges (skids, remote_instance = None )
   edges_by_index = [ [ indices[e.source_skid], indices[e.target_skid] ] for e in edges[ edges.weight >= threshold ].itertuples() ]    

   #Generate graph and assign custom properties
   g = Graph( directed = True )
   g.add_vertices( len(skids) )
   g.add_edges( edges_by_index )

   g.vs['node_id'] = skids
   g.vs['neuron_name'] = neuron_names   
   g.es['weight'] = edges.weight.tolist()

   return g

def matrix2graph( adj_matrix, **kwargs ):
   """ Takes an adjacency matrix and turns it into an iGraph object

   Parameters
   ----------
   adj_matrix :      Pandas dataframe 
                     Adjacency matrix - e.g. from :func:`pymaid.cluster.create_adjacency_matrix`   
   syn_threshold :   int, optional     
                     Edges with less connections will be ignored
   syn_cutoff :      int, optional
                     Edges with more connections will be maxed at syn_cutoff

   Returns
   -------
   iGraph object
                     Representation of network   

   Examples
   --------
   >>> from pymaid import pymaid, cluster, igraph_catmaid
   >>> from igraph import plot as gplot
   >>> remote_instance = pymaid.CatmaidInstance(   URL, 
   ...                                             HTTP_USER, 
   ...                                             HTTP_PW, 
   ...                                             TOKEN )
   >>> neurons = pymaid.get_skids_by_annotation( 'right_pns' ,remote_instance)
   >>> mat = cluster.matrix2graph(neurons,neurons,remote_instance)
   >>> g = igraph_catmaid.matrix2graph ( mat )
   >>> #Use fruchterman-Reingold algorithm
   >>> layout = g.layout('fr')
   >>> gplot( g, layout = layout )
   """   

   syn_threshold = kwargs.get('syn_threshold', 1 )
   syn_cutoff = kwargs.get('syn_cutoff', None )

   cols = adj_matrix.columns.tolist()
   rows = adj_matrix.index.tolist()

   #Extract values
   v = adj_matrix.values      
   #Set values < syn_threshold to 0 -> will not show up on v.nonzero()
   v [ v < syn_threshold ] = 0

   #Get unique neurons in adj matrix
   neurons = list( set( cols + rows ) )   

   #Generate dict containing the indices of neurons
   neurons_index = { n : i for i,n in enumerate( neurons ) }

   #nonzero(): First index is row, second is column

   #edges = [ ( neurons.index ( rows[ v.nonzero()[0][i] ] ), neurons.index( cols[ v.nonzero()[1][i] ] ) ) for i in range( len( v.nonzero()[0] ) ) if v[ v.nonzero()[0][i] ][ v.nonzero()[1][i] ] >= syn_threshold ]
   #weights = [ v[ v.nonzero()[0][i] ][ v.nonzero()[1][i] ] for i in range( len( v.nonzero()[0] ) ) if v[ v.nonzero()[0][i] ][ v.nonzero()[1][i] ] >= syn_threshold ]   

   #Get list of edges as indices of the vertices in the graph
   edges = [ ( neurons_index [ rows[r] ] , neurons_index [ cols[c] ] ) for r,c in zip( v.nonzero()[0], v.nonzero()[1] ) ]    
   weights = [ v[r][c] for r,c in zip( v.nonzero()[0], v.nonzero()[1] ) ]    

   if syn_cutoff:
      weights = [ min( e, syn_cutoff ) for e in weights ]

   g = Graph(directed = True)

   #Add vertices
   g.add_vertices( len(neurons) )
   g.vs['label'] = neurons

   #Add edges
   g.add_edges( edges )
   g.es['weight'] = weights

   return g

def neuron2graph( skdata, append=True ):
   """ Takes CATMAID single skeleton data and turns it into an iGraph object
   
   Parameters
   ----------
   skdata :          {Pandas dataframe, CatmaidNeuron}
                     Containing either a single or multiple neurons
   append :          bool, optional
                     Unless False, graph is automatically appended to the 
                     dataframe. Default = True

   Returns
   -------
   iGraph object
                     Representation of the neuron 

   """
   if isinstance(skdata, pd.DataFrame) or isinstance(skdata, core.CatmaidNeuronList):            
      return [ neuron2graph( skdata.ix[i] ) for i in range( skdata.shape[0] ) ]
   elif isinstance(skdata, pd.Series) or isinstance( skdata, core.CatmaidNeuron ):
      df = skdata

   module_logger.info('Generating graph from skeleton data...')

   #Generate list of vertices -> this order is retained
   vlist = df.nodes.treenode_id.tolist()

   #Get list of edges as indices (needs to exclude root node)
   tn_index_with_parent = df.nodes[ ~df.nodes.parent_id.isnull() ].index.tolist()
   parent_ids = df.nodes[ ~df.nodes.parent_id.isnull() ].parent_id.tolist()
   df.nodes['temp_index'] = df.nodes.index #add temporary column
   parent_index = df.nodes.set_index('treenode_id').ix[ parent_ids ]['temp_index'].tolist()
   df.nodes.drop('temp_index', axis = 1, inplace=True) #remove temporary column

   #Generate list of edges based on index of vertices   
   elist = list( zip( tn_index_with_parent , parent_index ) )      
   
   #Save this as backup
   #elist = [ [ n.Index, vlist.index( n.parent_id )  ] for n in df.nodes.itertuples() if n.parent_id != None ]     

   #Generate graph and assign custom properties
   g = Graph( elist , n = len( vlist ) ,  directed = True)     
   g.vs['node_id'] = df.nodes.treenode_id.tolist()
   g.vs['parent_id'] = df.nodes.parent_id.tolist()
   g.vs['X'] = df.nodes.x.tolist()
   g.vs['Y'] = df.nodes.y.tolist()
   g.vs['Z'] = df.nodes.z.tolist()

   #Find nodes with synapses and assign them the custom property 'has_synapse'
   nodes_w_synapses = df.connectors.treenode_id.tolist()
   g.vs['has_synapse'] = [ n in nodes_w_synapses for n in df.nodes.treenode_id.tolist() ]

   #Generate weights by calculating edge lengths = distance between nodes
   tn_coords = df.nodes.ix[ [ e[0] for e in elist ]  ][ ['x','y','z' ] ].reset_index()
   parent_coords = df.nodes.ix[ [ e[1] for e in elist ]  ][ [ 'x','y','z'] ].reset_index()
   w = np.sqrt( np.sum(( tn_coords[ ['x','y','z' ] ] - parent_coords[ ['x','y','z' ] ] ) **2, axis=1 )).tolist()
   g.es['weight'] = w

   if append:
      df.igraph = g      

   return g

def dist_from_root( data, synapses_only=False ):
   """ Get geodesic distance to root in nano meter (nm) for all treenodes 

   Parameters
   ----------
   data :            {iGraph object, pandas DataFrame, CatmaidNeuron}
                     Holds the skeleton data
   synapses_only :   bool, optional
                     If True, only distances for nodes with synapses will be 
                     returned (only makes sense if input is a Graph). Default = False

   Returns:
   -------     
   dict             
                     Only if ``data`` is a graph object. 
                     Format ``{node_id : distance_to_root }``

   pandas DataFrame 
                     Only if ``data`` is a pandas DataFrame:. With 
                     ``df.nodes.dist_to_root`` holding the distances to root. 

   """ 

   if isinstance(data, Graph):
      g = data
   elif isinstance(data, pd.DataFrame) or isinstance(data, core.CatmaidNeuronList): 
      return [ dist_from_root( data.ix[i] ) for i in range( data.shape[0] ) ]
   elif isinstance(data, pd.Series) or isinstance(data, core.CatmaidNeuron):
      g = data.igraph
      if g == None:
         g = neuron2graph( data )      
   else:
      raise Exception('Unexpected data type: %s' % str(type(data)))

   #Generate distance matrix.
   try:
      module_logger.info('Generating distance matrix for neuron %s #%s...' % ( data.neuron_name, str(data.skeleton_id) ) )
   except:
      module_logger.info('Generating distance matrix for igraph...'  )
      
   distance_matrix = g.shortest_paths_dijkstra ( mode = 'All', weights='weight' )

   if synapses_only:
      nodes = [ ( v.index, v['node_id'] ) for v in g.vs.select(has_synapse=True) ]
   else:
      nodes = [ ( v.index, v['node_id'] ) for v in g.vs ]

   root = [ v.index for v in g.vs if v['parent_id'] == None ][0]

   distances_to_root = {}

   for n in nodes:
      distances_to_root[ n[1] ] = distance_matrix[ n[0] ][ root ]

   if isinstance(data, Graph):
      return distances_to_root
   else:
      data.nodes['dist_to_root'] = [ distances_to_root[ n ] for n in data.nodes.treenode_id.tolist() ]  
      data.igraph = neuron2graph ( data )      
      return data

def cluster_nodes_w_synapses(data, plot_graph=True):
   """ Cluster nodes of an iGraph object based on distance

   Parameters
   ----------
   data :         {CatmaidNeuron, iGraph object}
   plot_graph :   boolean, optional
                  If true, plots a Graph. Default = True

   Returns
   -------
   cluster :      scipy clustering
                  Plots dendrogram + distance matrix and returns hierachical 
                  clustering
   """   

   if isinstance(data, Graph):
      g = data
   elif isinstance(data, pd.DataFrame) or isinstance(data, core.CatmaidNeuronList): 
      if data.shape[0] == 1:
         g = data.ix[0].igraph
      else:
         raise Exception('Please provide a SINGLE neuron.')
   elif isinstance(data, pd.Series) or isinstance(data, core.CatmaidNeuron):
      g = data.igraph
      if g == None:
         g = neuron2graph( data )      
   else:
      raise Exception('Unexpected data type: %s' % str(type(data)))

   module_logger.info('Generating distance matrix for neuron...')
   #Generate distance matrix.
   distance_matrix = g.shortest_paths_dijkstra ( mode = 'All', weights='weight' )

   #List of nodes without synapses
   not_synapse_nodes = [ v.index for v in g.vs.select(has_synapse=False) ]

   #Delete non synapse nodes from distance matrix (columns first, then rows)
   distance_matrix_syn = np.delete(distance_matrix,not_synapse_nodes,0)
   distance_matrix_syn = np.delete(distance_matrix_syn,not_synapse_nodes,1)   

   module_logger.info('Clustering nodes with synapses...')     
   Y_syn = cluster.hierarchy.ward(distance_matrix_syn)

   if plot_graph:
      module_logger.debug('Plotting graph')
      # Compute and plot first dendrogram for all nodes.
      fig = pylab.figure(figsize=(8,8))
      ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
      Y_all = cluster.hierarchy.ward(distance_matrix)
      Z1 = cluster.hierarchy.dendrogram(Y_all, orientation='left')
      ax1.set_xticks([])
      ax1.set_yticks([])
      module_logger.debug('Plotting graph.')

      # Compute and plot second dendrogram for synapse nodes only.
      ax2 = fig.add_axes([0.3,0.71,0.6,0.2])    
      Z2 = cluster.hierarchy.dendrogram(Y_syn)
      ax2.set_xticks([])
      ax2.set_yticks([])

      module_logger.debug('Plotting graph..')
      # Plot distance matrix.
      axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
      idx1 = Z1['leaves']
      idx2 = Z2['leaves']
      D = np.delete(distance_matrix,not_synapse_nodes,1)
      D = D[idx1,:]
      D = D[:,idx2]
      im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
      axmatrix.set_xticks([])
      axmatrix.set_yticks([])

      module_logger.debug('Plotting graph...')
      # Plot colorbar.
      axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
      pylab.colorbar(im, cax=axcolor)
      fig.show()

   return Y_syn