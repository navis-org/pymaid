""" Collection of tools to turn CATMAID neurons into iGraph objects to efficiently calculate distances and cluster synapses.

Basic example:
------------

from pymaid import CatmaidInstance, get_3D_skeleton
from catmaid_igraph import igraph_from_skeleton, cluster_nodes_w_synapses

remote_instance = CatmaidInstance(  'www.your.catmaid-server.org', 
                                    'user', 
                                    'password', 
                                    'token' 
                                 )

#Example skid
skid = '12345'

#Retrieve 3D skeleton data for neuron of interest
skdata = get_3D_skeleton ( [ example_skid ], 
                           remote_instance, 
                           connector_flag = 1, 
                           tag_flag = 0 )[0]

#Generate iGraph object from node data
g = igraph_from_skeleton( skdata, remote_instance)

#Cluster synapses - generates plot and returns clustering for nodes with synapses
syn_linkage = cluster_nodes_w_synapses( g, plot_graph = True )

#Find the last two clusters (= the two biggest):
clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters), len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))

"""

import math, pylab
import numpy as np
import matplotlib.pyplot as plt
from igraph import *
from scipy import cluster, spatial
import logging
import pandas as pd

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

def igraph_from_adj_mat( adj_matrix, **kwargs ):
   """ Takes an adjacency matrix and turns it into an iGraph object

   Parameters:
   ----------
   adj_matrix :         Pandas dataframe 
                        adjacency matrix - e.g. from pymaid.cluster

   Optional kwargs:
   syn_threshold :      edges with less connections will be ignored
   syn_cutoff :         edges with more connections will be maxed at syn_cutoff

   Returns:
   -------
   iGraph representation of network   

   Example:
   -------
   from pymaid import pymaid, cluster, igraph_catmaid
   from igraph import plot as gplot

   remote_instance = pymaid.CatmaidInstance( URL, HTTP_USER, HTTP_PW, TOKEN )

   neurons = pymaid.get_skids_by_annotation( 'right_pns' ,remote_instance)
   mat = cluster.create_adjacency_matrix( neurons, neurons, remote_instance )
   g = igraph_catmaid.igraph_from_adj_mat ( mat )

   #Fruchterman-Reingold algorithm
   layout = g.layout('fr')
   gplot( g, layout = layout )
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

def igraph_from_skeleton(skdata):
   """ Takes CATMAID single skeleton data and turns it into an iGraph object
   
   Parameters:
   ----------
   skdata :             Pandas dataframe containing a SINGLE neuron  

   Returns:
   -------
   iGraph representation of the neuron

   """   

   if type( skdata ) == type( pd.DataFrame() ):
      df = skdata.ix[0]
   elif type( skdata ) == type( pd.Series() ):
      df = skdata 

   module_logger.info('Generating graph from skeleton data...')

   #Generate list of vertices -> this order is retained
   vlist = skdata.nodes.treenode_id.tolist()   
   
   #Generate list of edges based on index of vertices   
   elist = [ [ n.Index, vlist.index( n.parent_id )  ] for n in skdata.nodes.itertuples() if n.parent_id != None ]  

   #Generate graph and assign custom properties
   g = Graph( elist , n = len( vlist ) ,  directed = True)     
   g.vs['node_id'] = skdata.nodes.treenode_id.tolist()
   g.vs['parent_id'] = skdata.nodes.parent_id.tolist()
   g.vs['X'] = skdata.nodes.x.tolist()
   g.vs['Y'] = skdata.nodes.y.tolist()
   g.vs['Z'] = skdata.nodes.z.tolist()

   #Find nodes with synapses and assign them the custom property 'has_synapse'
   nodes_w_synapses = skdata.connectors.treenode_id.tolist()
   g.vs['has_synapse'] = [ n[0] in nodes_w_synapses for n in skdata[0] ]

   #Generate weights by calculating edge lengths = distance between nodes
   tn_coords = skdata.nodes.ix[ [ e[0] for e in elist ]  ][ ['x','y','z' ] ].reset_index()
   parent_coords = skdata.nodes.ix[ [ e[1] for e in elist ]  ][ [ 'x','y','z'] ].reset_index()
   w = np.sqrt( np.sum(( tn_coords[ ['x','y','z' ] ] - parent_coords[ ['x','y','z' ] ] ) **2, axis=1 )).tolist()
   g.es['weight'] = w

   return g

def calculate_distance_from_root( g, synapses_only = False ):
   """ Get distance to root for nodes with synapses

   Parameters:
   ----------
   g :               iGraph object
                     Holds the skeleton
   synapses_only :   boolean
                     If True, only distances for nodes with synapses will be returned

   Returns:
   -------  
   dict :            {node_id : distance_to_root }
   
   """

   module_logger.info('Generating distance matrix for neuron...')

   #Generate distance matrix.
   distance_matrix = g.shortest_paths_dijkstra ( mode = 'All', weights='weight' )

   if synapses_only:
      nodes = [ ( v.index, v['node_id'] ) for v in g.vs.select(has_synapse=True) ]
   else:
      nodes = [ ( v.index, v['node_id'] ) for v in g.vs ]

   root = [ v.index for v in g .vs if v['parent_id'] == None ][0]

   distances_to_root = {}

   for n in nodes:
      distances_to_root[ n[1] ] = distance_matrix[ n[0] ][ root ]

   return distances_to_root

def cluster_nodes_w_synapses(g, plot_graph = True):
   """ Cluster nodes of an iGraph object based on distance

   Parameters:
   ----------
   g :            iGraph object
                  Holds the skeleton.  
   plot_graph :   boolean
                  If true, plots a Graph.

   Returns:
   -------
   Plots dendrogram and distance matrix
   Returns hierachical clustering
   """   

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