""" Collection of tools to turn CATMAID neurons into iGraph objects to efficiently calculate distances and cluster synapses.

Basic example:
------------

from pymaid import CatmaidInstance, get_3D_skeleton
from catmaid_igraph import igraph_from_skeleton, cluster_nodes_w_synapses

remote_instance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Example skid
skid = '12345'

#Retrieve 3D skeleton data for neuron of interest
skdata = get_3D_skeleton ( [ example_skid ], remote_instance, connector_flag = 1, tag_flag = 0 )[0]

#Generate iGraph object from node data
g = igraph_from_skeleton( skdata, remote_instance)

#Cluster synapses - creates and saves plot, and returns clustering for nodes with synapses
syn_linkage = cluster_nodes_w_synapses( g, filename = 'test.png' )

#Find the last two clusters (= the two biggest):
clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters),len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))

Contents:
-------

- igraph_from_skeleton
- synapse_distance_matrix
- cluster_nodes_w_synapses

"""

import math, pylab
import numpy as np
import matplotlib.pyplot as plt
from igraph import *
from scipy import cluster, spatial


def igraph_from_skeleton(skdata,remote_instance):
	""" Takes CATMAID skeleton data and turns it into an iGraph object
	
	Parameters:
	==========
	skdata :			list of 3D skeleton data
						As retrieved from CATMAID server.
	remote_instance :	CATMAID Instance
						See <pymaid> for example.

	Returns:
	=======
	iGraph object

	"""	

	print('Generating graph from skeleton data...')

	#Generate list of vertices -> this order is retained
	vlist = [n[0] for n in skdata[0]]
	
	#Generate list of edges based on index of vertices
	elist = []
	for i, n in enumerate( skdata[0] ):
		if n[1] == None:
			continue
		elist.append( ( vlist.index( n[1] ), i ) )	

	#Generate graph and assign custom properties
	g = Graph( elist , n = len( vlist ) ,  directed = True)	
	g.vs['node_id'] = [ n[0] for n in skdata[0] ]
	g.vs['parent_id'] = [ n[1] for n in skdata[0] ]
	g.vs['X'] = [ n[3] for n in skdata[0] ]
	g.vs['Y'] = [ n[4] for n in skdata[0] ]
	g.vs['Z'] = [ n[5] for n in skdata[0] ]

	#Find nodes with synapses and assign them the custom property 'has_synapse'
	nodes_w_synapses = [ n[0] for n in skdata[1] ]
	g.vs['has_synapse'] = [ n[0] in nodes_w_synapses for n in skdata[0] ]

	#Generate weights by calculating edge lengths
	w = [ math.sqrt( (skdata[0][e[0]][3]-skdata[0][e[1]][3])**2 + (skdata[0][e[0]][4]-skdata[0][e[1]][4])**2 + (skdata[0][e[0]][5]-skdata[0][e[1]][5])**2 ) for e in elist ]	
	g.es['weight'] = w

	return g

def synapse_distance_matrix(synapse_data, labels = None):
	""" Takes a list of CATMAID synapses [[parent_node, connector_id, 0/1 , x, y, z ],[],...], calculates euclidian distance matrix and clusters them (WARD)

	Parameters:
	=========
	synapse_data : 	list of synapses
					As received from CATMAID 3D skeleton [ [parent_node, connector_id, 0/1 , x, y, z ], ... ].
	labels :		list of strings
					Labels for each leaf of the dendrogram (e.g. connector ids).	

	Returns:
	=======
	Plots dendrogram and distance matrix
	Returns hierachical clustering
	"""		

	#Generate numpy array containing x, y, z coordinates
	s = np.array( [ [e[3],e[4],e[5] ] for e in synapse_data ] )

	#Calculate euclidean distance matrix 
	condensed_dist_mat = spatial.distance.pdist( s , 'euclidean' )
	squared_dist_mat = spatial.distance.squareform( condensed_dist_mat )

	# Compute and plot first dendrogram for all nodes.
	fig = pylab.figure(figsize=(8,8))
	ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
	Y = cluster.hierarchy.ward(squared_dist_mat)
	Z1 = cluster.hierarchy.dendrogram(Y, orientation='left', labels = labels)
	ax1.set_xticks([])
	ax1.set_yticks([])

	# Compute and plot second dendrogram.
	ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
	Y = cluster.hierarchy.ward(squared_dist_mat)
	Z2 = cluster.hierarchy.dendrogram(Y, labels = labels)
	ax2.set_xticks([])
	ax2.set_yticks([])

	# Plot distance matrix.
	axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	idx1 = Z1['leaves']
	idx2 = Z2['leaves']	
	D = squared_dist_mat
	D = D[idx1,:]
	D = D[:,idx2]
	im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])

	# Plot colorbar.
	axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
	pylab.colorbar(im, cax=axcolor)
	fig.show()	

	return Y

def cluster_nodes_w_synapses(g, plot_graph = True):
	""" Cluster nodes of an iGraph object based on distance

	Parameters:
	=========
	g : 			iGraph object
					Holds the skeleton.	
	plot_graph : 	boolean
					If true, plots a Graph.

	Returns:
	=======
	Plots dendrogram and distance matrix
	Returns hierachical clustering
	"""	

	print('Generating distance matrix for neuron...')
	#Generate distance matrix.
	distance_matrix = g.shortest_paths_dijkstra ( mode = 'All', weights='weight' )

	#List of nodes without synapses
	not_synapse_nodes = [ v.index for v in g.vs.select(has_synapse=False) ]

	#Delete non synapse nodes from distance matrix (columns first, then rows)
	distance_matrix_syn = np.delete(distance_matrix,not_synapse_nodes,0)
	distance_matrix_syn = np.delete(distance_matrix_syn,not_synapse_nodes,1)	

	print('Clustering nodes with synapses...')		
	Y_syn = cluster.hierarchy.ward(distance_matrix_syn)

	if plot_graph:

		# Compute and plot first dendrogram for all nodes.
		fig = pylab.figure(figsize=(8,8))
		ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
		Y_all = cluster.hierarchy.ward(distance_matrix)
		Z1 = cluster.hierarchy.dendrogram(Y_all, orientation='left')
		ax1.set_xticks([])
		ax1.set_yticks([])

		# Compute and plot second dendrogram for synapse nodes only.
		ax2 = fig.add_axes([0.3,0.71,0.6,0.2])		
		Z2 = cluster.hierarchy.dendrogram(Y_syn)
		ax2.set_xticks([])
		ax2.set_yticks([])

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

		# Plot colorbar.
		axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
		pylab.colorbar(im, cax=axcolor)
		fig.show()
		

	return Y_syn