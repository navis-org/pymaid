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

from pymaid import get_3D_skeleton, get_connectors, get_connector_details, retrieve_skids_by_annotation
import math

def generate_list_of_childs(skdata):
	""" Transforms list of nodes into a dictionary { parent: [child1,child2,...]}
	
	Parameter:
	---------
	skdata :			CATMAID skeleton for a single neuron

	Returns:
	list_of_childs :	dict()

	"""
	print('Generating list of childs...')
	list_of_childs = { n[0] : [] for n in skdata[0] }   

	for n in skdata[0]:
		try:
			list_of_childs[ n[1] ].append( n[0] )
		except:
			list_of_childs[None]=[None]

	print('Done')

	return list_of_childs

def downsample_neuron ( skdata, resampling_factor):
	""" Downsamples a neuron by a given factor. Preserves root, leafs, branchpoints and synapse nodes
	
	Parameter
	---------
	skdata : 			CATMAID skeleton data
	resampling_factor :	Factor by which to reduce the node count

	Returns
	-------
	skdata :			CATMAID skeleton data (reduced)

	"""

	list_of_childs  = generate_list_of_childs(skdata)
	list_of_parents = { n[0]:n[1] for n in skdata[0] }

	end_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) == 0 ]   
	branch_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) > 1 ]  
	root = [ n[0]  for n in skdata[0] if n[1] == None ][0]
	synapse_nodes = [ n[0] for n in skdata[1] ]

	fix_points = list ( set( end_nodes + branch_nodes + synapse_nodes ) )

	#Walk from all fix points to the root - jump N nodes on the way
	new_parents = {}

	print('Sampling neuron down by factor of', resampling_factor)
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

	new_nodes = [ [ n[0], new_parents[ n[0] ] ,n[1],n[2],n[3],n[4],n[5],n[6],n[7] ] for n in skdata[0] if n[0] in new_parents ]

	print('Node before:', len( skdata[0] ))
	print('Node after:', len( new_nodes ))

	return [ new_nodes, skdata[1] ]


def cut_neuron( skdata, cut_node ):
	""" Cuts the neurons at given point and returns two new neurons.

	Parameter
	---------
	skdata : 	CATMAID skeleton data
	cut_node :	ID of the node to cut

	Returns
	-------
	neuronA :	distal to the cut
	neuronB :	proximal to the cut
	"""

	list_of_childs  = generate_list_of_childs(skdata)
	list_of_parents = { n[0]:n[1] for n in skdata[0] }


	if len( list_of_childs[ cut_node ] ) == 0:
		print('Can not cut: cut_node is a leaf node!')
		return
	elif list_of_parents[ cut_node ] == None:
		print('Can not cut: cut_node is a root node!')
		return

	end_nodes = list ( set( [ n for n in list_of_childs if len(list_of_childs[n]) == 0 ] + [ cut_node ] ) ) 
	branch_nodes = [ n for n in list_of_childs if len(list_of_childs[n]) > 1 ]  
	root = [ n[0]  for n in skdata[0] if n[1] == None ][0]

	#Walk from all end points to the root - if you hit the cut node assign this branch to neuronA otherwise neuronB
	distal_nodes = []
	proximal_nodes = []

	print('Cutting neuron...')
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

	neuronA = [ [ n for n in skdata[0] if n[0] in distal_nodes ] , [ c for c in skdata[1] if c[0] in distal_nodes ] ]	
	neuronA[0].append( [ [ n[0], None , n[2], n[3], n[4], n[5], n[6], n[7] ] for n in skdata[0] if n[0] == cut_node ][0] )

	neuronB = [ [ n for n in skdata[0] if n[0] in proximal_nodes ] , [ c for c in skdata[1] if c[0] in proximal_nodes ] ]

	print('Done!')
	print('Distal to cut node: %i nodes' % len( neuronA[0] ) )
	print('Proximal to cut node: %i nodes' % len( neuronB[0] ) )

	return neuronA, neuronB

def synapse_root_distances(skid, skdata, remote_instance, pre_skid_filter = [], post_skid_filter = [] ):    
	""" Calculates distance of synapses to root (i.e. soma)

	Parameter
	---------
	skid :				this neuron's skeleton id
	skdata : 			CATMAID skeleton data
	pre_skid_filter :	(optional) if provided, only synapses from these neurons will be processed
	post_skid_filter :	(optional) if provided, only synapses to these neurons will be processed

	Returns
	-------
	pre_node_distances :	{'connector_id: distance_to_root[nm]'} for all presynaptic sistes of this neuron
	post_node_distances :	{'connector_id: distance_to_root[nm]'} for all postsynaptic sites of this neuron
	"""

	cn_details = get_connector_details ( [ c[1] for c in skdata[1] ] , remote_instance = remote_instance)   

	list_of_parents = { n[0]: (n[1], n[3], n[4], n[5] ) for n in skdata[0] }    

	if pre_skid_filter or post_skid_filter:
		#Filter connectors that are both pre- and postsynaptic to the skid in skid_filter 
		filtered_cn = [ c for c in cn_details if True in [ int(f) in c[1]['postsynaptic_to'] for f in post_skid_filter ] and True in [ int(f) == c[1]['presynaptic_to'] for f in pre_skid_filter ] ]
		print('%i of %i connectors left after filtering' % ( len( filtered_cn ) ,len( cn_details ) ) )
	else:
		filtered_cn = cn_details 

	pre_node_distances = {}
	post_node_distances = {}
	visited_nodes = {}

	print('Calculating distances to root')
	for i,cn in enumerate(filtered_cn):

		if i % 10 == 0:
			print('%i of %i' % ( i, len(filtered_cn) ) )   

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

def walk_to_root( start_node, list_of_parents, visited_nodes ):
    """
    start_node :        (node_id, x,y,z)
    list_of_parents :   {node_id: (parent_id, x,y,z) }
    visited_nodes :     {node_id: distance_to_root}
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
	from connect_catmaid import connect_adult_em
	from pymaid import get_3D_skeleton

	rm = connect_adult_em()

	skdata = get_3D_skeleton( [2333007], rm)[0]     

	nA, nB = cut_neuron( skdata, 4450214 )

	downsample_neuron ( nA , 10 )
