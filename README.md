pymaid
==================

Collection of [Python](ww.python.org "Python Homepage") 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.

## Required packages:
`pymaid` uses standard Python 3 libraries
`catmaid_igraph` requires iGraph, SciPy, Numpy, PyLab and Matplotlib

## Basic example:

### Retrieve 3D skeleton data
```python
from pymaid import CatmaidInstance, get_3D_skeleton

myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )
skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )
```
### Cluster synapses based on distance along the arbor using iGraph
```python
from pymaid import CatmaidInstance, get_3D_skeleton
from catmaid_igraph import igraph_from_skeleton, cluster_nodes_w_synapses

remote_instance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Example skid
skid = '12345'

#Retrieve 3D skeleton data for neuron of interest
skdata = get_3D_skeleton ( [ example_skid ], remote_instance, connector_flag = 1, tag_flag = 0 )[0]

#Generate iGraph object from node data
g = igraph_from_skeleton( skdata, remote_instance)

#Cluster synapses - generates plot and returns clustering for nodes with synapses
syn_linkage = cluster_nodes_w_synapses( g, plot_graph = True )

#Find the last two clusters (= the two largest clusters):
clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters),len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))
```

## Available wrappers:
Currently **pymaid** features a range of wrappers to conveniently fetch data from CATMAID servers.
Use help() to learn more about their function, parameters and usage.

- add_annotations
- get_3D_skeleton
- get_arbor
- get_edges
- get_connectors
- get_review
- get_neuron_annotation
- get_neurons_in_volume
- get_annotations_from_list
- get_contributor_statistics
- retrieve_annotation_id
- retrieve_skids_by_annotation
- retrieve_skeleton_list
- retrieve_history
- retrieve_partners
- retrieve_names
- retrieve_node_lists
- skid_exists

## License:
This code is under GNU GPL V3
