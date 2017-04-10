pymaid
==================

Collection of [Python](ww.python.org "Python Homepage") 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.

`pymaid.pymaid` is the low level library to connect to CATMAID servers and fetch data. Most data is not reformatted after response from the server.

`pymaid.igraph` contains a wrapper to turn CATMAID skeletons in [iGraph](http://www.igraph.org) objects which can then be used to e.g. quickly calculate geodesic (along the arbor) distances and cluster synapses. 

`pymaid.anatomy` contains wrappers to downsample or cut neurons

`pymaid.plot` contains a wrapper to generate 2D morphology plots of neurons

`pymaid.cluster` contains wrappers to cluster neurons 

## Installation
I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install pymaid.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:  

`pip3 install git+git://github.com/schlegelp/pymaid@master`  

This command should also work to update the package.

*Attention*: on Windows, the dependencies (i.e. Numpy and SciPy) will likely fail to install. Your best bet is to get a Python distribution that already includes them (e.g. [Anaconda](https://www.continuum.io/downloads)).

#### Dependencies:
`pymaid` uses standard Python 3 libraries

`catmaid_igraph` requires [iGraph](http://www.igraph.org), [SciPy](http://www.scipy.org), [Numpy](http://www.scipy.org) and [Matplotlib](http://www.matplotlib.org)

`plot` requires [matplotlib](http://matplotlib.org/) and [Plotly](http://plot.ly)

`anatomy` uses standard Python 3 libraries, [iGraph](http://www.igraph.org) is optional 

`cluster` requires [SciPy](http://www.scipy.org)

## Basic examples:

### Retrieve 3D skeleton data
```python
from pymaid.pymaid import CatmaidInstance, get_3D_skeleton

#Initialize Catmaid instance 
myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Retrieve skeleton data for two neurons
skdata = get_3D_skeleton ( ['12345','67890'] , myInstance )
#e.g. skdata[0][0] holds the nodes of skeleton '12345' 
#while skdata[0][1] holds connectors of skeleton '12345'
```
### Cluster synapses based on distance along the arbor using iGraph
```python
from pymaid.pymaid import CatmaidInstance, get_3D_skeleton
from pymaid.igraph_catmaid import igraph_from_skeleton, cluster_nodes_w_synapses

#Initiate Catmaid instance
remote_instance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Example skid
skid = '12345'

#Retrieve 3D skeleton data for neuron of interest
skdata = get_3D_skeleton ( [ example_skid ], remote_instance, connector_flag = 1, tag_flag = 0 )[0]

#(Optional) Consider downsampling for large neuronns (preverses branch points, end points, synapses, etc.)
from pymaid.anatomy import downsample_neuron
skdata = downsample_neuron( skdata, 4 )

#Generate iGraph object from node data
g = igraph_from_skeleton( skdata )

#Cluster synapses - generates plot and returns clustering for nodes with synapses
syn_linkage = cluster_nodes_w_synapses( g, plot_graph = True )

#Find the last two clusters (= the two largest clusters):
clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters),len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))
```

## Additional examples:
Check out [/examples/](https://github.com/schlegelp/PyMaid/tree/master/examples) for a growing list of Jupyter notebooks.

## Contents:
### pymaid.pymaid:
Currently **pymaid** features a range of wrappers to conveniently fetch data from CATMAID servers.
Use e.g. `help(get_edges)` to learn more about their function, parameters and usage.

- `add_annotations()`: use to add annotation(s) to neuron(s)
- `get_3D_skeleton()`: get neurons' skeleton(s) - i.e. what the 3D viewer in CATMAID shows
- `get_arbor()`: similar to get_3D_skeleton but more detailed information on connectors
- `get_edges()`: get edges (connections) between sets of neurons
- `get_connectors()`: get connectors (synapses, abutting and/or gap junctions) for set of neurons
- `get_connector_details()`: get details for connector (i.e. all neurons connected to it)
- `get_logs()`: get what the log widged shows (merges, splits, etc.)
- `get_review()`: get review status for set of neurons
- `get_review_details()`: get review status (reviewer + timestamp) for each individual node
- `get_neuron_annotation()`: get annotations of a **single** neuron (includes user and timestamp)
- `get_annotations_from_list()`: get annotations of a set of neurons (annotation only)
- `get_neurons_in_volume()`: get neurons in a defined box volume
- `get_contributor_statistics()`: get contributors (nodes, synapses, etc) for a set of neurons
- `get_skids_by_annotation()`: get skeleton IDs that are annotated with a given annotation
- `get_skids_by_name()`: get skeleton IDs of neurons with given names
- `get_skeleton_list()`: retrieve neurons that fit certain criteria (e.g. user, size, dates)
- `get_history()`: retrieve project history similar to the project statistics widget
- `get_partners()`: retrieve connected partners for a list of neurons
- `get_names()`: retrieve names of a set of skeleton IDs
- `get_node_lists()`: retrieve list of nodes within given volume
- `skid_exists()`: checks if a skeleton ID exists
- `get_volume()`: get volume (verts + faces) of CATMAID volumes

### pymaid.igraph_catmaid:
- `igraph_from_skeleton()`: generates iGraph object from CATMAID neurons
- `calculate_distance_from_root()`: calculates geodesic (along-the-arbor) distances for nodes to root node
- `cluster_nodes_w_synapses()`: uses iGraph's shortest_paths_dijkstra to cluster nodes with synapses

### pymaid.plot:
- `plot2d()`: generates 2D plots of neurons
- `plot3d()`: uses [Plotly](http://plot.ly) to generate 3D plots of neurons

### pymaid.cluster:
- `synapse_distance_matrix()`: cluster synapses based on eucledian distance
- `create_adjacency_matrix()`: create a Pandas dataframe containing the adjacency matrix for two sets of neurons
- `create_connectivity_distance_matrix()`: returns distance matrix based on connectivity similarity (Jarrell et al., 2012)

### pymaid.anatomy:
- `downsample_neuron()`: take skeleton data and reduces the number of nodes while preserving synapses, branch points, etc.
- `cut_neuron()`: virtually cuts a neuron at given treenode and returns the distal and the proximal part
- `cut_neuron2()`: similar to above but uses iGraph (slightly faster)
- `synapse_root_distances()`: similar to `pymaid.igraph.calculate_distance_from_root` but does not use iGraph
- `calc_cable()`: calculate cable length of given neuron

## License:
This code is under GNU GPL V3
