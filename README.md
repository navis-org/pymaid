pymaid
==================

Collection of [Python](ww.python.org "Python Homepage") 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.

`pymaid` is the low level library to connect to CATMAID servers and fetch data. Most data is not reformatted after response from the server.

`catmaid_igraph` contains a wrapper to turn CATMAID skeletons in [iGraph](http://www.igraph.org) objects which can then be used to e.g. quickly calculate geodesic (along the arbor) distances and cluster synapses. 

`natpy` contains wrappers to downsample or cut neurons

`plotneuron` is a wrapper to generate 2D morphology plots of neurons

## Installation
I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install pymaid.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:
`pip install git+git://github.com/schlegelp/pymaid@master`  
This command should also work to update the package.

*Attention*: on Windows, the dependencies (i.e. Numpy, SciPy) will likely fail to install unless the are already part of your Python distribution (e.g. [Anaconda](https://www.continuum.io/downloads)). In that case, you can choose to install pymaid manually by downloading them into a directory of your choice and adding that path to your Python `PATH` variable.

#### Dependencies (in case you need to install manually):
`pymaid` uses standard Python 3 libraries

`catmaid_igraph` requires [iGraph](http://www.igraph.org), [SciPy](http://www.scipy.org), [Numpy](http://www.scipy.org) and [Matplotlib](http://www.matplotlib.org)

`plotneuron` requires [matplotlib](http://matplotlib.org/)

`natpy` uses standard Python 3 libraries, [iGraph](http://www.igraph.org) is optional 

## Basic examples:

### Retrieve 3D skeleton data
```python
from pymaid import CatmaidInstance, get_3D_skeleton

#Initiate Catmaid instance 
myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Retrieve skeleton data for two neurons 
skeleton_data = get_3D_skeleton ( ['12345','67890'] , myInstance )
```
### Cluster synapses based on distance along the arbor using iGraph
```python
from pymaid import CatmaidInstance, get_3D_skeleton
from catmaid_igraph import igraph_from_skeleton, cluster_nodes_w_synapses

#Initiate Catmaid instance
remote_instance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Example skid
skid = '12345'

#Retrieve 3D skeleton data for neuron of interest
skdata = get_3D_skeleton ( [ example_skid ], remote_instance, connector_flag = 1, tag_flag = 0 )[0]

#(Optional) Consider downsampling for large neuronns (preverses branch points, end points, synapses, etc.)
from natpy import downsample_neuron
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

## Available wrappers:
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
- `retrieve_skids_by_annotation()`: get skeleton IDs that are annotated with a given annotation
- `retrieve_skids_by_name()`: get skeleton IDs of neurons with given names
- `retrieve_skeleton_list()`: retrieve neurons that fit certain criteria (e.g. user, size, dates)
- `retrieve_history()`: retrieve project history similar to the project statistics widget
- `retrieve_partners()`: retrieve connected partners for a list of neurons
- `retrieve_names()`: retrieve names of a set of skeleton IDs
- `retrieve_node_lists()`: retrieve list of nodes within given volume
- `skid_exists()`: checks if a skeleton ID exists

## License:
This code is under GNU GPL V3
