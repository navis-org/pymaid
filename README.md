pymaid
==================

Collection of [Python](http://www.python.org) 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.

Pymaid has been tested with CATMAID latest release version 2017.04.20 - if you are working with older versions, you may experience incompatibilities.

`pymaid.core` contains definition for CatmaidNeuron and CatmaidNeuronList classes.

`pymaid.pymaid` is the low-level library to connect to CATMAID servers and fetch data. 

`pymaid.igraph_catmaid` contains a wrapper to turn CATMAID skeletons in [iGraph](http://www.igraph.org) objects which can then be used to e.g. quickly calculate geodesic (along the arbor) distances and cluster synapses. 

`pymaid.morpho` contains wrappers to analyse or manipulate neuron morphology

`pymaid.plot` contains a wrapper to generate 2D or 3D morphology plots of neurons

`pymaid.cluster` contains wrappers to cluster neurons

`pymaid.rmaid` provides an interface with R libraries ([nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr), [catnat](https://github.com/alexanderbates/catnat) ) using [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/)

`pymaid.user_stats` contains functions for user stats and contributions

## Installation
I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install pymaid.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:  

`pip install git+git://github.com/schlegelp/pymaid@master`  

This command should also work to update the package.

**Attention**: on Windows, the dependencies (i.e. Numpy, Pandas and SciPy) will likely fail to install automatically. Your best bet is to get a Python distribution that already includes them (e.g. [Anaconda](https://www.continuum.io/downloads)).

#### External libraries used:
Installing via [PIP](https://pip.pypa.io/en/stable/installing/) should install all external dependencies. You may run into problems on Windows though. In that case, you need to install dependencies manually, here is a list of dependencies (check out `install_requires` in [setup.py](https://raw.githubusercontent.com/schlegelp/PyMaid/master/setup.py) for version info):

- [Pandas](http://pandas.pydata.org/)
- [iGraph](http://www.igraph.org) 
- [SciPy](http://www.scipy.org)
- [Numpy](http://www.scipy.org) 
- [Matplotlib](http://www.matplotlib.org)
- [vispy](http://vispy.org/) 
- [Plotly](http://plot.ly)
- [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/)

## Basic examples:

### Retrieve 3D skeleton data
```python
from pymaid.pymaid import CatmaidInstance, get_3D_skeleton
from pymaid.plot import plot3d
from pymaid.core import CatmaidNeuron, CatmaidNeuronList

#Initialize Catmaid instance 
myInstance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Initialise a single neuron with just its skeleton ID
n = CatmaidNeuron( '12345', remote_instance = myInstance )

#Retrieve a list of skeletons using an annotation
nl = get_3D_skeleton ( 'annotation:example_neurons' , myInstance )

#nl is a CatmaidNeuronList object that manages data:
#Notice that some entries show as 'NA' because that data has not yet been retrieved/calculated
print(nl)

#You can subset the neurons based on their properties
only_large = nl[ nl.n_nodes > 2000 ]
only_reviewed = nl [ nl.review_status == 100 ]
name_subset = nl[ ['name1','name2'] ]
skid_subset = nl[ [ 12345, 67890 ] ]

#To get more data, e.g. annotations, simply request that attribute explicitedly
nl[0].annotations

#To force an update of annotations
nl[0].get_annotations()

#Access treenodes of the first neuron
nodes = nl[0].nodes

#Get coordinates of all connectors
coords = nl[0].connectors[ ['x','y','z'] ].as_matrix()

#Get all branch oints:
branch_points = nl[0].nodes[ nl[0].nodes.type == 'branch' ].treenode_id

#Plot neuron -> see help(pymaid.plot.plot3d) for accepted kwargs
nl.plot3d()
```

### Cluster synapses based on distance along the arbor using iGraph
```python
from pymaid.pymaid import CatmaidInstance, get_3D_skeleton
from pymaid.igraph_catmaid import, cluster_nodes_w_synapses

#Initiate Catmaid instance
remote_instance = CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Retrieve 3D skeleton data for neuron of interest
nl = get_3D_skeleton ( [ '12345' ], remote_instance, connector_flag = 1, tag_flag = 0 )

#(Optional) Consider downsampling for large neurons (preverses branch points, end points, synapses, etc.)
nl.downsample( factor = 4 )

#Cluster synapses - generates plot and returns clustering for nodes with synapses
syn_linkage = cluster_nodes_w_synapses( ds_neuron, plot_graph = True )

#Find the last two clusters (= the two largest clusters):
clusters = cluster.hierarchy.fcluster( syn_linkage, 2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters),len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))
```

## Additional examples:
Check out [/examples/](https://github.com/schlegelp/PyMaid/tree/master/examples) for a growing list of Jupyter notebooks.

## Contents:

### pymaid.core.CatmaidNeuron:
Representation of a **single** Catmaid neuron. Can be minimally initialized with just a skeleton ID.
Data (e.g. nodes, connectors, name, review status, annotation) are retrieved/calculated on-demand the first time they are **explicitly** requested.

Primary attributes:
- `skeleton_id`: neuron's skeleton ID
- `neuron_name`: neuron's name
- `nodes`: pandas DataFrame of treenode table
- `connectors`: pandas DataFrame of connector table
- `tags`: node tags
- `annotations`: list of neuron(s) annotations
- `cable_length`: cable length(s) in nm
- `review_status`: review status of neuron(s)

### pymaid.core.CatmaidNeuronList:
Representation of a **list** of Catmaid neurons. Can be minimally initialized with just a skeleton ID.
Has the same attributes as `CatmaidNeuron` objects. Additionally it allows indexing similar to 
pandas DataFrames (see examples).

### pymaid.pymaid:
Currently **pymaid** features a range of wrappers to conveniently fetch data from CATMAID servers.
Use e.g. `help(get_edges)` to learn more about their function, parameters and usage.

- `add_annotations()`: use to add annotation(s) to neuron(s)
- `edit_tags()`: edit (add/remove) tags of treenodes or connectors
- `get_3D_skeleton()`: get neurons' skeleton(s) - i.e. what the 3D viewer in CATMAID shows
- `get_arbor()`: similar to get_3D_skeleton but more detailed information on connectors
- `get_annotations_from_list()`: get annotations of a set of neurons (annotation only)
- `get_connectors()`: get connectors (synapses, abutting and/or gap junctions) for set of neurons
- `get_connector_details()`: get details for connector (i.e. all neurons connected to it)
- `get_contributor_statistics()`: get contributors (nodes, synapses, etc) for a set of neurons
- `get_edges()`: get edges (connections) between sets of neurons
- `get_history()`: retrieve project history similar to the project statistics widget
- `get_logs()`: get what the log widged shows (merges, splits, etc.)
- `get_names()`: retrieve names of a set of skeleton IDs
- `get_neuron_annotation()`: get annotations of a **single** neuron (includes user and timestamp)
- `get_neurons_in_volume()`: get neurons in a defined box volume
- `get_node_lists()`: retrieve list of nodes within given volume
- `get_node_user_details()`: get details (creator, edition time, etc.) for individual nodes
- `get_partners()`: retrieve connected partners for a list of neurons
- `get_partners_in_volume()`: retrieve connected partners for a list of neurons within a given Catmaid volume
- `get_review()`: get review status for set of neurons
- `get_review_details()`: get review status (reviewer + timestamp) for each individual node
- `get_skids_by_annotation()`: get skeleton IDs that are annotated with a given annotation
- `get_skids_by_name()`: get skeleton IDs of neurons with given names
- `get_skeleton_list()`: retrieve neurons that fit certain criteria (e.g. user, size, dates)
- `get_user_list()`: get list of users in the project
- `get_volume()`: get volume (verts + faces) of CATMAID volumes
- `skid_exists()`: checks if a skeleton ID exists

### pymaid.igraph_catmaid:
- `calculate_distance_from_root()`: calculates geodesic (along-the-arbor) distances for nodes to root node
- `cluster_nodes_w_synapses()`: uses iGraph's `shortest_paths_dijkstra` to cluster nodes with synapses
- `igraph_from_adj_matrix()`: generates iGraph representation of network
- `igraph_from_skeleton()`: generates iGraph representation of neuron morphology

### pymaid.plot:
- `plot2d()`: generates 2D plots of neurons
- `plot3d()`: uses either [Vispy](http://vispy.org) or [Plotly](http://plot.ly) to generate 3D plots of neurons
- `plot_network()`: uses iGraph and [Plotly](http://plot.ly) to generate network plots

### pymaid.cluster:
- `create_adjacency_matrix()`: create a Pandas dataframe containing the adjacency matrix for two sets of neurons
- `create_connectivity_distance_matrix()`: returns distance matrix based on connectivity similarity (Jarrell et al., 2012)
- `group_matrix()`: groups matrix by columns or rows - use to e.g. collapse connectivity matrix into groups of neurons
- `synapse_distance_matrix()`: cluster synapses based on eucledian distance

### pymaid.morpho:
- `calc_cable()`: calculate cable length of given neuron
- `calc_strahler_index()`: calculate strahler index for each node
- `classify_nodes()`: adds a new column to a neuron's dataframe categorizing each node as branch, slab, leaf or root
- `cut_neuron()`: cut neuron at a node or node tag
- `downsample_neuron()`: takes skeleton data and reduces the number of nodes while preserving synapses, branch points, etc.
- `in_volume()`: test if points are within given CATMAID volume
- `longest_neurite()`: prunes neuron to its longest neurite
- `reroot_neuron()`: reroot neuron to a specific node
- `synapse_root_distances()`: similar to `pymaid.igraph_catmaid.calculate_distance_from_root()` but does not use iGraph

### pymaid.rmaid:
- `init_rcatmaid()`: initialize connection with Catmaid server in R
- `data2py()`: wrapper to convert R data to Python 
- `nblast()`: wrapper to use Nat's NBLAST on Pymaid neurons
- `neuron2py()`: converts R neuron and neuronlist objects to Pymaid neurons
- `neuron2r()`: converts Pymaid neuron and list of neurons to R neuron and neuronlist objects, respectively

### pymaid.user_stats:
- `get_time_invested()`: calculate the time users have spent working on a set of neurons
- `get_user_contributions()`: returns contributions per user for a set of neurons

## License:
This code is under GNU GPL V3
