pymaid
====== 
Collection of [Python](http://www.python.org) 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.
Tested with CATMAID latest release version 2017.05.17 - if you are working with older versions you may experience incompatibilities due to 
API changes.

## Documentation
PyMaid is on [ReadTheDocs](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs").

## Features

* collection of functions to fetch data via CATMAID's API
* custom neuron classes that perform on-demand data fetching
* 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery (cutting, pruning, rerooting)
* clustering methods (e.g. by connectivity or synapse placement)
* R bindings (e.g. for libraries ([nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr), [catnat](https://github.com/alexanderbates/catnat) ) )
* interface with iGraph 
* tools to analyse user stats (e.g. time invested)
* interface with Blender 3D
* and oh so much more...

## Installation
See [documentation](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs") for detailed instructions. For the impatient:

I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install pymaid.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:  

`pip install git+git://github.com/schlegelp/pymaid@master`  

This command should also work to update the package.

If your default distribution is Python 2, you have to explicitly tell [PIP](https://pip.pypa.io/en/stable/installing/) to install for Python 3:

`pip3 install git+git://github.com/schlegelp/pymaid@master`  

**Attention**: on Windows, the dependencies (i.e. Numpy, Pandas and SciPy) will likely fail to install automatically. Your best bet is to get a Python distribution that already includes them (e.g. [Anaconda](https://www.continuum.io/downloads)). Also: one of the dependencies, `pyoctree`, requires `numpy` to already be on the system to install properly. If pip fails with `ImportError: No module named 'numpy'`, you have to manually install numpy using `pip install numpy` and then retry installing pymaid.

#### External libraries used:
Installing via [PIP](https://pip.pypa.io/en/stable/installing/) should install all essential dependencies. You may run into problems on Windows though. In that case, you need to install dependencies manually, here is a list of dependencies (check out `install_requires` in [setup.py](https://raw.githubusercontent.com/schlegelp/PyMaid/master/setup.py) for version info):

Must have: 

- [Pandas](http://pandas.pydata.org/)
- [iGraph](http://www.igraph.org) 
- [SciPy](http://www.scipy.org)
- [Numpy](http://www.scipy.org) 
- [Matplotlib](http://www.matplotlib.org)
- [vispy](http://vispy.org/) - this also requires one of the supported backend. By default, [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/installation.html) is installed.
- [Plotly](http://plot.ly)
- [pypng](https://pythonhosted.org/pypng/)

Optional: 

- [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/) - in order to use the `pymaid.rmaid` module you must setup R and install rpy2 manually.
- [pyoctree](https://pypi.python.org/pypi/pyoctree/) 


## Quickstart:

### Retrieve 3D skeleton data
```python
import pymaid

#Initialize Catmaid instance 
myInstance = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Initialise a single neuron with just its skeleton ID
n = pymaid.CatmaidNeuron( '12345', remote_instance = myInstance )

#Retrieve a list of skeletons using an annotation
nl = pymaid.get_neuron ( 'annotation:example_neurons' , myInstance )

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

#Clear 3D viewer
plot.clear3d()
```

### Cluster synapses based on distance along the arbor using iGraph
```python
import pymaid
from scipy import cluster
import matplotlib.pyplot as plt

#Initiate Catmaid instance
remote_instance = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

#Retrieve 3D skeleton data for neuron of interest
nl = pymaid.get_neuron ( [ '12345' ], remote_instance, connector_flag = 1, tag_flag = 0 )

#(Optional) Consider downsampling for large neurons (preverses branch points, end points, synapses, etc.)
nl.downsample( factor = 4 )

#Cluster by synapse synapse placement
clust_res = pymaid.cluster_by_synapse_placement( ds_neuron )

#Plot dendrogram
clust_res.plot_dendrogram()
plt.show()

#Find the last two clusters (= the two largest clusters):
clusters = clust_res.get_clusters( k=2, criterion='maxclust')

#Print summary
print('%i nodes total. Cluster 1: %i. Cluster 2: %i' % (len(clusters),len([n for n in clusters if n==1]),len([n for n in clusters if n==2])))
```

## License:
This code is under GNU GPL V3
