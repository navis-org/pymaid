[![Documentation Status](https://readthedocs.org/projects/pymaid/badge/?version=latest)](http://pymaid.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/schlegelp/pyMaid.svg?branch=master)](https://travis-ci.org/schlegelp/pyMaid) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/pyMaid/master?urlpath=lab)

pymaid
======
Collection of [Python](http://www.python.org) 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.
Tested with CATMAID release version 2018.04.15 - if you are working with older versions you may experience incompatibilities due to API changes.

## Documentation
PyMaid is on [ReadTheDocs](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs").

## Features

* fetch data via CATMAID's API
* custom neuron classes that perform on-demand data fetching
* 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery: cutting, pruning, rerooting
* clustering methods (e.g. by connectivity or synapse placement)
* R bindings (e.g. for libraries [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr), [catnat](https://github.com/alexanderbates/catnat) )
* interface with NetworkX and iGraph
* tools to analyse user stats (e.g. time-invested, project history)
* interface with Blender 3D

## Binder

This repository is compatible with Binder: click on this badge [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/pyMaid/master?urlpath=lab) to try out pymaid interactively in Jupyterlab hosted by [mybinder](https://mybinder.org)!

## Installation
See the [documentation](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs") for detailed instructions. For the impatient:

I recommend using [Python Packaging Index (PIP)](https://pypi.python.org/pypi) to install pymaid.
First, get [PIP](https://pip.pypa.io/en/stable/installing/) and then run in terminal:

`pip install git+git://github.com/schlegelp/pymaid@master`

This command should also work to update the package.

If your default distribution is Python 2, you have to explicitly tell [PIP](https://pip.pypa.io/en/stable/installing/) to install for Python 3:

`pip3 install git+git://github.com/schlegelp/pymaid@master`

**Attention**

On Windows, the dependencies (i.e. Numpy, Pandas and SciPy) will likely fail to install automatically. Your best bet is to get a Python distribution that already includes them (e.g. [Anaconda](https://www.continuum.io/downloads)). Also: one of the dependencies, `pyoctree`, requires `numpy` to already be on the system to install properly. If pip fails with `ImportError: No module named 'numpy'`, you have to manually install numpy using `pip install numpy` and then retry installing pymaid.

#### External libraries used:
Installing via [PIP](https://pip.pypa.io/en/stable/installing/) should install all essential dependencies. You may run into problems on Windows though. In that case, you need to install dependencies manually. Here is a list of dependencies (check out `install_requires` in [setup.py](https://raw.githubusercontent.com/schlegelp/PyMaid/master/setup.py) for version info):

Must have:

- [Pandas](http://pandas.pydata.org/)
- [NetworkX](https://networkx.github.io)
- [SciPy](http://www.scipy.org)
- [Numpy](http://www.scipy.org)
- [Matplotlib](http://www.matplotlib.org)
- [vispy](http://vispy.org/) - this also requires one of the supported backends. By default, [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/installation.html) is installed.
- [Plotly](http://plot.ly)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [pypng](https://pythonhosted.org/pypng/)

Optional:

- [rpy2](https://rpy2.readthedocs.io/en/version_2.8.x/) - in order to use the `pymaid.rmaid` module you must setup R and install rpy2 manually.
- [pyoctree](https://pypi.python.org/pypi/pyoctree/) - used to calculate points in volume, highly recommended.
- [shapely](https://shapely.readthedocs.io/en/latest/) - required if you want 2D plots of CATMAID volumes.

## Quickstart:

### Retrieve and plot 3D skeleton data
```python
import pymaid

# Initialize Catmaid instance
myInstance = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 'user' , 'password', 'token' )

# Initialize a single neuron with just its skeleton ID
n = pymaid.CatmaidNeuron( '12345', remote_instance = myInstance )

# Retrieve a list of skeletons using an annotation
nl = pymaid.get_neuron ( 'annotation:example_neurons' , myInstance )

# nl is a CatmaidNeuronList object that manages data:
# Notice that some entries show as 'NA' because that data has not yet been retrieved/calculated
print(nl)

# You can subset the neurons based on their properties
only_large = nl[ nl.n_nodes > 2000 ]
only_reviewed = nl [ nl.review_status == 100 ]
name_subset = nl[ ['name1','name2'] ]
skid_subset = nl[ [ 12345, 67890 ] ]

# To get more data, e.g. annotations, simply request that attribute explicitedly
nl[0].annotations

# To force an update of annotations
nl[0].get_annotations()

# Access treenodes of the first neuron
nodes = nl[0].nodes

# Get coordinates of all connectors
coords = nl[0].connectors[ ['x','y','z'] ].as_matrix()

# Get all branch points:
branch_points = nl[0].nodes[ nl[0].nodes.type == 'branch' ].treenode_id

# Plot neuron -> see help(pymaid.plot3d) for accepted kwargs
nl.plot3d()

# Clear 3D viewer
pymaid.clear3d()
```

## License:
This code is under GNU GPL V3

## References:
PyMaid implements algorithms described in:

1. **Comparison of neurons based on morphology**: Cell. 2016 doi: 10.1016/j.neuron.2016.06.012
*NBLAST: Rapid, Sensitive Comparison of Neuronal Structure and Construction of Neuron Family Databases.*
Costa M, Manton JD, Ostrovsky AD, Prohaska S, Jefferis GSXE.
http://www.cell.com/abstract/S0092-8674(13)01476-1
2. **Comparison of neurons based on connectivity**: Science. 2012 Jul 27;337(6093):437-44. doi: 10.1126/science.1221762.
*The connectome of a decision-making neural network.*
Jarrell TA, Wang Y, Bloniarz AE, Brittin CA, Xu M, Thomson JN, Albertson DG, Hall DH, Emmons SW.
http://science.sciencemag.org/content/337/6093/437.long
3. **Comparison of neurons based on synapse distribution**: eLife. doi: 10.7554/eLife.16799
*Synaptic transmission parallels neuromodulation in a central food-intake circuit.*
Schlegel P, Texada MJ, Miroschnikow A, Schoofs A, Hückesfeld S, Peters M, … Pankratz MJ.
https://elifesciences.org/content/5/e16799
4. **Synapse flow centrality and segregation index**: eLife. doi: 10.7554/eLife.12059
*Quantitative neuroanatomy for connectomics in Drosophila.*
Schneider-Mizell CM, Gerhard S, Longair M, Kazimiers T, Li, Feng L, Zwart M … Cardona A.

