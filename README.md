[![Documentation Status](https://readthedocs.org/projects/pymaid/badge/?version=latest)](http://pymaid.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/schlegelp/pyMaid.svg?branch=master)](https://travis-ci.org/schlegelp/pyMaid) [![Coverage Status](https://coveralls.io/repos/github/schlegelp/pyMaid/badge.svg?branch=master)](https://coveralls.io/github/schlegelp/pyMaid?branch=master) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/pyMaid/master?urlpath=tree)

<img src="https://github.com/schlegelp/pyMaid/raw/master/docs/_static/favicon.png" height="30"> pymaid
======================================================================================================
Collection of [Python](http://www.python.org) 3 tools to interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo") servers.
Tested with CATMAID release version 2018.04.15 - if you are working with older versions you may experience incompatibilities due to API changes.

## Documentation
PyMaid is on [ReadTheDocs](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs").

## Features

* fetch data via CATMAID's API
* custom neuron classes that perform on-demand data fetching
* interactive 2D (matplotlib) and 3D (vispy or plotly) plotting of neurons
* virtual neuron surgery: cutting, pruning, rerooting
* clustering methods (e.g. by connectivity or synapse placement)
* R bindings (e.g. for libraries [nat](https://github.com/jefferis/nat), [rcatmaid](https://github.com/jefferis/rcatmaid), [elmr](https://github.com/jefferis/elmr), [catnat](https://github.com/alexanderbates/catnat) )
* interface with NetworkX and iGraph
* tools to analyse user stats (e.g. time-invested, project history)
* interface with Blender 3D

## Getting started
See the [documentation](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs") for detailed installation instructions, tutorials and examples. For the impatient:

`pip install git+git://github.com/schlegelp/pymaid@master`

Alternatively click on the *launch binder* badge above to try out pymaid hosted by [mybinder](https://mybinder.org)!

![pymaid example](https://user-images.githubusercontent.com/7161148/41200671-4e4320ec-6ca1-11e8-90a2-2feda2d9372d.gif)

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

