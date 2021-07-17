[![Documentation Status](https://readthedocs.org/projects/pymaid/badge/?version=latest)](http://pymaid.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/schlegelp/pymaid/actions/workflows/run-tests.yml/badge.svg)](https://github.com/schlegelp/pymaid/actions/workflows/run-tests.yml) [![Coverage Status](https://coveralls.io/repos/github/schlegelp/pyMaid/badge.svg?branch=master)](https://coveralls.io/github/schlegelp/pyMaid?branch=master) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/schlegelp/pyMaid/master?urlpath=tree) [![DOI](https://zenodo.org/badge/78551448.svg)](https://zenodo.org/badge/latestdoi/78551448)

<img src="https://github.com/schlegelp/pyMaid/raw/master/docs/_static/favicon.png" height="30"> pymaid
======================================================================================================
Python-Catmaid - or "pymaid" - is a [Python](http://www.python.org) 3 library to
interface with [CATMAID](https://github.com/catmaid/CATMAID "CATMAID Repo")
servers.

Tested with CATMAID release version 2020.02.15 - if you are working with older
versions you may run into issues due to API changes.

## Features
* pull and push data from/to a CATMAID server
* visualize and analyse neuron morphology via [navis](https://navis.readthedocs.io)
* tools to analyse user stats (e.g. time-invested, project history)
* clustering methods (e.g. by connectivity or synapse placement)

## Documentation
Pymaid is on [ReadTheDocs](http://pymaid.readthedocs.io/ "pymaid ReadTheDocs").

## Getting started
See the [documentation](http://pymaid.readthedocs.io/ "PyMaid ReadTheDocs") for
detailed installation instructions, tutorials and examples. For the impatient:

```bash
pip3 install python-catmaid
```

*Important*: there is a `pymaid` package on PyPI which has _nothing_ to do with
this pymaid!

To install the bleeding edge from Github:

```bash
pip3 install git+git://github.com/schlegelp/pymaid@master
```

Alternatively click on the *launch binder* badge above to try out pymaid hosted by [mybinder](https://mybinder.org)!

![pymaid example](https://user-images.githubusercontent.com/7161148/41200671-4e4320ec-6ca1-11e8-90a2-2feda2d9372d.gif)

## License:
This code is under GNU GPL V3

## References:
Pymaid implements/provides an interfaces with algorithms described in:

1. **Comparison of neurons based on connectivity**: Science. 2012 Jul 27;337(6093):437-44. doi: 10.1126/science.1221762.
*The connectome of a decision-making neural network.*
Jarrell TA, Wang Y, Bloniarz AE, Brittin CA, Xu M, Thomson JN, Albertson DG, Hall DH, Emmons SW.
[link](http://science.sciencemag.org/content/337/6093/437.long)
2. **Comparison of neurons based on synapse distribution**: eLife. doi: 10.7554/eLife.16799
*Synaptic transmission parallels neuromodulation in a central food-intake circuit.*
Schlegel P, Texada MJ, Miroschnikow A, Schoofs A, Hueckesfeld S, Peters M, ... Pankratz MJ.
[link](https://elifesciences.org/content/5/e16799)
