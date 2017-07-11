Introduction
************
This section will teach you the basics of how to use R in Python.

On a fundamental level, you can use every single R function from within Python. The trick is to manually convert data types when the R interface ``rpy2`` does not. See https://rpy2.readthedocs.io for an introduction to ``rpy2``.

Quickstart
==========
>>> from pymaid import pymaid
>>> from pymaid import rmaid
>>> import matplotlib.pyplot as plt
>>> from rpy2.robjects.packages import importr
>>> #Load nat as module
>>> nat = importr('nat')
>>> #Initialise Catmaid instance
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> #Fetch a neuron in Python CATMAID
>>> skeleton_id = 123456
>>> n = pymaid.pymaid.get_3D_skeleton( skeleton_id, rm )
>>> #Initialize R's rcatmaid 
>>> rcatmaid = rmaid.init_rcatmaid( rm )
>>> Convert pymaid neuron to R neuron (works with neuron + neuronlist objects)
>>> n_r = rmaid.neuron2r( n.ix[0] )
>>> #Use nat to prune the neuron
>>> n_pruned = nat.prune_by_strahler( n_r )
>>> #Convert back to pymaid object
>>> n_py = rmaid.neuron2py( n_pruned, rm )
>>> #Nblast pruned neuron (assumes FlyCircuit database is saved locally)
>>> results = rmaid.nblast( n_pruned )
>>> #Sort results by mu score
>>> results.sort('mu_score')
>>> #Plot top hits
>>> results.plot( hits = 3 )

Data conversion
===============
:module:`pymaid.rmaid` provides functions to convert data from Python to R:
1. :func:`pymaid.rmaid.data2py` converts general data from R to Python
2. :func:`pymaid.rmaid.neuron2py` converts R neuron or neuronlist object to Python :class:`pymaid.core.CatmaidNeuron` and :class:`pymaid.core.CatmaidNeuronList`, respectively
3. :func:`pymaid.rmaid.neuron2r` converts :class:`pymaid.core.CatmaidNeuron` or :class:`pymaid.core.CatmaidNeuronList` to R neuron or neuronlist object
4. :func:`pymaid.rmaid.dotprops2py` converts R dotprop objects to pandas DataFrame that can be passed to :func:`pymaid.plot.plot3d`

R catmaid
=========
:func:`rmaid.init_rcatmaid` is a wrapper to initialise R catmaid (https://github.com/jefferis/rcatmaid)

>>> from pymaid import pymaid, rmaid
>>> #Initialize a CatmaidInstance in Python
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> #Initialize R's rcatmaid with Python instance
>>> rcat = rmaid.init_rcatmaid( rm )
>>> #Check contents of that module
>>> dir(rcat)
['*_catmaidneuron', '+_catmaidneuron', '___NAMESPACE___', '___S3MethodsTable___', '__doc__', '__loader__', '__name__', '__package__', '__rdata__', '__rname__', '__spec__', '__version__', '_env', '_exported_names', '_packageName', '_package_statevars', '_rpy2r', '_symbol_check_after', '_symbol_r2python', '_translation', 'as_catmaidmesh', 'as_catmaidmesh_catmaidmesh', 
...
'read_neuron_catmaid', 'read_neurons_catmaid', 'server', 'somapos_catmaidneuron', 'summary_catmaidneuron', 'token', 'xform_catmaidneuron']
>>> #Get neurons as R catmaidneuron
>>> n = rcat.read_neurons_catmaid('annotation:glomerulus DA1' )

You can use other packages such as nat (https://github.com/jefferis/nat) to process that neuron

>>> from rpy2.robjects.packages import importr
>>> #Load nat as module
>>> nat = importr('nat')
>>> #Use nat to prune the neuron
>>> n_pruned = nat.prune_strahler( n[0] )

Now convert to PyMaid :class:`pymaid.core.CatmaidNeuron`

>>> #Convert to Python
>>> n_py = rmaid.neuron2py( n_pruned, remote_instance = rm)
>>> #Plot
>>> n_py.plot3d()

Nblasting
=========
:func:`pymaid.rmaid.nblast` provides a wrapper to nblast neurons.

>>> from pymaid.pymaid import CatmaidInstance
>>> from pymaid import rmaid
>>> #Initialize connection to Catmaid server
>>> rm = CatmaidInstance( 'url', 'http_user', 'http_pw', 'token' )
>>> #Blast a neuron against default (FlyCircuit) database
>>> nbl = rmaid.nblast( skid = 16, remote_instance = rm  )

:func:`pymaid.rmaid.nblast` returns nblast results as :class:`pymaid.rmaid.nbl_results` 

>>> #See contents of nblast_res object
>>> help(nbl)
>>> #Get results as Pandas Dataframe
>>> nbl.res
>>> #Plot histogram of results
>>> nbl.res.plot.hist(alpha=.5)
>>> #Sort and plot the first hits
>>> nbl.sort('mu_score')
>>> nbl.plot(hits = 4)

