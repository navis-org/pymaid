#    Copyright (C) 2017 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along

"""
A collection of tools to interace with CATMAID R libraries (e.g. nat, catnat,
elmr, rcatmaid).

Notes
-----
See https://github.com/jefferis and https://github.com/alexanderbates

Examples
--------
>>> import pymaid
>>> from pymaid import rmaid
>>> import matplotlib.pyplot as plt
>>> from rpy2.robjects.packages import importr
>>> # Load nat as module
>>> nat = importr('nat')
>>> # Initialise Catmaid instance
>>> rm = pymaid.CatmaidInstance('server_url', 'http_user', 'http_pw', 'token')
>>> # Fetch a neuron in Python CATMAID
>>> skeleton_id = 123456
>>> n = pymaid.get_neuron( skeleton_id )
>>> # Initialize R's rcatmaid
>>> rcatmaid = rmaid.init_rcatmaid( rm )
>>> # Convert pymaid neuron to R neuron (works with neuron + neuronlist objects)
>>> n_r = rmaid.neuron2r( n.ix[0] )
>>> # Use nat to prune the neuron
>>> n_pruned = nat.prune_strahler( n_r )
>>> # Convert back to pymaid object
>>> n_py = rmaid.neuron2py( n_pruned, rm )
>>> # Nblast pruned neuron (assumes FlyCircuit database is saved locally)
>>> results = rmaid.nblast( n_pruned )
>>> # Sort results by mu score
>>> results.sort('mu_score')
>>> # Plot top hits
>>> results.plot( hits = 3 )
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd
import numpy as np

from colorsys import hsv_to_rgb

import pymaid.cluster as pyclust
from pymaid import core, fetch, plotting, config, utils

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

cl = robjects.r('class')
names = robjects.r('names')

# Set up logging
logger = config.logger

try:
    nat = importr('nat')
    r_nblast = importr('nat.nblast')
    nat_templatebrains = importr('nat.templatebrains')
    nat_flybrains = importr('nat.flybrains')
    # even if not used, these packages are important!
    flycircuit = importr('flycircuit')
    elmr = importr('elmr')
except:
    logger.error(
        'R library "nat" not found! Please install from within R.')

__all__ = sorted(['neuron2r', 'neuron2py', 'init_rcatmaid', 'dotprops2py',
                  'data2py', 'NBLASTresults', 'nblast', 'nblast_allbyall',
                  'get_neuropil'])


def init_rcatmaid(**kwargs):
    """ Initialize the R catmaid package.

    R package by Greg Jefferis: https://github.com/jefferis/rcatmaid

    Parameters
    ----------
    remote_instance :   CATMAID instance
                        From pymaid.CatmaidInstance(). This is used to
                        extract credentials. Overrides other credentials
                        provided!
    server :            str, optional
                        Use this to set server URL if no remote_instance is
                        provided
    authname :          str, optional
                        Use this to set http user if no remote_instance is
                        provided
    authpassword :      str, optional
                        Use this to set http password if no remote_instance
                        is provided
    authtoken :         str, optional
                        Use this to set user token if no remote_instance is
                        provided

    Returns
    -------
    catmaid :           R library
                        R object representing the rcatmaid library
    """

    remote_instance = kwargs.get('remote_instance', None)
    server = kwargs.get('server', None)
    authname = kwargs.get('authname', None)
    authpassword = kwargs.get('authpassword', None)
    authtoken = kwargs.get('authtoken', None)

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if remote_instance:
        server = remote_instance.server
        authname = remote_instance.authname
        authpassword = remote_instance.authpassword
        authtoken = remote_instance.authtoken
    elif not remote_instance and None in (server, authname, authpassword, authtoken):
        logger.error('Unable to initialize. Missing credentials: %s' % ''.join(
            [n for n in ['server', 'authname', 'authpassword', 'authtoken'] if n not in kwargs]))
        return None

    # Import R Catmaid
    try:
        catmaid = importr('catmaid')
    except BaseException:
        logger.error(
            'RCatmaid not found. Please install before proceeding.')
        return None

    # Use remote_instance's credentials
    catmaid.server = server
    catmaid.authname = authname
    catmaid.authpassword = authpassword
    catmaid.token = authtoken

    # Create the connection
    con = catmaid.catmaid_connection(server=catmaid.server,
                                     authname=catmaid.authname,
                                     authpassword=catmaid.authpassword,
                                     token=catmaid.token)

    # Login
    catmaid.catmaid_login(con)

    logger.info('Rcatmaid successfully initiated.')

    return catmaid


def data2py(data, **kwargs):
    """ Takes data object from rcatmaid (e.g. ``catmaidneuron`` from
    ``read.neuron.catmaid``) and converts into Python Data.

    Notes
    -----
    (1) Most R data comes as list (even if only 1 entry). This is preserved.
    (2) R lists with headers are converted to dictionaries
    (3) R DataFrames are converted to Pandas DataFrames
    (4) R nblast results are converted to Pandas DataFrames but only the top
        100 hits for which we have reverse scores!

    Parameters
    ----------
    data
        Any kind of R data. Can be nested (e.g. list of lists)!

    Returns
    -------
    converted data
    """

    if 'neuronlistfh' in cl(data):
        logger.error('On-demand neuronlist found. Conversion cancelled to '
                     'prevent loading large datasets in memory. Please use '
                     'rmaid.dotprops2py() and its "subset" parameter.')
        return None
    elif 'neuronlist' in cl(data):
        if 'catmaidneuron' in cl(data[0]):
            return neuron2py(data)
        elif 'dotprops' in cl(data[0]):
            return dotprops2py(data)
        else:
            logger.error(
                'Dont know how to convert unknown R datatype %s' % cl(data[0]))
            return data
    elif cl(data)[0] == 'integer':
        if not names(data):
            return [int(n) for n in data]
        else:
            return {n: int(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'character':
        if not names(data):
            return [str(n) for n in data]
        else:
            return {n: str(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'numeric':
        if not names(data):
            return [float(n) for n in data]
        else:
            return {n: float(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'data.frame':
        df = pandas2ri.ri2py_dataframe(data)
        return df
    elif cl(data)[0] == 'matrix':
        mat = np.array(data)
        df = pd.DataFrame(data=mat)
        if data.names:
            if data.names[1] != robjects.r('NULL'):
                df.columns = data.names[1]
            if data.names[0] != robjects.r('NULL'):
                df.index = data.names[0]
        return df
    elif 'list' in cl(data):
        # If this is just a list, return a list
        if not names(data):
            return [data2py(n) for n in data]
        # If this list has headers, return as dictionary
        else:
            return {n: data2py(data[i]) for i, n in enumerate(names(data))}
    elif cl(data)[0] == 'NULL':
        return None
    elif 'nblastfafb' in cl(data):
        fw_scores = {n: data[0][i] for i, n in enumerate(names(data[0]))}
        rev_scores = {n: data[1][i] for i, n in enumerate(names(data[1]))}
        mu_scores = {n: (fw_scores[n] + rev_scores[n]) / 2 for n in rev_scores}

        df = pd.DataFrame([[n, fw_scores[n], rev_scores[n], mu_scores[n]]
                           for n in rev_scores],
                          columns=['gene_name', 'forward_score',
                                   'reverse_score', 'mu_score']
                          )

        logger.info('Returning only nblast results. Neuron object is stored '
                    'in your original_data[2].')
        return df
    else:
        logger.error(
            'Dont know how to convert unknown R datatype %s' % cl(data))
        return data


def neuron2py(neuron, convert_to_nm=False, remote_instance=None):
    """ Converts an rcatmaid ``neuron`` or ``neuronlist`` object to a PyMaid
    :class:`~pymaid.CatmaidNeuron`/:class:`~pymaid.CatmaidNeuronList`.

    Node creator and confidence are not included in R's neuron/neuronlist
    and will be imported as ``None``.

    Parameters
    ----------
    neuron :            R neuron | R neuronlist
                        Neuron to convert to Python
    convert_to_nm :     bool, optional
                        If True will convert from um to nm.
    remote_instance :   CATMAID instance, optional
                        Provide if you want neuron names to be updated from.

    Returns
    -------
    pymaid CatmaidNeuronList
    """

    if 'rpy2' in str(type(neuron)):
        if cl(neuron)[0] == 'neuronlist':
            neuron_list = pd.DataFrame(data=[[data2py(e) for e in n] for n in neuron],
                                       columns=list(neuron[0].names))
            # neuron_list.columns =  data.names #[ 'NumPoints',
            # 'StartPoint','BranchPoints','EndPoints','nTrees', 'NumSeqs',
            # 'SegList', 'd', 'skid', 'connectors', 'tags','url', 'headers'  ]
            if 'df' in neuron.slots:
                neuron_list['name'] = neuron.slots['df'][2]
            else:
                neuron_list['name'] = ['NA'] * neuron_list.shape[0]
        elif cl(neuron)[0] == 'catmaidneuron' or cl(neuron)[0] == 'neuron':
            neuron_list = pd.DataFrame(
                data=[[e for e in neuron]], columns=neuron.names)
            neuron_list = neuron_list.applymap(data2py)
            neuron_list['name'] = ['NA']
            # neuron_list.columns = neuron.names  #[ 'NumPoints',
            # 'StartPoint','BranchPoints','EndPoints','nTrees', 'NumSeqs',
            # 'SegList', 'd', 'skid', 'connectors', 'tags','url', 'headers'  ]
        neuron = neuron_list

    remote_instance = utils._eval_remote_instance(remote_instance,
                                                  raise_error=False)

    # Nat function may return neuron objects that have ONLY nodes - no
    # connectors, skeleton_id, name or tags!
    if 'skid' in neuron and remote_instance:
        neuron_names = fetch.get_names(
            [n[0] for n in neuron.skid.tolist()], remote_instance)
    elif 'skid' in neuron and not remote_instance:
        neuron_names = None
        logger.info(
            'Please provide a remote instance if you want to add neuron name.')
    else:
        logger.warning(
            'Neuron has only nodes (no name, skid, connectors or tags).')

    data = []
    for i in range(neuron.shape[0]):
        # Note that radius is divided by 2 -> this is because in rcatmaid the
        # original radius is doubled for some reason
        nodes = pd.DataFrame([[no.PointNo, no.Parent, None, no.X, no.Y,
                               no.Z, no.W / 2, None] for no in neuron.loc[i, 'd'].itertuples()], dtype=object)
        nodes.columns = ['treenode_id', 'parent_id',
                         'creator_id', 'x', 'y', 'z', 'radius', 'confidence']
        nodes.loc[nodes.parent_id == -1, 'parent_id'] = None

        if 'connectors' in neuron and isinstance(neuron.loc[i, 'connectors'], pd.DataFrame):
            connectors = pd.DataFrame([[cn.treenode_id, cn.connector_id, cn.prepost, cn.x, cn.y, cn.z]
                                       for cn in neuron.loc[i, 'connectors'].itertuples()], dtype=object)
            connectors.columns = ['treenode_id',
                                  'connector_id', 'relation', 'x', 'y', 'z']
        else:
            connectors = pd.DataFrame(columns=['treenode_id', 'connector_id',
                                               'relation', 'x', 'y', 'z'])

        if 'skid' in neuron:
            skid = neuron.loc[i, 'skid'][0]
        else:
            skid = 'NA'

        data.append([skid,
                     nodes,
                     connectors
                     ])

    df = pd.DataFrame(data=data,
                      columns=['skeleton_id', 'nodes', 'connectors'],
                      dtype=object
                      )
    df['igraph'] = None

    if 'tags' in neuron:
        df['tags'] = neuron.tags.tolist()
    else:
        df['tags'] = [{} for n in df.skeleton_id.values]

    if 'skid' in neuron and neuron_names is not None:
        df['neuron_name'] = [neuron_names.get(str(n), 'NA') for n in df.skeleton_id.values]
    else:
        df['neuron_name'] = ['NA' for n in df.skeleton_id.values]

    nl = core.CatmaidNeuronList(df, remote_instance=remote_instance)

    # Do some clean up
    for n in nl:
        if not isinstance(n.tags, dict):
            n.tags = {}

        if convert_to_nm:
            n.nodes[['x', 'y', 'z', 'radius']] *= 1000
            n.connectors[['x', 'y', 'z']] *= 1000

    return nl


def neuron2r(neuron, convert_to_um=False):
    """ Converts a pymaid neuron or neuronlist to the corresponding
    neuron/neuronlist object in R.

    The way this works is essentially converting the PyMaid object back
    into what rcatmaid expects as a response from a CATMAID server, then
    we are calling the same functions as in rcatmaid's
    ``read.neuron.catmaid()``.

    Attention: Currently, the project ID saved as part of R neuronlist objects
    is ALWAYS 1.

    Parameters
    ----------
    neuron :        CatmaidNeuron | CatmaidNeuronList | pandas DataFrame
    convert_to_um : bool, optional
                    If True, coordinates are divided by 1000.

    Returns
    -------
    R neuron
        Either R neuron or neuronlist depending on input.
    """

    if isinstance(neuron, pd.DataFrame) or isinstance(neuron, core.CatmaidNeuronList):
        """
        The way neuronlist are constructed is a bit more complicated:
        They are essentially named lists { 'neuronA' : neuronobject, ... }
        BUT they also contain a dataframe that holds a DataFrame as attribute
        ( attr('df') = df )
        This dataframe looks like this

                pid   skid     name
        skid1
        skid2

        In rpy2, these attributes are assigned using the .slots['df'] function
        """

        nlist = {}
        for i in range(neuron.shape[0]):
            nlist[neuron.ix[i].skeleton_id] = neuron2r(
                neuron.ix[i], convert_to_um=convert_to_um)

        nlist = robjects.ListVector(nlist)
        nlist.rownames = neuron.skeleton_id.tolist()

        df = robjects.DataFrame({'pid': robjects.IntVector([1] * neuron.shape[0]),
                                 'skid': robjects.IntVector(neuron.skeleton_id.tolist()),
                                 'name': robjects.StrVector(neuron.neuron_name.tolist())
                                 })
        df.rownames = neuron.skeleton_id.tolist()
        nlist.slots['df'] = df

        nlist.rclass = robjects.r('c("neuronlist","list")')

        return nlist

    elif isinstance(neuron, pd.Series) or isinstance(neuron, core.CatmaidNeuron):
        n = neuron

        if convert_to_um:
            n = n.copy()
            n.nodes[['x', 'y', 'z', 'radius']] /= 1000
            n.connectors[['x', 'y', 'z']] /= 1000

        # First convert into format that rcatmaid expects as server response

        # Prepare list of parents -> root node's parent "None" has to be
        # replaced with -1
        parents = np.array(n.nodes.parent_id.values)
        # should technically be robjects.r('-1L')
        parents[parents == None] = -1 # DO NOT turn this into "parents is None"!

        swc = robjects.DataFrame({'PointNo': robjects.IntVector(n.nodes.treenode_id.tolist()),
                                  'Label': robjects.IntVector([0] * n.nodes.shape[0]),
                                  'X': robjects.IntVector(n.nodes.x.tolist()),
                                  'Y': robjects.IntVector(n.nodes.y.tolist()),
                                  'Z': robjects.IntVector(n.nodes.z.tolist()),
                                  'W': robjects.FloatVector([w * 2 for w in n.nodes.radius.tolist()]),
                                  'Parent': robjects.IntVector(parents)
                                  })

        if n.nodes[n.nodes.radius > 500].shape[0] == 1:
            soma_id = int(
                n.nodes[n.nodes.radius > 500].treenode_id.tolist()[0])
        else:
            soma_id = robjects.r('NULL')

        # Generate nat neuron
        n_r = nat.as_neuron(swc, origin=soma_id, skid=n.skeleton_id)

        # Convert back to python dict so that we can add additional data
        n_py = {n_r.names[i]: n_r[i] for i in range(len(n_r))}

        if n.connectors.shape[0] > 0:
            n_py['connectors'] = robjects.DataFrame({'treenode_id': robjects.IntVector(n.connectors.treenode_id.tolist()),
                                                     'connector_id': robjects.IntVector(n.connectors.connector_id.tolist()),
                                                     'prepost': robjects.IntVector(n.connectors.relation.tolist()),
                                                     'x': robjects.IntVector(n.connectors.x.tolist()),
                                                     'y': robjects.IntVector(n.connectors.y.tolist()),
                                                     'z': robjects.IntVector(n.connectors.z.tolist())
                                                     })
        else:
            n_py['connectors'] = robjects.r('NULL')

        if n.tags:
            # Make sure we have integers (not e.g. np.int64)
            tags = {k : [int(t) for t in v] for k, v in n.tags.items()}
            n_py['tags'] = robjects.ListVector(tags)
        else:
            n_py['tags'] = robjects.r('NULL')

        # R neuron objects contain information about URL and response headers
        # -> since we don't have that (yet), we will create the entries but
        # leave them blank
        n_py['url'] = robjects.r('NULL')
        n_py['headers'] = robjects.ListVector({
            'server ': 'NA',
            'date': 'NA',
            'content-type': 'NA',
            'transfer-encoding': 'NA',
            'connections': 'NA',
            'vary': 'NA',
            'expires': 'NA',
            'cache-control': 'NA',
            'content-encoding': 'NA'
        })

        # Convert back to R object
        n_r = robjects.ListVector(n_py)
        n_r.rclass = robjects.r('c("catmaidneuron","neuron")')

        return n_r
    else:
        logger.error('Unknown DataFrame format: %s' % str(type(neuron)))


def dotprops2py(dp, subset=None):
    """ Converts dotprops into pandas DataFrame.

    Parameters
    ----------
    dp :        dotprops neuronlist | neuronlistfh
                Dotprops object to convert.
    subset :    list of str | list of indices, optional
                Neuron names or indices.

    Returns
    -------
    core.Dotprops
        Subclass of pandas DataFrame. Contains dotprops.
        Can be passed to `plotting.plot3d( dotprops )`
    """

    # Check if list is on demand
    if 'neuronlistfh' in cl(dp) and not subset:
        dp = dp.rx(robjects.IntVector([i + 1 for i in range(len(dp))]))
    elif subset:
        indices = [i for i in subset if isinstance(
            i, int)] + [dp.names.index(n) + 1 for n in subset if isinstance(n, str)]
        dp = dp.rx(robjects.IntVector(indices))

    df = data2py(dp.slots['df'])
    df.reset_index(inplace=True, drop=True)

    points = []
    for i in range(len(dp)):
        this_points = pd.concat([data2py(dp[i][0]), data2py(dp[i][2])], axis=1)
        this_points['alpha'] = dp[i][1]
        this_points.columns = ['x', 'y', 'z',
                               'x_vec', 'y_vec', 'z_vec', 'alpha']
        points.append(this_points)

    df['points'] = points

    return core.Dotprops(df)


def nblast_allbyall(x, normalize=True, remote_instance=None,
                    n_cores=os.cpu_count(), UseAlpha=False):
    """ Wrapper to use R's ``nat:nblast_allbyall``
    (https://github.com/jefferislab/nat.nblast/).

    Neurons are automatically resampled to 1 micron.

    Parameters
    ----------
    x :                 skeleton IDs | CatmaidNeuronList | RCatmaid neurons
                        Neurons to blast.
    remote_instance :   Catmaid Instance, optional
                        Only neccessary if only skeleton IDs are provided
    normalize :         bool, optional
                        If true, matrix is normalized using z-score.
    n_cores :           int, optional
                        Number of cores to use for nblasting. Default is
                        ``os.cpu_count()``.
    UseAlpha :          bool, optional
                        Emphasises neurons' straight parts (backbone) over
                        parts that have lots of branches.

    Returns
    -------
    nblast_results
        Instance of :class:`pymaid.rmaid.NBLASTresults` that holds distance
        matrix and contains wrappers to cluster and plot data. Please use
        ``help(nblast_results)`` to learn more and see example below.

    Examples
    --------
    >>> import pymaid
    >>> import matplotlib.pyplot as plt
    >>> # Initialize connection to Catmaid server
    >>> rm = pymaid.CatmaidInstance( url, http_user, http_pw, token )
    >>> # Get a bunch of neurons
    >>> nl = pymaid.get_neuron('annotation:glomerulus DA1')
    >>> # Blast against each other
    >>> res = pymaid.nblast_allbyall( nl )
    >>> # Cluster and create simple dendrogram
    >>> res.cluster(method='ward')
    >>> res.plot_matrix()
    >>> plt.show()
    """

    domc = importr('doMC')
    cores = robjects.r('registerDoMC(%i)' % n_cores)

    doParallel = importr('doParallel')
    doParallel.registerDoParallel(cores=n_cores)

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    if 'rpy2' in str(type(x)):
        rn = x
    elif isinstance(x, pd.DataFrame) or isinstance(x, core.CatmaidNeuronList):
        if x.shape[0] < 2:
            logger.warning(
                'You have to provide more than a single neuron.')
            raise ValueError('You have to provide more than a single neuron.')

        rn = neuron2r(x, convert_to_um=True)
    elif isinstance(x, pd.Series) or isinstance(x, core.CatmaidNeuron):
        logger.warning(
            'You have to provide more than a single neuron.')
        raise ValueError('You have to provide more than a single neuron.')
    elif isinstance(x, list):
        if not remote_instance:
            logger.error('You have to provide a CATMAID instance using the '
                         '<remote_instance> parameter. See help(rmaid.nblast) '
                         'for details.')
            return
        x = fetch.get_neuron(x, remote_instance)
        rn = neuron2r(x, convert_to_um=True)
    else:
        logger.error('Unable to intepret <neuron> parameter provided. See '
                     'help(rmaid.nblast) for details.')
        raise ValueError('Unable to intepret <neuron> parameter provided. See'
                         'help(rmaid.nblast) for details.')

    # Make dotprops and resample
    xdp = nat.dotprops(rn, k=5, resample=1)

    # Calculate scores
    scores = r_nblast.nblast(xdp, xdp, **{'normalised': False,
                                          '.parallel': True,
                                          'UseAlpha': UseAlpha})

    # Generate matrix with skeleton IDs as indices/columns
    matrix = pd.DataFrame(np.array(scores),
                          columns=names(xdp),
                          index=names(xdp)
                          )

    if normalize:
        # Perform z-score normalization
        matrix = (matrix - matrix.mean()) / matrix.std()

    logger.info('Done! Use results.plot_dendrgram() and '
                'matplotlib.pyplot.show() to plot dendrogram.')

    if isinstance(x, core.CatmaidNeuronList) or isinstance(x, pd.DataFrame):
        name_dict = x.summary().set_index('skeleton_id')[
            'neuron_name'].to_dict()
        res = pyclust.ClustResults(
            matrix, labels=[name_dict[n] for n in matrix.columns],
            mat_type='similarity')
        res.neurons = x
        return res
    else:
        return pyclust.ClustResults(matrix, mat_type='similarity')


def nblast(neuron, remote_instance=None, db=None, n_cores=os.cpu_count(),
           reverse=False, normalised=True, UseAlpha=False, mirror=True,
           reference='nat.flybrains::FCWB'):
    """ Wrapper to use R's nblast (https://github.com/jefferis/nat).

    Provide neuron to nblast either as skeleton ID or neuron object. This
    essentially recapitulates what `elmr's <https://github.com/jefferis/elmr>`_
    ``nblast_fafb`` does.

    Notes
    -----
    Neurons are automatically resampled to 1 micron.

    Parameters
    ----------
    x
                    Neuron to nblast. This can be either:
                    1. A single skeleton ID
                    2. PyMaid neuron from e.g. pymaid.get_neuron()
                    3. RCatmaid neuron object
    remote_instance :   Catmaid Instance, optional
                        Only neccessary if only a SKID is provided
    db :            database, optional
                    File containing dotproducts to blast against. This can be
                    either:

                    1. the name of a file in ``'flycircuit.datadir'``,
                    2. a path (e.g. ``'.../gmrdps.rds'``),
                    3. an R file object (e.g. ``robjects.r("load('.../gmrdps.rds')")``)
                    4. a URL to load the list from (e.g. ``'http://.../gmrdps.rds'``)

                    If not provided, will search for a 'dpscanon.rds' file in
                    'flycircuit.datadir'.
    n_cores :       int, optional
                    Number of cores to use for nblasting. Default is
                    ``os.cpu_count()``.
    reverse :       bool, optional
                    If True, treats the neuron as NBLAST target rather than
                    neurons of database. Makes sense for partial
                    reconstructions.
    UseAlpha :      bool, optional
                    Emphasises neurons' straight parts (backbone) over parts
                    that have lots of branches.
    mirror :        bool, optional
                    Whether to mirror the neuron or not b/c FlyCircuit neurons
                    are on fly's right.
    normalised :    bool, optional
                    Whether to return normalised NBLAST scores.
    reference :     string | R file object, optional
                    Default = 'nat.flybrains::FCWB'

    Returns
    -------
    nblast_results
        Instance of :class:`pymaid.rmaid.NBLASTresults` that holds nblast
        results and contains wrappers to plot/extract data. Please use
        help(NBLASTresults) to learn more and see example below.

    Examples
    --------
    >>> import pymaid
    >>> # Initialize connection to Catmaid server
    >>> rm = CatmaidInstance( url, http_user, http_pw, token )
    >>> # Blast a neuron against default (FlyCircuit) database
    >>> nbl = pymaid.nblast( skid = 16, remote_instance = rm  )
    >>> # See contents of nblast_res object
    >>> help(nbl)
    >>> # Get results as Pandas Dataframe
    >>> nbl.results
    >>> # Plot histogram of results
    >>> nbl.plot.hist(alpha=.5)
    >>> # Sort and plot the first hits
    >>> nbl.sort('mu_score')
    >>> nbl.plot(hits = 4)
    """

    start_time = time.time()

    domc = importr('doMC')
    cores = robjects.r('registerDoMC(%i)' % n_cores)

    doParallel = importr('doParallel')
    doParallel.registerDoParallel(cores=n_cores)

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']

    try:
        flycircuit = importr('flycircuit')
        datadir = robjects.r('getOption("flycircuit.datadir")')[0]
    except BaseException:
        logger.error('R Flycircuit not found.')

    if db is None:
        if not os.path.isfile(datadir + '/dpscanon.rds'):
            logger.error('Unable to find default DPS database dpscanon.rds in '
                         'flycircuit.datadir. Please provide database using '
                         'db parameter.')
            return
        logger.info('DPS database not explicitly provided. Loading local '
                    'FlyCircuit DB from dpscanon.rds')
        dps = robjects.r('read.neuronlistfh("%s")' %(datadir + '/dpscanon.rds'))
    elif isinstance(db, str):
        if db.startswith('http') or '/' in db:
            dps = robjects.r('read.neuronlistfh("%s")' % db)
        else:
            dps = robjects.r('read.neuronlistfh("%s")' % datadir + '/' + db)
    elif 'rpy2' in str(type(db)):
        dps = db
    else:
        logger.error('Unable to process the DPS database you have provided. '
                     'See help(rmaid.nblast) for details.')
        return

    if 'rpy2' in str(type(neuron)):
        rn = neuron
    elif isinstance(neuron, pd.DataFrame) or isinstance(neuron, core.CatmaidNeuronList):
        if neuron.shape[0] > 1:
            logger.warning('You provided more than a single neuron. Blasting '
                           'only against the first: %s' % neuron.ix[0].neuron_name)
        rn = neuron2r(neuron.ix[0], convert_to_um=False)
    elif isinstance(neuron, pd.Series) or isinstance(neuron, core.CatmaidNeuron):
        rn = neuron2r(neuron, convert_to_um=False)
    elif isinstance(neuron, str) or isinstance(neuron, int):
        if not remote_instance:
            logger.error('You have to provide a CATMAID instance using the '
                         '<remote_instance> parameter. See help(rmaid.nblast) '
                         'for details.')
            return
        rn = neuron2r(fetch.get_neuron(
            neuron, remote_instance), convert_to_um=False)
    else:
        logger.error('Unable to intepret <neuron> parameter provided. See '
                     'help(rmaid.nblast) for details.')
        return

    # Bring catmaid neuron into reference brain space -> this also converts to
    # um
    if isinstance(reference, str):
        reference = robjects.r(reference)
    rn = nat_templatebrains.xform_brain(
        nat.neuronlist(rn), sample='FAFB14', reference=reference)

    # Mirror neuron
    if mirror:
        rn = nat_templatebrains.mirror_brain(rn, reference)

    # Save template brain for later
    tb = nat_templatebrains.regtemplate(rn)

    # Get neuron object out of the neuronlist
    rn = rn.rx2(1)

    # Reassign template brain
    rn.slots['regtemplate'] = tb

    # The following step are from nat.dotprops_neuron()
    # xdp = nat.dotprops( nat.xyzmatrix(rn) )
    xdp = nat.dotprops(rn, resample=1, k=5)

    # number of reverse scores to calculate (max 100)
    nrev = min(100, len(dps))

    logger.info('Blasting neuron...')
    if reverse:
        sc = r_nblast.nblast(dps, nat.neuronlist(
            xdp), **{'normalised': normalised, '.parallel': True})

        # Have to convert to dataframe to sort them -> using
        # 'robjects.r("sort")' looses the names for some reason
        sc_df = pd.DataFrame([[sc.names[0][i], sc[i]] for i in range(len(sc))],
                             columns=['name', 'score'])
        sc_df.sort_values('score', ascending=False, inplace=True)

        # Use ".rx()" like "[]" and "rx2()" like "[[]]" to extract subsets of R
        # objects
        scr = r_nblast.nblast(nat.neuronlist(xdp),
                              dps.rx(robjects.StrVector(sc_df.name.tolist()[:nrev])),
                              **{'normalised': normalised,
                                 '.parallel': True,
                                 'UseAlpha': UseAlpha})
    else:
        sc = r_nblast.nblast(nat.neuronlist(xdp), dps, **
                             {'normalised': normalised, '.parallel': True})

        # Have to convert to dataframe to sort them -> using
        # 'robjects.r("sort")' looses the names for some reason
        sc_df = pd.DataFrame([[sc.names[0][i], sc[i]] for i in range(len(sc))],
                             columns=['name', 'score'])
        sc_df.sort_values('score', ascending=False, inplace=True)

        # Use ".rx()" like "[]" and "rx2()" like "[[]]" to extract subsets of R
        # objects
        scr = r_nblast.nblast(dps.rx(robjects.StrVector(sc_df.name.tolist()[:nrev])),
                              nat.neuronlist(xdp),
                              **{'normalised': normalised,
                                 '.parallel': True,
                                 'UseAlpha': UseAlpha})

    sc_df.set_index('name', inplace=True, drop=True)

    df = pd.DataFrame([[scr.names[i], sc_df.ix[scr.names[i]].score, scr[i], (sc_df.ix[scr.names[i]].score + scr[i]) / 2]
                       for i in range(len(scr))],
                      columns=['gene_name', 'forward_score',
                               'reverse_score', 'mu_score']
                      )

    logger.info('Blasting done in %s seconds' % round(time.time() - start_time))

    return NBLASTresults(df, sc, scr, rn, xdp, dps, {'mirror': mirror,
                                                     'reference': reference,
                                                     'UseAlpha': UseAlpha,
                                                     'normalised': normalised,
                                                     'reverse': reverse})


class NBLASTresults:
    """ Class that holds nblast results and contains wrappers that allow easy
    plotting.

    Attributes
    ----------
    results :   pandas.Dataframe
                Contains top N results.
    sc :        Robject
                Contains original RNblast forward scores.
    scr :       Robject
                Original R Nblast reverse scores (Top N only).
    neuron :    R ``catmaidneuron``
                The neuron that was nblasted transformed into reference space.
    xdp :       robject
                Dotproduct of the transformed neuron.
    param :     dict
                Contains parameters used for nblasting.
    db :        file robject
                Dotproduct database as R object "neuronlistfh".
    date :      datetime object
                Time of nblasting.

    Examples
    --------
    >>> import pymaid
    >>> # Blast neuron by skeleton ID
    >>> nbl = pymaid.nblast( skid, remote_instance = rm )
    >>> # Sort results by mu_score
    >>> nbl.sort( 'mu_score' )
    >>> # Show table
    >>> nbl.results
    >>> # 3D plot top 5 hits using vispy
    >>> canvas, view = nbl.plot(hits=5)
    >>> # Show distribution of results
    >>> import matplotlib.pyplot as plt
    >>> nbl.results.hist( layout=(3,1), sharex=True)
    >>> plt.show()
    """

    def __init__(self, results, sc, scr, neuron, xdp, dps_db, nblast_param):
        """ Init function."""
        self.results = results  # this is pandas Dataframe holding top N results
        self.sc = sc  # original Nblast forward scores
        self.scr = scr  # original Nblast reverse scores (Top N only)
        self.neuron = neuron  # the transformed neuron that was nblasted
        self.xdp = xdp  # dotproduct of the transformed neuron
        self.db = dps_db  # dotproduct database as R object "neuronlistfh"
        self.param = nblast_param  # parameters used for nblasting
        self.date = datetime.now()  # time of nblasting

    def sort(self, columns):
        """ Sort results by given column."""
        self.results.sort_values(columns, inplace=True, ascending=False)
        self.results.reset_index(inplace=True, drop=True)

    def plot3d(self, hits=5, plot_neuron=True, plot_brain=True, **kwargs):
        """ Wrapper to plot nblast hits using ``pymaid.plot3d()``

        Parameters
        ----------
        hits :  int | str | list thereof, optional
                Nblast hits to plot (default = 5). Can be:

                1. int: e.g. ``hits=5`` for top 5 hits
                2 .list of ints: e.g. ``hits=[2,5]`` to plot hits 2 and 5
                3. string: e.g. ``hits='THMARCM-198F_seg1'`` to plot this neuron
                4. list of strings: e.g. ``['THMARCM-198F_seg1', npfMARCM-'F000003_seg002']`` to plot multiple neurons by their gene name

        plot_neuron :   bool
                        If ``True``, the nblast query neuron will be plotted.
        plot_brain :    bool
                        If ``True``, the reference brain will be plotted.
        **kwargs
                        Parameters passed to :func:`~pymaid.plot3d`.
                        See ``help(pymaid.plot3d)`` for details.

        Returns
        -------
        Depending on the backends used by ``pymaid.plot3d()``:

        vispy (default) : canvas, view
        plotly : matplotlib figure

        You can specify the backend by using e.g. ``backend='plotly'`` in
        **kwargs. See ``help(pymaid.plot3d)`` for details.
        """

        nl = self.get_dps(hits)

        n_py = neuron2py(self.neuron)
        # We have to bring the soma radius down to um -> this may mess
        # up soma detection elsewhere, so be carefull!
        n_py.ix[0].nodes.radius /= 1000

        # Create colormap with the query neuron being black
        cmap = {n_py.ix[0].skeleton_id: (0, 0, 0)}

        colors = np.linspace(0, 1, len(nl) + 1)
        colors = np.array([hsv_to_rgb(c, 1, 1) for c in colors])
        colors *= 255
        cmap.update({e: colors[i] for i, e in enumerate(nl.names)})

        # Prepare brain
        if plot_brain:
            ref_brain = robjects.r(self.param['reference'][8][0] + '.surf')

            verts = data2py(ref_brain[0])[['X', 'Y', 'Z']].as_matrix().tolist()
            faces = data2py(ref_brain[1][0]).as_matrix()
            faces -= 1  # reduce indices by 1
            faces = faces.tolist()
            # [ [i,i+1,i+2] for i in range( int( len(verts)/3 ) ) ]

            volumes = {self.param['reference'][8][0]: {'verts': verts, 'faces': faces}}
        else:
            volumes = []

        kwargs.update({'colors': cmap,
                       'downsampling': 1})

        logger.info('Colormap:' + str(cmap))

        if nl:
            if plot_neuron is True:
                return plotting.plot3d([n_py, dotprops2py(nl), volumes], **kwargs)
            else:
                return plotting.plot3d([dotprops2py(nl), volumes], **kwargs)

    def get_dps(self, entries):
        """ Wrapper to retrieve dotproducts from DPS database (neuronlistfh)
        as neuronslist.

        Parameters
        ----------
        entries :   int | str | list thereof, optional
                    Neurons to extract from DPS database. Can be:

                    1. int: e.g. ``hits=5`` for top 5 hits
                    2 .list of ints: e.g. ``hits=[2,5]`` to plot hits 2 and 5
                    3. string: e.g. ``hits = 'THMARCM-198F_seg1'`` to plot this neuron
                    4. list of strings:
                       e.g. ``['THMARCM-198F_seg1', npfMARCM-'F000003_seg002']``
                       to plot multiple neurons by their gene name

        Returns
        -------
        neuronlist of dotproduct neurons
        """

        if isinstance(entries, int):
            return self.db.rx(robjects.StrVector(self.results.ix[:entries - 1].gene_name.tolist()))
        elif isinstance(entries, str):
            return self.db.rx(entries)
        elif isinstance(entries, (list, np.ndarray)):
            if isinstance(entries[0], int):
                return self.db.rx(robjects.StrVector(self.results.ix[entries].gene_name.tolist()))
            elif isinstance(entries[0], str):
                return self.db.rx(robjects.StrVector(entries))
        else:
            logger.error('Unable to intepret entries provided. See '
                         'help(NBLASTresults.plot3d) for details.')
            return None
