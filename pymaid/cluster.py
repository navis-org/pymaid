#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
#    Copyright (C) 2017 Philipp Schlegel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

import pylab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import logging
import scipy.cluster
import scipy.spatial
import colorsys
import sys

from concurrent.futures import ThreadPoolExecutor

import os
import json

from pymaid import fetch, core, plotting, utils

from tqdm import tqdm
if utils.is_jupyter():
    from tqdm import tqdm_notebook, tnrange
    tqdm = tqdm_notebook
    trange = tnrange

import matplotlib as mpl
from scipy.cluster.hierarchy import set_link_color_palette

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if len( module_logger.handlers ) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

__all__ = sorted([  'cluster_by_connectivity',
                    'cluster_by_synapse_placement','cluster_xyz',
                    'ClustResults'])

# Default settings for progress bars
pbar_hide = False
pbar_leave = True

def cluster_by_connectivity(x, remote_instance=None, upstream=True, downstream=True,
                            threshold=1, include_skids=None, exclude_skids=None, min_nodes=2,
                            similarity='vertex_normalized', connectivity_table=None, cluster_kws={}):
    """ Calculate connectivity similarity.

    Parameters
    ----------
    x
                         Neurons as single or list of either:

                         1. skeleton IDs (int or str)
                         2. neuron name (str, exact match)
                         3. annotation: e.g. 'annotation:PN right'
                         4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :    CATMAID instance, optional
    upstream :           bool, optional
                         If True, upstream partners will be considered.
                         Default = True
    downstream :         bool, optional
                         If True, downstream partners will be considered.
                         Default = True
    threshold :          int, optional
                         Only partners with >= this synapses are considered.
                         Default = 1. Attention: this might impair proper
                         comparison: e.g. neuron A and neuron B connect to
                         neuron C with 1 and 3 synapses, respectively. If
                         ``threshold=2``, connection from A to C will be
                         ignored!
    min_nodes :          int, optional
                         Minimum number of nodes for a partners to be
                         considered. Default = 2
    include_skids :      {see ``x``}, optional
                         If filter_skids is not empty, only neurons whose skids
                         are in filter_skids will be considered when
                         calculating similarity score
    exclude_skids :      {see ``x``}, optional
                         Neurons to exclude from calculation of connectivity
                         similarity
    connectivity_table : pd.DataFrame, optional
                         Connectivity table, e.g. from
                         :func:`~pymaid.get_partners`. If provided, will use 
                         this instead of querying CATMAID server. Filters 
                         still apply!

    Returns
    -------
    :func:`pymaid.ClustResults`
                         Custom cluster results class holding the distance
                         matrix and contains wrappers e.g. to plot dendograms.

    Examples
    --------
    >>> import pymaid
    >>> import matplotlib.pyplot as plt
    >>> # Initialise CatmaidInstance
    >>> rm = pymaid.CatmaidInstance( 'url','user','pw','token' )
    >>> # Cluster a set of neurons by their inputs (ignore small fragments)
    >>> res = pymaid.cluster_by_connectivity('annotation:PBG6 P-EN right', upstream=True, downstream=False, threshold=1, min_nodes=500)
    >>> # Get the adjacency matrix
    >>> print(res.mat)
    >>> # Plot a dendrogram
    >>> fig = res.plot_dendrogram()
    >>> plt.show()
    
    """

    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            remote_instance = sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            module_logger.error(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            raise Exception(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')

    # Extract skids from CatmaidNeuron, CatmaidNeuronList, DataFrame or Series
    neurons = utils.eval_skids(x, remote_instance=remote_instance)

    # Make sure neurons are strings, not integers
    neurons = [str(n) for n in list(set(neurons))]

    directions = []
    if upstream is True:
        directions.append('upstream')
    if downstream is True:
        directions.append('downstream')

    if isinstance(connectivity_table, type(None)):
        # Retrieve connectivity and apply filters
        connectivity = fetch.get_partners(
            neurons, remote_instance, min_size=min_nodes, threshold=threshold)
    else:
        connectivity = connectivity_table[ ( connectivity_table.num_nodes >= min_nodes ) &
                                           ( connectivity_table.total >= threshold ) ]

    if not isinstance( include_skids, type(None) ) or not isinstance(exclude_skids, type(None)):
        module_logger.info(
            'Filtering connectivity. %i entries before filtering' % (connectivity.shape[0]))

        if not isinstance(include_skids, type(None)):
            connectivity = connectivity[
                connectivity.skeleton_id.isin(utils.eval_skids(include_skids, remote_instance=remote_instance))]

        if not isinstance(exclude_skids, type(None)):
            connectivity = connectivity[
                ~connectivity.skeleton_id.isin(utils.eval_skids(exclude_skids, remote_instance=remote_instance))]

        module_logger.info('%i entries after filtering' %
                           (connectivity.shape[0]))

    # Calc number of partners used for calculating matching score (i.e. ratio of input to outputs)!
    # This is AFTER filtering! Total number of partners can be altered!
    number_of_partners = { n: {'upstream': connectivity[(connectivity[str(n)] > 0) & (connectivity.relation == 'upstream')].shape[0],
                              'downstream': connectivity[(connectivity[str(n)] > 0) & (connectivity.relation == 'downstream')].shape[0]
                              }
                          for n in neurons}

    # Retrieve names
    module_logger.debug('Retrieving neuron names')
    neuron_names = fetch.get_names(
        list(set(neurons + connectivity.skeleton_id.tolist())), remote_instance)

    matching_scores = {}

    if similarity in ['vertex_normalized','vertex']:
        vertex_score = True
    else:
        vertex_score = False

    # Calculate connectivity similarity by direction
    for d in directions:
        this_cn = connectivity[connectivity.relation == d]

        # Prepare connectivity subsets:
        cn_subsets = {n: this_cn[n] > 0 for n in neurons}

        module_logger.info('Calculating %s similarity scores' % d)
        matching_scores[d] = pd.DataFrame(
            np.zeros((len(neurons), len(neurons))), index=neurons, columns=neurons)
        if this_cn.shape[0] == 0:
            module_logger.warning('No %s partners found: filtered?' % d)

        combinations = [ (nA,nB,this_cn,vertex_score,cn_subsets[nA],cn_subsets[nB],cluster_kws) for nA in neurons for nB in neurons ]

        with ThreadPoolExecutor(max_workers=max(1, os.cpu_count())) as e:
            futures = e.map( _unpack_connectivity_helper, combinations  )

            matching_indices = [ n for n in tqdm(futures, total=len(combinations),
                                                          desc=d,
                                                          disable=pbar_hide,
                                                          leave=pbar_leave ) ]

        for i,v in enumerate(combinations):
            matching_scores[d].loc[ v[0],v[1] ] = matching_indices[i][similarity]

    # Attention! Averaging over incoming and outgoing pairing scores will give weird results with - for example -  sensory/motor neurons
    # that have predominantly either only up- or downstream partners!
    # To compensate, the ratio of upstream to downstream partners (after applying filters!) is considered!
    # Ratio is applied to neuronA of A-B comparison -> will be reversed at B-A
    # comparison
    module_logger.info('Finalizing scores')
    dist_matrix = pd.DataFrame(
        np.zeros((len(neurons), len(neurons))), index=neurons, columns=neurons)
    for neuronA in neurons:
        for neuronB in neurons:
            if len(directions) == 1:
                dist_matrix[neuronA][neuronB] = matching_scores[
                    directions[0]][neuronA][neuronB]
            else:
                try:
                    r_inputs = number_of_partners[neuronA][
                        'upstream'] / (number_of_partners[neuronA]['upstream'] + number_of_partners[neuronA]['downstream'])
                    r_outputs = 1 - r_inputs
                except:
                    module_logger.warning(
                        'Failed to calculate input/output ratio for %s assuming 50/50 (probably division by 0 error)' % str(neuronA))
                    r_inputs = 0.5
                    r_outputs = 0.5

                dist_matrix[neuronA][neuronB] = matching_scores['upstream'][neuronA][
                    neuronB] * r_inputs + matching_scores['downstream'][neuronA][neuronB] * r_outputs

    module_logger.info('All done.')

    # Rename rows and columns
    #dist_matrix.columns = [neuron_names[str(n)] for n in dist_matrix.columns]
    #dist_matrix.index = [ neuron_names[str(n)] for n in dist_matrix.index ]

    results = ClustResults(dist_matrix, labels = [ neuron_names[str(n)] for n in dist_matrix.columns ], mat_type='similarity' )

    if isinstance(x, core.CatmaidNeuronList):
        results.neurons = x

    return results

def _unpack_connectivity_helper(x):
    """Helper function to unpack values from pool"""
    return _calc_connectivity_matching_index( x[0], x[1], x[2], vertex_score=x[3], nA_cn=x[4], nB_cn=x[5], **x[6]  )

def _calc_connectivity_matching_index(neuronA, neuronB, connectivity, syn_threshold=1, min_nodes=1, **kwargs):
    """ Calculates and returns various matching indices between two neurons.

    Parameters
    ----------
    neuronA :         skeleton ID
    neuronB :         skeleton ID
    connectivity :    pandas DataFrame
                      Connectivity data as provided by ``pymaid.get_partners()``
    syn_threshold :   int, optional
                      Min number of synapses for a connection to be considered.
                      Default = 1
    min_nodes :       int, optional
                      Min number of nodes for a partner to be considered use
                      this to filter fragments. Default = 1
    vertex_score :    bool, optional
                      If False, no vertex score is returned (much faster!).
                      Default = True
    nA_cn/nB_cn :     list of bools
                      Subsets of the connectivity that connect to either
                      neuronA or neuronB -> if not provided, will be calculated
                      -> time consuming

    Returns
    -------
    dict
                      Containing all initially described matching indices

    Notes
    -----
    |matching_index =           Number of shared partners divided by total number
    |                           of partners

    |matching_index_synapses =  Number of shared synapses divided by total number
    |                           of synapses. Attention! matching_index_synapses
    |                           is tricky, because if neuronA has lots of
    |                           connections and neuronB only little, they will
    |                           still get a high matching index.
    |                           E.g. 100 of 200 / 1 of 50 = 101/250
    |                           -> ``matching index = 0.404``

    |matching_index_weighted_synapses = Similar to matching_index_synapses but
    |                           slightly less prone to above mentioned error:
    |                           % of shared synapses A * % of shared synapses
    |                           B * 2 / (% of shared synapses A + % of shared
    |                           synapses B)
    |                           -> value will be between 0 and 1; if one neuronB
    |                           has only few connections (percentage) to a shared
    |                           partner, the final value will also be small

    |vertex_normalized =        Matching index that rewards shared and punishes
    |                           non-shared partners. Vertex similarity based on
    |                           Jarrell et al., 2012:
    |                           f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    |                           x,y = edge weights to compare
    |                           vertex_similarity is the sum of f over all vertices
    |                           C1 determines how negatively a case where one edge
    |                           is much stronger than another is punished
    |                           C2 determines the point where the similarity
    |                           switches from negative to positive

    """

    if min_nodes > 1:
        connectivity = connectivity[connectivity.num_nodes > min_nodes]

    vertex_score = kwargs.get('vertex_score', True)
    nA_cn = kwargs.get('nA_cn', connectivity[neuronA] >= syn_threshold)
    nB_cn = kwargs.get('nB_cn', connectivity[neuronB] >= syn_threshold)

    total = connectivity[nA_cn | nB_cn]
    n_total = total.shape[0]

    shared = connectivity[nA_cn & nB_cn]
    n_shared = shared.shape[0]

    shared_sum = shared.sum()
    n_synapses_sharedA = shared_sum[neuronA]
    n_synapses_sharedB = shared_sum[neuronB]
    n_synapses_shared = n_synapses_sharedA + n_synapses_sharedB

    total_sum = total.sum()
    n_synapses_totalA = total_sum[neuronA]
    n_synapses_totalB = total_sum[neuronB]
    n_synapses_total = n_synapses_totalA + n_synapses_totalB

    # Vertex similarity based on Jarrell et al., 2012
    # f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    # x,y = edge weights to compare
    # vertex_similarity is the sum of f over all vertices
    # C1 determines how negatively a case where one edge is much stronger than another is punished
    # C2 determines the point where the similarity switches from negative to
    # positive
    C1 = kwargs.get('C1', 0.5)
    C2 = kwargs.get('C2', 1)
    vertex_similarity = 0
    max_score = 0
    similarity_indices = {}

    if vertex_score:
        # Get index of neuronA and neuronB -> itertuples unfortunately
        # scrambles the column
        nA_index = connectivity.columns.tolist().index(neuronA)
        nB_index = connectivity.columns.tolist().index(neuronB)

        # We only need the columns for neuronA and neuronB
        this_cn = total[ [neuronA, neuronB] ]

        # Get min and max between both neurons
        this_max = np.max(this_cn, axis=1)
        this_min = np.min(this_cn, axis=1)

        # The max possible score is when both synapse counts are the same:
        # in which case score = max(x,y) - C1 * max(x,y) * e^(-C2 * max(x,y))
        max_score = this_max - C1 * this_max * np.exp( - C2 * this_max )

        # The smallest possible score is when either synapse count is 0:
        # in which case score = -C1 * max(a,b)
        min_score = -C1 * this_max

        # Implement: f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
        v_sim = this_min - C1 * this_max * np.exp( - C2 * this_min )

        # Sum over all partners
        vertex_similarity = v_sim.sum()

        similarity_indices['vertex'] = vertex_similarity

        try:
            similarity_indices['vertex_normalized'] = (
                vertex_similarity - min_score.sum() ) / (max_score.sum() - min_score.sum())
        except:
            similarity_indices['vertex_normalized'] = 0

    if n_total != 0:
        similarity_indices['matching_index'] = n_shared / n_total
        similarity_indices[
            'matching_index_synapses'] = n_synapses_shared / n_synapses_total
        if n_synapses_sharedA != 0 and n_synapses_sharedB != 0:
            similarity_indices['matching_index_weighted_synapses'] = (
                n_synapses_sharedA / n_synapses_totalA) * (n_synapses_sharedB / n_synapses_totalB)
        else:
            # If no shared synapses at all:
            similarity_indices['matching_index_weighted_synapses'] = 0
    else:
        similarity_indices['matching_index'] = 0
        similarity_indices['matching_index_synapses'] = 0
        similarity_indices['matching_index_weighted_synapses'] = 0

    return similarity_indices

def _unpack_synapse_helper(x):
    """Helper function to unpack values from pool."""
    return _calc_synapse_similarity( x[0], x[1], x[2], x[3], x[4] )

def _calc_synapse_similarity(cnA, cnB, sigma=2000, omega=2000, restrict_cn=None):
    """ Calculates synapses similarity score.

    Notes
    -----
    Synapse similarity score is calculated by calculating for each synapse of
    neuron A: (1) the distance to the closest (eucledian) synapse in neuron B
    and (2) comparing the synapse density around synapse A and B. This is type
    sensitive: presynapses will only be matched with presynapses, post with
    post, etc. The formula is described in Schlegel et al., eLife (2017).

    Parameters
    ----------
    (cnA, cnB) :    CatmaidNeuron connector tables
    sigma :         int, optional
                    Distance in nanometer that is considered to be
                    "close"
    omega :         int, optional
                    Radius in nanometer over which to calculate
                    synapse density

    Returns
    -------
    synapse_similarity_score

    """

    all_values = []

    # Get the connector types that we want to compare between neuron A and B
    if isinstance( restrict_cn, type(None) ):
        # If no restrictions, get all cn types in neuron A
        cn_to_check = cnA.relation.unique()
    else:
        # Intersect restricted connectors and actually available types
        cn_to_check = set( cnA.relation.unique() ) & set( restrict_cn )

    # Iterate over all types of connectors
    for r in cn_to_check:
        # Skip if either neuronA or neuronB don't have this synapse type
        if cnB[ cnB.relation == r ].empty:
            all_values += [0] * cnA[ cnA.relation == r ].shape[0]
            continue

        #Get inter-neuron matrix
        dist_mat = scipy.spatial.distance.cdist(  cnA[ cnA.relation == r ][['x','y','z']],
                                                  cnB[ cnB.relation == r ][['x','y','z']] )

        #Get index of closest synapse in neuron B
        closest_ix = np.argmin(dist_mat,axis=1)

        #Get closest distances
        closest_dist = dist_mat.min(axis=1)

        #Get intra-neuron matrices for synapse density checking
        distA = scipy.spatial.distance.pdist( cnA[ cnA.relation == r ][['x','y','z']] )
        distA = scipy.spatial.distance.squareform(distA)
        distB = scipy.spatial.distance.pdist( cnB[ cnB.relation == r ][['x','y','z']] )
        distB = scipy.spatial.distance.squareform(distB)

        #Calculate number of synapses closer than OMEGA. This does count itself!
        closeA = (distA <= omega).sum(axis=1)
        closeB = (distB <= omega).sum(axis=1)

        #Now calculate the scores over all synapses
        for a in range(distA.shape[0]):
            this_synapse_value = math.exp( -1 * math.fabs(closeA[a] - closeB[ closest_ix[a] ]) / (closeA[a] + closeB[ closest_ix[a] ]) ) * math.exp( -1 * (closest_dist[a]**2) / (2 * sigma**2))
            all_values.append(this_synapse_value)

    score = sum(all_values)/len(all_values)

    return score

def cluster_by_synapse_placement(x, sigma=2000, omega=2000, mu_score=True, restrict_cn=None, remote_instance=None):
    """ Clusters neurons based on their synapse placement.

    Notes
    -----
    Distances score is calculated by calculating for each synapse of
    neuron A: (1) the distance to the closest (eucledian) synapse in neuron B
    and (2) comparing the synapse density around synapse A and B. 
    This is type-sensitive: presynapses will only be matched with presynapses, 
    post with post, etc. The formula is described in 
    Schlegel et al., eLife (2017).

    Parameters
    ----------
    x
                        Neurons as single or list of either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    sigma :             int, optional
                        Distance in nanometer between synapses that is
                        considered to be "close".
    omega :             int, optional
                        Radius in nanometer over which to calculate synapse
                        density.
    mu_score :          bool, optional
                        If True, score is calculated as mean between A->B and
                        B->A comparison.
    restrict_cn :       {int, list, None}, optional
                        Restrict to given connector types:
                            - 0: presynapses
                            - 1: postsynapses
                            - 2: gap junctions
                            - 3: abutting connectors
                        If None, will use all connectors. Use either single
                        integer or list. E.g. ``restrict_cn=[0,1]`` to use  
                        only pre- and postsynapses.
    remote_instance :   CatmaidInstance, optional
                        Need to provide if neurons are only skids or
                        annotation(s).

    Returns
    -------
    :class:`~pymaid.ClustResults`
                Object that contains distance matrix and methods to plot
                dendrograms.

    """

    if not isinstance(x, core.CatmaidNeuronList):
        if remote_instance is None:
            if 'remote_instance' in sys.modules:
                remote_instance = sys.modules['remote_instance']
            elif 'remote_instance' in globals():
                remote_instance = globals()['remote_instance']
            else:
                raise Exception(
                    'Please either pass a CATMAID instance or define globally as "remote_instance" ')
        neurons = fetch.get_neuron(x, remote_instance=remote_instance)
    else:
        neurons = x

    # If single value, turn into list
    if not isinstance(restrict_cn, ( type(None), list, set, np.ndarray )):
        restrict_cn = [ restrict_cn ]

    sim_matrix = pd.DataFrame(
        np.zeros((len(neurons), len(neurons))), index=neurons.skeleton_id, columns=neurons.skeleton_id)

    combinations = [ (nA.connectors,nB.connectors,sigma,omega,restrict_cn) for nA in neurons for nB in neurons ]
    comb_skids = [ (nA.skeleton_id,nB.skeleton_id) for nA in neurons for nB in neurons ]

    with ThreadPoolExecutor(max_workers=max(1, os.cpu_count())) as e:
        futures = e.map( _unpack_synapse_helper, combinations  )

        scores = [ n for n in tqdm(futures, total=len(combinations),
                                   desc='Processing',
                                   disable=pbar_hide,
                                   leave=pbar_leave ) ]

    for i,v in enumerate(combinations):
        sim_matrix.loc[ comb_skids[i][0], comb_skids[i][1] ] = scores[i]

    if mu_score:
        sim_matrix = (sim_matrix + sim_matrix.T) / 2

    res = ClustResults(sim_matrix, mat_type='similarity', labels = [ neurons.skid[ str(s) ].neuron_name for s in sim_matrix.columns ])
    res.neurons = neurons

    return res


def cluster_xyz(x, labels=None):
    """ Thin wrapper for scipy.scipy.spatial.distance. Takes a list of x,y,z
    coordinates and calculates EUCLEDIAN distance matrix.

    Parameters
    ----------
    x :             pandas.DataFrame
                    Must contain ``x``,``y``,``z`` columns.
    labels :        list of str, optional
                    Labels for each leaf of the dendrogram
                    (e.g. connector ids).

    Returns
    -------
    :class:`pymaid.ClustResults`
                      Contains distance matrix and methods to generate plots.

    Examples
    --------
    This examples assumes you understand the basics of using pymaid.

    >>> import pymaid
    >>> import matplotlib.pyplot as plt
    >>> pymaid.remote_instance = CatmaidInstance('server','user','pw','token')
    >>> n = pymaid.get_neuron(16)
    >>> rs = pymaid.cluster_xyz(n.connectors, labels=n.connectors.connector_id.tolist())
    >>> rs.plot_matrix()
    >>> plt.show()

    """

    # Generate numpy array containing x, y, z coordinates
    try:
        s = x[['x', 'y', 'z']].as_matrix()
    except:
        module_logger.error(
            'Please provide dataframe connector data of exactly a single neuron')
        return

    # Calculate euclidean distance matrix
    condensed_dist_mat = scipy.spatial.distance.pdist(s, 'euclidean')
    squared_dist_mat = scipy.spatial.distance.squareform(condensed_dist_mat)

    return ClustResults(squared_dist_mat, labels=labels, mat_type='distance')


class ClustResults:
    """ Class to handle, analyze and plot similarity/distance matrices.
    Contains thin wrappers for ``scipy.cluster``.

    Attributes
    ----------
    dist_mat :  Distance matrix (0=similar, 1=dissimilar)
    sim_mat :   Similarity matrix (0=dissimilar, 1=similar)
    linkage :   Hierarchical clustering. Run :func:`pymaid.ClustResults.cluster`
                to generate linkage. By default, WARD's algorithm is used.
    leafs :     list of skids

    Examples
    --------
    >>> import pymaid
    >>> import matplotlib.pyplot as plt
    >>> rm = pymaid.CatmaidInstance('server_url','user','password','token')
    >>> # Get a bunch of neurons
    >>> nl = pymaid.get_neuron('annotation:glomerulus DA1')
    >>> # Perform all-by-all nblast
    >>> res = pymaid.nblast_allbyall( nl )
    >>> # res is a ClustResults object
    >>> res.plot_matrix()
    >>> plt.show()
    >>> # Extract 5 clusters
    >>> res.get_clusters( 5, criterion = 'maxclust' )

    """

    _PERM_MAT_TYPES = ['similarity', 'distance']

    def __init__(self, mat, labels=None, mat_type='distance'):
        """ Initialize class instance.

        Parameters
        ----------
        mat :       {np.array, pandas.DataFrame}
                    Distance or similarity matrix.
        labels :    list, optional
                    Labels for matrix.
        mat_type :  {'distance','similarity'}, default = 'distance'
                    Sets the type of input matrix:
                      - 'similarity' = high values are more similar
                      - 'distance' = low values are more similar

                    The "missing" matrix type will be computed. For clustering,
                    plotting, etc. distance matrices are used.

        """
        if mat_type not in ClustResults._PERM_MAT_TYPES:
            raise ValueError('Matrix type "{0}" unkown.'.format(mat_type) )

        if mat_type == 'similarity':
            self.dist_mat = self._invert_mat( mat )
            self.sim_mat = mat
        else:
            self.dist_mat = mat
            self.sim_mat = self._invert_mat( mat )

        self.labels = labels
        self.mat_type = mat_type

        if isinstance(labels, type(None)) and isinstance(mat, pd.DataFrame):
            self.labels = mat.columns.tolist()

    def __getattr__(self, key):
        if key == 'linkage':
            self.cluster()
            return self.linkage
        elif key == 'condensed_dist_mat':
             return scipy.spatial.distance.squareform( self.dist_mat, checks=False )
        elif key in ['leafs', 'leaves']:
            return self.get_leafs()
        elif key == 'cophenet':
             return self.calc_cophenet()
        elif key == 'agg_coeff':
            return self.calc_agg_coeff()

    def get_leafs(self, use_labels=False):
        """ Use to retrieve labels.

        Parameters
        ----------
        use_labels :    bool, optional
                        If True, self.labels will be returned. If False, will
                        use either columns (if matrix is pandas DataFrame)
                        or indices (if matrix is np.ndarray)

        """

        if isinstance(self.dist_mat, pd.DataFrame):
            if use_labels:
                return [self.labels[i] for i in scipy.cluster.hierarchy.leaves_list(self.linkage)]
            else:
                return [self.dist_mat.columns.tolist()[i] for i in scipy.cluster.hierarchy.leaves_list(self.linkage)]
        else:
            return scipy.cluster.hierarchy.leaves_list(self.linkage)

    def calc_cophenet(self):
        """ Returns Cophenetic Correlation coefficient of your clustering.
        This (very very briefly) compares (correlates) the actual pairwise
        distances of all your samples to those implied by the hierarchical
        clustering. The closer the value is to 1, the better the clustering
        preserves the original distances.

        """

        return scipy.cluster.hierarchy.cophenet( self.linkage, self.condensed_dist_mat )

    def calc_agg_coeff(self):
        """ Returns the agglomerative coefficient. This measures the clustering
        structure of the linkage matrix. Because it grows with the number of
        observations, this measure should not be used to compare datasets of
        very different sizes.

        For each observation i, denote by m(i) its dissimilarity to the first
        cluster it is merged with, divided by the dissimilarity of the merger
        in the final step of the algorithm. The agglomerative coefficient is
        the average of all 1 - m(i).

        """

        # Turn into pandas DataFrame for fancy indexing
        Z = pd.DataFrame(self.linkage, columns = ['obs1','obs2','dist','n_org'] )

        # Get all distances at which an original observation is merged
        all_dist = Z[ ( Z.obs1.isin(self.leafs) ) | (Z.obs2.isin(self.leafs) ) ].dist.values

        # Divide all distances by last merger
        all_dist /= self.linkage[-1][2]

        # Calc final coefficient
        coeff = np.mean( 1 - all_dist )

        return coeff

    def _invert_mat(self, sim_mat):
        """ Inverts matrix."""
        if isinstance( sim_mat, pd.DataFrame ):
            return ( sim_mat - sim_mat.max().max() ) * -1
        else:
            return ( sim_mat - sim_mat.max() ) * -1

    def cluster(self, method='ward'):
        """ Cluster distance matrix. This will automatically be called when
        attribute linkage is requested for the first time.

        Parameters
        ----------
        method :    str, optional
                    Clustering method (see scipy.cluster.hierarchy.linkage
                    for reference)

        """

        # Use condensed distance matrix - otherwise clustering thinks we are
        # passing observations instead of final scores
        self.linkage = scipy.cluster.hierarchy.linkage(self.condensed_dist_mat, method=method)

        # Save method in case we want to look it up later
        self.cluster_method = method

        module_logger.info('Clustering done using method "{0}"'.format(method) )

    def plot_dendrogram(self, color_threshold=None, return_dendrogram=False, labels=None, fig=None, **kwargs):
        """ Plot dendrogram using matplotlib.

        Parameters
        ----------
        color_threshold :   {int,float}, optional
                            Coloring threshold for dendrogram
        return_dendrogram : bool, optional
                            If True, dendrogram object is returned instead of figure
        labels :            list of str, dict
                            Labels in order of original observation or
                            dictionary with mapping original labels
        kwargs
                            Passed to ``scipy.cluster.hierarchy.dendrogram()``

        Returns
        -------
        matplotlib.figure
                            If ``return_dendrogram=False`.
        sciyp.cluster.hierarchy.dendrogram
                            If ``return_dendrogram=True``.

        """

        if isinstance( labels, type(None)):
            labels = self.labels
        elif isinstance(labels, dict):
            labels = [ labels[l] for l in self.labels ]

        if not fig:
            fig = plt.figure()

        dn_kwargs = {'leaf_rotation':90,
                     'above_threshold_color':'k'}
        dn_kwargs.update(kwargs)

        dn = scipy.cluster.hierarchy.dendrogram(self.linkage,
                                          color_threshold=color_threshold,
                                          labels=labels,
                                          **dn_kwargs)
        module_logger.info(
            'Use matplotlib.pyplot.show() to render dendrogram.')

        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        try:
            plt.tight_layout()
        except:
            pass

        if return_dendrogram:
            return dn
        else:
            return fig

    def plot_matrix2(self, **kwargs):
        """ Plot distance matrix and dendrogram using seaborn. This package
        needs to be installed manually.

        Parameters
        ----------
        kwargs      dict
                    Keyword arguments to be passed to seaborn.clustermap. See
                    http://seaborn.pydata.org/generated/seaborn.clustermap.html


        Returns
        -------
        seaborn.clustermap

        """

        try:
            import seaborn as sns
        except:
            raise ImportError('Need seaborn package installed.')

        cg = sns.clustermap(
            self.dist_mat, row_linkage=self.linkage, col_linkage=self.linkage, **kwargs)

        # Rotate labels
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

        # Make labels smaller
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), fontsize=4)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), fontsize=4)

        # Increase padding
        cg.fig.subplots_adjust(right=.8, top=.95, bottom=.2)

        module_logger.info(
            'Use matplotlib.pyplot.show() to render figure.')

        return cg

    def plot_matrix(self):
        """ Plot distance matrix and dendrogram using matplotlib.

        Returns
        -------
        matplotlib figure

        """

        # Compute and plot first dendrogram for all nodes.
        fig = pylab.figure(figsize=(8, 8))
        ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
        Z1 = scipy.cluster.hierarchy.dendrogram(
            self.linkage, orientation='left', labels=self.labels)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
        Z2 = scipy.cluster.hierarchy.dendrogram(self.linkage, labels=self.labels)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = self.dist_mat.copy()

        if isinstance(D, pd.DataFrame):
            D = D.as_matrix()

        D = D[idx1, :]
        D = D[:, idx2]
        im = axmatrix.matshow(
            D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
        pylab.colorbar(im, cax=axcolor)

        module_logger.info(
            'Use matplotlib.pyplot.show() to render figure.')

        return fig

    def plot3d(self, k=5, criterion='maxclust', **kwargs):
        """Plot neuron using :func:`pymaid.plot.plot3d`. Will only work if
        instance has neurons attached to it.

        Parameters
        ----------
        k :         {int, float}
        criterion : {'maxclust','distance'}, optional
                    If `maxclust`, `k` clusters will be formed. If `distance`,
                    clusters will be created at threshold `k`.
        **kwargs
                will be passed to plot.plot3d()
                see help(plot.plot3d) for a list of keywords

        See Also
        --------
        :func:`pymaid.plot.plot3d`
                    Function called to generate 3d plot

        """

        if 'neurons' not in self.__dict__:
            module_logger.error(
                'This works only with cluster results from neurons')
            return None

        cmap = self.get_colormap(k=k, criterion=criterion)

        kwargs.update({'color': cmap})

        return plotting.plot3d(self.neurons, **kwargs)

    def to_json(self, fname='cluster.json', k=5, criterion='maxclust'):
        """ Convert clustered neurons into json file that can be loaded into
        CATMAID selection table.

        Parameters
        ----------
        fname :     str, optional
                    Filename to save selection to
        k :         {int, float}
        criterion : {'maxclust','distance'}, optional
                    If `maxclust`, `k` clusters will be formed. If `distance`,
                    clusters will be created at threshold `k`.

        See Also
        --------
        :func:`pymaid.plot.plot3d`
                    Function called to generate 3d plot

        """

        cmap = self.get_colormap(k=k, criterion=criterion)

        # Convert to 0-255
        cmap = { n : [ int(v*255) for v in cmap[n] ] for n in cmap }

        data = [ dict(skeleton_id=int(n),
                     color="#{:02x}{:02x}{:02x}".format( cmap[n][0],cmap[n][1],cmap[n][2] ),
                     opacity=1
                     ) for n in cmap ]

        with open(fname, 'w') as outfile:
            json.dump(data, outfile)

        module_logger.info('Selection saved as %s in %s' % (fname, os.getcwd()))

        return

    def get_colormap(self, k=5, criterion='maxclust'):
        """Generate colormap based on clustering.

        Parameters
        ----------
        k :         {int, float}
        criterion : {'maxclust','distance'}, optional
                    If `maxclust`, `k` clusters will be formed. If `distance`,
                    clusters will be created at threshold `k`.

        Returns
        -------
        dict
                    {'skeleton_id': (r,g,b),...}

        """

        cl = self.get_clusters(k, criterion, return_type='indices')

        cl = [[self.dist_mat.index.tolist()[i] for i in l] for l in cl]

        colors = [colorsys.hsv_to_rgb(1 / len(cl) * i, 1, 1)
                  for i in range(len(cl) + 1)]

        return {n: colors[i] for i in range(len(cl)) for n in cl[i]}

    def get_clusters(self, k, criterion='maxclust', return_type='labels'):
        """ Wrapper for cluster.hierarchy.fcluster to get clusters.

        Parameters
        ----------
        k :             {int, float}
        criterion :     {'maxclust','distance'}, optional
                        If `maxclust`, `k` clusters will be formed. If
                        `distance`, clusters will be created at threshold `k`.
        return_type :   {'labels','indices','columns','rows'}
                        Determines what to construct the clusters of. 'labels'
                        only works if labels are provided. 'indices' refers
                        to index in distance matrix. 'columns'/'rows' works
                        if distance matrix is pandas DataFrame

        Returns
        -------
        list
                    list of clusters [ [leaf1, leaf5], [leaf2, ...], ... ]

        """

        cl = scipy.cluster.hierarchy.fcluster(self.linkage, k, criterion=criterion)

        if self.labels and return_type.lower()=='labels':
            return [[self.labels[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        elif return_type.lower() == 'rows':
            return [[self.dist_mat.columns.tolist()[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        elif return_type.lower() == 'columns':
            return [[self.dist_mat.index.tolist()[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        else:
            return [[j for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]

    def to_tree(self):
        """ Turns linkage to ete3 tree.

        See Also
        --------
        http://etetoolkit.org/
            Ete3 homepage

        Returns
        -------
        ete 3 tree

        """
        try:
            import ete3
        except:
            raise ImportError('Please install ete3 package to use this function.')

        max_dist = self.linkage[-1][2]
        n_original_obs = self.dist_mat.shape[0]

        list_of_childs = { n_original_obs+i : e[:2] for i,e in enumerate(self.linkage) }
        list_of_parents = { int(e[0]) : int(n_original_obs+i) for i,e in enumerate(self.linkage) }
        list_of_parents.update( { int(e[1]) : int(n_original_obs+i) for i,e in enumerate(self.linkage) } )

        total_dist = { n_original_obs+i : e[2] for i,e in enumerate(self.linkage) }
        # Process total distance into distances between nodes (root = 0)
        dist_to_parent = { n : max_dist - total_dist[ list_of_parents[n] ] for n in list_of_parents }

        names = { i : n for i,n in enumerate(self.dist_mat.columns.tolist())}

        # Create empty tree
        tree = ete3.Tree()

        # Start with root node
        root = sorted ( list( list_of_childs.keys() ), reverse=True)[0]
        treenodes = { root : tree.add_child() }

        for k in sorted(list( list_of_childs.keys()), reverse=True):
            e = list_of_childs[k]
            treenodes[ e[0] ] = treenodes[k].add_child(dist=dist_to_parent[e[0]], name=names.get(e[0],None) )
            treenodes[ e[1] ] = treenodes[k].add_child(dist=dist_to_parent[e[1]], name=names.get(e[1],None) )

        return tree

def _calc_sparseness(x, mode='activity_ratio'):
    """ Calculates sparseness for a set of neurons.

    Parameters
    ----------
    x :         Adjacency matrix
                Pandas DataFrame or numpy array in which rows are sources and
                columns are targets. Sparseness is calculated for targets.
    mode :      str, optional
                Sparseness comes in three different flavours:
                    (1) "activity_ratio" after Rolls and Tovee is used to
                        describes distributions with heavy tails
                    (2) "lifetime_sparseness" after XY
                    (3) "kurtosis"

    """

    if mode not in ['activity_ratio','lifetime_sparseness','kurtosis']:
        raise ValueError('Unknown mode: {0}'.format(mode))

    if isinstance(x, pandas.DataFrame):
        mat = x.as_matrix()
        names = x.columns.tolist()
    elif isinstance(x, np.ndarray):
        mat = x
        names = list(range(x.shape[1]))
    else:
        raise TypeError('Unable to process data of type {0}'.format(type(x)))

    for i in range( mat.shape[1] ):
        this_col = mat[:,i].T
        this_col = this_col[ this_col > 0 ]

        if mode == 'activity_ratio':
            a = ( sum(this_col) / len(this_col) ) **2 / ( sum(this_col ** 2) / len(this_col) )
            S = 1-a



