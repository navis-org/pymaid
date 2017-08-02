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
from scipy import cluster, spatial
import colorsys

from pymaid import pymaid, core, plot

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if not module_logger.handlers:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)


def create_adjacency_matrix(n_a, n_b, remote_instance=None, row_groups={}, col_groups={}, syn_threshold=1, syn_cutoff=None):
    """ Wrapper to generate a matrix for synaptic connections between neuronsA
    -> neuronsB (unidirectional!)

    Parameters
    ----------
    n_a           
                        Source neurons as single or list of either:
                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    n_b            
                        Target neurons as single or list of either:
                        1. skeleton IDs (int or str)
                        2. neuron name (str, exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   CATMAID instance, optional
    syn_cutoff :        int, optional
                        If set, will cut off synapses above given value.                          
    syn_threshold :     int, optional
                        If set, will cut off synapses below given value.                      
    row_groups :        dict, optional
                        Use to collapse neuronsA/B into groups:
                        ``{'Group1': [skid1,skid2,skid3], 'Group2' : [] }``
    col_groups :        dict, optional
                        See row_groups

    Returns
    -------
    matrix :          pandas.Dataframe

    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            module_logger.error(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            raise Exception(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')

    neuronsA = pymaid.eval_skids(n_a, remote_instance=remote_instance)
    neuronsB = pymaid.eval_skids(n_b, remote_instance=remote_instance)

    # Make sure neurons are strings, not integers
    neurons = list(set([str(n) for n in list(set(neuronsA + neuronsB))]))
    neuronsA = [ str(n) for n in neuronsA ]
    neuronsB = [ str(n) for n in neuronsB ]

    #Make sure neurons are unique
    neuronsA = sorted(set(neuronsA), key=neuronsA.index)
    neuronsB = sorted(set(neuronsB), key=neuronsB.index)

    neuron_names = pymaid.get_names(neurons, remote_instance)

    module_logger.info('Retrieving and filtering connectivity')

    edges = pymaid.get_edges(neurons, remote_instance=remote_instance)

    if row_groups or col_groups:
        rows_grouped = {str(n): g for g in row_groups for n in row_groups[g]}
        cols_grouped = {str(n): g for g in col_groups for n in col_groups[g]}

        # Groups are sorted alphabetically
        neuronsA = sorted(list(row_groups.keys())) + \
            [n for n in neuronsA if n not in list(rows_grouped.keys())]
        neuronsB = sorted(list(col_groups.keys())) + \
            [n for n in neuronsB if n not in list(cols_grouped.keys())]

        edge_dict = {n: {} for n in neuronsA}
        for e in edges.itertuples():
            if str(e.source_skid) in rows_grouped:
                source_string = rows_grouped[str(e.source_skid)]
            elif str(e.source_skid) in neuronsA:
                source_string = str(e.source_skid)
            else:
                continue

            if str(e.target_skid) in cols_grouped:
                target_string = cols_grouped[str(e.target_skid)]
            elif str(e.target_skid) in neuronsB:
                target_string = str(e.target_skid)
            else:
                continue

            try:
                edge_dict[source_string][target_string] += e.weight
            except:
                edge_dict[source_string][target_string] = e.weight

    else:
        edge_dict = {n: {} for n in neuronsA}
        for e in edges.itertuples():
            if str(e.source_skid) in neuronsA:
                edge_dict[str(e.source_skid)][str(e.target_skid)] = e.weight

    matrix = pd.DataFrame(
        np.zeros((len(neuronsA), len(neuronsB))), index=neuronsA, columns=neuronsB)

    for nA in neuronsA:
        for nB in neuronsB:
            try:
                e = edge_dict[nA][nB]
            except:
                e = 0

            if syn_cutoff:
                e = min(e, syn_cutoff)

            if e < syn_threshold:
                e = 0

            matrix[nB][nA] = e

    module_logger.info('Finished')

    return matrix


def group_matrix(mat, row_groups={}, col_groups={}, method='AVERAGE'):
    """ Takes a matrix or a pandas Dataframe and groups values by keys provided.

    Parameters
    ----------
    mat :               {pandas.DataFrame, numpy.array}
                        Matrix to group
    row_groups :        dict, optional
                        For pandas DataFrames members need to be column
                        or index, for np they need to be slices indices:
                        ``{ 'group name' : [ member1, member2, ... ], .. }``
    col_groups :        dict, optional
                        See row_groups.
    method :            {'AVERAGE', 'MAX', 'MIN'}
                        Method by which groups are collapsed.

    Returns
    -------
    pandas.DataFrame    
    """

    # Convert numpy array to DataFrame
    if isinstance(mat, np.ndarray):
        mat = pd.DataFrame(mat)

    if not row_groups:
        row_groups = {r: [r] for r in mat.index.tolist()}
    if not col_groups:
        col_groups = {c: [c] for c in mat.columns.tolist()}

    clean_col_groups = {}
    clean_row_groups = {}

    not_found = []
    for row in row_groups:
        not_found += [r for r in row_groups[row]
                      if r not in mat.index.tolist()]
        clean_row_groups[row] = [r for r in row_groups[
            row] if r in mat.index.tolist()]
    for col in col_groups:
        not_found += [c for c in col_groups[col]
                      if c not in mat.columns.tolist()]
        clean_col_groups[col] = [c for c in col_groups[
            col] if c in mat.columns.tolist()]

    module_logger.warning(
        'Unable to find the following indices - will skip them: %s' % ', '.join(list(set(not_found))))

    new_mat = pd.DataFrame(np.zeros((len(clean_row_groups), len(
        clean_col_groups))), index=clean_row_groups.keys(), columns=clean_col_groups.keys())

    for row in clean_row_groups:
        for col in clean_col_groups:
            values = [[mat.ix[r][c] for c in clean_col_groups[col]]
                      for r in clean_row_groups[row]]
            flat_values = [v for l in values for v in l]
            try:
                if method == 'AVERAGE':
                    new_mat.ix[row][col] = sum(flat_values) / len(flat_values)
                if method == 'MAX':
                    new_mat.ix[row][col] = max(flat_values)
                if method == 'MIN':
                    new_mat.ix[row][col] = min(flat_values)
            except:
                new_mat.ix[row][col] = 0

    return new_mat


def create_connectivity_distance_matrix(x, remote_instance=None, upstream=True, downstream=True, threshold=1, filter_skids=[], exclude_skids=[], plot_matrix=True, min_nodes=2, similarity='vertex_normalized'):
    """ Wrapper to calculate connectivity similarity and creates a distance
    matrix for a set of neurons. Uses Ward's algorithm for clustering.

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
                         comparison: e.g. neuronA and neuronB connect to neuronC
                         with 1 and 3 synapses, respectively. If threshold=2,
                         then connection from A to C will be ignored!
    min_nodes :          int, optional
                         Minimum number of nodes for a partners to be
                         considered. Default = 2
    filter_skids :       list of skeleton IDs, optional
                         If filter_skids is not empty, only neurons whose skids
                         are in filter_skids will be considered when
                         calculating similarity score
    exclude_skids :      list of skeleton IDs, optional
                         Neurons to exclude from calculation of connectivity
                         similarity
    plot_matrix :        bool, optional
                         If True, a plot will be generated. Default = True

    Returns
    -------
    dist_matrix :        Pandas dataframe
                         Distance matrix containing all-by-all connectivity
    cg :                 Seaborn cluster grid plot
                         Only if ``plot_matrix = True``
    """

    if remote_instance is None:
        if 'remote_instance' in globals():
            remote_instance = globals()['remote_instance']
        else:
            module_logger.error(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')
            raise Exception(
                'Please either pass a CATMAID instance or define globally as "remote_instance" ')

    # Extract skids from CatmaidNeuron, CatmaidNeuronList, DataFrame or Series
    neurons = pymaid.eval_skids(x, remote_instance=remote_instance)

    # Make sure neurons are strings, not integers
    neurons = [str(n) for n in list(set(neurons))]

    directions = []
    if upstream is True:
        directions.append('upstream')
    if downstream is True:
        directions.append('downstream')

    module_logger.info('Retrieving and filtering connectivity')

    # Retrieve connectivity and apply filters
    connectivity = pymaid.get_partners(
        neurons, remote_instance, min_size=min_nodes, threshold=threshold)

    # Filter direction
    # connectivity = connectivity[ connectivity.relation.isin(directions) ]

    if filter_skids or exclude_skids:
        module_logger.info(
            'Filtering connectivity. %i entries before filtering' % (connectivity.shape[0]))

        if filter_skids:
            connectivity = connectivity[
                connectivity.skeleton_id.isin(filter_skids)]

        if exclude_skids:
            connectivity = connectivity[
                ~connectivity.skeleton_id.isin(exclude_skids)]

        module_logger.info('%i entries after filtering' %
                           (connectivity.shape[0]))

    # Calc number of partners used for calculating matching score (i.e. ratio of input to outputs)!
    # This is AFTER filtering! Total number of partners can be altered!
    number_of_partners = {n: {'upstream': connectivity[(connectivity[str(n)] > 0) & (connectivity.relation == 'upstream')].shape[0],
                              'downstream': connectivity[(connectivity[str(n)] > 0) & (connectivity.relation == 'downstream')].shape[0]
                              }
                          for n in neurons}

    module_logger.debug('Retrieving neuron names')
    # Retrieve names
    neuron_names = pymaid.get_names(
        list(set(neurons + connectivity.skeleton_id.tolist())), remote_instance)

    matching_scores = {}

    if similarity == 'vertex_normalized':
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

        # Compare all neurons vs all neurons
        for i, neuronA in enumerate(neurons):
            print('%s (%i of %i)' % (str(neuronA), i, len(neurons)), end=', ')
            for neuronB in neurons:
                matching_indices = _calc_connectivity_matching_index(
                    neuronA, neuronB, this_cn, vertex_score=vertex_score, nA_cn=cn_subsets[neuronA], nB_cn=cn_subsets[neuronB])
                matching_scores[d][neuronA][
                    neuronB] = matching_indices[similarity]

    # Attention! Averaging over incoming and outgoing pairing scores will give weird results with - for example -  sensory/motor neurons
    # that have predominantly either only up- or downstream partners!
    # To compensate, the ratio of upstream to downstream partners (after applying filters!) is considered!
    # Ratio is applied to neuronA of A-B comparison -> will be reversed at B-A
    # comparison
    module_logger.info('Calculating average scores')
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
    dist_matrix.columns = [neuron_names[str(n)] for n in dist_matrix.columns]
    # dist_matrix.index = [ neuron_names[str(n)] for n in dist_matrix.index ]

    if plot_matrix:
        import seaborn as sns

        linkage = cluster.hierarchy.ward(dist_matrix.as_matrix())
        cg = sns.clustermap(
            dist_matrix, row_linkage=linkage, col_linkage=linkage)

        # Rotate labels
        plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

        # Increase padding
        cg.fig.subplots_adjust(right=.8, top=.95, bottom=.2)

        return dist_matrix, cg

    return dist_matrix


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
    |
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

    n_synapses_sharedA = shared.sum()[neuronA]
    n_synapses_sharedB = shared.sum()[neuronB]
    n_synapses_shared = n_synapses_sharedA + n_synapses_sharedB
    n_synapses_totalA = total.sum()[neuronA]
    n_synapses_totalB = total.sum()[neuronB]
    n_synapses_total = n_synapses_totalA + n_synapses_totalB

    # Vertex similarity based on Jarrell et al., 2012
    # f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
    # x,y = edge weights to compare
    # vertex_similarity is the sum of f over all vertices
    # C1 determines how negatively a case where one edge is much stronger than another is punished
    # C2 determines the point where the similarity switches from negative to
    # positive
    C1 = 0.5
    C2 = 1
    vertex_similarity = 0
    max_score = 0
    similarity_indices = {}

    if vertex_score:
        # Get index of neuronA and neuronB -> itertuples unfortunately
        # scrambles the column
        nA_index = connectivity.columns.tolist().index(neuronA)
        nB_index = connectivity.columns.tolist().index(neuronB)

        # Go over all entries in which either neuron has at least a single connection
        # If both have 0 synapses, similarity score would not change at all
        # anyway
        # index=False neccessary otherwise nA_index is off by +1
        for entry in total.itertuples(index=False):
            a = entry[nA_index]
            b = entry[nB_index]

            max_score += max([a, b])

            vertex_similarity += (
                min([a, b]) - C1 * max([a, b]) * math.exp(- C2 * min([a, b]))
            )

        try:
            similarity_indices['vertex_normalized'] = (
                vertex_similarity + C1 * max_score) / ((1 + C1) * max_score)
            # Reason for (1+C1) is that by increasing vertex_similarity first
            # by C1*max_score, we also increase the maximum reachable value
        except:
            similarity_indices['vertex_normalized'] = 0

    if n_total != 0:
        similarity_indices['matching_index'] = n_shared / n_total
        similarity_indices[
            'matching_index_synapses'] = n_synapses_shared / n_synapses_total
        try:
            similarity_indices['matching_index_weighted_synapses'] = (
                n_synapses_sharedA / n_synapses_totalA) * (n_synapses_sharedB / n_synapses_totalB)
            # * 2 / ((n_synapses_sharedA/n_synapses_totalA) + (n_synapses_sharedB/n_synapses_totalB))
        except:
            # If no shared synapses at all:
            similarity_indices['matching_index_weighted_synapses'] = 0
    else:
        similarity_indices['matching_index'] = 0
        similarity_indices['matching_index_synapses'] = 0
        similarity_indices['matching_index_weighted_synapses'] = 0

    return similarity_indices


def synapse_distance_matrix(synapse_data, labels=None, plot_matrix=True, method='ward'):
    """ Takes a list of CATMAID synapses, calculates EUCLEDIAN distance matrix
    and clusters them (WARD algorithm)

    Parameters
    ----------
    synapse_data :    pandas.DataFrame
                      Contains the connector data (df.connectors)
    labels :          list of str, optional
                      Labels for each leaf of the dendrogram
                      (e.g. connector ids).
    plot_matrix :     boolean, optional
                      If True, matrix figure is generated and returned
    method : {'single', 'ward', 'complete', 'average', 'weighted', 'centroid'}
                      Method used for hierarchical clustering from
                      ``(scipy.cluster.hierarchy.linkage)``

    Returns
    -------
    dist_matrix :     np.array 
                      Distance matrix
    fig :             matplotlib object
                      Only if plot_matrix = True
    """

    # Generate numpy array containing x, y, z coordinates
    try:
        s = synapse_data[['x', 'y', 'z']].as_matrix()
    except:
        module_logger.error(
            'Please provide dataframe connector data of exactly a single neuron')
        return

    # Calculate euclidean distance matrix
    condensed_dist_mat = spatial.distance.pdist(s, 'euclidean')
    squared_dist_mat = spatial.distance.squareform(condensed_dist_mat)

    if plot_matrix:
        # Compute and plot first dendrogram for all nodes.
        fig = pylab.figure(figsize=(8, 8))
        ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
        Y = cluster.hierarchy.linkage(squared_dist_mat, method=method)
        Z1 = cluster.hierarchy.dendrogram(Y, orientation='left', labels=labels)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
        Y = cluster.hierarchy.linkage(squared_dist_mat, method=method)
        Z2 = cluster.hierarchy.dendrogram(Y, labels=labels)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Plot distance matrix.
        axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
        idx1 = Z1['leaves']
        idx2 = Z2['leaves']
        D = squared_dist_mat
        D = D[idx1, :]
        D = D[:, idx2]
        im = axmatrix.matshow(
            D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
        pylab.colorbar(im, cax=axcolor)

        return squared_dist_mat, fig
    else:
        return squared_dist_mat


class clust_results:
    """ Class to handle, analyze and plot distance matrices. Contains thin
    wrappers for scipy.cluster

    Parameters
    ----------
    mat :       {np.array, pandas.DataFrame}
                Distance matrix
    labels :    list, optional
                labels for matrix

    Attributes
    ----------
    mat :       Distance matrix
    linkage :   Hierarchical clustering. Run :func:`pymaid.cluster.clustres.cluster`
                to generate linkage.res
    leafs :     list of skids

    Examples
    --------
    >>> from pymaid import pymaid, rmaid
    >>> import matplotlib.pyplot as plt
    >>> rm = pymaid.CatmaidInstance('server_url','user','password','token')
    >>> pymaid.remote_instance = rm
    >>> #Get a bunch of neurons
    >>> nl = pymaid.get_3D_skeleton('annotation:glomerulus DA1')
    >>> #Perform all-by-all nblast
    >>> res = rmaid.nblast_allbyall( nl )
    >>> #res is a clustres object
    >>> res.plot_mpl()
    >>> plt.show()
    >>> #Extract 5 clusters
    >>> res.get_clusters( 5, criterion = 'maxclust' )
    """

    def __init__(self, mat, labels=None):
        self.mat = mat
        self.labels = labels

        if not labels and isinstance(self.mat, pd.DataFrame):
            self.labels = self.mat.columns.tolist()

    def __getattr__(self, key):
        if key == 'linkage':
            self.cluster()
            return self.linkage
        elif key in ['leafs','leaves']:
            return [ self.mat.columns.tolist()[i] for i in cluster.hierarchy.leaves_list(self.linkage) ]

    def cluster(self, method='single'):
        """ Cluster distance matrix. This will automatically be called when
        attribute linkage is requested for the first time.

        Parameters
        ----------
        method :    str
                    Clustering method (see scipy.cluster.hierarchy.linkage 
                    for reference)
        """
        self.linkage = cluster.hierarchy.linkage(self.mat, method=method)

    def plot_mpl(self, color_threshold=None, return_dendrogram = False):
        """ Plot dendrogram using matplotlib

        Parameters
        ----------
        color_threshold :   {int,float}, optional
                            Coloring threshold for dendrogram
        return_dendrogram : bool, optional
                            If true, dendrogram object is returned
        """

        plt.figure()
        dn = cluster.hierarchy.dendrogram( self.linkage, 
                                           color_threshold=color_threshold, 
                                           labels=self.labels,
                                           leaf_rotation=90)

        module_logger.info(
            'Use matplotlib.pyplot.show() to render dendrogram.')

        if return_dendrogram:
            return dn

    def plot3d(self, k=5, criterion='maxclust', **kwargs):
        """Plot neuron using :func:`pymaid.plot.plot3d`. Will only work if
        instance has neurons attached to it.

        Parameters
        ---------
        k :         {int, float}
        criterion : str
                    Either 'maxclust' or 'distance'. If maxclust, k clusters
                    will be formed. If distance, clusters will be created at
                    threshold k.
        **kwargs
                will be passed to plot.plot3d() 
                see help(plot.plot3d) for a list of keywords      

        See Also
        --------
        :func:`pymaid.plot.plot3d` 
                    Function called to generate 3d plot                  
        """
        if 'neurons' not in self.__dict__:
            module_logger.error('This works only with cluster results from neurons')
            return None

        cl = self.get_clusters(k, criterion, use_labels=False)

        cl = [ [ self.mat.index.tolist()[i] for i in l ] for l in cl ]

        colors = [ colorsys.hsv_to_rgb(1/len(cl)*i,1,1) for i in range( len(cl) + 1 ) ]

        cmap = { n : colors[i] for i in range(len(cl)) for n in cl[i] }        

        kwargs.update({'colormap':cmap})

        return plot.plot3d(skdata=self.neurons, **kwargs)
 
    def get_clusters(self, k, criterion='maxclust', use_labels=True):
        """ Wrapper for cluster.hierarchy.fcluster to get clusters

        Parameters
        ----------
        k :         {int, float}
        criterion : str
                    Either 'maxclust' or 'distance'. If maxclust, k clusters
                    will be formed. If distance, clusters will be created at
                    threshold k.
        use_labels : bool, optional
                     If true and labels have been provided, they will be used
                     to return clusters. Otherwise, indices of the original
                     matrix (self.mat) is returned.    

        Returns
        -------
        list 
                    list of clusters [ [leaf1, leaf5], [leaf2, ...], ... ]
        """
        cl = cluster.hierarchy.fcluster(self.linkage, k, criterion=criterion)

        if self.labels and use_labels:
            return [[self.labels[j] for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl) + 1)]
        else:
            return [[j for j in range(len(cl)) if cl[j] == i] for i in range(min(cl), max(cl))]
