"""
    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
    Copyright (C) 2017 Philipp Schlegel

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along
"""

import pylab
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math
import logging
from scipy import cluster, spatial

from pymaid import pymaid, core


#Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)
if not module_logger.handlers:
   #Generate stream handler
   sh = logging.StreamHandler()
   sh.setLevel(logging.INFO)
   #Create formatter and add it to the handlers
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
   sh.setFormatter(formatter)
   module_logger.addHandler(sh)

def create_adjacency_matrix( neuronsA, neuronsB, remote_instance, syn_cutoff = None, row_groups = {}, col_groups = {}, syn_threshold = 1 ):
   """ Wrapper to generate a matrix for synaptic connections between neuronsA 
   -> neuronsB (unidirectional!)

   Parameters:
   -----------
   neuronsA :        list of skids
   neuronsB :        list of skids
   remote_instance : CATMAID instance
   syn_cutoff  :     integer (default = None)
                     if set, will cut off synapses above given value
   syn_threshold  :  integer (default = 1)
                     if set, will cut off synapses below given value
   row_groups /:     dict (optional)
   col_groups        use to collapse neuronsA/B into groups
                     example: {'Group1': [skid1,skid2,skid3], 'Group2' : [] }

   Returns:
   -------
   matrix :          pandas Dataframe
   """

   #Extract skids from CatmaidNeuron, CatmaidNeuronList, DataFrame or Series
   try:   
      neuronsA = list( neuronsA.skeleton_id )   
      neuronsB = list( neuronsB.skeleton_id )
   except:
      pass

   #Make sure neurons are strings, not integers
   neurons = list( set( [str(n) for n in list(set(neuronsA + neuronsB))] ) )
   neuronsA = list( set( [str(n) for n in  neuronsA ] ) )
   neuronsB = list( set( [str(n) for n in  neuronsB ] ) )

   neuron_names = pymaid.get_names( neurons, remote_instance )

   module_logger.info('Retrieving and filtering connectivity')
   
   edges = pymaid.get_edges ( neurons , remote_instance = remote_instance)   

   if row_groups or col_groups:
      rows_grouped = { str(n) : g for g in row_groups for n in row_groups[g] }
      cols_grouped = { str(n) : g for g in col_groups for n in col_groups[g] }

      #Groups are sorted alphabetically
      neuronsA = sorted( list(row_groups.keys())) + [ n for n in neuronsA if n not in list(rows_grouped.keys()) ]
      neuronsB = sorted( list(col_groups.keys())) + [ n for n in neuronsB if n not in list(cols_grouped.keys()) ]

      edge_dict = { n:{} for n in neuronsA }
      for e in edges.itertuples():         
         if str(e.source_skid) in rows_grouped:
            source_string = rows_grouped[ str(e.source_skid) ]
         elif str(e.source_skid) in neuronsA:
            source_string = str(e.source_skid)
         else:
            continue
         
         if str(e.target_skid) in cols_grouped:
            target_string = cols_grouped[ str(e.target_skid) ]         
         elif str(e.target_skid) in neuronsB:
            target_string = str(e.target_skid)
         else:
            continue
            
         try:
            edge_dict[ source_string ][ target_string ] += e.weight
         except:            
            edge_dict[ source_string ][ target_string ] = e.weight

   else:
      edge_dict = { n:{} for n in neuronsA }
      for e in edges.itertuples():
         if str(e.source_skid) in neuronsA:
            edge_dict[str(e.source_skid)][str(e.target_skid)] = e.weight    
   
   matrix = pd.DataFrame( np.zeros( ( len( neuronsA ), len( neuronsB ) ) ), index = neuronsA , columns = neuronsB )

   for nA in neuronsA:
      for nB in neuronsB:
         try:
            e = edge_dict[nA][nB]
         except:
            e = 0

         if syn_cutoff:
            e = min( e , syn_cutoff )

         if e < syn_threshold:
            e = 0

         matrix[nB][nA] = e 

   module_logger.info('Finished')

   return matrix

def group_matrix ( mat, row_groups = {} , col_groups = {}, method = 'AVERAGE' ):
   """ Takes a matrix or a pandas Dataframe and groups values by keys provided.  

   Parameters:
   ---------
   mat :          pandas or numpy matrix
   row_groups/ :  dictionaries =  { 'group name' : [ member1, member2, ... ], .. }
   col_groups     for pandas DataFrames members need to be column or index, for
                  np they need to be slices indices
   method :       method by which groups are collapsed
                  can be 'AVERAGE', 'MAX' or 'MIN'

   Returns:
   -------
   pandas DataFrame
   """

   #Convert numpy array to DataFrame
   if type(mat) == type( np.zeros( [0] ) ):
      mat = pd.DataFrame( mat )

   if not row_groups:      
      row_groups = { r : [r] for r in mat.index.tolist() }
   if not col_groups:
      col_groups = { c : [c] for c in mat.columns.tolist() }

   clean_col_groups = {}
   clean_row_groups = {}

   not_found = []
   for row in row_groups:      
      not_found += [ r for r in row_groups[row] if r not in mat.index.tolist() ]
      clean_row_groups[row] = [ r for r in row_groups[row] if r in mat.index.tolist() ]
   for col in col_groups:
      not_found += [ c for c in col_groups[col] if c not in mat.columns.tolist() ]
      clean_col_groups[col] = [ c for c in col_groups[col] if c in mat.columns.tolist() ]

   module_logger.warning('Unable to find the following indices - will skip them: %s' % ', '.join( list(set(not_found))) )

   new_mat = pd.DataFrame( np.zeros( ( len( clean_row_groups ), len( clean_col_groups ) ) ), index = clean_row_groups.keys() , columns = clean_col_groups.keys() )

   for row in clean_row_groups:
      for col in clean_col_groups:
         values = [ [ mat.ix[ r ][ c ] for c in clean_col_groups[col] ] for r in clean_row_groups[row]  ]
         flat_values =  [ v for l in values for v in l ]         
         try:
            if method == 'AVERAGE':           
               new_mat.ix[ row ][ col ] = sum(flat_values)/len(flat_values)
            if method == 'MAX':
               new_mat.ix[ row ][ col ] = max(flat_values)
            if method == 'MIN':
               new_mat.ix[ row ][ col ] = min(flat_values)
         except:
               new_mat.ix[ row ][ col ] = 0

   return new_mat

def create_connectivity_distance_matrix( neurons, remote_instance, upstream=True, downstream=True, threshold=1, filter_skids=[], exclude_skids=[], plot_matrix = True, min_nodes = 2, similarity = 'vertex_normalized'):
   """ Wrapper to calculate connectivity similarity and creates a distance 
   matrix for a set of neurons. Uses Ward's algorithm for clustering.

   Parameters:
   -----------
   neurons :         list of skids
   remote_instance : CATMAID instance
   upstream :        boolean (default=True)
                     if True, upstream partners will be considered
   downstream :      boolean (default=True)
                     if True, downstream partners will be considered
   threshold :       int (default = 1)
                     Only partners with >= this synapses are considered. 
                     Attention: this might impair proper comparison:
                     e.g. neuronA and neuronB connect to neuronC with 1 and 3 
                     synapses, respectively. 
                     If threshold=2, then connection from A to C will be ignored!
   min_nodes :       int (default = 2)
                     minimum number of nodes for a partners to be considered
   filter_skids :    list of skids (optional)
                     If filter_skids is not empty, only neurons whose skids are 
                     in filter_skids will be considered when calculating 
                     similarity score
   exclude_skids :   list of skids (optional)
                     skids to exclude from calculation of connectivity similarity
   plot_matrix :     if True, a plot will be generated (default = True)


   Returns:
   --------
   dist_matrix :     Pandas dataframe containing all-by-all connectivity 
                     distance matrix 
   cg :              (only if plot_matrix = True) Seaborn cluster grid plot 
   """

   #Extract skids from CatmaidNeuron, CatmaidNeuronList, DataFrame or Series
   try:
      neurons = list( neurons.skeleton_id )
   except:
      pass

   #Make sure neurons are strings, not integers
   neurons = [ str(n) for n in list(set(neurons)) ]

   directions = []
   if upstream is True:
      directions.append('upstream')
   if downstream is True:
      directions.append('downstream')

   module_logger.info('Retrieving and filtering connectivity')

   #Retrieve connectivity and apply filters
   connectivity = pymaid.get_partners( neurons, remote_instance, min_size = min_nodes, threshold = threshold )

   #Filter direction
   #connectivity = connectivity[ connectivity.relation.isin(directions) ]

   if filter_skids or exclude_skids:
      module_logger.info('Filtering connectivity. %i entries before filtering' % ( connectivity.shape[0] ) )

      if filter_skids:
         connectivity = connectivity[ connectivity.skeleton_id.isin( filter_skids )  ]

      if exclude_skids:
         connectivity = connectivity[ ~connectivity.skeleton_id.isin( exclude_skids )  ]

      module_logger.info('%i entries after filtering' % ( connectivity.shape[0] ) )   

   #Calc number of partners used for calculating matching score (i.e. ratio of input to outputs)! 
   #This is AFTER filtering! Total number of partners can be altered!      
   number_of_partners = { n : { 'upstream' : connectivity[ (connectivity[str(n)] > 0) & (connectivity.relation == 'upstream') ].shape[0],
                                'downstream' : connectivity[ (connectivity[str(n)] > 0) & (connectivity.relation == 'downstream') ].shape[0]
                              }
                               for n in neurons }  

   module_logger.debug('Retrieving neuron names')
   #Retrieve names
   neuron_names = pymaid.get_names( list( set( neurons + connectivity.skeleton_id.tolist() ) ), remote_instance)   
   
   matching_scores = {}

   if similarity == 'vertex_normalized':
      vertex_score = True
   else:
      vertex_score = False

   #Calculate connectivity similarity by direction
   for d in directions:  
      this_cn = connectivity[ connectivity.relation == d ]

      #Prepare connectivity subsets:
      cn_subsets = { n : this_cn[n] > 0 for n in neurons }

      module_logger.info('Calculating %s similarity scores' % d)
      matching_scores[d] = pd.DataFrame( np.zeros( ( len( neurons ), len( neurons ) ) ), index = neurons, columns = neurons )
      if this_cn.shape[0] == 0:
         module_logger.warning('No %s partners found: filtered?' % d)
      
      #Compare all neurons vs all neurons
      for i, neuronA in enumerate(neurons):  
         print('%s (%i of %i)' % ( str( neuronA), i, len(neurons) ), end = ', ')         
         for neuronB in neurons:          
            matching_indices = _calc_matching_index ( neuronA, neuronB, this_cn, vertex_score = vertex_score, nA_cn = cn_subsets[neuronA], nB_cn = cn_subsets[neuronB] )                
            matching_scores[d][neuronA][neuronB] = matching_indices[similarity]

   #Attention! Averaging over incoming and outgoing pairing scores will give weird results with - for example -  sensory/motor neurons
   #that have predominantly either only up- or downstream partners!
   #To compensate, the ratio of upstream to downstream partners (after applying filters!) is considered!
   #Ratio is applied to neuronA of A-B comparison -> will be reversed at B-A comparison
   module_logger.info('Calculating average scores')
   dist_matrix = pd.DataFrame( np.zeros( ( len( neurons ), len( neurons ) ) ), index = neurons, columns = neurons )
   for neuronA in neurons:
      for neuronB in neurons:
         if len(directions) == 1:
            dist_matrix[neuronA][neuronB] = matching_scores[directions[0]][neuronA][neuronB]
         else:
            try:
               r_inputs = number_of_partners[neuronA]['upstream']/(number_of_partners[neuronA]['upstream']+number_of_partners[neuronA]['downstream'])
               r_outputs = 1 - r_inputs
            except:
               module_logger.warning('Failed to calculate input/output ratio for %s assuming 50/50 (probably division by 0 error)' % str(neuronA) )
               r_inputs = 0.5
               r_outputs = 0.5

            dist_matrix[neuronA][neuronB] = matching_scores['upstream'][neuronA][neuronB] * r_inputs + matching_scores['downstream'][neuronA][neuronB] * r_outputs

   module_logger.info('All done.')

   #Rename rows and columns
   dist_matrix.columns = [ neuron_names[str(n)] for n in dist_matrix.columns ]
   #dist_matrix.index = [ neuron_names[str(n)] for n in dist_matrix.index ]

   if plot_matrix:
      import seaborn as sns   

      linkage = cluster.hierarchy.ward( dist_matrix.as_matrix() )
      cg = sns.clustermap(dist_matrix, row_linkage=linkage, col_linkage=linkage)    

      #Rotate labels
      plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
      plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

      #Increase padding
      cg.fig.subplots_adjust(right=.8, top=.95, bottom=.2)

      return dist_matrix, cg

   return dist_matrix


def _calc_matching_index( neuronA, neuronB, connectivity, syn_threshold = 1, min_nodes = 1, **kwargs ): 
   """ Calculates and returns various matching indices between two neurons.   
   
   matching_index =           Number of shared partners divided by total number 
                              of partners
   
   matching_index_synapses =  Number of shared synapses divided by total number 
                              of synapses. Attention! matching_index_synapses is 
                              tricky, because if neuronA has lots of connections 
                              and neuronB only little, they will still get a 
                              high matching index. 
                              E.g. 100 of 200 / 1 of 50 = 101/250 
                              -> matching index = 0.404
   
   matching_index_weighted_synapses = Similar to matching_index_synapses but 
                              slightly less prone to above mentioned error: 
                              % of shared synapses A * % of shared synapses 
                              B * 2 / (% of shared synapses A + % of shared 
                              synapses B)
                              -> value will be between 0 and 1; if one neuronB 
                              has only few connections (percentage) to a shared 
                              partner, the final value will also be small
   
   vertex_normalized =        Matching index that rewards shared and punishes 
                              non-shared partners. Vertex similarity based on 
                              Jarrell et al., 2012: 
                              f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
                              x,y = edge weights to compare
                              vertex_similarity is the sum of f over all vertices
                              C1 determines how negatively a case where one edge 
                              is much stronger than another is punished
                              C2 determines the point where the similarity 
                              switches from negative to positive                        

   Parameters:
   -----------
   neuronA :         skeleton ID
   neuronB :         skeleton ID
   connectivity :    connectivity data as provided by pymaid.get_partners()
   syn_threshold :   min number of synapses for a connection to be considered
   min_nodes :       min number of nodes for a partner to be considered
                     use this to filter fragments   
   vertex_score :    (default = True)
                     if False, no vertex score is returned (much faster!)
   nA_cn/nB_cn :     list of booleans
                     subsets of the connectivity that connect to either neuronA
                     or neuronB -> if not provided, will be calculated -> time
                     consuming

   Returns:
   -------
   dict containing all initially described matching indices
   """   

   if min_nodes > 1:
      connectivity = connectivity[ connectivity.num_nodes > min_nodes ]

   vertex_score = kwargs.get('vertex_score', True )   
   nA_cn = kwargs.get('nA_cn', connectivity[neuronA] >= syn_threshold ) 
   nB_cn = kwargs.get('nB_cn', connectivity[neuronB] >= syn_threshold )    

   total = connectivity[ nA_cn | nB_cn ]
   n_total = total.shape[0]

   shared = connectivity[ nA_cn & nB_cn ]
   n_shared = shared.shape[0]  

   n_synapses_sharedA = shared.sum()[neuronA]
   n_synapses_sharedB = shared.sum()[neuronB]    
   n_synapses_shared = n_synapses_sharedA + n_synapses_sharedB
   n_synapses_totalA = total.sum()[neuronA]
   n_synapses_totalB = total.sum()[neuronB]
   n_synapses_total = n_synapses_totalA + n_synapses_totalB

   #Vertex similarity based on Jarrell et al., 2012
   # f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
   # x,y = edge weights to compare
   # vertex_similarity is the sum of f over all vertices
   # C1 determines how negatively a case where one edge is much stronger than another is punished
   # C2 determines the point where the similarity switches from negative to positive
   C1 = 0.5
   C2 = 1
   vertex_similarity = 0
   max_score = 0 
   similarity_indices = {}

   if vertex_score:
      #Get index of neuronA and neuronB -> itertuples unfortunately scrambles the column
      nA_index = connectivity.columns.tolist().index(neuronA) 
      nB_index = connectivity.columns.tolist().index(neuronB) 

      #Go over all entries in which either neuron has at least a single connection
      #If both have 0 synapses, similarity score would not change at all anyway
      for entry in total.itertuples(index=False): #index=False neccessary otherwise nA_index is off by +1
         a = entry[nA_index]
         b = entry[nB_index]

         max_score += max( [ a , b ] )

         vertex_similarity += ( 
                           min([ a , b ]) - C1 * max([ a , b ]) * math.exp(- C2 * min([ a , b ]))
                              )      

      try: 
         similarity_indices['vertex_normalized'] = ( vertex_similarity + C1 * max_score ) / ( ( 1 + C1 ) * max_score) 
         #Reason for (1+C1) is that by increasing vertex_similarity first by C1*max_score, we also increase the maximum reachable value      
      except:     
         similarity_indices['vertex_normalized'] = 0

   if n_total != 0:
      similarity_indices['matching_index'] = n_shared/n_total
      similarity_indices['matching_index_synapses'] = n_synapses_shared/n_synapses_total
      try:
         similarity_indices['matching_index_weighted_synapses'] = (n_synapses_sharedA/n_synapses_totalA) * (n_synapses_sharedB/n_synapses_totalB) 
         # * 2 / ((n_synapses_sharedA/n_synapses_totalA) + (n_synapses_sharedB/n_synapses_totalB))
      except:
         #If no shared synapses at all:
         similarity_indices['matching_index_weighted_synapses'] = 0
   else:
      similarity_indices['matching_index'] = 0
      similarity_indices['matching_index_synapses'] = 0
      similarity_indices['matching_index_weighted_synapses'] = 0

   return similarity_indices

def synapse_distance_matrix(synapse_data, labels = None, plot_matrix = True, method = 'ward'):
   """ Takes a list of CATMAID synapses, calculates EUCLEDIAN distance matrix 
   and clusters them (WARD algorithm)

   Parameters:
   ----------
   synapse_data :    Pandas dataframe
                     Contains the connector data (df.connectors)
   labels :          list of strings
                     Labels for each leaf of the dendrogram (e.g. connector ids). 
   plot_matrix :     boolean
                     if True, matrix figure is generated and returned
   method :          method used for hierarchical clustering 
                     (scipy.cluster.hierarchy.linkage)
                     possible values: 'single', 'ward', 'complete', 'average', 
                     'weighted', 'centroid'

   Returns:
   -------
   dist_matrix :     numpy distance matrix
   fig :             (only if plot_matrix = True) matplotlib object 
   """      

   #Generate numpy array containing x, y, z coordinates
   try:
      s = synapse_data[ ['x','y','z'] ].as_matrix()
   except:
      module_logger.error('Please provide dataframe connector data of exactly a single neuron')
      return  

   #Calculate euclidean distance matrix 
   condensed_dist_mat = spatial.distance.pdist( s , 'euclidean' )
   squared_dist_mat = spatial.distance.squareform( condensed_dist_mat )

   if plot_matrix:
      # Compute and plot first dendrogram for all nodes.
      fig = pylab.figure(figsize=(8,8))
      ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
      Y = cluster.hierarchy.linkage(squared_dist_mat, method = method)
      Z1 = cluster.hierarchy.dendrogram(Y, orientation='left', labels = labels)
      ax1.set_xticks([])
      ax1.set_yticks([])

      # Compute and plot second dendrogram.
      ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
      Y = cluster.hierarchy.linkage(squared_dist_mat, method = method)
      Z2 = cluster.hierarchy.dendrogram(Y, labels = labels)
      ax2.set_xticks([])
      ax2.set_yticks([])

      # Plot distance matrix.
      axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
      idx1 = Z1['leaves']
      idx2 = Z2['leaves']  
      D = squared_dist_mat
      D = D[idx1,:]
      D = D[:,idx2]
      im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
      axmatrix.set_xticks([])
      axmatrix.set_yticks([])

      # Plot colorbar.
      axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
      pylab.colorbar(im, cax=axcolor)

      return squared_dist_mat, fig
   else:
      return squared_dist_mat



