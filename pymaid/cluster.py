import pylab
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import math

from scipy import cluster, spatial
try:
   from pymaid import get_partners, get_names, get_edges
except:
   from pymaid.pymaid import get_partners, get_names, get_edges

import logging

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

def create_adjacency_matrix( neuronsA, neuronsB, remote_instance, syn_cutoff = None, row_groups = {}, column_groups = {} ):
   """ Wrapper to generate a matrix for synaptic connections between neuronsA 
       -> neuronsB (unidirectional!)

   Parameters:
   -----------
   neuronsA :        list of skids
   neuronsB :        list of skids
   remote_instance : CATMAID instance
   syn_cutoff  :     integer (default = None)
                     if set, will cut off synapses at given value
   row_groups /:     dict (optional)
   make_groupsB      use to collapse neuronsA/B into groups
                     example: {'Group1': [skid1,skid2,skid3], 'Group2' : [] }

   Returns:
   -------
   matrix :       pandas Dataframe
   """

   #Make sure neurons are strings, not integers
   neurons = [str(n) for n in list(set(neuronsA + neuronsB))]

   neuron_names = get_names( neurons, remote_instance )

   module_logger.info('Retrieving and filtering connectivity')
   
   edges = get_edges ( neurons , remote_instance = remote_instance)['edges']     

   if row_groups:
      neurons_grouped = { str(n) : g  for g in row_groups for n in row_groups[g] }

      #Groups are sorted alphabetically
      neuronsA = sorted(list(row_groups.keys())) + [ n for n in neuronsA if n not in list(neurons_grouped.keys()) ]

      edge_dict = { n:{} for n in neuronsA }
      for e in edges:
         if str(e[0]) in neuronsA:
            edge_dict[str(e[0])][str(e[1])] = sum(e[2])
         elif str(e[0]) in neurons_grouped:
            try:
               edge_dict[ neurons_grouped[ str(e[0]) ] ][str(e[1])] += sum(e[2])
            except:
               edge_dict[ neurons_grouped[ str(e[0]) ] ][str(e[1])] = sum(e[2])

   else:
      edge_dict = { n:{} for n in neuronsA }
      for e in edges:
         if str(e[0]) in neuronsA:
            edge_dict[str(e[0])][str(e[1])] = sum(e[2])

   
   matrix = pd.DataFrame( np.zeros( ( len( neuronsA ), len( neuronsB ) ) ), index = neuronsA , columns = neuronsB )

   for nA in neuronsA:
      for nB in neuronsB:
         try:
            e = edge_dict[nA][nB]
         except:
            e = 0

         if syn_cutoff:
            e = min( e , syn_cutoff )

         matrix[nB][nA] = e 

   module_logger.info('Finished')

   return matrix  

def create_connectivity_distance_matrix( neurons, remote_instance, upstream=True, downstream=True, threshold=1, filter_skids=[], exclude_skids=[], plot_matrix = True):
   """   Wrapper to calculate connectivity similarity and creates a distance 
         matrix for a set of neurons

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
                     If threshold = 2, then connection from A to C will be ignored!
   filter_skids :    list of skids (optional)
                     If filter_skids is not empty, only neurons whose skids are in filter_skids will be considered 
                     when calculating similarity score
   exclude_skids :   list of skids (optional)
                     skids to exclude from calculation of connectivity similarity
   plot_matrix :     if True, a plot will be generated (default = True)

   Returns:
   --------
   dist_matrix :     Pandas dataframe containing all-by-all connectivity distance matrix 
   cg :              Seaborn cluster grid plot (only if plot_matrix = True)
   """

   #Make sure neurons are strings, not integers
   neurons = [str(n) for n in list(set(neurons))]

   directions = []
   if upstream is True:
      directions.append('incoming')
   if downstream is True:
      directions.append('outgoing')

   module_logger.info('Retrieving and filtering connectivity')
   #Retrieve connectivity and apply filters
   connectivity = get_partners(neurons,remote_instance)
   for d in directions:
      if filter_skids or exclude_skids:
         to_delete = []
         module_logger.info('Filtering %s connectivity. %i entries before filtering' % (d,len(connectivity[d])))
         for entry in connectivity[d]:
            if filter_skids and entry not in filter_skids:
               to_delete.append(entry)
            if exclude_skids and entry in exclude_skids:
               to_delete.append(entry) 

         for skid in list(set(to_delete)):
            connectivity[d].pop(skid) 
         module_logger.info('%s: %i entries after filtering' % ( d, len(connectivity[d]) ) )

   module_logger.info('Retrieving neuron names')
   #Retrieve names
   neuron_names = get_names( list(set(neurons+
                                 list(connectivity['incoming']) +
                                 list(connectivity['outgoing'])
                                 )
                         ), remote_instance)

   number_of_partners = dict([(e,{'incoming':0,'outgoing':0}) for e in neurons])
   
   matching_scores = {} 

   #Calculate connectivity similarity by direction
   for d in directions:    
      module_logger.info('Calculating %s similarity scores' % d)
      matching_scores[d] = pd.DataFrame( np.zeros( ( len( neurons ), len( neurons ) ) ), index = neurons, columns = neurons )

      if len(connectivity) == 0:
         module_logger.warning('No %s partners found: filtered?' % d)   

      #Calc number of partners used for calculating matching score (i.e. ratio of input to outputs)! 
      #This is AFTER filtering! Total number of partners can be altered!      
      for entry in connectivity[d]:
         for skid in connectivity[d][entry]['skids']:          
            number_of_partners[skid][d] += 1             
      
      #Compare all neurons vs all neurons
      for neuronA in neurons:       
         for neuronB in neurons:          
            matching_indices = calc_matching_index ( neuronA, neuronB, connectivity[d], threshold, min_nodes = 500 )                
            matching_scores[d][neuronA][neuronB] = matching_indices['vertex_normalized']

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
               r_inputs = number_of_partners[neuronA]['incoming']/(number_of_partners[neuronA]['incoming']+number_of_partners[neuronA]['outgoing'])
               r_outputs = 1 - r_inputs
            except:
               module_logger.warning('Failed to calculate input/output ratio for %s assuming 50/50 (probably division by 0 error)' % str(neuronA) )
               r_inputs = 0.5
               r_outputs = 0.5

            dist_matrix[neuronA][neuronB] = matching_scores['incoming'][neuronA][neuronB] * r_inputs + matching_scores['outgoing'][neuronA][neuronB] * r_outputs

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


def calc_matching_index( neuronA, neuronB, connectivity, syn_threshold = 1, min_nodes = 1 ): 
   """ Calculates various matching indices between two neurons:
   
   matching_index =           number of shared partners divided by total number of partners
   
   matching_index_synapses =  number of shared synapses divided by total number of synapses 
                              Attention! matching_index_synapses is tricky, because if neuronA 
                              has lots of connections and neuronB only little, they will still 
                              get a high matching index. E.g. 100 of 200 / 1 of 50 = 101/250 
                              -> matching index = 0.404
   
   matching_index_weighted_synapses =  similar to matching_index_synapses but slightly less prone to
                              above mentioned error: 
                              % of shared synapses A * % of shared synapses B * 2 / (% of shared synapses A + % of shared synapses B)
                              -> value will be between 0 and 1; if one neuronB has only few connections (percentage) to a shared partner, the final value will also be small
   
   vertex_normalized =           matching index that rewards shared and punishes non-shared partners
                              Vertex similarity based on Jarrell et al., 2012: 
                              f(x,y) = min(x,y) - C1 * max(x,y) * e^(-C2 * min(x,y))
                              x,y = edge weights to compare
                              vertex_similarity is the sum of f over all vertices
                              C1 determines how negatively a case where one edge is much stronger than another is punished
                              C2 determines the point where the similarity switches from negative to positive

   Parameters:
   -----------
   neuronA :         skeleton ID
   neuronB :         skeleton ID
   connectivity :    connectivity data as provided by pymaid.get_partners()
   syn_threshold :   min number of synapses for a connection to be considered
   min_nodes :       min number of nodes for a partner to be considered
                     use this to filter fragments

   Returns:
   -------
   dict containing all initially described matching indices
   """   

   n_total = 0
   n_shared = 0
   n_synapses_shared = 0
   n_synapses_sharedA = 0
   n_synapses_sharedB = 0
   n_synapses_total = 0
   n_synapses_totalA = 0
   n_synapses_totalB = 0

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

   for entry in connectivity:          
      if connectivity[entry]['num_nodes'] < min_nodes:
         continue

      A_connected = False
      B_connected = False

      if neuronA in connectivity[entry]['skids'] and entry is not neuronB and entry is not neuronA:
         if connectivity[entry]['skids'][neuronA] >= syn_threshold:
            n_total += 1
            n_synapses_total += connectivity[entry]['skids'][neuronA]
            n_synapses_totalA += connectivity[entry]['skids'][neuronA]
            A_connected = True
      if neuronB in connectivity[entry]['skids'] and entry is not neuronA and entry is not neuronB:
         if connectivity[entry]['skids'][neuronB] >= syn_threshold:           
            B_connected = True
            n_synapses_total += connectivity[entry]['skids'][neuronB]
            n_synapses_totalB += connectivity[entry]['skids'][neuronB]
            #Make sure to not count this entry again if it is already connected to A
            if A_connected is False:
               n_total += 1

      if A_connected is True and B_connected is True:
         n_shared += 1
         n_synapses_shared += connectivity[entry]['skids'][neuronA] + connectivity[entry]['skids'][neuronB]
         n_synapses_sharedA += connectivity[entry]['skids'][neuronA]
         n_synapses_sharedB += connectivity[entry]['skids'][neuronB]

      if A_connected is True:
         a = connectivity[entry]['skids'][neuronA]
      else: 
         a = 0
      if B_connected is True:
         b = connectivity[entry]['skids'][neuronB]
      else: 
         b = 0

      #This keeps track of what the maximal vertex_index can be (for later normalisation)
      max_score += max([a,b])

      vertex_similarity += ( 
                        min([a,b]) - C1 * max([a,b]) * math.exp(- C2 * min([a,b]))
                           )

   similarity_indices = {}

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

def synapse_distance_matrix(synapse_data, labels = None):
   """ Takes a list of CATMAID synapses [[parent_node, connector_id, 0/1 , x, y, z ],[],...], 
      calculates euclidian distance matrix and clusters them (WARD)

   Parameters:
   ----------
   synapse_data :    list of synapses
                     As received from CATMAID 3D skeleton:
                     [ [parent_node, connector_id, 0/1 , x, y, z ], ... ]
   labels :          list of strings
                     Labels for each leaf of the dendrogram (e.g. connector ids).   

   Returns:
   -------
   Plots dendrogram and distance matrix
   Returns hierachical clustering
   """      

   #Generate numpy array containing x, y, z coordinates
   s = np.array( [ [ e[3],e[4],e[5] ] for e in synapse_data ] )

   #Calculate euclidean distance matrix 
   condensed_dist_mat = spatial.distance.pdist( s , 'euclidean' )
   squared_dist_mat = spatial.distance.squareform( condensed_dist_mat )

   # Compute and plot first dendrogram for all nodes.
   fig = pylab.figure(figsize=(8,8))
   ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
   Y = cluster.hierarchy.ward(squared_dist_mat)
   Z1 = cluster.hierarchy.dendrogram(Y, orientation='left', labels = labels)
   ax1.set_xticks([])
   ax1.set_yticks([])

   # Compute and plot second dendrogram.
   ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
   Y = cluster.hierarchy.ward(squared_dist_mat)
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
   fig.show()  

   return Y



