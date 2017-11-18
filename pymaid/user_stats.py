#    This script is part of pymaid (http://www.github.com/schlegelp/pymaid).
#    Copyright (C) 2017 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along

""" This module contains functions to retrieve user statistics for sets of 
neurons.

Examples
--------
>>> import pymaid
>>> myInstance = pymaid.CatmaidInstance( 'www.your.catmaid-server.org' , 
...                                      'HTTP_USER' , 
...                                      'HTTP_PASSWORD', 
...                                      'TOKEN' )
>>> skeleton_ids = pymaid.get_skids_by_annotation('Hugin')
>>> cont = pymaid.get_user_contributions( skeleton_ids )
>>> cont
             user  nodes  presynapses  postsynapses
0        Schlegel  47221          470          1408
1            Tran   1645            7             4
2           Lacin   1300            1            20
3              Li   1244            5            45
...
>>> # Get the time that each user has invested
>>> time_inv = pymaid.get_time_invested(  skeleton_ids,
...                                       remote_instance = myInstance )
>>> time_inv
            user  total  creation  edition  review
0       Schlegel   4649      3224     2151    1204
1           Tran    174       125       59       0
2             Li    150       114       65       0
3          Lacin    133       119       30       0
...
>>> # Plot contributions as pie chart
>>> import plotly
>>> fig = { "data" : [ { "values" : time_inv.total.tolist(), 
...         "labels" : time_inv.user.tolist(), 
...         "type" : "pie" } ] } 
>>> plotly.offline.plot(fig) 
"""

from pymaid.pymaid import get_neuron, get_user_list, get_node_user_details, get_contributor_statistics, eval_skids, _eval_remote_instance
from pymaid import core
import logging
import pandas as pd
from tqdm import tqdm
import sys

# Set up logging
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

if len( module_logger.handlers ) == 0:
    # Generate stream handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
                '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    module_logger.addHandler(sh)

__all__ = ['get_user_contributions','get_time_invested',]

def get_user_contributions(x, remote_instance=None):
    """ Takes a list of skeleton IDs and returns nodes and synapses contributed 
    by each user.

    Notes
    -----
    This is essentially a wrapper for pymaid.get_contributor_statistics() - if 
    you are also interested in e.g. construction time, review time, etc. you 
    may want to consider using :func:`pymaid.pymaid.get_contributor_statistics` 
    instead.

    Parameters
    ----------
    x
                        Which neurons to check. Can be either:
                            1. skeleton IDs (int or str)
                            2. neuron name (str, must be exact match)
                            3. annotation: e.g. 'annotation:PN right'
                            4. CatmaidNeuron or CatmaidNeuronList object
    remote_instance :   Catmaid Instance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a user

        >>> df
        ...   user nodes presynapses  postsynapses
        ... 0
        ... 1

    See Also
    --------
    :func:`pymaid.pymaid.get_contributor_statistics`
                           Gives you more basic info on neurons of interest 
                           such as total reconstruction/review time.
    """

    remote_instance = _eval_remote_instance(remote_instance)

    skids = eval_skids(x, remote_instance)

    user_list = get_user_list(remote_instance).set_index('id')
    
    cont = get_contributor_statistics(
        skids, remote_instance, separate=False).ix[0]

    all_users = set(list(cont.node_contributors.keys(
    )) + list(cont.pre_contributors.keys()) + list(cont.post_contributors.keys()))

    stats = {
        'nodes': {u: 0 for u in all_users},
        'presynapses': {u: 0 for u in all_users},
        'postsynapses': {u: 0 for u in all_users}
    }

    for u in cont.node_contributors:
        stats['nodes'][u] = cont.node_contributors[u]
    for u in cont.pre_contributors:
        stats['presynapses'][u] = cont.pre_contributors[u]
    for u in cont.post_contributors:
        stats['postsynapses'][u] = cont.post_contributors[u]    

    return pd.DataFrame([[user_list.ix[int(u)].last_name, stats['nodes'][u], stats['presynapses'][u], stats['postsynapses'][u]] for u in all_users], columns=['user', 'nodes', 'presynapses', 'postsynapses']).sort_values('nodes', ascending=False).reset_index(drop=True)


def get_time_invested(x, remote_instance=None, minimum_actions=10, treenodes=True, connectors=True, mode='SUM', max_inactive_time=3):
    """ Takes a list of skeleton IDs and calculates the time individual users 
    have spent working on this set of neurons.

    Parameters
    ----------
    x
                        Which neurons to check. Can be either:
                            1. skeleton IDs (int or str)
                            2. neuron name (str, must be exact match)
                            3. annotation: e.g. 'annotation:PN right'
                            4. CatmaidNeuron or CatmaidNeuronList object
                        If you pass skeleton data (i.e. CatmaidNeurons), this
                        data is used to calculate time invested. You can 
                        exploit this to get time invested into a given 
                        compartment of a neurons, e.g. by pruning it to a 
                        corresponding volume.
    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.
    minimum_actions :   int, optional
                        Minimum number of actions per minute to be counted as 
                        active.
    treenodes :         bool, optional
                        If False, treenodes will not be taken into account
    connectors :        bool, optional
                        If False, connectors will not be taken into account
    mode :              {'SUM','OVER_TIME','ACTIONS'}, optional
                        (1) 'SUM' will return total time invested (in minutes) 
                            per user. 
                        (2) 'OVER_TIME' will return minutes invested/day over 
                            time. 
                        (3) 'ACTIONS' will return actions 
                            (node/connectors placed/edited) per day.
    max_inactive_time : int, optional
                        Maximal time inactive in minutes. 

    Returns
    -------
    pandas.DataFrame
        If mode=='SUM', values represent minutes invested.

        >>> df
        ...       total  creation  edition  review
        ... user1
        ... user2

        If mode=='OVER_TIME' or mode=='ACTIONS': 

        >>> df
        ...       date date date ...
        ... user1
        ... user2

        For `OVER_TIME`, values respresent minutes invested on that day. For 
        `ACTIONS`, values represent actions (creation, edition, review) on that 
        day.


    Important
    ---------
    Creation/Edition/Review times can overlap! This is why total time spent 
    is not just creation + edition + review.

    Please note that this does currently not take placement of postsynaptic 
    nodes or creation of connector links into account!

    Be aware of the ``minimum_actions`` parameter: at low settings even 
    a single actions (e.g. connecting a node) will add considerably to time 
    invested. To keep total reconstruction time comparable to what Catmaid
    calculates, you should consider about 10 actions/minute (= a click every
    6 seconds) and ``max_inactive_time`` of 3 mins.

    CATMAID gives reconstruction time across all users. Here, we calculate
    the time spent tracing for individuals. This may lead to a discrepancy
    between sum of time invested over of all users from this function vs.
    CATMAID's reconstruction time.

    Examples
    --------
    Plot pie chart of contributions per user using Plotly. This example 
    assumes that you have already imported and set up pymaid.

    >>> import plotly    
    >>> stats = pymaid.get_time_invested( skids, remote_instance )
    >>> # Use plotly to generate pie chart
    >>> fig = { "data" : [ { "values" : stats.total.tolist(), 
    ...         "labels" : stats.user.tolist(), "type" : "pie" } ] } 
    >>> plotly.offline.plot(fig)    

    Plot reconstruction efforts over time

    >>> stats = pymaid.get_time_invested( skids, mode='OVER_TIME' )    
    >>> # Plot time invested over time
    >>> stats.T.plot()
    >>> # Plot cumulative time invested over time
    >>> stats.T.cumsum(axis=0).plot()
    >>> # Filter for major contributors
    >>> stats[ stats.sum(axis=1) > 20 ].T.cumsum(axis=0).plot()

    """

    if mode not in ['SUM','OVER_TIME','ACTIONS']:
        raise ValueError('Unknown mode %s' % str(mode))    

    remote_instance = _eval_remote_instance(remote_instance)

    skids = eval_skids(x, remote_instance)

    # Maximal inactive time is simply translated into binning
    # We need this later for pandas.TimeGrouper() anyway
    interval = max_inactive_time
    bin_width = '%iMin' % interval

    # Update minimum_actions to reflect actions/interval instead of actions/minute
    minimum_actions *= interval

    user_list = get_user_list(remote_instance).set_index('id')

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        x = get_neuron(skids, remote_instance=remote_instance)

    if isinstance(x, core.CatmaidNeuron):
        skdata = core.CatmaidNeuronList(x)
    elif isinstance(x, core.CatmaidNeuronList):
        skdata = x

    # Extract connector and node IDs
    node_ids = []
    connector_ids = []
    for n in skdata.itertuples():
        if treenodes:
            node_ids += n.nodes.treenode_id.tolist()
        if connectors:
            connector_ids += n.connectors.connector_id.tolist()

    # Get node details
    node_details = get_node_user_details( 
        node_ids + connector_ids, remote_instance=remote_instance )

    # Dataframe for creation (i.e. the actual generation of the nodes)
    creation_timestamps = pd.DataFrame( 
        node_details[['user', 'creation_time']].values, columns=['user', 'timestamp'])

    # Dataframe for edition times 
    edition_timestamps = pd.DataFrame( 
        node_details[['editor', 'edition_time']].values, columns=['user', 'timestamp'])

    # Generate dataframe for reviews
    reviewers = [u for l in node_details.reviewers.tolist() for u in l]
    timestamps = [ts for l in node_details.review_times.tolist() for ts in l]
    review_timestamps = pd.DataFrame([[u, ts] for u, ts in zip(
        reviewers, timestamps)], columns=['user', 'timestamp'])

    # Merge all timestamps
    all_timestamps = pd.concat(
        [creation_timestamps, edition_timestamps, review_timestamps], axis=0)    

    all_timestamps.sort_values('timestamp', inplace=True)

    if mode == 'SUM':
        stats = {
            'total': {u: 0 for u in all_timestamps.user.unique()},
            'creation': {u: 0 for u in all_timestamps.user.unique()},
            'edition': {u: 0 for u in all_timestamps.user.unique()},
            'review': {u: 0 for u in all_timestamps.user.unique()}
        }

        # Get total time spent
        for u in tqdm(all_timestamps.user.unique(), desc='Calc. total', disable=module_logger.getEffectiveLevel()>=40):
            stats['total'][u] += sum(all_timestamps[all_timestamps.user == u].timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.TimeGrouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval
        # Get reconstruction time spent
        for u in tqdm(creation_timestamps.user.unique(), desc='Calc. reconst.', disable=module_logger.getEffectiveLevel()>=40):
            stats['creation'][u] += sum(creation_timestamps[creation_timestamps.user == u].timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.TimeGrouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval
        # Get edition time spent
        for u in tqdm(edition_timestamps.user.unique(), desc='Calc. edition', disable=module_logger.getEffectiveLevel()>=40):
            stats['edition'][u] += sum(edition_timestamps[edition_timestamps.user == u].timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.TimeGrouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval
        # Get time spent reviewing
        for u in tqdm(review_timestamps.user.unique(), desc='Calc. review', disable=module_logger.getEffectiveLevel()>=40):
            stats['review'][u] += sum(review_timestamps[review_timestamps.user == u].timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.TimeGrouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval

        module_logger.info(
            'Done! Use e.g. plotly to generate a plot: \n stats = get_time_invested( skids, remote_instance ) \n fig = { "data" : [ { "values" : stats.total.tolist(), "labels" : stats.user.tolist(), "type" : "pie" } ] } \n plotly.offline.plot(fig) ')

        return pd.DataFrame([[user_list.ix[u].login, stats['total'][u], stats['creation'][u], stats['edition'][u], stats['review'][u]] for u in all_timestamps.user.unique()], columns=['user', 'total', 'creation', 'edition', 'review']).sort_values('total', ascending=False).reset_index(drop=True).set_index('user')

    elif mode == 'ACTIONS':
        all_ts = all_timestamps.set_index('timestamp', drop=False).timestamp.groupby(pd.TimeGrouper(freq='1d')).count().to_frame()
        all_ts.columns = ['all_users'] 
        all_ts = all_ts.T
        # Get total time spent
        for u in tqdm(all_timestamps.user.unique(), desc='Calc. total', disable=module_logger.getEffectiveLevel()>=40):
            this_ts = all_timestamps[all_timestamps.user==u].set_index('timestamp', drop=False).timestamp.groupby(pd.TimeGrouper(freq='1d')).count().to_frame()
            this_ts.columns=[ user_list.ix[u].login ]

            all_ts = pd.concat( [all_ts, this_ts.T] )

        return all_ts.fillna(0)

    elif mode == 'OVER_TIME':
        #First count all minutes with minimum number of actions
        minutes_counting = ( all_timestamps.set_index('timestamp', drop=False).timestamp.groupby(pd.TimeGrouper(freq=bin_width)).count().to_frame() > minimum_actions )
        #Then remove the minutes that have less
        minutes_counting = minutes_counting[ minutes_counting.timestamp == True ]
        #Now group by hour
        all_ts = minutes_counting.groupby(pd.TimeGrouper(freq='1d')).count()       
        all_ts.columns = ['all_users']
        all_ts = all_ts.T
        # Get total time spent        
        for u in tqdm(all_timestamps.user.unique(), desc='Calc. total', disable=module_logger.getEffectiveLevel()>=40):            
            minutes_counting = ( all_timestamps[all_timestamps.user==u].set_index('timestamp', drop=False).timestamp.groupby(pd.TimeGrouper(freq=bin_width)).count().to_frame() > minimum_actions )
            minutes_counting = minutes_counting[ minutes_counting.timestamp == True ]
            this_ts = minutes_counting.groupby(pd.TimeGrouper(freq='1d')).count()

            this_ts.columns=[ user_list.ix[u].login ]

            all_ts = pd.concat( [all_ts, this_ts.T] )

        all_ts.fillna(0, inplace=True)

        return all_ts



