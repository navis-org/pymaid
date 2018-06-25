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

# TODOs
# - Github punch card-like figure

from pymaid import core, fetch, utils, config
import pandas as pd
import numpy as np
import datetime

# Set up logging
logger = config.logger

__all__ = ['get_user_contributions', 'get_time_invested', 'get_user_actions',
           'get_team_contributions']


def get_team_contributions(teams, neurons=None, remote_instance=None):
    """ Get contributions by teams.

    Notes
    -----
     1. Time calculation uses defaults from :func:`pymaid.get_time_invested`.
     2. Time does not take edits into account.
     3. ``total_reviews`` > ``total_nodes`` is possible if nodes have been
        reviewed multiple times by different users. Similarly,
        ``total_reviews`` = ``total_nodes`` does not imply that the neuron
        is fully reviewed!

    Parameters
    ----------
    teams               dict
                        Teams to group contributions for. Users must be logins.
                        Format can be either:

                          1. Simple user assignments. For example::

                              ``{'teamA': ['user1', 'user2'], 'team2': ['user3'], ...]}``

                          2. Users with start and end dates. Start and end date
                             must be either ``datetime.date`` or a single
                             ``pandas.date_range`` object. For example::

                              {
                               'team1': {
                                        'user1': (datetime.date(2017, 1, 1), datetime.date(2018, 1, 1)),
                                        'user2': (datetime.date(2016, 6, 1), datetime.date(2018, 1, 1)
                                        }
                               'team2': {
                                        'user3': pandas.date_range('2017-1-1', '2018-1-1'),
                                        }
                              }

                        Mixing both styles is permissible. For second style,
                        use e.g. ``'user1': None`` for no date restrictions
                        on that user.

    neuron              skeleton ID(s) | CatmaidNeuron/List, optional
                        Restrict check to given set of neurons. If
                        CatmaidNeuron/List, will use this neurons nodes/
                        connectors. You subset contributions e.g. to a given
                        neuropil by pruning neurons.
    remote_instance :   Catmaid Instance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron. Example for two teams,
        ``teamA`` and ``teamB``:

        >>> df
           skeleton_id  total_nodes  teamA_nodes  teamB_nodes  ...
        0
        1
           total_reviews teamA_reviews  teamB_reviews  ...
        0
        1
           total_connectors  teamA_connectors  teamB_connectors ...
        0
        1
           total_time  teamA_time  teamB_time
        0
        1

    Examples
    --------
    >>> from datetime import date
    >>> import pandas as pd
    >>> teams = {'teamA': ['user1', 'user2'],
    ...          'teamB': {'user3': None,
    ...                    'user4': (date(2017, 1, 1), date(2018, 1, 1))},
    ...          'teamC': {'user5': pd.date_range('2015-1-1', '2018-1-1')}}
    >>> stats = pymaid.get_team_contributions(teams)

    See Also
    --------
    :func:`~pymaid.get_contributor_statistics`
                           Gives you more basic info on neurons of interest
                           such as total reconstruction/review time.
    :func:`~pymaid.get_time_invested`
                           Time invested by individual users. Gives you more
                           control over how time is calculated.
    """

    remote_instance = utils._eval_remote_instance(remote_instance)

    # Prepare teams
    if not isinstance(teams, dict):
        raise TypeError('Expected teams of type dict, got {}'.format(type(teams)))

    beginning_of_time = datetime.date(1900, 1, 1)
    today = datetime.date.today()
    all_time = pd.date_range(beginning_of_time, today)

    for t in teams:
        if isinstance(teams[t], list):
            teams[t] = {u: all_time for u in teams[t]}
        elif isinstance(teams[t], dict):
            for u in teams[t]:
                if isinstance(teams[t][u], type(None)):
                    teams[t][u] = all_time
                elif isinstance(teams[t][u], (tuple, list)):
                    try:
                        teams[t][u] = pd.date_range(*teams[t][u])
                    except BaseException:
                        raise Exception('Error converting "{}" to pandas.date_range'.format(teams[t][u]))
                elif isinstance(teams[t][u],
                                pd.core.indexes.datetimes.DatetimeIndex):
                    pass
                else:
                    TypeError('Expected user dates to be either None, tuple of datetimes or pandas.date_range, got {}'.format(type(teams[t][u])))
        else:
            raise TypeError('Expected teams to be either lists or dicts of users, got {}'.format(type(teams[t])))

    # Get all users
    all_users = [u for t in teams for u in teams[t]]

    # Prepare neurons - download if neccessary
    if not isinstance(neurons, type(None)):
        if isinstance(neurons, core.CatmaidNeuron):
            neurons = core.CatmaidNeuronList(neurons)
        elif isinstance(neurons, core.CatmaidNeuronList):
            pass
        else:
            neurons = fetch.get_neurons(neurons)
    else:
        all_dates = [d.date() for t in teams for u in teams[t] for d in teams[t][u]]
        neurons = fetch.find_neurons(users=all_users,
                                     from_date=min(all_dates),
                                     to_date=max(all_dates))
        neurons.get_skeletons()

    # Get user list
    user_list = fetch.get_user_list(remote_instance).set_index('login')

    for u in all_users:
        if u not in user_list.index:
            raise ValueError('User "{}" not found in user list'.format(u))

    interval = 3
    bin_width = '%iMin' % interval
    minimum_actions = 10 * interval
    stats = []
    for n in config.tqdm(neurons, desc='Processing',
                         disable=config.pbar_hide, leave=config.pbar_leave):
        # Get node details
        tn_ids = n.nodes.treenode_id.values.astype(str)
        cn_ids = n.connectors.connector_id.values.astype(str)

        current_status = config.pbar_hide
        config.pbar_hide=True
        node_details = fetch.get_node_details(np.append(tn_ids, cn_ids),
                                              remote_instance=remote_instance)
        config.pbar_hide = current_status

        # Extract node creation
        node_creation = node_details.loc[node_details.treenode_id.isin(tn_ids),
                                         ['user', 'creation_time']].values
        node_creation = np.c_[node_creation, ['node_creation'] * node_creation.shape[0]]

        # Extract connector creation
        cn_creation = node_details.loc[node_details.treenode_id.isin(cn_ids),
                                         ['user', 'creation_time']].values
        cn_creation = np.c_[cn_creation, ['cn_creation'] * cn_creation.shape[0]]

        # Extract review times
        reviewers = [u for l in node_details.reviewers.tolist() for u in l]
        timestamps = [ts for l in node_details.review_times.tolist() for ts in l]
        node_review = np.c_[reviewers, timestamps, ['review'] * len(reviewers)]

        # Merge all timestamps (ignore edits for now) to get time_invested
        all_ts = pd.DataFrame(np.vstack([node_creation,
                                         node_review,
                                         cn_creation]),
                              columns=['user', 'timestamp', 'type'])

        # Add column with just the date and make it the index
        all_ts['date'] = [v.date() for v in all_ts.timestamp.astype(datetime.date).values]
        all_ts.index = pd.to_datetime(all_ts.date)

        # Fill in teams for each timestamp based on user + date
        all_ts['team'] = None
        for t in teams:
            for u in teams[t]:
                # Assign all timestamps by this user in the right time to
                # this team
                existing_dates = (teams[t][u] & all_ts.index).unique()
                ss = (all_ts.index.isin(existing_dates)) & (all_ts.user.values == user_list.loc[u, 'id'])
                all_ts.loc[ss, 'team'] = t

        # Get total
        total_time = sum(all_ts.timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.Grouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval

        this_neuron = [n.skeleton_id, n.n_nodes, n.n_connectors,
                       node_review.shape[0], total_time]
        # Go over the teams and collect values
        for t in teams:
            # Subset to team
            this_team = all_ts[all_ts.team == t]
            if this_team.shape[0] > 0:
                # Subset to user ID
                team_time = sum(this_team.timestamp.to_frame().set_index(
                    'timestamp', drop=False).groupby(pd.Grouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval
                team_nodes = this_team[this_team['type'] == 'node_creation'].shape[0]
                team_cn = this_team[this_team['type'] == 'cn_creation'].shape[0]
                team_rev = this_team[this_team['type'] == 'review'].shape[0]
            else:
                team_nodes = team_cn = team_rev = team_time = 0

            this_neuron += [team_nodes, team_cn, team_rev, team_time]

        stats.append(this_neuron)

    cols = ['skeleton_id', 'total_nodes', 'total_connectors',
            'total_reviews', 'total_time']

    for t in teams:
        for s in ['nodes', 'connectors', 'reviews', 'time']:
            cols += ['{}_{}'.format(t, s)]

    stats = pd.DataFrame(stats, columns=cols)

    cols_ordered = ['skeleton_id'] + ['{}_{}'.format(t, v) for v in
                    ['nodes', 'connectors', 'reviews', 'time']for t in ['total'] + list(teams)]
    stats = stats[cols_ordered]

    return stats


def get_user_contributions(x, teams=None, remote_instance=None):
    """ Takes a list of neurons and returns nodes and synapses contributed
    by each user.

    Notes
    -----
    This is essentially a wrapper for :func:`~pymaid.get_contributor_statistics`
    - if you are also interested in e.g. construction time, review time, etc.
    you may want to consider using :func:`~pymaid.get_contributor_statistics`
    instead.

    Parameters
    ----------
    x
                        Which neurons to check. Can be either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, must be exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object
    teams               dict, optional
                        Teams to group contributions for. Users must be logins::

                            {'teamA': ['user1', 'user2'], 'team2': ['user3'], ...]}

                        Users not part of any team, will be grouped as team
                        ``'others'``.

    remote_instance :   Catmaid Instance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a user

        >>> df
        ...   user  nodes  presynapses  postsynapses  nodes_reviewed
        ... 0
        ... 1

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Get contributors for a single neuron
    >>> cont = pymaid.get_user_contributions(2333007)
    >>> # Get top 10 (by node contribution)
    >>> top10 = cont.iloc[:10].set_index('user')
    >>> # Plot as bar chart
    >>> ax = top10.plot(kind='bar')
    >>> plt.show()

    >>> # Plot relative contributions
    >>> cont = pymaid.get_user_contributions(2333007)
    >>> cont = cont.set_index('user')
    >>> # Normalise
    >>> cont_rel = cont / cont.sum(axis=0).values
    >>> # Plot contributors with >5% node contributions
    >>> ax = cont_rel[cont_rel.nodes > .05].plot(kind='bar')
    >>> plt.show()

    See Also
    --------
    :func:`~pymaid.get_contributor_statistics`
                           Gives you more basic info on neurons of interest
                           such as total reconstruction/review time.

    """

    if not isinstance(teams, type(None)):
        # Prepare teams
        if not isinstance(teams, dict):
            raise TypeError('Expected teams of type dict, got {}'.format(type(teams)))

        for t in teams:
            if not isinstance(teams[t], list):
                raise TypeError('Teams need to list of user logins, got {}'.format(type(teams[t])))

        # Turn teams into a login -> team dict
        teams = {u : t for t in teams for u in teams[t]}

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance)

    cont = fetch.get_contributor_statistics(
        skids, remote_instance, separate=False)

    all_users = set(list(cont.node_contributors.keys(
    )) + list(cont.pre_contributors.keys()) + list(cont.post_contributors.keys()))

    stats = {
        'nodes': {u: 0 for u in all_users},
        'presynapses': {u: 0 for u in all_users},
        'postsynapses': {u: 0 for u in all_users},
        'nodes_reviewed': {u: 0 for u in all_users}
    }

    for u in cont.node_contributors:
        stats['nodes'][u] = cont.node_contributors[u]
    for u in cont.pre_contributors:
        stats['presynapses'][u] = cont.pre_contributors[u]
    for u in cont.post_contributors:
        stats['postsynapses'][u] = cont.post_contributors[u]
    for u in cont.review_contributors:
        stats['nodes_reviewed'][u] = cont.review_contributors[u]

    stats = pd.DataFrame([[u, stats['nodes'][u],
                             stats['presynapses'][u],
                             stats['postsynapses'][u],
                             stats['nodes_reviewed'][u]] for u in all_users],
                         columns=['user', 'nodes', 'presynapses',
                                  'postsynapses', 'nodes_reviewed']
                        ).sort_values('nodes', ascending=False).reset_index(drop=True)

    if isinstance(teams, type(None)):
        return stats

    stats['team'] = [teams.get(u, 'others') for u in stats.user.values]
    return stats.groupby('team').sum()


def get_time_invested(x, remote_instance=None, minimum_actions=10,
                      treenodes=True, connectors=True, mode='SUM',
                      max_inactive_time=3):
    """ Takes a list of neurons and calculates the time individual users
    have spent working on this set of neurons.

    Parameters
    ----------
    x
                        Which neurons to check. Can be either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, must be exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object

                        If you pass a CatmaidNeuron/List, its data is used
                        calculate time invested. You can exploit this to get
                        time invested into a given compartment of a neurons,
                        e.g. by pruning it to a volume.
    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.
    minimum_actions :   int, optional
                        Minimum number of actions per minute to be counted as
                        active.
    treenodes :         bool, optional
                        If False, treenodes will not be taken into account
    connectors :        bool, optional
                        If False, connectors will not be taken into account
    mode :              'SUM' | 'OVER_TIME' | 'ACTIONS', optional
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
        If ``mode='SUM'``, values represent minutes invested.

        >>> df
        ...       total  creation  edition  review
        ... user1
        ... user2

        If ``mode='OVER_TIME'`` or ``mode='ACTIONS'``:

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

    Please note that this does currently not take placement of
    pre-/postsynaptic nodes into account!

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

    def _extract_timestamps(ts, desc='Calc'):
        temp_stats = {}
        for u in config.tqdm( set(ts.user.unique()) & set(relevant_users), desc=desc, disable=config.pbar_hide, leave=False):
            temp_stats[u] = sum(ts[ts.user == u].timestamp.to_frame().set_index(
                'timestamp', drop=False).groupby(pd.Grouper(freq=bin_width)).count().values >= minimum_actions)[0] * interval
        return temp_stats

    if mode not in ['SUM', 'OVER_TIME', 'ACTIONS']:
        raise ValueError('Unknown mode "%s"' % str(mode))

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance)

    # Maximal inactive time is simply translated into binning
    # We need this later for pandas.TimeGrouper() anyway
    interval = max_inactive_time
    bin_width = '%iMin' % interval

    # Update minimum_actions to reflect actions/interval instead of actions/minute
    minimum_actions *= interval

    user_list = fetch.get_user_list(remote_instance).set_index('id')

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        x = fetch.get_neuron(skids, remote_instance=remote_instance)

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
    node_details = fetch.get_node_details(
        node_ids + connector_ids, remote_instance=remote_instance)

    # Get details for links
    link_details = fetch.get_connector_links(skdata)

    # Dataframe for creation (i.e. the actual generation of the nodes)
    creation_timestamps = np.append(node_details[['user','creation_time']].values,
                                    link_details[['creator_id','creation_time']].values,
                                    axis=0)
    creation_timestamps = pd.DataFrame(creation_timestamps,
                                       columns=['user', 'timestamp'])

    # Dataframe for edition times
    edition_timestamps = np.append(node_details[['user','edition_time']].values,
                                    link_details[['creator_id','edition_time']].values,
                                    axis=0)
    edition_timestamps = pd.DataFrame(edition_timestamps,
                                       columns=['user', 'timestamp'])

    # Generate dataframe for reviews
    reviewers = [u for l in node_details.reviewers.tolist() for u in l]
    timestamps = [ts for l in node_details.review_times.tolist() for ts in l]
    review_timestamps = pd.DataFrame([[u, ts] for u, ts in zip(
        reviewers, timestamps)], columns=['user', 'timestamp'])

    # Merge all timestamps
    all_timestamps = pd.concat(
        [creation_timestamps, edition_timestamps, review_timestamps], axis=0)

    all_timestamps.sort_values('timestamp', inplace=True)

    relevant_users = all_timestamps.groupby('user').count()
    relevant_users = relevant_users[ relevant_users.timestamp >= minimum_actions ].index.values

    if mode == 'SUM':
        stats = {
            'total': {u: 0 for u in relevant_users},
            'creation': {u: 0 for u in relevant_users},
            'edition': {u: 0 for u in relevant_users},
            'review': {u: 0 for u in relevant_users}
        }
        stats['total'].update(_extract_timestamps(all_timestamps, desc='Calc total'))
        stats['creation'].update(_extract_timestamps(creation_timestamps, desc='Calc creation'))
        stats['edition'].update(_extract_timestamps(edition_timestamps, desc='Calc edition'))
        stats['review'].update(_extract_timestamps(review_timestamps, desc='Calc review'))

        return pd.DataFrame([[user_list.loc[u, 'login'],
                              stats['total'][u],
                              stats['creation'][u],
                              stats['edition'][u],
                              stats['review'][u]] for u in relevant_users],
                            columns=['user', 'total', 'creation', 'edition', 'review']).sort_values('total', ascending=False).reset_index(drop=True).set_index('user')

    elif mode == 'ACTIONS':
        all_ts = all_timestamps.set_index('timestamp', drop=False).timestamp.groupby(
            pd.Grouper(freq='1d')).count().to_frame()
        all_ts.columns = ['all_users']
        all_ts = all_ts.T
        # Get total time spent
        for u in config.tqdm(all_timestamps.user.unique(), desc='Calc. total', disable=config.pbar_hide, leave=False):
            this_ts = all_timestamps[all_timestamps.user == u].set_index(
                'timestamp', drop=False).timestamp.groupby(pd.Grouper(freq='1d')).count().to_frame()
            this_ts.columns = [user_list.loc[u, 'login']]

            all_ts = pd.concat([all_ts, this_ts.T])

        return all_ts.fillna(0)

    elif mode == 'OVER_TIME':
        # First count all minutes with minimum number of actions
        minutes_counting = (all_timestamps.set_index('timestamp', drop=False).timestamp.groupby(
            pd.Grouper(freq=bin_width)).count().to_frame() > minimum_actions)
        # Then remove the minutes that have less
        minutes_counting = minutes_counting[minutes_counting.timestamp == True]
        # Now group by hour
        all_ts = minutes_counting.groupby(pd.Grouper(freq='1d')).count()
        all_ts.columns = ['all_users']
        all_ts = all_ts.T
        # Get total time spent
        for u in config.tqdm(all_timestamps.user.unique(), desc='Calc. total', disable=config.pbar_hide, leave=False):
            minutes_counting = (all_timestamps[all_timestamps.user == u].set_index(
                'timestamp', drop=False).timestamp.groupby(pd.Grouper(freq=bin_width)).count().to_frame() > minimum_actions)
            minutes_counting = minutes_counting[minutes_counting.timestamp == True]
            this_ts = minutes_counting.groupby(pd.Grouper(freq='1d')).count()

            this_ts.columns = [user_list.loc[u, 'login']]

            all_ts = pd.concat([all_ts, this_ts.T])

        all_ts.fillna(0, inplace=True)

        return all_ts


def get_user_actions(users=None, neurons=None, start_date=None, end_date=None,
                     remote_instance=None):
    """ Get timestamps of users' actions (creations, editions, reviews).

    Important
    ---------
    This function returns most but not all user actions::

      1. The API endpoint used for finding neurons worked on by a given user
         (:func:`pymaid.find_neurons`) does not return single-node neurons.
         Hence, placing e.g. postsynaptic nodes is not taken into account.
      2. Connecting a node to a connector is not taken into account as there
         is no API endpoint for getting timestamps of the creation of
         connector links.

    Parameters
    ----------
    users :           str | list, optional
                      Users login(s) for which to return timestamps.
    neurons :         list of skeleton IDs | CatmaidNeuron/List, optional
                      Neurons for which to return timestamps. If None, will
                      find neurons by user.
    start_date :      tuple | datetime.date, optional
    end_date :        tuple | datetime.date, optional
                      Start and end date of time window to check.
    remote_instance : CatmaidInstance, optional

    Return
    ------
    pandas.DataFrame
            >>> df
                user      timestamp     action
            0
            1

    Examples
    --------
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> # Get all actions for a single user
    >>> actions = pymaid.get_user_actions(users='schlegelp', start_date=(2017,11,1))
    >>> # Group by hour and see what time of the day user is usually active
    >>> actions.set_index(pd.DatetimeIndex(actions.timestamp), inplace=True )
    >>> hours = actions.groupby(actions.index.hour).count()
    >>> hours.action.plot()
    >>> plt.show()

    >>> # Plot day-by-day activity
    >>> ax = plt.subplot()
    >>> ax.scatter(actions.timestamp.date.tolist(),
    ...            actions.timestamp.time.tolist(),
    ...            marker='_')

    """

    if not neurons and not users and not (start_date or end_date):
        raise ValueError(
            'Query must be restricted by at least a single parameter!')

    if users and not isinstance(users, (list, np.ndarray)):
        users = [users]

    # Get user dictionary (needed later)
    user_list = fetch.get_user_list(
        remote_instance=remote_instance).set_index('id')
    user_dict = user_list.login.to_dict()

    if isinstance(neurons, type(None)):
        neurons = fetch.find_neurons(users=users,
                                     from_date=start_date, to_date=end_date,
                                     reviewed_by=users,
                                     remote_instance=remote_instance)
        # Get skeletons
        neurons.get_skeletons()
    elif not isinstance(neurons, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        neurons = fetch.get_neuron(neurons)

    if not isinstance(end_date, (datetime.date, type(None))):
        end_date = datetime.date(*end_date)

    if not isinstance(start_date, (datetime.date, type(None))):
        start_date = datetime.date(*start_date)

    node_ids = neurons.nodes.treenode_id.tolist()
    connector_ids = neurons.connectors.connector_id.tolist()

    # Get node details
    node_details = fetch.get_node_details(
        node_ids + connector_ids, remote_instance=remote_instance)

    # Get details for links
    link_details = fetch.get_connector_links(neurons)

    # Dataframe for creation (i.e. the actual generation of the nodes)
    creation_timestamps = node_details[['user', 'creation_time']]
    creation_timestamps.loc[:, 'action'] = 'creation'
    creation_timestamps.columns = ['user', 'timestamp', 'action']

    # Dataframe for edition times
    edition_timestamps = node_details[['editor', 'edition_time']]
    edition_timestamps.loc[:, 'action'] = 'edition'
    edition_timestamps.columns = ['user', 'timestamp', 'action']

    # DataFrame for linking
    linking_timestamps =  link_details[['creator_id','creation_time']]
    linking_timestamps.loc[:, 'action'] = 'linking'
    linking_timestamps.columns = ['user', 'timestamp', 'action']

    # Generate dataframe for reviews
    reviewers = [u for l in node_details.reviewers.tolist() for u in l]
    timestamps = [ts for l in node_details.review_times.tolist() for ts in l]
    review_timestamps = pd.DataFrame([[u, ts, 'review'] for u, ts in zip(
        reviewers, timestamps)], columns=['user', 'timestamp', 'action'])

    # Merge all timestamps
    all_timestamps = pd.concat([creation_timestamps,
                                edition_timestamps,
                                review_timestamps,
                                linking_timestamps],
                                axis=0).reset_index(drop=True)

    # Map login onto user ID
    all_timestamps.user = [user_dict[u] for u in all_timestamps.user.values]

    # Remove other users
    all_timestamps = all_timestamps[all_timestamps.user.isin(users)]

    # Remove timestamps outside of date range (if provided)
    if start_date:
        all_timestamps = all_timestamps[all_timestamps.timestamp.values >= np.datetime64(
            start_date)]
    if end_date:
        all_timestamps = all_timestamps[all_timestamps.timestamp.values <= np.datetime64(
            end_date)]

    return all_timestamps.sort_values('timestamp').reset_index(drop=True)
