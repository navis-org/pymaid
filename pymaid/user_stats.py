#    This script is part of pymaid (http://www.github.com/navis-org/pymaid).
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

"""This module contains functions to retrieve user statistics.

Examples
--------
>>> import pymaid
>>> myInstance = pymaid.CatmaidInstance('https://www.your.catmaid-server.org',
...                                     api_token='YOURTOKEN',
...                                     http_user='HTTP_PASSWORD', # omit if not required
...                                     http_password='TOKEN')
>>> skeleton_ids = pymaid.get_skids_by_annotation('Hugin')
>>> cont = pymaid.get_user_contributions(skeleton_ids)
>>> cont
             user  nodes  presynapses  postsynapses
0        Schlegel  47221          470          1408
1            Tran   1645            7             4
2           Lacin   1300            1            20
3              Li   1244            5            45
...
>>> # Get the time that each user has invested
>>> time_inv = pymaid.get_time_invested(skeleton_ids,
...                                     remote_instance = myInstance)
>>> time_inv
            user  total  creation  edition  review
0       Schlegel   4649      3224     2151    1204
1           Tran    174       125       59       0
2             Li    150       114       65       0
3          Lacin    133       119       30       0
...
>>> # Plot contributions as pie chart
>>> import plotly
>>> fig = {"data": [{"values": time_inv.total.tolist(),
...        "labels": time_inv.user.tolist(),
...        "type": "pie"}]}
>>> plotly.offline.plot(fig)

"""

# TODOs
# - Github punch card-like figure

import datetime

import pandas as pd
import numpy as np

from . import core, fetch, utils, config

# Set up logging
logger = config.logger

__all__ = ['get_user_contributions', 'get_time_invested', 'get_user_actions',
           'get_team_contributions', 'get_user_stats']


def get_user_stats(start_date=None, end_date=None, remote_instance=None):
    """Get user stats similar to the pie chart statistics widget in CATMAID.

    Returns cable [nm], nodes created/reviewed and connector links created.

    Parameters
    ----------
    start_date :        tuple | datetime.date, optional
    end_date :          tuple | datetime.date, optional
                        Start and end date of time window to check. If
                        ``None``, will use entire project history.
    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
                Dataframe in which each row represents a user::

                          cable  nodes_created  nodes_reviewed  links_created
                 username
                    user1  ...
                    user2  ...

    Examples
    --------
    Create a pie chart similar to the stats widget in CATMAID:

    >>> import matplotlib.pyplot as plt
    >>> stats = pymaid.get_user_stats()
    >>> stats_to_plot = ['cable', 'nodes_created', 'nodes_reviewed',
    ...                  'links_created']
    >>> fig, axes = plt.subplots(1, len(stats_to_plot), figsize=(12, 4))
    >>> for s, ax in zip(stats_to_plot, axes):
    ...     # Get the top 10 contributors for this stat
    ...     this_stats = stats[s].sort_values(ascending=False).iloc[:10]
    ...     # Calculate "others"
    ...     this_stats.loc['others'] = stats[s].sort_values(ascending=False).iloc[10:].sum()
    ...     # Plot
    ...     this_stats.plot.pie(ax=ax, textprops={'size': 6},
    ...                         explode=[.05] * this_stats.shape[0],
    ...                         rotatelabels=True)
    ...     # Make labels a bit smaller
    ...     ax.set_ylabel(s.replace('_', ' '), fontsize=8)
    >>> plt.show()

    See Also
    --------
    :func:`~pymaid.get_history`
            Returns day-by-day stats.

    """
    remote_instance = utils._eval_remote_instance(remote_instance)

    if isinstance(start_date, type(None)):
        start_date = datetime.date(2010, 1, 1)
    elif not isinstance(start_date, datetime.date):
        start_date = datetime.date(*start_date)

    if isinstance(end_date, type(None)):
        end_date = datetime.date.today()
    elif not isinstance(end_date, datetime.date):
        end_date = datetime.date(*end_date)

    # Get and summarize other stats
    hist = fetch.get_history(remote_instance=remote_instance,
                             start_date=start_date,
                             end_date=end_date)

    stats = pd.concat([hist.cable.sum(axis=1),
                       hist.treenodes.sum(axis=1),
                       hist.reviewed.sum(axis=1),
                       hist.connector_links.sum(axis=1)],
                      axis=1, sort=True).fillna(0).astype(int)

    stats.index.name = 'username'
    stats.columns = ['cable', 'nodes_created', 'nodes_reviewed',
                     'links_created']

    stats.sort_values('nodes_created', ascending=False, inplace=True)

    return stats


def get_team_contributions(teams, neurons=None, remote_instance=None):
    """Get contributions by teams (nodes, reviews, connectors, time invested).

    Notes
    -----
     1. Time calculation uses defaults from :func:`pymaid.get_time_invested`.
     2. ``total_reviews`` > ``total_nodes`` is possible if nodes have been
        reviewed multiple times by different users. Similarly,
        ``total_reviews`` = ``total_nodes`` does not imply that the neuron
        is fully reviewed!

    Parameters
    ----------
    teams               dict
                        Teams to group contributions for. Users must be logins.
                        Format can be either:

                          1. Simple user assignments. For example::

                              {'teamA': ['user1', 'user2'],
                               'team2': ['user3'], ...]}

                          2. Users with start and end dates. Start and end date
                             must be either ``datetime.date`` or a single
                             ``pandas.date_range`` object. For example::

                               {'team1': {
                                        'user1': (datetime.date(2017, 1, 1),
                                                  datetime.date(2018, 1, 1)),
                                        'user2': (datetime.date(2016, 6, 1),
                                                  datetime.date(2017, 1, 1)
                                        }
                                'team2': {
                                        'user3': pandas.date_range('2017-1-1',
                                                                   '2018-1-1'),
                                        }}

                        Mixing both styles is permissible. For second style,
                        use e.g. ``'user1': None`` for no date restrictions
                        on that user.

    neurons             skeleton ID(s) | CatmaidNeuron/List, optional
                        Restrict check to given set of neurons. If
                        CatmaidNeuron/List, will use this neurons nodes/
                        connectors. Use to subset contributions e.g. to a given
                        neuropil by pruning neurons before passing to this
                        function.
    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a neuron. Example for two teams,
        ``teamA`` and ``teamB``::

           skeleton_id  total_nodes  teamA_nodes  teamB_nodes  ...
         0
         1
           total_reviews  teamA_reviews  teamB_reviews  ...
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
        raise TypeError('Expected teams of type dict, got '
                        '{}'.format(type(teams)))

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
                        raise Exception('Error converting "{}" to pandas.'
                                        'date_range'.format(teams[t][u]))
                elif isinstance(teams[t][u],
                                pd.core.indexes.datetimes.DatetimeIndex):
                    pass
                else:
                    TypeError('Expected user dates to be either None, tuple '
                              'of datetimes or pandas.date_range, '
                              'got {}'.format(type(teams[t][u])))
        else:
            raise TypeError('Expected teams to be either lists or dicts of '
                            'users, got {}'.format(type(teams[t])))

    # Get all users
    all_users = [u for t in teams for u in teams[t]]

    # Prepare neurons - download if neccessary
    if not isinstance(neurons, type(None)):
        if isinstance(neurons, core.CatmaidNeuron):
            neurons = core.CatmaidNeuronList(neurons)
        elif isinstance(neurons, core.CatmaidNeuronList):
            pass
        else:
            neurons = fetch.get_neurons(neurons,
                                        remote_instance=remote_instance)
    else:
        all_dates = [d.date() for t in teams for u in teams[t] for d in teams[t][u]]
        neurons = fetch.find_neurons(users=all_users,
                                     from_date=min(all_dates),
                                     to_date=max(all_dates),
                                     remote_instance=remote_instance)
        neurons.get_skeletons()

    # Get user list
    user_list = fetch.get_user_list(remote_instance=remote_instance).set_index('login')

    for u in all_users:
        if u not in user_list.index:
            raise ValueError('User "{}" not found in user list'.format(u))

    # Get all node details
    all_node_details = fetch.get_node_details(neurons,
                                              remote_instance=remote_instance)

    # Get connector links
    link_details = fetch.get_connector_links(neurons, remote_instance=remote_instance)

    # link_details contains all links. We have to subset this to existing
    # connectors in case the input neurons have been pruned
    link_details = link_details[link_details.connector_id.isin(neurons.connectors.connector_id.values)]

    interval = 3
    bin_width = '%iMin' % interval
    minimum_actions = 10 * interval
    stats = []
    for n in config.tqdm(neurons, desc='Processing',
                         disable=config.pbar_hide, leave=config.pbar_leave):
        # Get node details
        tn_ids = n.nodes.node_id.values.astype(str)
        cn_ids = n.connectors.connector_id.values.astype(str)

        current_status = config.pbar_hide
        config.pbar_hide = True
        node_details = all_node_details[all_node_details.node_id.isin(np.append(tn_ids, cn_ids))]
        config.pbar_hide = current_status

        # Extract node creation
        node_creation = node_details.loc[node_details.node_id.isin(tn_ids),
                                         ['creator', 'creation_time']].values
        node_creation = np.c_[node_creation, ['node_creation'] * node_creation.shape[0]]

        # Extract connector creation
        cn_creation = node_details.loc[node_details.node_id.isin(cn_ids),
                                       ['creator', 'creation_time']].values
        cn_creation = np.c_[cn_creation, ['cn_creation'] * cn_creation.shape[0]]

        # Extract edition times (treenodes + connectors)
        node_edits = node_details.loc[:, ['editor', 'edition_time']].values
        node_edits = np.c_[node_edits, ['editor'] * node_edits.shape[0]]

        # Link creation
        link_creation = link_details.loc[link_details.connector_id.isin(cn_ids),
                                         ['creator', 'creation_time']].values
        link_creation = np.c_[link_creation, ['link_creation'] * link_creation.shape[0]]

        # Extract review times
        reviewers = [u for l in node_details.reviewers.values for u in l]
        timestamps = [ts for l in node_details.review_times.values for ts in l]
        node_review = np.c_[reviewers, timestamps, ['review'] * len(reviewers)]

        # Merge all timestamps (ignore edits for now) to get time_invested
        all_ts = pd.DataFrame(np.vstack([node_creation,
                                         node_review,
                                         cn_creation,
                                         link_creation,
                                         node_edits]),
                              columns=['user', 'timestamp', 'type'])

        return all_ts

        # Add column with just the date and make it the index
        all_ts['date'] = all_ts.timestamp.values.astype('datetime64[D]')
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
                                      ['nodes', 'connectors',
                                       'reviews', 'time'] for t in ['total'] + list(teams)]
    stats = stats[cols_ordered]

    return stats


def get_user_contributions(x, teams=None, remote_instance=None):
    """Return number of nodes and synapses contributed by each user.

    This is essentially a wrapper for :func:`pymaid.get_contributor_statistics`
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

    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        DataFrame in which each row represents a user::

            user  nodes  presynapses  postsynapses  nodes_reviewed
         0
         1
         ...

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
    >>> # Normalize
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
            raise TypeError('Expected teams of type dict, got '
                            '{}'.format(type(teams)))

        for t in teams:
            if not isinstance(teams[t], list):
                raise TypeError('Teams need to list of user logins, '
                                'got {}'.format(type(teams[t])))

        # Turn teams into a login -> team dict
        teams = {u: t for t in teams for u in teams[t]}

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    cont = fetch.get_contributor_statistics(skids,
                                            remote_instance=remote_instance,
                                            separate=False)

    all_users = set(list(cont.node_contributors.keys()) + list(cont.pre_contributors.keys()) + list(cont.post_contributors.keys()))

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


def get_time_invested(x, mode='SUM', by='USER', minimum_actions=10,
                      max_inactive_time=3, treenodes=True, connectors=True,
                      links=True, start_date=None, end_date=None,
                      remote_instance=None):
    """Calculate the time spent working on a set of neurons.

    Use ``minimum_actions`` and ``max_inactive_time`` to fine tune how time
    invested is calculated: by default, time is binned over 3 minutes in
    which a user has to perform 3x10 actions for that interval to be
    counted towards the time spent tracing.

    Important
    ---------
    Creation/Edition/Review times can overlap! This is why total time spent
    is not just creation + edition + review.

    Please note that this does currently not take placement of
    pre-/postsynaptic nodes into account!

    Be aware of the ``minimum_actions`` parameter: at low values even
    a single action (e.g. connecting a node) will add considerably to time
    invested. To keep total reconstruction time comparable to what Catmaid
    calculates, you should consider about 10 actions/minute (= a click every
    6 seconds) and ``max_inactive_time`` of 3 mins.

    CATMAID gives reconstruction time across all users. Here, we calculate
    the time spent tracing for individuals. This may lead to a discrepancy
    between sum of time invested over of all users from this function vs.
    CATMAID's reconstruction time.


    Parameters
    ----------
    x
                        Which neurons to check. Can be either:

                        1. skeleton IDs (int or str)
                        2. neuron name (str, must be exact match)
                        3. annotation: e.g. 'annotation:PN right'
                        4. CatmaidNeuron or CatmaidNeuronList object

                        If you pass a CatmaidNeuron/List, its node/connectors
                        are used to calculate time invested. You can exploit
                        this to get time spent reconstructing in given
                        compartment of a neurons, e.g. by pruning it to a
                        volume before passing it to ``get_time_invested``.
    mode :              'SUM' | 'SUM2' | 'OVER_TIME' | 'ACTIONS', optional
                        (1) 'SUM' will return total time invested (in minutes)
                            broken down by creation, edition and review.
                        (2) 'SUM2' will return total time invested (in
                            minutes) broken down by `treenodes`, `connectors`
                            and `links`.
                        (3) 'OVER_TIME' will return minutes invested/day over
                            time.
                        (4) 'ACTIONS' will return actions
                            (node/connectors placed/edited) per day.
    by :                'USER' | 'NEURON', optional
                        Determines whether the stats are broken down by user or
                        by neuron.
    minimum_actions :   int, optional
                        Minimum number of actions per minute to be counted as
                        active.
    max_inactive_time : int, optional
                        Interval in minutes over which time invested is
                        binned. Essentially determines how much time can be
                        between bouts of activity.
    treenodes :         bool, optional
                        If False, treenodes will not be taken into account.
    connectors :        bool, optional
                        If False, connectors will not be taken into account.
    links :             bool, optional
                        If False, connector links will not be taken into account.
    start_date :        iterable | datetime.date | numpy.datetime64, optional
                        Restricts time invested to window. Applies to creation
                        but not edition time! If iterable, must be year, month
                        day, e.g. ``[2018, 1, 1]``.
    end_date :          iterable | datetime.date | numpy.datetime64, optional
                        See ``start_date``.
    remote_instance :   CatmaidInstance, optional
                        Either pass explicitly or define globally.

    Returns
    -------
    pandas.DataFrame
        If ``mode='SUM'``, values represent minutes invested::

                 total  creation  edition  review
          user1
          user2
          ..
          .

        If ``mode='SUM2'``, values represent minutes invested::

                 total  treenodes  connectors  links
          user1
          user2
          ..
          .

        If ``mode='OVER_TIME'`` or ``mode='ACTIONS'``::

                 date1  date2  date3  ...
          user1
          user2
          ..
          .

        For `OVER_TIME`, values respresent minutes invested on that day. For
        `ACTIONS`, values represent actions (creation, edition, review) on that
        day.


    Examples
    --------
    Get time invested for a set of neurons:

    >>> da1 = pymaid.get_neurons('annotation:glomerulus DA1')
    >>> time = pymaid.get_time_invested(da1)

    Get time spent tracing in a specific compartment:

    >>> da1_lh = pymaid.prune_by_volume('LH_R', inplace=False)
    >>> time_lh = pymaid.get_time_invested(da1_lh)

    Get contributions within a given time window:

    >>> time_jan = pymaid.get_time_invested(da1,
    ...                                     start_date=[2018, 1, 1],
    ...                                     end_date=[2018, 1, 31])


    Plot pie chart of contributions per user using Plotly:

    >>> import plotly
    >>> stats = pymaid.get_time_invested(skids, remote_instance)
    >>> # Use plotly to generate pie chart
    >>> fig = {"data": [{"values": stats.total.tolist(),
    ...        "labels": stats.user.tolist(), "type" : "pie" }]}
    >>> plotly.offline.plot(fig)

    Plot reconstruction efforts over time:

    >>> stats = pymaid.get_time_invested(skids, mode='OVER_TIME')
    >>> # Plot time invested over time
    >>> stats.T.plot()
    >>> # Plot cumulative time invested over time
    >>> stats.T.cumsum(axis=0).plot()
    >>> # Filter for major contributors
    >>> stats[stats.sum(axis=1) > 20].T.cumsum(axis=0).plot()

    """
    def _extract_timestamps(ts, restrict_groups, desc='Calc'):
        if ts.empty:
            return {}
        grouped = ts.set_index('timestamp',
                               drop=False).groupby(['group',
                                                    pd.Grouper(freq=bin_width)]).count() >= minimum_actions
        temp_stats = {}
        for g in config.tqdm(set(ts.group.unique()) & set(restrict_groups),
                             desc=desc, disable=config.pbar_hide, leave=False):
            temp_stats[g] = sum(grouped.loc[g].values)[0] * interval
        return temp_stats

    if mode not in ['SUM', 'SUM2', 'OVER_TIME', 'ACTIONS']:
        raise ValueError('Unknown mode "{}"'.format(mode))

    if by not in ['NEURON', 'USER']:
        raise ValueError('Unknown by "{}"'.format(by))

    remote_instance = utils._eval_remote_instance(remote_instance)

    skids = utils.eval_skids(x, remote_instance=remote_instance)

    # Maximal inactive time is simply translated into binning
    # We need this later for pandas.TimeGrouper() anyway
    interval = max_inactive_time
    bin_width = '%iMin' % interval

    # Update minimum_actions to reflect actions/interval instead of
    # actions/minute
    minimum_actions *= interval

    user_list = fetch.get_user_list(remote_instance=remote_instance).set_index('id')
    user_dict = user_list.login.to_dict()

    if not isinstance(x, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        x = fetch.get_neuron(skids, remote_instance=remote_instance)

    if isinstance(x, core.CatmaidNeuron):
        skdata = core.CatmaidNeuronList(x)
    elif isinstance(x, core.CatmaidNeuronList):
        skdata = x

    if not isinstance(end_date, (datetime.date, np.datetime64, type(None))):
        end_date = datetime.date(*end_date)

    if not isinstance(start_date, (datetime.date, np.datetime64, type(None))):
        start_date = datetime.date(*start_date)

    # Extract connector and node IDs
    node_ids = []
    connector_ids = []
    for n in skdata.itertuples():
        if treenodes:
            node_ids += n.nodes.node_id.tolist()
        if connectors:
            connector_ids += n.connectors.connector_id.tolist()

    # Get node details
    node_details = fetch.get_node_details(node_ids + connector_ids,
                                          remote_instance=remote_instance)

    # Get details for links
    if links:
        link_details = fetch.get_connector_links(skdata,
                                                 remote_instance=remote_instance)

        # link_details contains all links. We have to subset this to existing
        # connectors in case the input neurons have been pruned
        link_details = link_details[link_details.connector_id.isin(connector_ids)]
    else:
        link_details = pd.DataFrame([], columns=['creator', 'creation_time'])

    # Remove timestamps outside of date range (if provided)
    if start_date:
        node_details = node_details[node_details.creation_time >= np.datetime64(start_date)]
        link_details = link_details[link_details.creation_time >= np.datetime64(start_date)]
    if end_date:
        node_details = node_details[node_details.creation_time <= np.datetime64(end_date)]
        link_details = link_details[link_details.creation_time <= np.datetime64(end_date)]

    # If we want to group by neuron, we need to add a "skeleton ID" column and
    # make check if we need to duplicate rows with connectors
    if by == 'NEURON':
        # Need to add a column with the skeleton ID
        node_details['skeleton_id'] = None
        node_details['node_type'] = 'connector'
        col_name = 'skeleton_id'

        for n in skdata:
            cond = node_details.node_id.isin(n.nodes.node_id.values.astype(str))
            node_details.loc[cond, 'skeleton_id'] = n.skeleton_id
            node_details.loc[cond, 'node_type'] = 'treenode'

        # Connectors can show up in more than one neuron -> we need to duplicate
        # those rows for each of the associated neurons
        cn_details = []
        for n in skdata:
            cond1 = node_details.node_type == 'connector'
            cond2 = node_details.node_id.isin(n.connectors.connector_id.values.astype(str))
            node_details.loc[cond1 & cond2, 'skeleton_id'] = n.skeleton_id
            this_cn = node_details.loc[cond1 & cond2]
            cn_details.append(this_cn)
        cn_details = pd.concat(cn_details, axis=0)

        # Merge the node details again
        cond1 = node_details.node_type == 'treenode'
        node_details = pd.concat([node_details.loc[cond1], cn_details],
                                 axis=0).reset_index(drop=True)

        # Note that link_details already has a "skeleton_id" column
        # but we need to make sure it's strings
        link_details['skeleton_id'] = link_details.skeleton_id.astype(str)

        create_group = edit_group = 'skeleton_id'
    else:
        create_group = 'creator'
        edit_group = 'editor'
        col_name = 'user'

    # Dataframe for creation (i.e. the actual generation of the nodes)
    creation_timestamps = np.append(node_details[[create_group,
                                                  'creation_time']].values,
                                    link_details[[create_group,
                                                  'creation_time']].values,
                                    axis=0)
    creation_timestamps = pd.DataFrame(creation_timestamps,
                                       columns=['group', 'timestamp'])

    # Dataframe for edition times - can't use links as there is no editor
    # Because creation of a node counts as an edit, we are removing
    # timestamps where creation and edition time are less than 100ms apart
    is_edit = (node_details.edition_time - node_details.creation_time) > np.timedelta64(200, 'ms')
    edition_timestamps = node_details.loc[is_edit, [edit_group, 'edition_time']]
    edition_timestamps.columns = ['group', 'timestamp']

    # Generate dataframe for reviews -> here we have to unpack
    if by == 'USER':
        groups = [u for l in node_details.reviewers.values for u in l]
    else:
        groups = [s for l, s in zip(node_details.review_times.values,
                                    node_details.skeleton_id.values) for ts in l]
    timestamps = [ts for l in node_details.review_times.values for ts in l]
    review_timestamps = pd.DataFrame([groups, timestamps]).T
    review_timestamps.columns = ['group', 'timestamp']

    # Change user ID to login
    if by == 'USER':
        if mode == 'SUM2':
            node_details['creator'] = node_details.creator.map(lambda x: user_dict.get(x, f'Anonymous{x}'))
            node_details['editor'] = node_details.editor.map(lambda x: user_dict.get(x, f'Anonymous{x}'))

            link_details['creator'] = link_details.creator.map(lambda x: user_dict.get(x, f'Anonymous{x}'))

        creation_timestamps['group'] = creation_timestamps.group.map(lambda x: user_dict.get(x, f'Anonymous{x}'))
        edition_timestamps['group'] = edition_timestamps.group.map(lambda x: user_dict.get(x, f'Anonymous{x}'))
        review_timestamps['group'] = review_timestamps.group.map(lambda x: user_dict.get(x, f'Anonymous{x}'))

    # Merge all timestamps
    all_timestamps = pd.concat([creation_timestamps,
                                edition_timestamps,
                                review_timestamps],
                               axis=0)

    all_timestamps.sort_values('timestamp', inplace=True)

    if by == 'USER':
        # Extract the users that are relevant for us
        relevant_users = all_timestamps.groupby('group').count()
        groups = relevant_users[relevant_users.timestamp >= minimum_actions].index.values
    else:
        groups = skdata.skeleton_id

    if mode == 'SUM':
        # This breaks it down by time spent on creation, edition and review
        stats = {k: {g: 0 for g in groups} for k in ['total',
                                                     'creation',
                                                     'edition',
                                                     'review']}

        stats['total'].update(_extract_timestamps(all_timestamps,
                                                  groups,
                                                  desc='Calc total'))
        stats['creation'].update(_extract_timestamps(creation_timestamps,
                                                     groups,
                                                     desc='Calc creation'))
        stats['edition'].update(_extract_timestamps(edition_timestamps,
                                                    groups,
                                                    desc='Calc edition'))
        stats['review'].update(_extract_timestamps(review_timestamps,
                                                   groups,
                                                   desc='Calc review'))

        return pd.DataFrame([[g,
                              stats['total'][g],
                              stats['creation'][g],
                              stats['edition'][g],
                              stats['review'][g]] for g in groups],
                            columns=[col_name, 'total',
                                     'creation', 'edition',
                                     'review']
                            ).sort_values('total',
                                          ascending=False
                                          ).reset_index(drop=True).set_index(col_name)

    elif mode == 'SUM2':
        # This breaks it down by time spent on nodes, connectors and links
        stats = {k: {g: 0 for g in groups} for k in ['total',
                                                     'treenodes',
                                                     'connectors',
                                                     'links']}

        stats['total'].update(_extract_timestamps(all_timestamps,
                                                  groups,
                                                  desc='Calc total'))

        # We need to construct separate DataFrames for nodes, connectors + links
        # Note that we are using only edits that do not stem from the creation
        is_tn = node_details.node_id.astype(int).isin(node_ids)
        conc = np.concatenate([node_details.loc[is_tn,
                                                [create_group, 'creation_time']
                                                ].values,
                               node_details.loc[is_edit & is_tn,
                                                [edit_group, 'edition_time']
                                                ].values
                               ],
                              axis=0)
        treenode_timestamps = pd.DataFrame(conc, columns=['group', 'timestamp'])

        stats['treenodes'].update(_extract_timestamps(treenode_timestamps,
                                                      groups,
                                                      desc='Calc treenodes'))

        # Now connectors
        # Note that we are using only edits that do not stem from the creation
        is_cn = node_details.node_id.astype(int).isin(connector_ids)
        conc = np.concatenate([node_details.loc[is_cn,
                                                [create_group, 'creation_time']
                                                ].values,
                               node_details.loc[is_edit & is_cn,
                                                [edit_group, 'edition_time']
                                                ].values
                               ],
                              axis=0)
        connector_timestamps = pd.DataFrame(conc, columns=['group', 'timestamp'])

        stats['connectors'].update(_extract_timestamps(connector_timestamps,
                                                       groups,
                                                       desc='Calc connectors'))

        # Now links
        link_timestamps = pd.DataFrame(link_details[[create_group,
                                                     'creation_time']].values,
                                       columns=['group', 'timestamp'])

        stats['links'].update(_extract_timestamps(link_timestamps,
                                                  groups,
                                                  desc='Calc links'))

        return pd.DataFrame([[g,
                              stats['total'][g],
                              stats['treenodes'][g],
                              stats['connectors'][g],
                              stats['links'][g]] for g in groups],
                            columns=[col_name, 'total',
                                     'treenodes', 'connectors',
                                     'links']
                            ).sort_values('total', ascending=False
                                          ).reset_index(drop=True
                                                        ).set_index(col_name)

    elif mode == 'ACTIONS':
        all_ts = all_timestamps.set_index('timestamp', drop=False
                                          ).timestamp.groupby(pd.Grouper(freq='1d')
                                                              ).count().to_frame()
        all_ts.columns = ['all_groups']
        all_ts = all_ts.T
        # Get total time spent
        for g in config.tqdm(all_timestamps.group.unique(), desc='Calc. total',
                             disable=config.pbar_hide, leave=False):
            this_ts = all_timestamps[all_timestamps.group == g].set_index(
                'timestamp', drop=False).timestamp.groupby(pd.Grouper(freq='1d')).count().to_frame()
            this_ts.columns = [g]

            all_ts = pd.concat([all_ts, this_ts.T])

        return all_ts.fillna(0)

    elif mode == 'OVER_TIME':
        # Go over all users and collect time invested
        all_ts = []
        for g in config.tqdm(all_timestamps.group.unique(), desc='Calc. total', disable=config.pbar_hide, leave=False):
            # First count all minutes with minimum number of actions
            minutes_counting = (all_timestamps[all_timestamps.group == g].set_index(
                'timestamp', drop=False).timestamp.groupby(pd.Grouper(freq=bin_width)).count().to_frame() >= minimum_actions)
            # Then remove the minutes that have less than minimum actions
            minutes_counting = minutes_counting[minutes_counting.timestamp]

            # Now group timestamps by day
            this_ts = minutes_counting.groupby(pd.Grouper(freq='1d')).count()

            # Rename columns to user login
            this_ts.columns = [g]

            # Append if an and move on
            if not this_ts.empty:
                all_ts.append(this_ts.T)

        # Turn into DataFrame
        all_ts = pd.concat(all_ts).sort_index()

        # Replace NaNs with 0
        all_ts.fillna(0, inplace=True)

        # Add all users column
        all_users = all_ts.sum(axis=0)
        all_users.name = 'all_groups'

        all_ts = pd.concat([all_users, all_ts.T], axis=1).T

        return all_ts


def get_user_actions(users=None, neurons=None, start_date=None, end_date=None,
                     remote_instance=None):
    """Get timestamps of user actions (creations, editions, reviews, linking).

    Important
    ---------
    This function returns most but not all user actions::

      1. The API endpoint used for finding neurons worked on by a given user
         (:func:`pymaid.find_neurons`) does not return single-node neurons.
         Hence, placing e.g. postsynaptic nodes is not taken into account.
      2. Any creation is also an edit. However, only the last edit is kept
         track of. So each creation counts as an edit for the creator until a
         different user makes an edit.

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
            DataFrame in which each row is a user action::

                user   timestamp   action
             0
             1
             ...

    Examples
    --------
    In the first example we will have a look at how active a user is over
    the course of a day.

    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> # Get all actions for a single user
    >>> actions = pymaid.get_user_actions(users='schlegelp',
    ....                                  start_date=(2017, 11, 1))
    >>> # Group by hour and see what time of the day user is usually active
    >>> actions.set_index(pd.DatetimeIndex(actions.timestamp), inplace=True)
    >>> hours = actions.groupby(actions.index.hour).count()
    >>> ax = hours.action.plot()
    >>> plt.show()

    >>> # Plot day-by-day activity
    >>> ax = plt.subplot()
    >>> ax.scatter(actions.timestamp.date.values,
    ...            actions.timestamp.time.values,
    ...            marker='_')

    """
    if not neurons and not users and not (start_date or end_date):
        raise ValueError('Query must be restricted by at least a single '
                         'parameter!')

    if users and not isinstance(users, (list, np.ndarray)):
        users = [users]

    # Get user dictionary (needed later)
    user_list = fetch.get_user_list(remote_instance=remote_instance)
    user_dict = user_list.set_index('id').login.to_dict()

    if isinstance(neurons, type(None)):
        neurons = fetch.find_neurons(users=users,
                                     from_date=start_date, to_date=end_date,
                                     reviewed_by=users,
                                     remote_instance=remote_instance)
        # Get skeletons
        neurons.get_skeletons()
    elif not isinstance(neurons, (core.CatmaidNeuron, core.CatmaidNeuronList)):
        neurons = fetch.get_neuron(neurons, remote_instance=remote_instance)

    if not isinstance(end_date, (datetime.date, type(None))):
        end_date = datetime.date(*end_date)

    if not isinstance(start_date, (datetime.date, type(None))):
        start_date = datetime.date(*start_date)

    node_ids = neurons.nodes.node_id.tolist()
    connector_ids = neurons.connectors.connector_id.tolist()

    # Get node details
    node_details = fetch.get_node_details(node_ids + connector_ids,
                                          remote_instance=remote_instance)

    # Get details for links
    link_details = fetch.get_connector_links(neurons,
                                             remote_instance=remote_instance)

    # Dataframe for creation (i.e. the actual generation of the nodes)
    creation_timestamps = node_details[['creator', 'creation_time']].copy()
    creation_timestamps['action'] = 'creation'
    creation_timestamps.columns = ['user', 'timestamp', 'action']

    # Dataframe for edition times
    edition_timestamps = node_details[['editor', 'edition_time']].copy()
    edition_timestamps['action'] = 'edition'
    edition_timestamps.columns = ['user', 'timestamp', 'action']

    # DataFrame for linking
    linking_timestamps = link_details[['creator', 'creation_time']].copy()
    linking_timestamps['action'] = 'linking'
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
        all_timestamps = all_timestamps[all_timestamps.timestamp.values >= np.datetime64(start_date)]
    if end_date:
        all_timestamps = all_timestamps[all_timestamps.timestamp.values <= np.datetime64(end_date)]

    return all_timestamps.sort_values('timestamp').reset_index(drop=True)
