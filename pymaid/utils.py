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

import collections
import itertools
import os
import six
import sys
import warnings

import pandas as pd
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import vispy.visuals

from . import core, fetch, config, client

# Set up logging
logger = config.logger

__all__ = ['set_loggers', 'set_pbars', 'eval_skids', 'clear_cache', 'shorten_name']


def clear_cache():
    """Clear cache of global CatmaidInstance."""
    if 'remote_instance' in sys.modules:
        rm = sys.modules['remote_instance']
    elif 'remote_instance' in globals():
        rm = globals()['remote_instance']
    else:
        raise ValueError('No global CatmaidInstance found.')

    rm.clear_cache()


def _type_of_script():
    """Return context in which pymaid is run."""
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyterlab():
    """Test if we are inside Jupyter lab."""
    import psutil
    return any(['jupyter-lab' in x for x in psutil.Process().parent().cmdline()])


def has_plotly_extension():
    """Check if Jupyter lab plotly renderer extension is installed."""
    import subprocess
    # This is the old plotly renderer
    result = subprocess.run(['jupyter',
                             'labextension',
                             'check',
                             '@jupyterlab/plotly-extension'])
    if result.returncode == 0:
        return True

    # This is the new one
    result = subprocess.run(['jupyter',
                             'labextension',
                             'check',
                             'jupyterlab-plotly'])
    if result.returncode == 0:
        return True
    return False


def is_headless():
    """Check if Display is available."""
    return 'DISPLAY' not in os.environ


def is_jupyter():
    """Test if pymaid is run in a Jupyter notebook."""
    return _type_of_script() == 'jupyter'


def ipywidgets_installed():
    """Test if pymaid is run in a Jupyter notebook."""
    try:
        import ipywidgets
        return True
    except ImportError:
        return False
    except BaseException as e:
        logger.error('Error importing ipytwidgets: {}'.format(str(e)))


def set_loggers(level='INFO'):
    """Helper function to set levels for all associated module loggers."""
    config.logger.setLevel(level)


def set_pbars(hide=None, leave=None, jupyter=None):
    """Set global progress bar behaviors.

    Parameters
    ----------
    hide :      bool, optional
                Set to True to hide all progress bars.
    leave :     bool, optional
                Set to False to clear progress bars after they have finished.
    jupyter :   bool, optional
                Set to False to force using of classic tqdm even if in
                Jupyter environment.

    Returns
    -------
    Nothing

    """
    if isinstance(hide, bool):
        config.pbar_hide = hide

    if isinstance(leave, bool):
        config.pbar_leave = leave

    if isinstance(jupyter, bool):
        if jupyter:
            if not is_jupyter():
                logger.error('Unable to use fancy Jupyter progress: '
                             'No Jupyter environment detected.')
            elif not ipywidgets_installed():
                logger.error('Unable to use fancy Jupyter progress: '
                             'ipywidgets not installed .')
            else:
                config.tqdm = config.tqdm_notebook
                config.trange = config.tnrange
        else:
            config.tqdm = config.tqdm_classic
            config.trange = config.trange_classic


def _make_iterable(x, force_type=None):
    """Convert input into a np.ndarray, if it isn't already.

    For dicts, keys will be turned into array.

    """
    if not isinstance(x, collections.Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, dict) or isinstance(x, set):
        x = list(x)

    if force_type:
        return np.array(x).astype(force_type)
    else:
        return np.array(x)


def _make_non_iterable(x):
    """Convert input into non-iterable, if it isn't already.

    Will raise error if len(x) > 1.

    """
    if not _is_iterable(x):
        return x
    elif len(x) == 1:
        return x[0]
    else:
        raise ValueError('Iterable must not contain more than one entry.')


def _is_iterable(x):
    """Check is input is iterable but not str, dictionary or pandas DataFrame.
    """
    if isinstance(x, collections.Iterable) and not isinstance(x, (six.string_types, pd.DataFrame)):
        return True
    else:
        return False


def _eval_conditions(x):
    """Split list of strings into positive (no ~) and negative (~) conditions.
    """

    x = _make_iterable(x, force_type=str)

    return [i for i in x if not i.startswith('~')], [i[1:] for i in x if i.startswith('~')]


def _eval_remote_instance(remote_instance, raise_error=True):
    """Evaluates remote instance.

    If input is None, checks for globally defined remote instances as fall
    back.

    Parameters
    ----------
    remote_instance :   CatmaidInstance | None
                        Input to be evaluated.
    raise_on_error :    bool, optional
                        If True will raise error if input is ``None`` and
                        no global CatmaidInstance was found.

    Returns
    -------
    CatmaidInstance

    """
    if remote_instance is None:
        if 'remote_instance' in sys.modules:
            return sys.modules['remote_instance']
        elif 'remote_instance' in globals():
            return globals()['remote_instance']
        else:
            if raise_error:
                raise Exception('No pymaid.CatmaidInstance found. Please '
                                'either define globally or pass explicitly '
                                'as "remote_instance". See '
                                '`help(pymaid.CatmaidInstance) for details.')
            else:
                logger.warning('No global remote instance found.')
    elif not isinstance(remote_instance, client.CatmaidInstance):
        error = 'Expected None or CatmaidInstance, got {}'.format(type(remote_instance))
        if raise_error:
            raise TypeError(error)
        else:
            logger.warning(error)

    return remote_instance


def eval_skids(x, remote_instance=None, warn_duplicates=True):
    """Extract skeleton IDs from input.

    Will turn annotations and neuron names into skeleton IDs.

    Parameters
    ----------
    x :             int | str | CatmaidNeuron | CatmaidNeuronList | DataFrame
                    Your options are either::
                    1. int or list of ints:
                        - will be assumed to be skeleton IDs
                    2. str or list of str:
                        - if convertible to int, will be interpreted as x
                        - if starts with 'annotation:' will be assumed to be
                          annotations
                        - else will be assumed to be neuron names
                    3. For CatmaidNeuron/List or pandas.DataFrames/Series:
                        - will look for ``skeleton_id`` attribute
    remote_instance :  CatmaidInstance, optional
                       If not passed directly, will try using global.
    warn_duplicates :  bool, optional
                       If True, will warn if duplicate skeleton IDs are found.
                       Only applies to CatmaidNeuronLists.

    Returns
    -------
    list
                    List containing skeleton IDs as strings.

    """
    remote_instance = _eval_remote_instance(remote_instance,
                                            raise_error=False)

    if isinstance(x, (int, np.int64, np.int32)):
        return [str(x)]
    elif isinstance(x, str):
        try:
            int(x)
            return [str(x)]
        except BaseException:
            if x.startswith('annotation:') or x.startswith('annotations:'):
                an = x[x.index(':') + 1:]
                return fetch.get_skids_by_annotation(an,
                                                     remote_instance=remote_instance)
            elif x.startswith('name:'):
                return fetch.get_skids_by_name(x[5:],
                                               remote_instance=remote_instance,
                                               allow_partial=False
                                               ).skeleton_id.tolist()
            else:
                return fetch.get_skids_by_name(x,
                                               remote_instance=remote_instance,
                                               allow_partial=False
                                               ).skeleton_id.tolist()
    elif isinstance(x, (list, np.ndarray, set)):
        skids = []
        for e in x:
            temp = eval_skids(e, remote_instance=remote_instance)
            if isinstance(temp, (list, np.ndarray)):
                skids += temp
            else:
                skids.append(temp)
        return sorted(set(skids), key=skids.index)
    elif isinstance(x, core.CatmaidNeuron):
        return [x.skeleton_id]
    elif isinstance(x, core.CatmaidNeuronList):
        if len(x.skeleton_id) != len(set(x.skeleton_id)) and warn_duplicates:
            logger.warning('Duplicate skeleton IDs found in neuronlist. '
                           'The function you are using might not respect '
                           'fragments of the same neuron. For explanation see '
                           'http://pymaid.readthedocs.io/en/latest/source/conn'
                           'ectivity_analysis.html.')
        return list(x.skeleton_id)
    elif isinstance(x, pd.DataFrame):
        if 'skeleton_id' not in x.columns:
            raise ValueError('Expect "skeleton_id" column in pandas DataFrames')
        return x.skeleton_id.tolist()
    elif isinstance(x, pd.Series):
        if x.name == 'skeleton_id':
            return x.tolist()
        elif 'skeleton_id' in x:
            return [x.skeleton_id]
        else:
            raise ValueError('Unable to extract skeleton ID from Pandas '
                             'series {0}'.format(x))
    elif isinstance(x, type(None)):
        return None
    else:
        logger.error(
            'Unable to extract x from type %s' % str(type(x)))
        raise TypeError('Unable to extract skids from type %s' % str(type(x)))


def eval_user_ids(x, user_list=None, remote_instance=None):
    """Check a list of users and turns them into user IDs.

    Always returns a list! Will attempt converting in the following order:

        (1) user ID
        (2) login name
        (3) last name
        (4) full name
        (5) first name

    Important
    ---------
    Last, first and full names are case-sensitive!

    Parameters
    ----------
    x :         int | str | list of either
                Users to check.
    user_list : pd.DataFrame, optional
                User list from :func:`~pymaid.get_user_list`. If you
                already have it, pass it along to save time.

    """
    remote_instance = _eval_remote_instance(remote_instance)

    if x and not isinstance(x, (list, np.ndarray)):
        x = [x]

    try:
        # Test if we have any non IDs (i.e. logins) in users
        user_ids = [int(u) for u in x]
    except BaseException:
        # Get list of users if we don't already have it
        if not user_list:
            user_list = fetch.get_user_list(
                remote_instance=remote_instance)

        # Now convert individual entries to user IDs
        user_ids = []
        for u in x:
            try:
                user_ids.append(int(u))
            except BaseException:
                for col in ['login', 'last_name', 'full_name', 'first_name']:
                    found = []
                    if u in user_list[col].values:
                        found = user_list[user_list[col] == u].id.tolist()
                        break
                if not found:
                    logger.warning(
                        'User "{0}" not found. Skipping...'.format(u))
                elif len(found) > 1:
                    logger.warning('Multiple matching entries for '
                                   '"{0}" found. Skipping...'.format(u))
                else:
                    user_ids.append(int(found[0]))

    return user_ids


def eval_node_ids(x, connectors=True, nodes=True):
    """Extract node or connector IDs.

    Parameters
    ----------
    x :             int | str | CatmaidNeuron | CatmaidNeuronList | DataFrame
                    Your options are either::
                    1. int or list of ints will be assumed to be node IDs
                    2. str or list of str will be checked if convertible to int
                    3. For CatmaidNeuron/List or pandas.DataFrames will try
                       to extract node IDs
    connectors :    bool, optional
                    If True will return connector IDs from neuron objects
    nodes :         bool, optional
                    If True will return node IDs from neuron objects

    Returns
    -------
    list
                    List containing nodes as strings.

    """
    if isinstance(x, (int, np.int64, np.int32)):
        return [x]
    elif isinstance(x, str):
        try:
            return [int(x)]
        except BaseException:
            raise TypeError(
                'Unable to extract node ID from string <%s>' % str(x))
    elif isinstance(x, (set, list, np.ndarray)):
        # Check non-integer entries
        ids = []
        for e in x:
            temp = eval_node_ids(e, connectors=connectors,
                                 nodes=nodes)
            if isinstance(temp, (list, np.ndarray)):
                ids += temp
            else:
                ids.append(temp)
        # Preserving the order after making a set is super costly
        # return sorted(set(ids), key=ids.index)
        return list(set(ids))
    elif isinstance(x, core.CatmaidNeuron):
        to_return = []
        if nodes:
            to_return += x.nodes.treenode_id.tolist()
        if connectors:
            to_return += x.connectors.connector_id.tolist()
        return to_return
    elif isinstance(x, core.CatmaidNeuronList):
        to_return = []
        for n in x:
            if nodes:
                to_return += n.nodes.treenode_id.tolist()
            if connectors:
                to_return += n.connectors.connector_id.tolist()
        return to_return
    elif isinstance(x, (pd.DataFrame, pd.Series)):
        to_return = []
        if nodes and 'node_id' in x:
            to_return += x.node_id.tolist()
        if connectors and 'connector_id' in x:
            to_return += x.connector_id.tolist()

        if 'connector_id' not in x and 'node_id' not in x:
            to_return = x.tolist()

        return to_return
    else:
        raise TypeError(f'Unable to extract node IDs from type {type(x)}')


def _unpack_neurons(x, raise_on_error=True):
    """Unpack neurons and returns a list of individual neurons."""
    neurons = []

    if isinstance(x, (list, np.ndarray, tuple)):
        for l in x:
            neurons += _unpack_neurons(l)
    elif isinstance(x, core.CatmaidNeuron):
        neurons.append(x)
    elif isinstance(x, core.CatmaidNeuronList):
        neurons += x.neurons
    elif raise_on_error:
        raise TypeError('Unknown neuron format: "{}"'.format(type(x)))

    return neurons


def _parse_objects(x, remote_instance=None):
    """Parse objects into different types.

    Returns
    -------
    skids :     list
    skdata :    pymaid.CatmaidNeuronList
    dotprops :  pd.DataFrame
    volumes :   list
    points :    list of arrays
    visuals :   list of vispy visuals

    """
    if not isinstance(x, list):
        x = [x]

    # If any list in x, flatten first
    if any([isinstance(i, list) for i in x]):
        # We need to be careful to preserve order because of colors
        y = []
        for i in x:
            y += i if isinstance(i, list) else [i]
        x = y

    # Check for skeleton IDs
    skids = []
    for ob in x:
        if isinstance(ob, (str, int)):
            try:
                skids.append(int(ob))
            except BaseException:
                pass

    # Collect neuron objects and collate to single Neuronlist
    neuron_obj = [ob for ob in x if isinstance(ob,
                                               (core.CatmaidNeuron,
                                                core.CatmaidNeuronList))]
    skdata = core.CatmaidNeuronList(neuron_obj, make_copy=False)

    # Collect visuals
    visuals = [ob for ob in x if isinstance(ob, vispy.visuals.Visual)]

    # Collect dotprops
    dotprops = [ob for ob in x if isinstance(ob, core.Dotprops)]

    if len(dotprops) == 1:
        dotprops = dotprops[0]
    elif len(dotprops) == 0:
        dotprops = core.Dotprops()
        dotprops['gene_name'] = []
    elif len(dotprops) > 1:
        dotprops = pd.concat(dotprops)

    # Collect and parse volumes
    volumes = [ob for ob in x if isinstance(ob, (core.Volume, str))]

    # Collect dataframes with X/Y/Z coordinates
    # Note: dotprops and volumes are instances of pd.DataFrames
    dataframes = [ob for ob in x if isinstance(ob, pd.DataFrame) and not isinstance(ob, (core.Dotprops, core.Volume))]
    if [d for d in dataframes if False in [c in d.columns for c in ['x', 'y', 'z']]]:
        logger.warning('DataFrames must have x, y and z columns.')
    # Filter to and extract x/y/z coordinates
    dataframes = [d for d in dataframes if False not in [c in d.columns for c in ['x', 'y', 'z']]]
    dataframes = [d[['x', 'y', 'z']].values for d in dataframes]

    # Collect arrays
    arrays = [ob.copy() for ob in x if isinstance(ob, np.ndarray)]
    # Remove arrays with wrong dimensions
    if [ob for ob in arrays if ob.shape[1] != 3]:
        logger.warning('Point objects need to be of shape (n,3).')
    arrays = [ob for ob in arrays if ob.shape[1] == 3]

    points = dataframes + arrays

    return skids, skdata, dotprops, volumes, points, visuals


def __guess_sentiment(x):
    """Classify a list of words.

    Tries sorting words into either <type>, <nickname>, <tracer> or <generic>
    annotations.

    """
    sent = []
    for i, w in enumerate(x):
        # If word is a number, it's most likely something generic
        if w.isdigit():
            sent.append('generic')
        elif w == 'neuron':
            # If there is a lonely "neuron" followed by a number, it's generic
            if i != len(x) and x[i + 1].isdigit():
                sent.append('generic')
            # If not, it's probably type
            else:
                sent.append('type')
        # If there is a short, all upper case word after the generic information
        elif w.isupper() and len(w) > 1 and w.isalpha() and 'generic' in sent:
            # If there is no number in that word, it's probably tracer initials
            sent.append('tracer')
        else:
            # If the word is AFTER the generic number, it's probably a nickname
            if 'generic' in sent:
                sent.append('nickname')
            # If not, it's likely type information
            else:
                sent.append('type')

    return sent


def parse_neuronname(x):
    """Parse neuron names into type, nickname, tracer and generic information.

    This works best if neuron name follows this convention::

      {type} {generic} {nickname} {tracer initials}

    Parameters
    ----------
    x :     str | CatmaidNeuron
            Neuron name.

    Returns
    -------
    type :          str
    nickname :      str
    tracer :        str
    generic :       str

    Examples
    --------
    >>> pymaid.utils.parse_neuronname('AD1b2#7 3080184 Dust World JJ PS')
    ('AD1b2#7', 'Dust World', 'JJ PS', '3080184')

    """
    if isinstance(x, core.CatmaidNeuron):
        x = x.neuron_name

    if not isinstance(x, str):
        raise TypeError('Unable to parse name: must be str, not {}'.format(type(x)))

    # Split name into single words
    words = x.split(' ')
    sentiments = __guess_sentiment(words)

    type_str = [w for w, s in zip(words, sentiments) if s == 'type']
    nick_str = [w for w, s in zip(words, sentiments) if s == 'nickname']
    tracer_str = [w for w, s in zip(words, sentiments) if s == 'tracer']
    gen_str = [w for w, s in zip(words, sentiments) if s == 'generic']

    return ' '.join(type_str), ' '.join(nick_str), ' '.join(tracer_str), ' '.join(gen_str)


def shorten_name(x, max_len=30):
    """Shorten a neuron name by iteratively removing non-essential bits.

    Prioritises generic -> tracer -> nickname -> type information when removing
    until target length is reached. This works best if neuron name follows
    this convention::

      {type} {generic} {nickname} {tracers}

    Parameters
    ----------
    x :         str | CatmaidNeuron
                Neuron name.
    max_len :   int, optional
                Max length of shortened name.

    Returns
    -------
    shortened name :    str

    Examples
    --------
    >>> pymaid.shorten_name('AD1b2#7 3080184 Dust World JJ PS', 30)
    'AD1b2#7 Dust World [..]'

    """
    if isinstance(x, core.CatmaidNeuron):
        x = x.neuron_name

    # Split into individual words and guess their type
    words = x.split(' ')
    sentiments = __guess_sentiment(words)

    # Make sure we're working on a copy of the original neuron name
    short = str(x)

    ty = ['generic', 'tracer', 'nickname', 'type']
    # Iteratively remove generic -> tracer -> nickname -> type information
    for t, (w, sent) in itertools.product(ty, zip(words[::-1], sentiments[::-1])):
        # Stop if we are below max length
        if len(short) <= max_len:
            break
        # Stop if there is only a single word left
        elif len(short.replace('[..]', '').strip().split(' ')) == 1:
            break
        # Skip if this word is not of the right sentiment
        elif t != sent:
            continue
        # Remove this word
        short = short.replace(w, '[..]').strip()
        # Make sure to merge consecutive '[..]'
        while '[..] [..]' in short:
            short = short.replace('[..] [..]', '[..]')

    return short


def to_float(x):
    """Convert input to float."""
    try:
        return float(x)
    except ValueError:
        return None
    except BaseException:
        raise
