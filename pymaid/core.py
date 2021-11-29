#    Copyright (C) 2017 Philipp Schlegel

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

"""This module contains neuron and neuronlist classes returned and accepted
by many functions within pymaid. CatmaidNeuron and CatmaidNeuronList objects
also provided quick access to many other PyMaid functions.

Examples
--------
>>> # Get a bunch of neurons from CATMAID server as CatmaidNeuronList
>>> nl = pymaid.get_neuron('annotation:uPN right')
>>> # CatmaidNeuronLists work in, many ways, like pandas DataFrames
>>> nl.head()
                            neuron_name skeleton_id  n_nodes  n_connectors  \
0              PN glomerulus VA6 017 DB          16    12721          1878
1          PN glomerulus VL2a 22000 JMR       21999    10740          1687
2            PN glomerulus VC1 22133 BH       22132     8446          1664
3  PN putative glomerulus VC3m 22278 AA       22277     6228           674
4          PN glomerulus DL2v 22423 JMR       22422     4610           384

   n_branch_nodes  n_end_nodes  open_ends  cable_length review_status  soma
0             773          822        280   2863.743284            NA  True
1             505          537        194   2412.045343            NA  True
2             508          548        110   1977.235899            NA  True
3             232          243        100   1221.985849            NA  True
4             191          206         93   1172.948499            NA  True
>>> # Plot neurons
>>> nl.plot3d()
>>> # Neurons in a list can be accessed by index, ...
>>> nl[0]
>>> # ... by skeleton ID, ...
>>> nl.skid[16]
>>> # ... or by attributes
>>> nl[nl.cable_length > 2000]
>>> # Each neuron has a bunch of useful attributes
>>> print(nl[0].skeleton_id, nl[0].soma, nl[0].n_open_ends)
>>> # Attributes can also be accessed for the entire neuronslist
>>> nl.skeleton_id

:class:`~pymaid.CatmaidNeuron` and :class:`~pymaid.CatmaidNeuronList`
also allow quick access to other PyMaid functions:

>>> # This ...
>>> pymaid.reroot_neuron(nl[0], nl[0].soma, inplace=True)
>>> # ... is essentially equivalent to this
>>> nl[0].reroot(nl[0].soma)
>>> # Similarly, CatmaidNeurons do on-demand data fetching for you:
>>> # So instead of this ...
>>> an = pymaid.get_annotations(nl[0])
>>> # ..., you can do just this:
>>> an = nl[0].annotations

"""

import datetime
import functools
import hashlib
import json
import navis
import numbers

import numpy as np
import pandas as pd
import matplotlib.colors as mcl
import navis.config as nsconfig

from . import (fetch, utils, config, client, cache)

try:
    import trimesh
except ImportError:
    trimesh = None

__all__ = ['CatmaidNeuron', 'CatmaidNeuronList', 'Dotprops', 'Volume']

# Set up logging
logger = config.logger


def inject_connection(func):
    """Raise error if no local or global connection."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        remote_instance = kwargs.get('remote_instance')
        if not remote_instance:
            if not self._remote_instance:
                raise Exception(f"{func.__name__}: Need CatmaidInstance to "
                                "fetch data. Either pass to function as "
                                "`remote_instance` or use neuron's "
                                "`set_remote_instance()` method.")
            remote_instance = self._remote_instance
        kwargs['remote_instance'] = remote_instance
        return func(*args, **kwargs)
    return wrapper


class CatmaidNeuron(navis.TreeNeuron):
    """Catmaid neuron object holding neuron data (nodes, connectors, name,
    etc) and providing quick access to various PyMaid functions.

    CatmaidNeuron can be minimally constructed from just a skeleton ID
    and a CatmaidInstance. Other parameters (nodes, connectors, neuron name,
    annotations, etc.) will then be retrieved from the server 'on-demand'.

    The easiest way to construct a CatmaidNeuron is by using
    :func:`~pymaid.get_neuron`.

    Manually, a complete CatmaidNeuron can be constructed from a pandas
    DataFrame (df) containing: df.nodes, df.connectors, df.skeleton_id,
    df.neuron_name, df.tags

    Using a CatmaidNeuron to initialise a CatmaidNeuron will automatically
    make a copy.

    Attributes
    ----------
    skeleton_id :       str
                        This neuron's skeleton ID.
    neuron_name :       str
                        This neuron's name.
    nodes :             ``pandas.DataFrame``
                        Contains complete node table.
    connectors :        ``pandas.DataFrame``
                        Contains complete connector table.
    presynapses :       ``pandas.DataFrame``
                        All presynaptic connectors.
    postsynapses :      ``pandas.DataFrame``
                        All postsynaptic connectors.
    gap_junctions :     ``pandas.DataFrame``
                        All gap junction connectors.
    date_retrieved :    ``datetime`` object
                        Timestamp of data retrieval.
    tags :              dict
                        Node tags.
    connector_tags :    dict
                        Connector tags.
    annotations :       list
                        This neuron's annotations.
    graph :             ``network.DiGraph``
                        Graph representation of this neuron.
    igraph :            ``igraph.Graph``
                        iGraph representation of this neuron. Returns ``None``
                        if igraph library not installed.
    review_status :     int
                        This neuron's review status.
    n_connectors :      int
                        Total number of synapses.
    n_presynapses :     int
                        Total number of presynaptic sites.
    n_postsynapses :    int
                        Total number of presynaptic sites.
    n_branch_nodes :    int
                        Number of branch nodes.
    n_end_nodes :       int
                        Number of end nodes.
    n_open_ends :       int
                        Number of open end nodes = leaf nodes that are not
                        tagged with either: ``ends``, ``not a branch``,
                        ``uncertain end``, ``soma`` or
                        ``uncertain continuation``.
    cable_length :      float
                        Cable length in micrometers [um].
    segments :          list of lists
                        Node IDs making up linear segments. Maximizes
                        segment lengths (similar to CATMAID's review widget).
    small_segments :    list of lists
                        Node IDs making up linear segments between
                        end/branch points.
    soma :              node ID of soma
                        Returns ``None`` if no soma or 'NA' if data not
                        available.
    root :              numpy.array
                        Node ID(s) of root.
    color :             tuple
                        Color of neuron. Used for e.g. export to json.
    partners :          pd.DataFrame
                        Connectivity table of this neuron.

    Examples
    --------
    >>> # Initialize a new neuron
    >>> n = pymaid.CatmaidNeuron(123456)
    >>> # Retrieve node data from server on-demand
    >>> n.nodes
    CatmaidNeuron - INFO - Retrieving skeleton data...
      node_id  parent_id  creator_id  x  y  z radius confidence
    0  ...
    >>> # Initialize with skeleton data
    >>> n = pymaid.get_neuron(123456)
    >>> # Get annotations from server
    >>> n.annotations
    ['annotation1', 'annotation2']
    >>> # Force update of annotations
    >>> n.get_annotations()

    """
    #: Minimum radius for soma detection. Set to ``None`` if no tag needed.
    #: Default = 1 micron
    soma_detection_radius = .5 * nsconfig.ureg.um
    #: Soma radius (e.g. for plotting). If string, must be column in nodes
    #: table. Default = 'radius'.
    soma_radius = 'radius'
    # Set default function for soma finding.
    _soma = None
    # Tag to detect soma
    soma_detection_tag = 'soma'

    #: Attributes to be used when comparing two neurons.
    EQ_ATTRIBUTES = ['n_nodes', 'n_connectors', 'soma', 'root', 'skeleton_id',
                     'n_branches', 'n_leafs', 'cable_length', 'name']

    #: Attributes used for neuron summary
    SUMMARY_PROPS = ['type', 'name', 'n_nodes', 'n_connectors',
                     'n_branches', 'n_leafs', 'cable_length', 'soma', 'units']

    # Default value for lazy loading
    _lazy_loading = True

    def __init__(self, x, remote_instance=None, units='nm', **metadata):
        """Initialize CatmaidNeuron.

        Parameters
        ----------
        x
                            Data to construct neuron from:
                             - `pandas.DataFrame` is expected to be SWC table
                             - `pandas.Series` is expected to have a DataFrame
                               as `.nodes` - additional properties will be
                               attached as meta data
                             - `str` is treated as SWC file name
                             - `BufferedIOBase` e.g. from `open(filename)`
                             - `networkx.DiGraph` parsed by `navis.nx2neuron`
        remote_instance :   CatmaidInstance, optional
                            Storing this makes it more convenient to retrieve
                            e.g. neuron annotations, review status, etc. If
                            not provided, will try using global CatmaidInstance.
        units :             str | pint.Units | pint.Quantity
                            Units for coordinates. Defaults to ``None`` (dimensionless).
                            Strings must be parsable by pint: e.g. "nm", "um",
                            "micrometer" or "8 nanometers".
        metadata
                            Any additional data to attach to neuron.

        """
        if (isinstance(x, str) and x.isnumeric()) or isinstance(x, numbers.Number):
            # Initialize empty neuron
            metadata.update({'skeleton_id': x})
            super().__init__(None, units=units, **metadata)
        else:
            super().__init__(x, units=units, **metadata)

        if remote_instance is None:
            remote_instance = getattr(x, 'remote_instance', None)

        remote_instance = utils._eval_remote_instance(remote_instance,
                                                      raise_error=False)

        # These will be overriden if x is a CatmaidNeuron
        self._remote_instance = remote_instance
        self.date_retrieved = datetime.datetime.now().isoformat()

    def __eq__(self, other):
        """Implement neuron comparison."""
        # Deactivate lazy loading during comparison
        self._lazy_loading = False
        if isinstance(other, CatmaidNeuron):
            other._lazy_loading = False

        try:
            res = super().__eq__(other)
        except BaseException:
            raise
        finally:
            self._lazy_loading = True
            if isinstance(other, CatmaidNeuron):
                other._lazy_loading = True

        return res

    def __mul__(self, other, *args, **kwargs):
        """Implement multiplication for coordinates (nodes, connectors)."""
        # Exclude missing radii from multiplication
        is_missing = self.nodes.radius == -1
        n = super().__mul__(other, *args, **kwargs)
        n.nodes.loc[is_missing, 'radius'] = -1

        return n

    def __truediv__(self, other, *args, **kwargs):
        """Implement division for coordinates (nodes, connectors)."""
        # Exclude missing radii from division
        is_missing = self.nodes.radius == -1
        n = super().__truediv__(other, *args, **kwargs)
        n.nodes.loc[is_missing, 'radius'] = -1

        return n

    def __hash__(self):
        # DO NOT REMOVE THIS! When defining __eq__ in subclass, _hash__ will
        # not be inherited and we have to explicitly define it
        return super().__hash__()

    @property
    def annotations(self):
        """Neuron annotations."""
        if not hasattr(self, '_annotations'):
            self._annotations = self.get_annotations()
        return self._annotations

    @annotations.setter
    def annotations(self, v):
        if not isinstance(v, (list, np.ndarray)):
            raise TypeError(f'Expected annotations as list or array, got {type(v)}')
        self._annotations = v

    @property
    def core_md5(self) -> str:
        """MD5 of core information for the neuron.

        Generated from ``nodes`` and ``connectors`` table.

        Returns
        -------
        md5 :   string
                MD5 of node and connector table. None if no such data.

        """
        data = []
        if self.has_nodes:
            data.append(self.nodes[['node_id', 'parent_id',
                                    'x', 'y', 'z']].values)
        if self.has_connectors:
            data.append(self.connectors[['node_id',
                                         'connector_id',
                                         'x', 'y', 'z']].values)

        if data:
            data = np.ascontiguousarray(np.concatenate(data, axis=0).astype(float))
            return hashlib.md5(data).hexdigest()

    @property
    def gap_junctions(self):
        """Table with gap junctions.

        Requires a "type" column in connector table. Will look for type labels
        that include "gap" or that equal 2 or "2".
        """
        if not self.has_connectors:
            raise ValueError('No connector table found.')
        # Make an educated guess what gap junctions are
        types = self.connectors['type'].unique()
        gap = [t for t in types if 'gap' in str(t) or t in [2, "2"]]

        if len(gap) == 0:
            logger.debug(f'Unable to find gap junctions in types: {types}')
            return self.connectors.iloc[0:0]  # return empty DataFrame
        elif len(gap) > 1:
            raise ValueError(f'Found ambigous labels for gap junctions: {gap}')

        return self.connectors[self.connectors['type'] == gap[0]]

    @property
    def name(self):
        """Neuron name."""
        return self.neuron_name

    @name.setter
    def name(self, v):
        """Neuron name."""
        self.neuron_name = v

    @property
    def neuron_name(self):
        """Neuron name (legacy - please use .name instead)."""
        if not hasattr(self, '_name'):
            if not self._lazy_loading:
                return 'NA'
            self._name = self.get_name()
        return self._name

    @neuron_name.setter
    def neuron_name(self, v):
        self._name = v

    @property
    def open_ends(self):
        """Node IDs of open ends."""
        if self.has_nodes:
            closed = set(self.tags.get('ends', [])
                         + self.tags.get('uncertain end', [])
                         + self.tags.get('uncertain continuation', [])
                         + self.tags.get('not a branch', [])
                         + self.tags.get('soma', []))
            ends = self.ends
            return ends[~ends.node_id.isin(closed)]
        else:
            logger.info('No skeleton data available. Use .get_skeleton() '
                        'to fetch.')
            return 'NA'

    @property
    def partners(self):
        """Get connected partners."""
        if not hasattr(self, '_partners'):
            if not self._lazy_loading:
                return 'NA'
            self._partners = self.get_partners()
        return self._partners

    @property
    def review_status(self):
        """Review status in percent."""
        if not hasattr(self, '_review_status'):
            if not self._lazy_loading:
                return 'NA'
            self._review_status = self.get_review()
        return self._review_status

    @property
    def skeleton_id(self):
        """Skeleton ID."""
        return self.id

    @skeleton_id.setter
    def skeleton_id(self, value):
        if not isinstance(value, (str, numbers.Number)):
            raise TypeError(f'Skeleton ID must be number or string, got {type(value)}')
        self.id = str(value)

    @property
    def tags(self):
        """Node tags."""
        if not hasattr(self, '_tags'):
            if not self._lazy_loading:
                return 'NA'
            self.get_skeleton()
        return self._tags

    @tags.setter
    def tags(self, v):
        if not isinstance(v, dict):
            raise TypeError(f'Expected tags as dict, got {type(v)}')
        self._tags = v

    @property
    def connector_tags(self):
        """Connector tags."""
        if not hasattr(self, '_connector_tags'):
            if not self._lazy_loading:
                return 'NA'
            self.get_connector_tags()
        return self._connector_tags

    @connector_tags.setter
    def connector_tags(self, v):
        if not isinstance(v, dict):
            raise TypeError(f'Expected connector tags as dict, got {type(v)}')
        self._connector_tags = v

    @property
    def type(self) -> str:
        """Return type."""
        return 'CatmaidNeuron'

    def _get_nodes(self):
        # This function redefines how the .nodes property is retrieved
        if not hasattr(self, '_nodes'):
            if not self._lazy_loading:
                return 'NA'
            self.get_skeleton()
        return self._nodes

    def _get_connectors(self):
        # This function redefines how the .connectors property is retrieved
        if not hasattr(self, '_connectors'):
            if not self._lazy_loading:
                return 'NA'
            self.get_skeleton()
        return self._connectors

    def copy(self, deepcopy=False):
        """Return a copy of this neuron."""
        # Use TreeNeuron's copy method
        x = super().copy(deepcopy=deepcopy)

        # Remote instance is excluded from copy -> otherwise we are *silently*
        # creating a new CatmaidInstance that will be identical to the original
        # but will have it's own cache!
        x._remote_instance = self._remote_instance

        return x

    @inject_connection
    def get_skeleton(self, remote_instance=None, **fetch_kwargs):
        """Get/update skeleton data for neuron.

        Parameters
        ----------
        **fetch_kwargs
                    Will be passed to :func:`pymaid.get_neuron` e.g. to get
                    the full node history use::

                        n.get_skeleton(with_history = True)

                    or to get abutting connectors::

                        n.get_skeleton(get_abutting = True)

        See Also
        --------
        :func:`~pymaid.get_neuron`
                    Function called to get skeleton information

        """
        func = cache.never_cache(fetch.get_neuron)
        skeleton = func(self.skeleton_id,
                        remote_instance=remote_instance,
                        return_df=True,
                        fetch_kwargs=fetch_kwargs).iloc[0]

        self._nodes = skeleton.nodes
        self._connectors = skeleton.connectors
        self._tags = skeleton.tags
        self._name = skeleton.neuron_name
        self.date_retrieved = datetime.datetime.now().isoformat()

        # Delete outdated attributes
        self._clear_temp_attr()

        if 'type' not in self.nodes:
            navis.classify_nodes(self)

        return

    @inject_connection
    def get_connector_tags(self, remote_instance=None, **fetch_kwargs):
        """Fetch tags on connectors of a neuron.

        After running, ``neuron.connector_tags`` will store a list of
        tag->connector IDs mappings analogous to ``neuron.tags``.
        """
        logger.debug('Retrieving connector tags...')
        self._connector_tags = fetch.get_connector_tags(self,
                                                        remote_instance=remote_instance)
        return

    def _soma(self):
        """Search for soma and return node ID of soma.

        Uses either a node tag or node radius or a combination of both
        to identify the soma. This is set in the class attributes
        ``soma_detection_radius`` and ``soma_detection_tag``. The default
        values for these are::


                soma_detection_radius = 100
                soma_detection_tag = 'soma'


        Returns
        -------
        node_id
            Returns node ID if soma was found, None if no soma.

        """
        tn = self.nodes

        if self.soma_detection_radius:
            is_large = tn.radius.values * self.units >= self.soma_detection_radius
            tn = tn[is_large]

        if tn.empty:
            return None

        if self.soma_detection_tag:
            if self.soma_detection_tag not in self.tags:
                return None
            else:
                tn = tn[tn.node_id.isin(self.tags[self.soma_detection_tag])]

        if tn.empty:
            return None

        return tn.node_id.values

    @inject_connection
    def get_partners(self, remote_instance=None):
        """Get connectivity table for this neuron."""
        # Get partners
        func = cache.never_cache(fetch.get_partners)
        self._partners = func(self.skeleton_id, remote_instance=remote_instance)

        return self.partners

    @inject_connection
    def get_review(self, remote_instance=None):
        """Get/Update review status for neuron."""
        func = cache.never_cache(fetch.get_review)
        self._review_status = func(self.skeleton_id,
                                   remote_instance=remote_instance).loc[0, 'percent_reviewed']
        return self._review_status

    @inject_connection
    def get_annotations(self, remote_instance=None):
        """Retrieve annotations for neuron."""
        func = cache.never_cache(fetch.get_annotations)
        self._annotations = func(self.skeleton_id,
                                 remote_instance=remote_instance).get(str(self.skeleton_id), [])
        return self.annotations

    @inject_connection
    def get_name(self, remote_instance=None):
        """Retrieve/update name of neuron."""
        func = cache.never_cache(fetch.get_names)
        self._name = func(self.skeleton_id,
                          remote_instance=remote_instance)[str(self.skeleton_id)]
        return self.name

    @inject_connection
    def reload(self, remote_instance=None):
        """Reload neuron from server.

        Currently only updates name, nodes, connectors and tags, not e.g.
        annotations.

        """
        func = cache.never_cache(fetch.get_neuron)
        n = func(self.skeleton_id,
                 remote_instance=remote_instance)
        self.__dict__.update(n.__dict__)

        # Clear temporary attributes
        self._clear_temp_attr()

    def set_remote_instance(self, remote_instance=None, api_token=None,
                            server_url=None, http_user=None, http_password=None):
        """Assign remote_instance to neuron.

        Provide either existing CatmaidInstance OR your credentials.

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        api_token :             str, optional
        http_user :             str, optional
        http_password :         str, optional

        See Also
        --------
        :class:`~pymaid.CatmaidInstance`

        """
        if remote_instance:
            self._remote_instance = remote_instance
        elif server_url:
            self._remote_instance = client.CatmaidInstance(server=server_url,
                                                           api_token=api_token,
                                                           http_user=http_user,
                                                           http_password=http_password
                                                           )
        else:
            raise ValueError('Provide either CatmaidInstance or credentials.')

    def summary(self, add_props=None):
        """Get a summary of this neuron."""
        # Suppress lazy loading during summary
        self._lazy_loading = False
        try:
            res = super().summary()
        except BaseException:
            raise
        finally:
            self._lazy_loading = True

        return res

    def to_dataframe(self):
        """Turn this CatmaidNeuron into a pandas DataFrame with original CATMAID data.

        Returns
        -------
        pandas DataFrame
                neuron_name  skeleton_id  nodes  connectors  tags
            0

        """
        return pd.DataFrame([[self.neuron_name, self.skeleton_id,
                              self.nodes, self.connectors, self.tags]],
                            columns=['neuron_name', 'skeleton_id', 'nodes',
                                     'connectors', 'tags'])


class CatmaidNeuronList(navis.NeuronList):
    """Compilation of :class:`~pymaid.CatmaidNeuron` that allow quick
    access to neurons' attributes/functions. They are designed to work in many
    ways much like a pandas.DataFrames by, for example, supporting ``.iloc[ ]``,
    ``.itertuples()``, ``.empty`` or ``.copy()``.

    CatmaidNeuronList can be minimally constructed from just skeleton IDs.
    Other parameters (nodes, connectors, neuron name, annotations, etc.)
    will then be retrieved from the server 'on-demand'.

    The easiest way to get a CatmaidNeuronList is by using
    :func:`~pymaid.get_neuron` (see examples).

    Attributes
    ----------
    skeleton_id :       array of str
    name :              array of str
    nodes :             ``pandas.DataFrame``
                        Merged node table.
    connectors :        ``pandas.DataFrame``
                        Merged connector table. This also works for
                        `presynapses`, `postsynapses` and `gap_junctions`.
    tags :              np.array of dict
                        Node tags.
    annotations :       np.array of list
    partners :          pd.DataFrame
                        Connectivity table for these neurons.
    graph :             np.array of ``networkx`` graph objects
    igraph :            np.array of ``igraph`` graph objects
    review_status :     np.array of int
    n_connectors :      np.array of int
    n_presynapses :     np.array of int
    n_postsynapses :    np.array of int
    n_branch_nodes :    np.array of int
    n_end_nodes :       np.array of int
    n_open_ends :       np.array of int
    cable_length :      np.array of float
                        Cable lengths in micrometers [um].
    soma :              np.array of node IDs
    root :              np.array of node IDs
    n_cores :           int
                        Number of cores to use. Default ``os.cpu_count()-1``.
    use_threading :     bool (default=True)
                        If True, will use parallel threads. Should be slightly
                        up to a lot faster depending on the numbers of cores.
                        Switch off if you experience performance issues.

    Examples
    --------
    >>> # Initialize with just a Skeleton ID
    >>> nl = pymaid.CatmaidNeuronList([123456, 45677])
    >>> # Retrieve review status from server on-demand
    >>> nl.review_status
    array([90, 23])
    >>> # Initialize with skeleton data
    >>> nl = pymaid.get_neuron([123456, 45677])
    >>> # Get annotations from server
    >>> nl.annotations
    [['annotation1', 'annotation2'], ['annotation3', 'annotation4']]
    >>> Index using node count
    >>> subset = nl[nl.n_nodes > 6000]
    >>> # Get neuron by its skeleton ID
    >>> n = nl.skid[123456]
    >>> # Index by multiple skeleton ID
    >>> subset = nl[['123456', '45677']]
    >>> # Index by neuron name
    >>> subset = nl['name1']
    >>> # Index using annotation
    >>> subset = nl['annotation:uPN right']
    >>> # Concatenate lists
    >>> nl += pymaid.get_neuron([912345])

    """

    def __init__(self, x, make_copy=False, **kwargs):
        if isinstance(x, pd.DataFrame):
            # Break DataFrame into Series
            x = [x.iloc[i] for i in range(x.shape[0])]

        super().__init__(x, make_copy=make_copy, make_using=CatmaidNeuron, **kwargs)

        # Legacy indexer
        self.skid = self.idx

    @property
    def _remote_instance(self):
        """Find a single remote instance for all neurons in this NeuronList."""
        all_instances = [n._remote_instance for n in self.neurons if hasattr(n, '_remote_instance')]

        if len(set(all_instances)) > 1:
            # Note that multiprocessing causes remote_instances to be pickled
            # and thus not be the same anymore
            logger.debug('Neurons are using multiple remote_instances! '
                         'Returning first entry.')
        elif len(set(all_instances)) == 0:
            logger.warning('No remote_instance found. Use '
                           '.set_remote_instance() to assign one to all '
                           'neurons.')
            return None
        return all_instances[0]

    def copy(self, **kwargs):
        """Make copy of this neuronlist."""
        c = super().copy(**kwargs)
        c.skid = c.idx
        return c

    def reload(self):
        """Update neuron skeletons from server."""
        self.get_skeletons(skip_existing=False)

    def get_annotations(self, skip_existing=False):
        """Get/update annotations for neurons."""
        if self.empty:
            logger.warning('Unable to fetch annotations: CatmaidNeuronList is empty.')
            return

        if skip_existing:
            to_update = [n.skeleton_id for n in self.neurons if 'annotations' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            func = cache.never_cache(fetch.get_annotations)
            annotations = func(to_update,
                               remote_instance=self._remote_instance)
            for n in self.neurons:
                n.annotations = annotations.get(str(n.skeleton_id), [])

    def get_names(self, skip_existing=False):
        """Get/update neuron names."""
        if self.empty:
            logger.warning('Unable to fetch names: CatmaidNeuronList is empty.')
            return

        if skip_existing:
            to_update = [n.skeleton_id for n in self.neurons if 'neuron_name' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            func = cache.never_cache(fetch.get_names)
            names = func(to_update, remote_instance=self._remote_instance)
            for n in self.neurons:
                if str(n.skeleton_id) in names:
                    n.neuron_name = names[str(n.skeleton_id)]

    def get_skeletons(self, skip_existing=False):
        """Fill in/update skeleton data of neurons.

        Updates ``.nodes``, ``.connectors``, ``.tags``, ``.date_retrieved`` and
        ``.neuron_name``. Will also generate new graph representation to match
        nodes/connectors.

        """
        if self.empty:
            logger.warning('Unable to fetch skeletons: CatmaidNeuronList is empty.')
            return

        if skip_existing:
            to_update = [n for n in self.neurons if 'nodes' not in n.__dict__]
        else:
            to_update = self.neurons

        if to_update:
            func = cache.never_cache(fetch.get_neuron)
            skdata = func([n.skeleton_id for n in to_update],
                          remote_instance=self._remote_instance,
                          return_df=True).set_index('skeleton_id')
            for n in config.tqdm(to_update,
                                 desc='Processing neurons',
                                 disable=config.pbar_hide,
                                 leave=config.pbar_leave):

                n.nodes = skdata.loc[str(n.skeleton_id), 'nodes']
                n.connectors = skdata.loc[str(n.skeleton_id), 'connectors']
                n.tags = skdata.loc[str(n.skeleton_id), 'tags']
                n.neuron_name = skdata.loc[str(n.skeleton_id), 'neuron_name']
                n.date_retrieved = datetime.datetime.now().isoformat()

                # Delete and update attributes
                n._clear_temp_attr()

    def get_review(self, skip_existing=False):
        """Get/update review status."""
        if self.empty:
            logger.warning('CatmaidNeuronList is empty - no review status to fetch.')
            return

        if skip_existing:
            to_update = [n.skeleton_id for n in self.neurons if 'review_status' not in n.__dict__]
        else:
            to_update = self.skeleton_id.tolist()

        if to_update:
            func = cache.never_cache(fetch.get_review)
            rev = func(to_update, remote_instance=self._remote_instance)
            rev.set_index('skeleton_id', inplace=True)
            for n in self.neurons:
                if str(n.skeleton_id) in rev:
                    n.review_status = rev.loc[str(n.skeleton_id),
                                              'percent_reviewed']
                else:
                    logger.warning('No review status found for neuron '
                                   f'{n.skeleton_id}')

    def set_remote_instance(self, remote_instance=None, server_url=None,
                            api_token=None, http_user=None, http_password=None):
        """Assign remote_instance to all neurons.

        Provide either existing CatmaidInstance OR your credentials.

        Parameters
        ----------
        remote_instance :       pymaid.CatmaidInstance, optional
        server_url :            str, optional
        api_token :             str, optional
        http_user :             str, optional
        http_password :         str, optional

        """
        if not remote_instance and server_url and api_token:
            remote_instance = client.CatmaidInstance(server_url,
                                                     api_token=api_token,
                                                     http_user=http_user,
                                                     http_password=http_password)
        elif not remote_instance:
            raise Exception('Provide either CatmaidInstance or credentials.')

        for n in self.neurons:
            n._remote_instance = remote_instance

    def summary(self, N=None, add_props=[]):
        """Get summary over all neurons in this NeuronList.

        Parameters
        ----------
        N :         int | slice, optional
                    If int, get only first N entries.
        add_props : list, optional
                    Additional properties to add to summary. If attribute not
                    available will return 'NA'.

        Returns
        -------
        pandas DataFrame

        """
        if not self.empty:
            # Fetch a union of all summary props (keep order)
            all_props = [p for l in self.SUMMARY_PROPS for p in l]
            props = np.unique(all_props)
            props = sorted(props, key=lambda x: all_props.index(x))
        else:
            props = []

        # Add skeleton ID
        props = np.insert(props, 2, 'skeleton_id')

        if add_props:
            props = np.append(props, add_props)

        if not isinstance(N, slice):
            N = slice(N)

        try:
            for n in self.neurons:
                n._lazy_loading = False

            summary = pd.DataFrame(data=[[getattr(n, a, 'NA') for a in props]
                                         for n in self.neurons[N]],
                                   columns=props)
        except BaseException:
            raise
        finally:
            for n in self.neurons:
                n._lazy_loading = True

        return summary

    def has_annotation(self, x, intersect=False, partial=False,
                       raise_not_found=True):
        """Filter neurons by their annotations.

        Parameters
        ----------
        x :               str | list of str
                          Annotation(s) to filter for. Use tilde (~) as prefix
                          to look for neurons WITHOUT given annotation(s).
        intersect :       bool, optional
                          If True, neuron must have ALL positive annotations to
                          be included and ALL negative (~) annotations to be
                          excluded. If False, must have at least one positive
                          to be included and one of the negative annotations
                          to be excluded.
        partial :         bool, optional
                          If True, allow partial match of annotation.
        raise_not_found : bool, optional
                          If True, will raise exception if no match is found.
                          If False, will simply return empty list.

        Returns
        -------
        :class:`pymaid.CatmaidNeuronList`
                          Neurons that have given annotation(s).

        Examples
        --------
        >>> # Get neurons that have "test1" annotation
        >>> nl.has_annotation('test1')
        >>> # Get neurons that have either "test1" or "test2"
        >>> nl.has_annotation(['test1', 'test2'])
        >>> # Get neurons that have BOTH "test1" and "test2"
        >>> nl.has_annotation(['test1', 'test2'], intersect=True)
        >>> # Get neurons that have "test1" but NOT "test2"
        >>> nl.has_annotation(['test1', '~test2'])

        """
        # This makes sure nobody accidentally forgets brackets around
        # multiple annotations
        for v in [intersect, partial, raise_not_found]:
            if not isinstance(v, bool):
                raise TypeError('Expected boolean, got {}'.format(type(v)))

        inc, exc = utils._eval_conditions(x)

        if not inc and not exc:
            raise ValueError('Must provide at least a single annotation')

        # Make sure we have annotations to begin with
        self.get_annotations(skip_existing=True)

        selection = []
        for n in self.neurons:
            if inc:
                if not partial:
                    pos = [a in n.annotations for a in inc]
                else:
                    pos = [any(a in b for b in n.annotations) for a in inc]

                # Skip if any positive annotation is missing
                if intersect and not all(pos):
                    continue
                # Skip if none of the positive annotations are there
                elif not intersect and not any(pos):
                    continue

            if exc:
                if not partial:
                    neg = [a in n.annotations for a in exc]
                else:
                    neg = [any(a in b for b in n.annotations) for a in exc]

                # Skip if all negative annotations are present
                if intersect and all(neg):
                    continue
                # Skip if any of the negative annotations are present
                elif not intersect and any(neg):
                    continue

            selection.append(n)

        if not selection and raise_not_found:
            raise ValueError('No neurons with matching annotation(s) found')
        else:
            return CatmaidNeuronList(selection, make_copy=self.copy_on_subset)

    @classmethod
    def from_selection(cls, fname):
        """Generate NeuronList from CATMAID JSON file."""
        # Read data from file
        with open(fname, 'r') as f:
            data = json.load(f)

        # Generate NeuronLost
        nl = cls([e['skeleton_id'] for e in data])

        # Parse colors
        for k, e in enumerate(data):
            nl[k].color = mcl.to_rgb(e['color'])

        return nl

    def to_selection(self, save_to='selection.json'):
        """Generate JSON file which can be loaded in CATMAID selection tables.

        Uses neurons' ``.color`` attribute (if exists).

        Parameters
        ----------
        save_to :   str | None, optional
                    Filename to save selection to. If ``None``, will
                    return the json data instead.

        """
        colors = [getattr(n, 'color', config.default_color) for n in self.neurons]
        data = [dict(skeleton_id=int(n.skeleton_id),
                     color=mcl.to_hex(c, keep_alpha=False),
                     opacity=1) for n, c in zip(self.neurons, colors)]

        if save_to:
            with open(save_to, 'w') as outfile:
                json.dump(data, outfile)

            logger.info('Selection saved as {}.'.format(save_to))
        else:
            return data

    def to_dataframe(self):
        """Turn this CatmaidneuronList into a pandas DataFrame.

        Returns
        -------
        pandas DataFrame
                neuron_name  skeleton_id  nodes  connectors  tags
            0
            1

        """
        return pd.DataFrame([[n.neuron_name, n.skeleton_id, n.nodes,
                              n.connectors, n.tags] for n in self.neurons],
                            columns=['neuron_name', 'skeleton_id', 'nodes',
                                     'connectors', 'tags'])

    def remove_duplicates(self, key='skeleton_id', inplace=False):
        """Remove duplicate neurons from list.

        Parameters
        ----------
        key :       str | list, optional
                    Attribute(s) by which to identify duplicates. In case of
                    multiple, all attributes must match to flag a neuron as
                    duplicate.
        inplace :   bool, optional
                    If False will return a copy of the original with
                    duplicates removed.

        """
        return super().remove_duplicates(key=key, inplace=inplace)


# This is for legacy but will be removed soon-ish
Dotprops = navis.Dotprops
Volume = navis.Volume
