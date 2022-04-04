#    This script is part of pymaid (http://www.github.com/navis-org/pymaid).
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

import os
import sys
import urllib

import requests
from requests_futures.sessions import FuturesSession
from requests.exceptions import HTTPError

import pandas as pd

from . import utils, config, cache

try:
    import ujson as json
except ImportError:
    import json
except BaseException:
    raise

__all__ = sorted(['CatmaidInstance', 'connect_catmaid'])

# Set up logging
logger = config.logger


def connect_catmaid(**kwargs):
    """Connect to CATMAID server using environmental variables.

    Pulls credentials from environmental variables and feeds them to
    :class:`~pymaid.CatmaidInstance`.
      - ``CATMAID_SERVER`` for ``server`` (required)
      - ``CATMAID_HTTP_USER`` for ``http_user`` (optional)
      - ``CATMAID_HTTP_PASSWORD`` for ``http_password`` (optional)
      - ``CATMAID_API_TOKEN`` for ``api_token`` (optional)

    User ``**kwargs`` to override those.

    Parameters
    ----------
    **kwargs
                Keyword arguments passed to CatmaidInstance.

    Returns
    -------
    CatmaidInstance

    Examples
    --------
    This assumes you have stored credentials as environment variables
    >>> import pymaid
    >>> # Initialize connection with stored credentials
    >>> con1 = pymaid.connect_catmaid()
    >>> # Same server, different project
    >>> con2 = pymaid.connect_catmaid(project_id=2)
    >>> # Different server, same credentials
    >>> con3 = pymaid.connect_catmaid(server="https://other.catmaid.server")

    """
    if 'server' not in kwargs:
        kwargs['server'] = os.environ['CATMAID_SERVER']

    if 'http_user' not in kwargs and 'CATMAID_HTTP_USER' in os.environ:
        kwargs['http_user'] = os.environ['CATMAID_HTTP_USER']

    if 'http_password' not in kwargs and 'CATMAID_HTTP_PASSWORD' in os.environ:
        kwargs['http_password'] = os.environ['CATMAID_HTTP_PASSWORD']

    if 'api_token' not in kwargs and 'CATMAID_API_TOKEN' in os.environ:
        kwargs['api_token'] = os.environ['CATMAID_API_TOKEN']

    return CatmaidInstance(**kwargs)


class CatmaidInstance:
    """Class representing connection to a CATMAID project.

    Holds base url, credentials and project ID. Fetches data and takes care of
    caching results. When initialised, a CatmaidInstance is made the "global"
    default connection for fetching data (see ``set_global`` argument).
    Alternatively, pymaid functions accept a ``remote_instance`` argument that
    lets you pass a CatmaidInstance explicitly.

    Attributes
    ----------
    server :        str
                    The url for a CATMAID server.
    api_token :     str | None
                    API token - see CATMAID `documentation <https://catmaid.
                    readthedocs.io/en/stable/api.html#api-token>`_ on how to
                    get it. If your CATMAID service is public and does not
                    require a token set to ``None``.
    http_user :     str | None, optional
                    Use this if your server requires a basic HTTP authentication
                    before it lets you through to CATMAID.
    http_password : str | None, optional
                    Use this if your server requires a basic HTTP authentication
                    before it lets you through to CATMAID.
    project_id :    int, optional
                    ID of your project. Default = 1.
    max_threads :   int | None
                    Maximum parallel threads to be used. Note that some
                    functions (e.g. :func:`pymaid.get_skid_from_node`)
                    override this parameter. If this is set too high, you
                    might experience connection errors when fetching data.
    set_global :    bool, optional
                    If True, this instance will be set as global (default)
                    CatmaidInstance. This overrides pre-existing global
                    instances.
    caching :       bool, optional
                    If True, will cache server responses for this session.
                    Use :func:`CatmaidInstance.setup_cache` to set size or
                    time limit.

    Examples
    --------
    Initialise a CatmaidInstance. Note that ``HTTP_USER`` and ``HTTP_PASSWORD``
    are only necessary if your server requires HTTP authentification.

    >>> rm = pymaid.CatmaidInstance('https://your.catmaid.server.org/',
    ...                             api_token='TOKEN')
    INFO  : Global CATMAID instance set. (pymaid.fetch)

    If your server requires HTTP authentification, just pass user and password
    as ``http_user`` and ``http_password``:

    >>> rm = pymaid.CatmaidInstance('https://your.catmaid.server.org/',
    ...                             api_token='TOKEN')
    INFO  : Global CATMAID instance set. (pymaid.fetch)


    >>> rm = pymaid.CatmaidInstance('https://your.catmaid.server.org/',
    ...                             api_token='TOKEN')
    INFO  : Global CATMAID instance set. (pymaid.fetch)

    As you instanciate a CatmaidInstance, it is made the default (“global”)
    remote instance and you don’t need to worry about it anymore.

    By default, a CatmaidInstance will refer to the first project on your
    server. To illustrate, let's assume you have two projects and you want to
    fetch data from both:

    >>> p1 = pymaid.CatmaidInstance('https://your.catmaid.server.org/',
    ...                             api_token='TOKEN')
    >>> # Make copy of CatmaidInstance and change project ID
    >>> p2 = p1.copy()
    >>> p2.project_id = 2
    >>> # Fetch a neuron from project 1 and another from project 2 by
    >>> # passing the CatmaidInstance explicitly via `remote_instance`
    >>> n1 = pymaid.get_neuron(16, remote_instance=p1)
    >>> n2 = pymaid.get_neuron(233007, remote_instance=p2)

    Manually make one CatmaidInstance the global one.

    >>> p2.make_global()

    Ordinarily, you would use one of the wrapper functions to fetch data
    from the server (e.g. :func:`pymaid.get_neuron`). If however you want
    to get the **raw data**, here is how:

    >>> # 1. Fetch raw skeleton data for a single neuron
    >>> rm = pymaid.CatmaidInstance('https://your.catmaid.server.org/',
    ...                             api_token='TOKEN')
    >>> skeleton_id = 16
    >>> url = rm._get_compact_details_url(skeleton_id)
    >>> raw_data = rm.fetch(url)
    >>> # 2. Query for neurons matching given criteria using GET request
    >>> GET = {'nodecount_gt': 1000, # min node size
    ...        'created_by': 16}     # user ID
    >>> url = rm._get_list_skeletons_url(**GET)
    >>> raw_data = rm.fetch(url)
    >>> # 3. Fetch contributions using POST request
    >>> url = rm._get_contributions_url()
    >>> POST = {'skids[0]': 16, 'skids[1]': 2333007}
    >>> raw_data = rm.fetch(url, POST)

    """

    def __init__(self, server, api_token, http_user=None, http_password=None,
                 project_id=1, max_threads=10, make_global=True, caching=True):
        # Catch too many backslashes
        if server.endswith('/'):
            server = server[:-1]

        self.server = server
        self.project_id = project_id
        self._http_user = http_user
        self._http_password = http_password
        self._api_token = api_token
        self.__max_threads = max_threads

        # Make some sanity checks - this is also to catch issues with
        # users using the old order of arguments
        if self._api_token and len(self._api_token) < 20:
            logger.warning("The provided API token looks suspiciously "
                           "short: '" + self._api_token + "'\nPlease note "
                           "that the name and order of arguments in "
                           "CatmaidInstance's signature has changed in "
                           "version 1.1.0 and is now `CatmaidInstance(server, "
                           "api_token, http_user=None, http_password=None, ...)`")

        self.caching = caching
        self._cache = cache.Cache(size_limit=128)
        self.pickle_cache = False

        self._session = requests.Session()
        self._future_session = FuturesSession(session=self._session,
                                              max_workers=self.max_threads)

        self.update_credentials()

        if make_global:
            self.make_global()

    def __getstate__(self):
        """Get state (used e.g. for pickling).

        Note that by default we clear the cache on purpose to speed up
        pickling. This is particularly relevant as remote instances are often
        attached to neurons which produces a lot of overhead.

        """
        state = {k: v for k, v in self.__dict__.items() if not callable(v)}

        # Empty cache (or rather replace cache with empty Cache)
        if not getattr(self, 'pickle_cache', False):
            if '_cache' in state:
                state['_cache'] = cache.Cache(size_limit=128)

        return state

    def __setstate__(self, d):
        """Set state (used e.g. for pickling)."""
        self.__dict__ = d

        # There are issues with pickling/copying the FutureSession and we have
        # to effectively re-create it from scratch.
        self._future_session = FuturesSession(session=self._session,
                                              max_workers=self.max_threads)

    def update_credentials(self):
        """Update session headers."""
        if self.http_user and self.http_password:
            self._session.auth = (self.http_user, self.http_password)

        if self.api_token:
            self._session.headers['X-Authorization'] = 'Token ' + self.api_token
        else:
            # If no api token, we have to get a CSRF token instead
            r = self._session.get(self.server)
            r.raise_for_status()
            # Extract token
            key = [k for k in r.cookies.keys() if 'csrf' in k.lower()]

            if not key:
                logger.warning("No CSRF Token found. You won't be able to "
                               "do POST requests to this server.")
            else:
                csrf = r.cookies[key[0]]
                self._session.headers['referer'] = self.server
                self._session.headers['X-CSRFToken'] = csrf

    @property
    def http_user(self):
        return self._http_user

    @http_user.setter
    def http_user(self, v):
        self._http_user = v
        self.update_credentials()

    @property
    def http_password(self):
        return self._http_password

    @http_password.setter
    def http_password(self, v):
        self._http_password = v
        self.update_credentials()

    @property
    def api_token(self):
        return self._api_token

    @api_token.setter
    def api_token(self, v):
        self._api_token = v
        self.update_credentials()

    def setup_cache(self, caching=True, size_limit=128, time_limit=None):
        """Set up a cache for responses from the CATMAID server.

        Parameters
        ----------
        caching :       bool, optional
                        Use to activate/deactivate caching. Deactivating does
                        not clear the existing cache, it's just not used
                        anymore.
        size_limit :    int | None, optional
                        Max amount memory used to cache responses in mb.
                        Set to ``None`` to for no limit.
        time_limit :    int, optional
                        Maximal time in seconds before cached responses are
                        discarded. Set to ``None`` to for no limit.

        """
        self.caching = caching
        self._cache.size_limit = size_limit
        self._cache.time_limit = time_limit

    def clear_cache(self):
        """Clear cache."""
        self._cache = cache.Cache(size_limit=self._cache.size_limit,
                                  time_limit=self._cache.time_limit)
        logger.info('Cached cleared.')

    def load_cache(self, filename):
        """ Load cache from file. """
        self._cache = cache.Cache.load(filename)

        # Deactivate time limit - otherwise might not use data
        self._cache.time_limit = False

        if not self.caching:
            logger.info('Cache loaded but caching is disabled.')

    def save_cache(self, filename='cache.pickle'):
        """ Save cache to file. """
        self._cache.save(filename)

    @property
    def cache_size(self):
        """ Size of cache in mb. """
        return self._cache.size

    @property
    def max_threads(self):
        return self.__max_threads

    @max_threads.setter
    def max_threads(self, v):
        if not isinstance(v, int):
            raise TypeError('max_threads has to be integer')
        if v < 1:
            raise ValueError('max_threads must be > 0')

        self.__max_threads = v
        self._future_session = FuturesSession(session=self._session,
                                              max_workers=self.__max_threads)

    def make_global(self):
        """Sets this variable as global by attaching it as ``sys.module``"""
        sys.modules['remote_instance'] = self
        if self.caching:
            logger.info('Global CATMAID instance set. Caching is ON.')
        else:
            logger.info('Global CATMAID instance set. Caching is OFF.')

    def fetch(self, url, post=None, files=None, on_error='raise', desc='Fetching',
              disable_pbar=False, leave_pbar=True, return_type='json'):
        """Fetch data from given URL(s).

        Parameters
        ----------
        url :           str, list of str
                        URL or list of URLs to fetch data from.
        post :          None | dict | list of dict
                        If provided, will send POST request. Must provide one
                        dictionary for each url.
        files :         str, optional
                        Files to be sent alongside POST request.
        on_error :      "raise" | "log" | "pass"
                        What to do if request returns an error code: raise
                        an exception, log the error but continue or silently
                        pass.
        desc :          str, optional
                        Message for progress bar.
        disable_pbar :  bool, optional
                        If True, won't show progress bar.
        leave_pbar :    bool, optional
                        If True, will not remove pbar after finishing.
        return_type :   "json" | "raw" | "request"
                        Set how to return data::

                          json: return json parsed data (default)
                          raw: return unparsed response content
                          request: return request object

        """
        assert on_error in ['raise', 'log', 'pass']
        assert return_type in ['json', 'raw', 'request']

        # Make sure url and post are iterables
        was_single = isinstance(url, str)
        url = utils._make_iterable(url)
        # Do not use _make_iterable here as it will turn dictionaries into keys
        post = [post] * len(url) if isinstance(post, (type(None), dict, bool)) else post

        # Warn if many individual queries with caching activated
        if len(url) > 1e4 and self.caching:
            logger.warning('You are making a lot of individual queries with '
                           'caching activated. The overhead from managing the '
                           'cache could notably slow down fetching of the '
                           'data. Consider deactivating caching.')

        if len(url) != len(post):
            raise ValueError('POST needs to be provided for each url.')

        # Generate futures
        futures = []
        for u, p in zip(url, post):
            # Try getting url from cache
            if self.caching:
                f = self._cache.get_cached_url(u, self._future_session,
                                               post=p, files=files)
            # If no caching, generate request
            elif not isinstance(p, type(None)):
                f = self._future_session.post(u, data=p, files=files)
            else:
                f = self._future_session.get(u, params=None)
            futures.append(f)

        # Get the responses
        resp = [f.result() for f in config.tqdm(futures,
                                                desc=desc,
                                                disable=(disable_pbar
                                                         or config.pbar_hide
                                                         or len(futures) == 1),
                                                leave=leave_pbar & config.pbar_leave)]

        # Check responses for errors
        errors = []
        details = []
        if on_error in ['raise', 'log']:
            for r in resp:
                # Skip if all is well
                if r.status_code == 200:
                    continue
                # CATMAID internal server errors return useful error messages
                if str(r.status_code).startswith('5'):
                    # Try extracting error:
                    try:
                        msg = r.json().get('error', 'No error message.')
                        det = r.json().get('detail', 'No details provided.')
                    except BaseException:
                        msg = r.reason
                        det = 'No details provided.'
                    errors.append('{} Server Error: {} for url: {}'.format(r.status_code,
                                                                           msg,
                                                                           r.url))
                    details.append(det)
                # Parse all other errors
                else:
                    errors.append('{} Server Error: {} for url: {}'.format(r.status_code,
                                                                           r.reason,
                                                                           r.url))
                    details.append('')

        if errors:
            if on_error == 'raise':
                raise HTTPError('{} errors encountered: {}'.format(len(errors),
                                                                   '\n'.join(errors)))
            else:
                for e, d in zip(errors, details):
                    logger.error(e)
                    logger.debug('{}. Details: {}'.format(e, d))

        # Add new responses to cache
        if self.caching:
            self._cache.update_responses(url, post, resp)

            # Flag if any data is from cache
            if True in [getattr(r, 'is_cached', False) for r in resp]:
                logger.debug('Cached url: {}'.format(url))
                logger.info('Cached data used. Use `pymaid.clear_cache()` '
                            'to clear.')

        # Return requested data
        if return_type.lower() == 'json':
            parsed = []
            for r in resp:
                content = r.content
                if isinstance(content, bytes):
                    content = content.decode()
                try:
                    parsed.append(json.loads(content))
                except BaseException:
                    logger.error('Error decoding json in response:\n{}'.format(content))
                    raise
        elif return_type.lower() == 'raw':
            parsed = [r.content for r in resp]
        else:
            parsed = resp

        return parsed[0] if was_single else parsed

    def make_url(self, *args, **GET):
        """Generates URL.

        Parameters
        ----------
        *args
                    Will be turned into the URL. For example::

                        >>> remote_instance.make_url('skeleton', 'list')
                        'http://my-server.com/skeleton/list'

        **GET
                    Keyword arguments are assumed to be GET request queries
                    and will be encoded in the url. For example::

                        >>> remote_instance.make_url('skeleton', node_gt: 100)
                        'http://my-server.com/skeleton?node_gt=100'

        Returns
        -------
        url :       str

        """
        # Generate the URL
        url = self.server
        for arg in args:
            arg_str = str(arg)
            joiner = '' if url.endswith('/') else '/'
            relative = arg_str[1:] if arg_str.startswith('/') else arg_str
            url = requests.compat.urljoin(url + joiner, relative)
        if GET:
            url += '?{}'.format(urllib.parse.urlencode(GET))
        return url

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self):
        return self.copy()

    def copy(self):
        """Returns a copy of this CatmaidInstance. Does not copy cache."""
        return CatmaidInstance(server=self.server,
                               api_token=self.api_token,
                               http_user=self.http_user,
                               http_password=self.http_password,
                               project_id=self.project_id,
                               max_threads=self.max_threads,
                               make_global=False)

    def __repr__(self):
        s = 'CatmaidInstance at {}.\nServer: {}\nProject: {}\nCaching {}'
        s = s.format(id(self),
                     self.server,
                     self.project_id,
                     self.caching)
        if self.caching:
            s += ' (size limit {}; time limit {})\n'.format(self._cache.size_limit,
                                                            self._cache.time_limit)
            s += 'Cache size: {}'.format(self.cache_size)

        return s

    @property
    def catmaid_version(self):
        """Version of CATMAID your server is running."""
        return self.fetch(self._get_catmaid_version())['SERVER_VERSION']

    @property
    def user_permissions(self):
        """List per-project permissions and groups of user with given token."""
        return self.fetch(self.make_url('permission'))

    @property
    def available_projects(self):
        """List of projects hosted on your server.

        This depends on your user's permission!
        """
        return pd.DataFrame(self.fetch(self._get_projects_url())).sort_values('id')

    @property
    def image_stacks(self):
        """Image stacks available under this project id."""
        stacks = self.fetch(self._get_stacks_url())
        details = self.fetch([self._get_stack_info_url(s['id']) for s in stacks])

        # Add details to stacks
        for s, d in zip(stacks, details):
            s.update(d)

        # Return as DataFrame
        return pd.DataFrame(stacks).set_index('id')

    def _get_catmaid_version(self, **GET):
        """Generate url for retrieving CATMAID server version."""
        return self.make_url('version', **GET)

    def _get_stack_info_url(self, stack_id, **GET):
        """Generate url for retrieving stack infos."""
        return self.make_url(self.project_id, 'stack', stack_id, 'info', **GET)

    def _get_projects_url(self, **GET):
        """Generate URL to get list of available projects on server."""
        return self.make_url('projects', **GET)

    def _get_stacks_url(self, **GET):
        """Generate URL to get list of available image stacks for the project."""
        return self.make_url(self.project_id, 'stacks', **GET)

    def _get_single_node_info_url(self, tn_id, **GET):
        """Generate url for retrieving skeleton info for a single node."""
        return self.make_url(self.project_id, 'treenodes', tn_id, 'info',
                             **GET)

    def _update_node_radii(self, **GET):
        """Generate url for updating node radii (POST)."""
        return self.make_url(self.project_id, 'treenodes', 'radius', **GET)

    def _get_node_labels_url(self, **GET):
        """Generate url for retrieving node infos (POST)."""
        return self.make_url(self.project_id, 'labels-for-nodes', **GET)

    def _get_skeleton_nodes_url(self, skid, **GET):
        """Generate url for retrieving skeleton nodes.

        Does not include info on parents or synapses. Does need post data.

        """
        return self.make_url(self.project_id, 'skeletons', skid,
                             'node-overview', **GET)

    def _get_skeleton_for_3d_viewer_url(self, skid, **GET):
        """Generate url for retrieving all info the 3D viewer gets.

        ATTENTION: this url doesn't work properly anymore as of 07/07/14
        use compact-skeleton instead.

        Does NOT need post data. Format: name, nodes, tags, connectors,
        reviews.

        """
        return self.make_url(self.project_id, 'skeleton', skid, 'compact-json',
                             **GET)

    def _get_add_annotations_url(self, **GET):
        """Generate url to add annotations to skeleton IDs (POST)."""
        return self.make_url(self.project_id, 'annotations', 'add', **GET)

    def _get_remove_annotations_url(self, **GET):
        """Generate url to remove annotations to skeleton IDs (POST)."""
        return self.make_url(self.project_id, 'annotations', 'remove', **GET)

    def _get_connectivity_url(self, **GET):
        """Generate url for retrieving connectivity (POST)."""
        return self.make_url(self.project_id, 'skeletons', 'connectivity',
                             **GET)

    def _get_connector_links_url(self, **GET):
        """Generate url to list of connectors.

        Either pre- or postsynaptic to a set of neurons - GET request Format::

            {'links': [skeleton_id, connector_id, x,y,z, S(?), confidence,
                       creator, node_id, creation_date ], 'tags':[] }

        """
        return self.make_url(self.project_id, 'connectors', 'links/', **GET)

    def _get_connectors_url(self, **GET):
        """Generate url to to retrieve list of connectors (POST)."""
        return self.make_url(self.project_id, 'connectors/', **GET)

    def _get_connector_types_url(self, **GET):
        """Generate URL to retrieve list of connectors (POST)."""
        return self.make_url(self.project_id, 'connectors/types/', **GET)

    def _get_connectors_between_url(self, **GET):
        """Generate url to retrieve connectors linking sets of neurons."""
        return self.make_url(self.project_id, 'connector', 'list',
                             'many_to_many', **GET)

    def _get_connector_details_url(self, **GET):
        """Generate url for retrieving info connectors (POST)."""
        return self.make_url(self.project_id, 'connector', 'skeletons', **GET)

    def _get_neuronnames(self, **GET):
        """Generate url for names for a list of skeleton ids (POST)."""
        return self.make_url(self.project_id, 'skeleton', 'neuronnames', **GET)

    def _get_list_skeletons_url(self, **GET):
        """Generate url to get neuron names (GET)."""
        return self.make_url(self.project_id, 'skeletons/', **GET)

    def _get_graph_dps_url(self, **GET):
        """Generate url for getting connections between source and targets."""
        return self.make_url(self.project_id, 'graph', 'dps', **GET)

    def _get_completed_connector_links(self, **GET):
        """Generate url to get completed connector links by given user (GET).
        """
        return self.make_url(self.project_id, 'connector', 'list', **GET)

    def _get_user_list_url(self, **GET):
        """Generate url to get list of users."""
        return self.make_url('user-list', **GET)

    def _get_single_neuronname_url(self, skid, **GET):
        """Generate url to get a SINGLE neuron."""
        return self.make_url(self.project_id, 'skeleton', skid, 'neuronname',
                             **GET)

    def _get_review_status_url(self, **GET):
        """Generate URL to get review status."""
        return self.make_url(self.project_id, 'skeletons', 'review-status',
                             **GET)

    def _get_review_details_url(self, skid, **GET):
        """Generate url to retrieve review status for individual nodes."""
        return self.make_url(self.project_id, 'skeletons', skid, 'review',
                             **GET)

    def _get_annotation_table_url(self, **GET):
        """Generate url to get annotations for given neuron (POST)."""
        return self.make_url(self.project_id, 'annotations', 'table-list',
                             **GET)

    def _get_intersects(self, vol_id, x, y, z, **GET):
        """Generate to test if point intersects with volume."""
        GET.update({'x': x, 'y': y, 'z': z})
        return self.make_url(self.project_id, 'volumes', vol_id, 'intersect',
                             **GET)

    def _get_volumes(self, **GET):
        """Generate url to list of all volumes in project."""
        return self.make_url(self.project_id, 'volumes/', **GET)

    def _get_volume_details(self, volume_id, **GET):
        """Generate url to get details on a given volume."""
        return self.make_url(self.project_id, 'volumes', volume_id, **GET)

    def _get_annotations_for_skid_list(self, **GET):
        """Generate url to get annotations for given neuron (POST)."""
        return self.make_url(self.project_id, 'skeleton', 'annotationlist',
                             **GET)

    def _get_logs_url(self, **GET):
        """Generate url to get logs (POST)."""
        return self.make_url(self.project_id, 'logs', 'list', **GET)

    def _get_transactions_url(self, **GET):
        """Generate url to get transactions (GET)."""
        return self.make_url(self.project_id, 'transactions/', **GET)

    def _get_annotation_list(self, **GET):
        """Generate url to retrieve list of all annotations."""
        return self.make_url(self.project_id, 'annotations/', **GET)

    def _get_contributions_url(self, **GET):
        """Generate url to retrieve contributor statistics."""
        return self.make_url(self.project_id, 'skeleton',
                             'contributor_statistics_multiple', **GET)

    def _get_annotated_url(self, **GET):
        """Generate url to retrieve annotated neurons (POST)."""
        return self.make_url(self.project_id, 'annotations', 'query-targets',
                             **GET)

    def _get_skid_from_tnid(self, node_id, **GET):
        """Generate url to retrieve the skeleton ID to a single node ID.
        """
        return self.make_url(self.project_id, 'skeleton', 'node', node_id,
                             'node_count', **GET)

    def _get_node_list_url(self, **GET):
        """Generate url for retrieving list of nodes (POST)."""
        return self.make_url(self.project_id, 'node', 'list', **GET)

    def _get_node_info_url(self, **GET):
        """Generate url for retrieving user info on nodes (POST)."""
        return self.make_url(self.project_id, 'node', 'user-info', **GET)

    def _node_add_tag_url(self, node_id, **GET):
        """Generate url for adding labels (tags) to a given node (POST)."""
        return self.make_url(self.project_id, 'label', 'treenode', node_id,
                             'update', **GET)

    def _delete_neuron_url(self, neuron_id, **GET):
        """Generate url to delete a neuron."""
        return self.make_url(self.project_id, 'neuron', neuron_id, 'delete',
                             **GET)

    def _delete_node_url(self, **GET):
        """Generate url for deleting nodes."""
        return self.make_url(self.project_id, 'treenode', 'delete', **GET)

    def _delete_connector_url(self, **GET):
        """Generate url for deleting connectors."""
        return self.make_url(self.project_id, 'connector', 'delete', **GET)

    def _connector_add_tag_url(self, node_id, **GET):
        """Generate url for adding labels (tags) to a node (POST)."""
        return self.make_url(self.project_id, 'label', 'connector',
                             node_id, 'update', **GET)

    def _get_compact_skeleton_url(self, skid, connector_flag=1, tag_flag=1,
                                  **GET):
        """Generate url to retrieve all info the 3D viewer gets (GET).

        Deprecated but kept for backwards compability!

        """
        return self.make_url(self.project_id, skid, connector_flag, tag_flag,
                             'compact-skeleton', **GET)

    def _get_compact_details_url(self, skid, **GET):
        """Generate url to get skeleton info.

        Similar to compact-skeleton but if 'with_history':True is passed
        as GET request, returned data will include all positions a
        nodes/connector has ever occupied plus the creation time and last
        modified.

        """
        return self.make_url(self.project_id, 'skeletons', skid,
                             'compact-detail', **GET)

    def _get_compact_arbor_url(self, skid, nodes_flag=1, connector_flag=1,
                               tag_flag=1, **GET):
        """Generate url to get skeleton info.

        The difference between this function and get_compact_skeleton is
        that the connectors contain the whole chain from the skeleton of
        interest to the partner skeleton: contains [node_id,
        confidence_to_connector, connector_id, confidence_from_connector,
        connected_node_id, connected_skeleton_id, relation1, relation2]
        relation1 = 1 means presynaptic (this neuron is upstream), 0 means
        postsynaptic (this neuron is downstream)

        """
        return self.make_url(self.project_id, skid, nodes_flag,
                             connector_flag, tag_flag, 'compact-arbor', **GET)

    def _get_edges_url(self, **GET):
        """Generate url for retrieving edges between neurons (POST)."""
        return self.make_url(self.project_id, 'skeletons',
                             'confidence-compartment-subgraph', **GET)

    def _get_skeletons_from_neuron_id(self, neuron_id, **GET):
        """Generate url to get all skeletons of a given neuron."""
        return self.make_url(self.project_id, 'neuron', neuron_id,
                             'get-all-skeletons', **GET)

    def _get_history_url(self, **GET):
        """Generate url to get user history."""
        return self.make_url(self.project_id, 'stats', 'user-history', **GET)

    def _get_stats_node_count(self, **GET):
        """Generate url to get nodecounts per user."""
        return self.make_url(self.project_id, 'stats', 'nodecount', **GET)

    def _rename_neuron_url(self, neuron_id, **GET):
        """Generate url to rename a single neuron (POST)."""
        return self.make_url(self.project_id, 'neurons', neuron_id, 'rename',
                             **GET)

    def _get_label_list_url(self, **GET):
        """Generte url to get a list of all labels."""
        return self.make_url(self.project_id, 'labels', 'stats', **GET)

    def _get_circles_of_hell_url(self, **GET):
        """Generate url to to get n-th order partners for a set of neurons."""
        return self.make_url(self.project_id, 'graph', 'circlesofhell', **GET)

    def _get_node_table_url(self, **GET):
        """Generate url to get node table (POST)."""
        return self.make_url(self.project_id, 'treenodes', 'compact-detail',
                             **GET)

    def _get_node_location_url(self, **GET):
        """Generate url to get node location (POST)."""
        return self.make_url(self.project_id, 'nodes', 'location', **GET)

    def _import_skeleton_url(self, **GET):
        """Generate url to import skeleton into Catmaid Instance (POST)."""
        return self.make_url(self.project_id, 'skeletons', 'import', **GET)

    def _get_skeletons_in_bbox(self, **GET):
        """Generate url to get list of skeleton in bounding box (POST)."""
        return self.make_url(self.project_id, 'skeletons', 'in-bounding-box',
                             **GET)

    def _get_connector_in_bbox_url(self, **GET):
        """Generate url for retrieving list of connectors in bounding box."""
        return self.make_url(self.project_id, 'connectors', 'in-bounding-box', **GET)

    def _get_neuron_ids_url(self, **GET):
        """Generate url for retrieving neuron IDs from skeleton IDs."""
        return self.make_url(self.project_id, 'neurons', 'from-models', **GET)

    def _upload_volume_url(self, **GET):
        """Generate url for uploading volumes."""
        return self.make_url(self.project_id, 'volumes', 'add', **GET)

    def _create_link_url(self, **GET):
        """Generate url for creating connector links."""
        return self.make_url(self.project_id, 'link', 'create', **GET)

    def _create_connector_url(self, **GET):
        """Generate url for creating connectors."""
        return self.make_url(self.project_id, 'connector', 'create', **GET)

    def _join_skeletons_url(self, **GET):
        """Generate url for joining skeletons."""
        return self.make_url(self.project_id, 'skeleton', 'join', **GET)

    def _get_login_info_url(self, **GET):
        """Generate url for getting login information for self."""
        return self.make_url('accounts', 'login', **GET)

    def _update_node_url(self, **GET):
        """Generate url for updating node locations."""
        return self.make_url(self.project_id, 'node', 'update', **GET)

    def _reroot_skeleton_url(self, **GET):
        """Generate url for rerooting skeletons."""
        return self.make_url(self.project_id, 'skeleton', 'reroot', **GET)

    def _create_node_url(self, **GET):
        """Generate url for generating nodes."""
        return self.make_url(self.project_id, 'treenode', 'create', **GET)

    def _get_neuron_cable_url(self, **GET):
        """Generate url for fetching neuron cable lengths."""
        return self.make_url(self.project_id, 'skeletons', 'cable-length', **GET)

    def _update_node_confidence_url(self, node_id, **GET):
        """Generate url for fetching neuron cable lengths."""
        return self.make_url(self.project_id, 'treenodes', node_id, 'confidence', **GET)

    def _get_connectivity_counts_url(self, **GET):
        """Generate url for fetching connectivity counts (POST)."""
        return self.make_url(self.project_id, 'skeletons', 'connectivity-counts', **GET)

    def _get_connectivity_matrix_url(self, **GET):
        """Generate url for fetching adjacency matrices (POST)."""
        return self.make_url(self.project_id, 'skeleton', 'connectivity_matrix', **GET)

    def _get_import_info_url(self, **GET):
        """Generate url for fetching imported nodes for a given skeleton."""
        return self.make_url(self.project_id, 'skeletons', 'import-info', **GET)

    def _get_skeleton_origin_url(self, **GET):
        """Generate url for fetching origin info for given skeleton."""
        return self.make_url(self.project_id, 'skeletons', 'origin', **GET)

    def _get_skeleton_by_origin_url(self, **GET):
        """Generate url for fetching skeleton by their origin."""
        return self.make_url(self.project_id, 'skeletons', 'from-origin', **GET)

    def _get_sampler_list_url(self, **GET):
        """Generate url for fetching list of reconstruction samplers."""
        return self.make_url(self.project_id, 'samplers', **GET)

    def _get_sampler_domains_url(self, sampler, **GET):
        """Generate url for fetching domains for given sampler."""
        return self.make_url(self.project_id, 'samplers', sampler, 'domains', **GET)

    def _get_sampler_counts_url(self, **GET):
        """Generate url for fetching domains for given sampler."""
        return self.make_url(self.project_id, 'skeletons', 'sampler-count', **GET)

    def _get_skeleton_change_url(self, **GET):
        """Generate url for fetching skeleton change history."""
        return self.make_url(self.project_id, 'skeletons', 'change-history', **GET)

    def _set_nodes_reviewed_url(self, node_id, **GET):
        """Generate url for fetching skeleton change history."""
        return self.make_url(self.project_id, 'node', node_id, 'reviewed', **GET)
