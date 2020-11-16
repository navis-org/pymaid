# A collection of tools to remotely access a CATMAID server via its API
#
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


""" This module contains classes and decorators to set up a basic cache
for responses from the CATMAID server.
"""

import pickle
import sys
import datetime
from functools import wraps
from collections import OrderedDict

from . import utils, config

# Set up logging
logger = config.logger


class Cache(OrderedDict):
    """Custom dictionary for handling the caching of request.responses.

    Implements a maximum size [mb] and a time limit [s].

    """
    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", None)
        self.time_limit = kwargs.pop("time_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)

        # This will keep track of all queries to the cache
        self.request_log = []

        self._check_size_limit()

    def __setitem__(self, key, value):
        # Add timestamp to value if not present
        if not isinstance(value, list):
            value = [value, datetime.datetime.now()]
        elif len(value) != 2 or not isinstance(value[1], datetime.datetime):
            value = [value, datetime.datetime.now()]

        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def __getitem__(self, key):
        # Log this request
        self.request_log.append(key)

        value = OrderedDict.__getitem__(self, key)

        if self.time_limit:
            if (datetime.datetime.now() - value[1]).seconds > self.time_limit:
                # Pop response and raise - for this we have to temporarily
                # deactivate the time restrictions. Otherwise we enter a
                # infinite loop because .pop calls __getitem__
                temp = self.time_limit
                self.time_limit = False
                _ = self.pop(key)
                self.time_limit = temp
                raise KeyError('{} exists but is outdated.'.format(key))

        # Extract response and flag as cached
        resp = value[0]
        resp.is_cached = True

        return resp

    def get_cached_url(self, url, future, post=None, files=None):
        """Look for cached url.

        If not cached, will return request futures for fetching the data from
        server.

        """
        try:
            # If response is cached, return a mock future object
            return _mock_future(self.__getitem__((url, str(post))))
        except KeyError:
            if post:
                return future.post(url, data=post, files=files)
            else:
                return future.get(url, params=None, files=files)

    def clear_cached_url(self, url, post=None):
        """Clear cached url for given url."""
        try:
            _ = self.pop((url, str(post)))
        except KeyError:
            pass
        except BaseException:
            raise

    def get(self, key, fallback=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return fallback

    def _check_size_limit(self):
        """Check size limit. Pop items if size limit reached."""
        if self.size_limit is not None:
            while self.size > self.size_limit and len(self) > 0:
                self.popitem(last=False)

    def update_responses(self, urls, posts, responses):
        """Update cached responses.

        Only overwrites responses not already cached.

        """
        if isinstance(posts, type(None)):
            posts = [posts] * len(urls)

        for u, p, r in zip(urls, posts, responses):
            # Update only if not already cached
            if not self.get(u):
                self[(u, str(p))] = r

    def __repr__(self):
        return 'Cache at {} (size limit: {}; time limit[s]: {}). {} items ({}mb).'.format(id(self),
                                                                                          self.size_limit,
                                                                                          self.time_limit,
                                                                                          len(self),
                                                                                          self.size)

    @property
    def size(self):
        """Size [mb] of cached responses."""
        return round(sum([sys.getsizeof(r[0].content) for r in OrderedDict.values(self)]) / 1000 ** 2, 1)

    def save(self, filename='cache.pickle'):
        """ Save cache to file. """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(self, filename):
        """Load cache from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


class _mock_future:
    """Class to emulate futures."""

    def __init__(self, response):
        self.response = response

    def result(self):
        return self.response


def never_cache(function):
    """Decorator to prevent caching of server responses."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Get remote instance either from kwargs or global
        rm = utils._eval_remote_instance(kwargs.get('remote_instance', None))
        # Keep track of old caching settings
        old_value = rm.caching
        # Set caching to False
        rm.caching = False
        # Execute function
        res = function(*args, **kwargs)
        # Set caching to old value
        rm.caching = old_value
        # Return result
        return res
    return wrapper


def wipe_and_retry(function):
    """Decorator that clears the cache of all data requested by a function
    and retries if said function fails on the first run (only if caching is
    enabled)."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Get remote instance either from kwargs or global
        rm = utils._eval_remote_instance(kwargs.get('remote_instance', None))
        try:
            # Keep track of what point in the query log we are
            n_queries = len(rm._cache.request_log)
            old_cache = set(rm._cache.keys())

            # Execute function the first time (make sure no new data is added
            # if exception is raised)
            res = undo_on_error(function)(*args, **kwargs)
        except BaseException:
            # If caching is on, try the function without caching
            if rm.caching:
                # If function failed without even using cached data, raise
                if not set(rm._cache.request_log[n_queries:]) & old_cache:
                    raise

                # Remove requested data before retrying
                for q in rm._cache.request_log[n_queries:]:
                    if q in rm._cache:
                        rm._cache.pop(q)

                logger.info('Failed using cached data. Clearing cache and '
                            'retrying...')

                # Retry function
                res = undo_on_error(function)(*args, **kwargs)

            # If caching is off, raise right away
            else:
                raise
        # Return result
        return res
    return wrapper


def clear_url_and_retry(*to_clear):
    """Decorator that clears specific URL(s) from the cache and retries if a
    function fails on the first run (only if caching is enabled).

    Parameters
    ----------
    *to_clear
                Variable length list of strings. Must be a CatmaidInstance
                URL function (e.g. `._get_annotation_list`). Upon failure,
                these URLs are cleared from the cache.

    """
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            # Get remote instance either from kwargs or global
            rm = utils._eval_remote_instance(kwargs.get('remote_instance',
                                                        None))
            try:
                # Execute function
                res = function(*args, **kwargs)
            except BaseException:
                # If caching is on, try the function without caching
                if rm.caching:
                    logger.info('Failed using cached data. Clearing relevant '
                                'entries and retrying...')
                    for f in to_clear:
                        url = getattr(rm, f)()
                        rm._cache.clear_cached_url(url)
                    try:
                        res = function(*args, **kwargs)
                    except BaseException:
                        raise
                # If caching is off, raise right away
                else:
                    raise
            # Return result
            return res
        return wrapper
    return decorator


def retry_no_caching(function):
    """Decorator that disables caching and retries if a function fails
    on the first run (only if caching is enabled).

    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Get remote instance either from kwargs or global
        rm = utils._eval_remote_instance(kwargs.get('remote_instance', None))
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            # If caching is on, try the function without caching
            if rm.caching:
                logger.info('Failed using cached data. Retrying without '
                            'caching...')
                rm.caching = False
                try:
                    res = function(*args, **kwargs)
                except BaseException:
                    raise
                finally:
                    # Make sure to re-enable caching
                    rm.caching = True
            # If caching is off, raise right away
            else:
                raise
        # Return result
        return res
    return wrapper


def undo_on_error(function):
    """Decorator to catch exceptions and undo caching of (potentially)
    erroneous data."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Get remote instance either from kwargs or global
        rm = utils._eval_remote_instance(kwargs.get('remote_instance', None))

        # Keep existing entries
        # DO NOT remove the list()
        old = list(rm._cache.keys())

        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            # If error was raised, remove new entries from cache
            new_entries = [k for k in rm._cache.keys() if k not in old]
            # Remove new entries from cache
            for k in new_entries:
                _ = rm._cache.pop(k)
            raise
        return res
    return wrapper
