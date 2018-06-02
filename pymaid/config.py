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

import logging
logger = logging.getLogger('pymaid')
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(levelname)-5s : %(message)s (%(name)s)')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

# Default settings for progress bars
pbar_hide = False
pbar_leave = True

# Default setting for igraph:
#   If True, will use iGraph if possible
#   If False, will ignore iGraph even if present
# Primarily used for debugging
use_igraph = True


def _type_of_script():
    """ Returns context in which pymaid is run. """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except BaseException:
        return 'terminal'


def is_jupyter():
    """ Test if pymaid is run in a Jupyter notebook."""
    return _type_of_script() == 'jupyter'

# Here, we import tqdm and determine whether we use classic notebook tbars
from tqdm import tqdm_notebook, tnrange
from tqdm import tqdm as tqdm_classic
from tqdm import trange as trange_classic

# Keep this because `tqdm_notebook` is only a wrapper (type "function")
tqdm_class = tqdm_classic

if is_jupyter():
    from tqdm import tqdm_notebook, tnrange
    tqdm = tqdm_notebook
    trange = tnrange
else:
    tqdm = tqdm_classic
    trange = trange_classic
