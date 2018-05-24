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