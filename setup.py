import itertools
from setuptools import setup, find_packages
import re
from pathlib import Path

from extreqs import parse_requirement_files


VERSIONFILE = "pymaid/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

install_requires, extras_require = parse_requirement_files(Path("requirements.txt"))
extras_require["all"] = list(set(itertools.chain.from_iterable(extras_require.values())))

setup(
    name='python-catmaid',
    version=verstr,
    packages=find_packages(),
    license='GNU GPL V3',
    description='Python interface to CATMAID servers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/navis-org/pymaid',
    project_urls={
     "Documentation": "http://pymaid.readthedocs.io",
     "Source": "https://github.com/navis-org/pymaid",
     "Changelog": "https://pymaid.readthedocs.io/en/latest/source/whats_new.html",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='CATMAID interface neuron navis',
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.9',
    zip_safe=False
)
