try:
    from setuptools import setup
except:
    print('Setuptools not found - falling back to distutils')
    from distutils.core import setup

import re


VERSIONFILE="pymaid/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name='pymaid',
    version=verstr,
    packages=['pymaid',],
    license='GNU GPL V3',
    description='Python interface with CATMAID',
    long_description=open('README.md').read(),
    url = 'https://github.com/schlegelp/pymaid',
    author='Philipp Schlegel',
    author_email = 'pms70@cam.ac.uk',
    keywords='CATMAID interface neuron blender3d',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    #setup_requires=[ 'numpy>=1.12.1' ], #pyoctree's install requires numpy to install but didn't specify that, so we have to do that for it

    install_requires=[
        "networkx>=2.1",
        "scipy>=1.0.0",
        "numpy>=1.13.0",
        "matplotlib>=2.2.0",
        "plotly>=2.4.0",
        "pandas>=0.22.0",
        "vispy>=0.5.3",
        "tqdm>=4.14.0",
        "pyqt5",
        "pypng"
        #"rpy2>=2.8.5", #This throws an error when no R is installed on the system
    ],

    python_requires='>=3.4',
)
