from distutils.core import setup
import re


VERSIONFILE="pymaid/_version.py"
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
    long_description=open('README.md').read(),
    url = 'https://github.com/schlegelp/pymaid',
    author='Philipp Schlegel',
    author_email = 'pms70@cam.ac.uk',
    keywords='CATMAID interface neurons',

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        
        'Intended Audience :: CATMAID Users',
        'Topic :: CATMAID :: Interface',

        'License :: GNU GPL V3',
        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    install_requires=[
        "python-igraph>=0.7.1",
        "scipy>=0.18.1",
        "numpy>=1.12.1",
        "matplotlib>=2.0.0",
        "plotly>=2.0.6",
        "pandas>=0.18.1"
    ],
)
