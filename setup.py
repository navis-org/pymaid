from setuptools import setup, find_packages
import re


VERSIONFILE = "pymaid/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    requirements = [l for l in requirements if not l.startswith('#')]

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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=requirements,
    extras_require={'extras': ['fuzzywuzzy[speedup]~=0.17.0',
                               'ujson~=1.35']},
    python_requires='>=3.6',
    zip_safe=False
)
