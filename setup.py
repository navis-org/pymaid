from distutils.core import setup

setup(
    name='pymaid',
    version='0.12',
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
    ],

    install_requires=[
        "igraph",
        "scipy",
        "numpy",
        "matplotlib",
    ],
)