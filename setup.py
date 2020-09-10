#!/usr/bin/env python
import os
from setuptools import setup, find_packages

with open(os.path.join('README.md')) as f:
    readme = f.read()

with open(os.path.join('LICENSE')) as f:
    license = f.read()


adcirc_nn_cmds = ['adcirc_nn = adcirc_nn.__main__:main']

setup(
    name='adcirc_nn',
    version='0.1.0',
    description='Software for physics based machine learning with ADCIRC',
    keywords='ADCIRC, pyADCIRC, machine learning, neural networks',
    long_description=readme,
    author=['Gajanan Choudhary','Wei Li'],
    author_email=['gajananchoudhary91@gmail.com', 'wei@oden.utexas.edu'],
    url='https://github.com/gajanan-choudhary/adcirc_nn',
    license=license,
    packages=find_packages(exclude=('tests', 'doc')),
    entry_points={'console_scripts': adcirc_nn_cmds},
)
