#!/usr/bin/env python
# Copyright (c) 2018-2019.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/hepstats for details.

from setuptools import setup
from setuptools import find_packages
import os

python_requires = "!=2.*, >=3.6"
install_requires = ["scipy", "numpy", "pandas", "asdf"]
setup_requires = ["pytest-runner"]
tests_requires = ["pytest", "zfit"]


def get_version():
    g = {}
    exec(open(os.path.join("hepstats", "version.py")).read(), g)
    return g["__version__"]


setup(
    name='hepstats',
    author='Matthieu Marinangeli',
    author_email='matthieu.marinangeli@cern.ch',
    maintainer='The Scikit-HEP admins',
    maintainer_email='scikit-hep-admins@googlegroups.com',
    version=get_version(),
    description='statistics tools and utilities',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/scikit-hep/hepstats',
    license='BSD 3-Clause License',
    packages=find_packages(),
    test_suite="tests",
    python_requires=python_requires,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_requires,
    keywords=[
            'HEP', 'statistics',
            ],
    classifiers=[
                'Topic :: Scientific/Engineering',
                'Intended Audience :: Science/Research',
                'Intended Audience :: Developers',
                'Operating System :: OS Independent',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Python',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                'Development Status :: 4 - Beta'
                ],
    platforms="Any",
)
