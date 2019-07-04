#!/usr/bin/env python
# Copyright (c) 2018-2019.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/scikit-stats for details.

from setuptools import setup
from setuptools import find_packages

install_requires = ["scipy", "numpy", "zfit"]
tests_requires = ["pytest"]

setup(
    name='scikit-stats',
    author='Matthieu Marinangeli',
    author_email='matthieu.marinangeli@cern.ch',
    maintainer='The Scikit-HEP admins',
    maintainer_email='scikit-hep-admins@googlegroups.com',
    version="0.0.0",
    description='statistics tools and utilities',
    long_description=open('README.md').read(),
    url='https://github.com/scikit-hep/statutils',
    license='BSD 3-Clause License',
    packages=find_packages(),
    test_suite="tests",
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
    tests_require=tests_requires,
    keywords=[
            'HEP', 'statistics',
            ],
    classifiers=[]
)
