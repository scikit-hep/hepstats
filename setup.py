#!/usr/bin/env python
# Copyright (c) 2018-2019.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/scikit-stats for details.

from setuptools import setup
from setuptools import find_packages
import os

install_requires = ["scipy", "numpy", "pandas"]
tests_requires = ["pytest", "zfit"]


def get_version():
    g = {}
    exec(open(os.path.join("skstats", "version.py")).read(), g)
    return g["__version__"]


setup(
    name='scikit-stats',
    author='Matthieu Marinangeli',
    author_email='matthieu.marinangeli@cern.ch',
    maintainer='The Scikit-HEP admins',
    maintainer_email='scikit-hep-admins@googlegroups.com',
    version=get_version(),
    description='statistics tools and utilities',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/scikit-hep/scikit-stats',
    license='BSD 3-Clause License',
    packages=find_packages(),
    test_suite="tests",
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
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
                'Development Status :: 5 - Production/Stable',
                ],
    platforms="Any",
)
