#!/usr/bin/env python
# Copyright (c) 2018-2019.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/statutils for details.

from setuptools import setup
from setuptools import find_packages

setup(
	name = 'statutils',
	author = 'Matthieu Marinangeli',
	author_email = 'matthieu.marinangeli@cern.ch',
	maintainer = 'The Scikit-HEP admins',
	maintainer_email = 'scikit-hep-admins@googlegroups.com',
	version = "0.0.0",
	description = 'statistics tools and utilities',
	long_description = open('README.md').read(),
	url = 'https://github.com/scikit-hep/statutils',
	license='BSD 3-Clause License',
	packages = find_packages(),
	install_requires = [],
	setup_requires = [], 
	tests_require = [],
	extras_require = [],
	keywords = [
		'HEP', 'statistics',
	],
	classifiers = []
)