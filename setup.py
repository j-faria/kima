#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='pykima',
	  version='0.1',
	  description='Analysis of results from kima',
	  author='João Faria',
	  author_email='joao.faria@astro.up.pt',
	  entry_points={
	  	'console_scripts': [
	  		'kima-showresults = pykima.showresults:showresults',
	  		]
	  	},
	 )