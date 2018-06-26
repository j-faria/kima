#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='pykima',
      version=open('VERSION').read().strip(), # same as kima
      description='Analysis of results from kima',
      author='João Faria',
      author_email='joao.faria@astro.up.pt',
      license='MIT',
      url='https://github.com/j-faria/kima/tree/master/pykima',
      packages=['pykima'],
      install_requires=[
        'numpy',
        'scipy',
        'matplotlib>=1.5.3',
        'corner',
      ],
      entry_points={
        'console_scripts': [
            'kima-showresults = pykima.showresults:showresults',
            'kima-checkpriors = pykima.check_priors:main',
            'kima-template = pykima.make_template:main',
            ]
        },
     )
