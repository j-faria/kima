#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
from setuptools import setup


def fill_make_template():
    thisdir = os.path.dirname(os.path.realpath(__file__))
    maketemplate = os.path.join(thisdir, 'pykima', 'make_template.template')
    maketemplatepy = os.path.join(thisdir, 'pykima', 'make_template.py')
    pycode = open(maketemplate).read().format(kimadir=thisdir)
    print(pycode, file=open(maketemplatepy, 'w'))


fill_make_template()


setup(name='pykima',
      version=open('VERSION').read().strip(),  # same as kima
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
          'loguniform',
          'kumaraswamy',
          'celerite',
          'urepr',
          'psutil',
      ],

      entry_points={
          'console_scripts': [
              'kima-showresults = pykima.showresults:showresults2',
              'kima-checkpriors = pykima.check_priors:main',
              'kima-template = pykima.make_template:main',
              'kima-run = pykima.run:run_local',
              'kima-kill = pykima.run:kill',
              'kima-gui = pykima.kimaGUI:main',
              'kima-report = pykima.report_template:main',
          ]
      },
      package_data={'pykima': ['template/*']},
      include_package_data=True,
      )
