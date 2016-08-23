import sys
import os
import numpy as np 

sys.path.append('/home/joao/Work/OPEN')
from OPEN.classes import rvSeries

data_path = '/home/joao/Work/bicerin/data/'


def get_system(filename=None, number=None, ms=True):
	if filename is not None:
		assert os.path.exists(filename)
		system = rvSeries(filename, skip=2)

	if number is not None:
		filename = os.path.join(data_path, 'PlSy%d.rdb' % number)
		assert os.path.exists(filename)
		system = rvSeries(filename, skip=2)
		system.number = number

	if ms:
		mean_vrad = system.vrad.mean()
		system.vrad = (system.vrad - mean_vrad)*1e3
		system.error *= 1e3

	return system
