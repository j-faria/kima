import pytest
import numpy as np
import numpy.testing as npt

import os

def example_priors_run():
	""" 
	Run kima with maxlevels=1 to sample from the default priors
	"""
	os.system('make -S' + ' >/dev/null 2>&1') # compile
	os.system('./run' + ' >/dev/null 2>&1') # run

	assert os.path.exists('sample.txt')
	assert os.path.exists('sample_info.txt')


def cleanup():
	os.system('make -S cleanout' + ' >/dev/null 2>&1')
	assert not os.path.exists('sample.txt')


@pytest.mark.slow
def test_priors():
	""" 
	After running kima with maxlevels=1 to sample from the default priors,
	test some basic statistics of the samples
	"""
	topdir = os.getcwd()
	os.chdir('examples/default_priors')
	
	example_priors_run()

	extra_sigma, vsys, P, ecc = np.loadtxt('sample.txt', unpack=True,
		                                   usecols=(0, -1, 4, 7))

	npt.assert_allclose(extra_sigma.min(), 0., rtol=0, atol=1e-1)
	npt.assert_allclose(extra_sigma.max(), 99., rtol=1e-1, atol=0)

	npt.assert_allclose(vsys.min(), -1000., rtol=0, atol=1)
	npt.assert_allclose(vsys.max(), 1000., rtol=0, atol=1)	

	npt.assert_allclose(P.min(), 1., rtol=1e-2, atol=0)
	npt.assert_allclose(P.max(), 1e5, rtol=1e-2, atol=0)
	npt.assert_allclose(P.mean(), (1e5 - 1.)/(np.log(1e5) - np.log(1.)),
	                    rtol=1, atol=0)

	npt.assert_allclose(ecc.min(), 0., rtol=0, atol=1e-4)
	npt.assert_allclose(ecc.max(), 1., rtol=0, atol=1e-4)
	npt.assert_allclose(ecc.mean(), 0.5, rtol=0, atol=1e-1)


	cleanup()
	os.chdir(topdir)
