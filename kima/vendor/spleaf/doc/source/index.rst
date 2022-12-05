
S+LEAF documentation
====================

S+LEAF is an `open-source <https://gitlab.unige.ch/Jean-Baptiste.Delisle/spleaf>`_
software that provides a flexible noise model with fast and scalable methods.
It is largely inspired by the
`celerite <https://github.com/dfm/celerite>`_ / `celerite2 <https://github.com/exoplanet-dev/celerite2>`_
model proposed by [1]_, [3]_.
In particular the modeling of gaussian processes is similar,
and uses the same semiseparable matrices representation as celerite.
S+LEAF extends the celerite model in two ways:

- It allows to account
  for close to diagonal (LEAF) noises such as instrument calibration errors
  (see [2]_ for more details).
- It allows to model simulatenously several time series
  with the same Gaussian processes and their derivatives
  (see [4]_ for more details).

Please cite [2]_ and [4]_ if you use S+LEAF in a publication.

Installation
------------

Using conda
~~~~~~~~~~~

The S+LEAF package can be installed using conda with the following command:

``conda install -c conda-forge spleaf``

Using pip
~~~~~~~~~

It can also be installed using pip with:

``pip install spleaf``

Usage
-----

S+LEAF covariance matrices are generated using the
:doc:`_autosummary/spleaf.cov.Cov` class.
The covariance matrix is modeled as the sum of different components (or terms),
which split into two categories:
noise terms and kernel terms (gaussian processes).
See the :ref:`API reference<api_ref>` for a list of available terms.

The low level implementation of
S+LEAF matrices as defined by [2]_
is available as the :doc:`_autosummary/spleaf.Spleaf` class,
but one typically does not need to directly deal with it.

Examples
--------

.. toctree::
   calib
   multi

.. _api_ref:

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/custom_module.rst
   :recursive:

   spleaf.cov
   spleaf.term
   spleaf

Contribute
----------

Everyone is welcome to open issues and/or contribute code via pull-requests.
A SWITCH edu-ID account is necessary to sign in to `<https://gitlab.unige.ch>`_.
If you don't have an account, you can easily create one at `<https://eduid.ch>`_.
Then you can sign in to `<https://gitlab.unige.ch>`_ by selecting "SWITCH edu-ID" as your organisation.


References
----------

.. [1] `Foreman-Mackey et al., "Fast and Scalable Gaussian Process Modeling with Applications to Astronomical Time Series", 2017 <http://adsabs.harvard.edu/abs/2017AJ....154..220F>`_.
.. [2] `Delisle, J.-B., Hara, N., and Ségransan, D., "Efficient modeling of correlated noise. II. A flexible noise model with fast and scalable methods", 2020 <https://ui.adsabs.harvard.edu/abs/2020A\&A...638A..95D>`_.
.. [3] `Gordon, T. A., Agol, E., Foreman-Mackey, D., "A Fast, Two-dimensional Gaussian Process Method Based on Celerite: Applications to Transiting Exoplanet Discovery and Characterization", 2020 <https://ui.adsabs.harvard.edu/abs/2020AJ....160..240G>`_.
.. [4] `Delisle, J.-B., Unger, N., Hara, N., and Ségransan, D., "Efficient modeling of correlated noise. III. Scalable methods for jointly modeling several observables' time series with Gaussian processes", 2022 <https://ui.adsabs.harvard.edu/abs/2022A\&A...659A.182D>`_.
