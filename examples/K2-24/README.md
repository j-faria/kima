This example demonstrates the new *KO* (known objects) feature in **kima**,
which allows for a number of "background" Keplerians with independent priors for
their orbital parameters. This can be useful in the situation where one or more
transiting planets have been detected, or when one is searching for circumbinary
planets.

We will use the Keck/HIRES radial velocities of K2-24 used by [Petigura et al.
(2016)](https://arxiv.org/abs/1511.04497) to confirm the discovery of two
sub-Saturn planets. There are 32 RVs in the file `K2-24.rv`, obtained directly
from the repository for [`radvel`](https://radvel.readthedocs.io/en/latest/).

Since this has become a fairly standard system, the results from **kima** can be
compared with those obtained with `radvel`
([here](https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting+MCMC.html))
and `exoplanet` ([here](https://docs.exoplanet.codes/en/stable/tutorials/rv/)).


In the `kima_setup.cpp` file, we set

```c++
const bool known_object = true;
const int n_known_object = 2;
```

and then define the priors for the respective orbital parameters. Here, we will
use information on the orbital periods and times of transit derived from the K2
lightcurve.

First, K2-24 b:

```c++
// priors for first known object
KO_Pprior[0] = make_prior<Gaussian>(20.8851, 0.0003); // orbital period
KO_Kprior[0] = make_prior<LogUniform>(1, 20);         // semi-amplitude (m/s)
KO_eprior[0] = make_prior<Uniform>(0, 1);             // eccentricity
KO_wprior[0] = make_prior<Uniform>(-PI, PI);          // argument of periastron
// and finally the mean anomaly at the epoch
KO_phiprior[0] = make_prior<Gaussian_from_Tc>(2072.7948, 0.0007, 20.8851, 0.0003);
```

where we used the special distribution `Gaussian_from_Tc` as a prior for `phi`,
the mean anomaly at the epoch. This is slightly different from the other prior
distributions in that it uses the known values of Tc and P to assign a Gaussian
prior for `phi`, which is the parameter used internally by **kima**. Note that
the prior for `phi` (in radians) could have been set with any other distribution, ignoring this extra information from the transit.

For K2-24 c, the code is very similar:

```c++
// // priors for first known_object
KO_Pprior[1] = make_prior<Gaussian>(42.3633, 0.0006);  // orbital period
KO_Kprior[1] = make_prior<LogUniform>(1, 20);          // semi-amplitude (m/s)
KO_eprior[1] = make_prior<Uniform>(0, 1);              // eccentricity
KO_wprior[1] = make_prior<Uniform>(-PI, PI);           // argument of periastron
KO_phiprior[1] = make_prior<Gaussian_from_Tc>(2082.6251, 0.0004, 42.3633, 0.0006); // mean anomaly at the epoch
```

---

To compile and run the example, type

```
kima-run
```
