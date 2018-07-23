---
title: 'kima: Exoplanet detection in radial velocities'
tags:
  - exoplanets
  - radial-velocity
  - bayesian
  - gaussian-processes
authors:
 - name: J. P. Faria
   orcid: 0000-0002-6728-244X
   affiliation: "1, 2" # (Multiple affiliations must be quoted)
 - name: N. C. Santos
   orcid: 0000
   affiliation: "1, 2"
 - name: P. Figueira
   orcid: 0000
   affiliation: 1
 - name: B. J. Brewer
   orcid: 0000
   affiliation: 3

affiliations:
 - name: Instituto de Astrofísica e Ciências do Espaço, Universidade do Porto, CAUP, Rua das Estrelas, 4150-762, Porto, Portugal
   index: 1
 - name: Departamento de Física e Astronomia, Faculdade de Ciências, Universidade do Porto, Rua do Campo Alegre, 4169-007, Porto, Portugal
   index: 2
 - name: Department of Statistics, The University of Auckland, Private Bag 92019, Auckland 1142, New Zealand
   index: 3

date: 11 November 2017
bibliography: paper.bib
---

# Summary

The radial-velocity (RV) method is one of the most 
successful in the detection of exoplanets.
An orbiting planet induces a gravitational pull on its host star,
which is observed as a periodic variation of the velocity of the star 
in the direction of the line of sight.
By measuring the associated wavelength shifts in stellar spectra, 
a RV timeseries is constructed.
These data provide information about the presence of (one or more) 
planets and allow for the planet mass(es) and 
several orbital parameters to be determined
[e.g. @Fischer2016].

One of the main barriers to the detection of Earth-like planets with RVs
is the intrinsic variations of the star,
which can easily mimic or hide true RV signals of planets.
Gaussian processes (GP) are now seen as a promising tool to model 
the correlated noise that arises from stellar-induced RV variations. 
[e.g. @Haywood2014].


**kima** is a package for the detection and characterization of 
exoplanets using RV data.
It fits a sum of Keplerian curves to a timeseries of RV measurements, 
using the Diffusive Nested Sampling algorithm [@Brewer2011]
to sample from the posterior distribution of the model parameters. 
This algorithm can sample the multimodal and correlated posteriors 
that often arise in this problem [e.g. @Brewer2015].

<!-- Additionally, -->
Unlike similar open-source packages, 
**kima** calculates the fully marginalized likelihood, or _evidence_,
<!--; see @Kass1995)  -->
both for a model with a fixed number $N_p$ of Keplerian signals,
or after marginalising over $N_p$.
For this latter task, $N_p$ itself is a free parameter
and we sample from its posterior distribution
using the trans-dimensional method proposed by @Brewer2014.
<!-- [see also @Brewer2015, @Faria2016] -->
<!--  -->
<!-- This means that **kima** can be used to compare models.  -->
Because **kima** uses the Diffusive Nested Sampling algorithm,
the evidence values are still accurate
when the likelihood function contains phase changes
which would make other algorithms 
(such as thermodynamic integration) unreliable [@Skilling2006].
<!--  -->

Moreover, **kima** can use a GP with a quasi-periodic kernel
as a noise model, to deal with activity-induced signals.
The hyperparameters of the GP are inferred together with the orbital parameters.
Priors for each of the parameters can be easily set by the user,
with a broad choice of standard probability distributions already implemented.


The code is written in C\texttt{++}, but also includes a helper Python package,
`pykima`, which facilitates the analysis of the results.
It depends on the `DNest4` <!-- [@Brewer2016]  -->
and the `Eigen` <!-- [@eigenweb]  -->
packages,
which are included as submodules in the repository.
Other (Python) dependencies are the 
`numpy`, `scipy`, `matplotlib`, and `corner` packages.
Documentation can be found in the main repository,
that also contains a set of examples 
of how use **kima**, serving as the package's test suite.

Initial versions of this package were used in the analysis 
of HARPS RV data of the active planet-host CoRoT-7 [@Faria2016],
in which the orbital parameters of the two exoplanets CoRoT-7b and CoRoT-7c,
as well as the rotation period of the star and the typical lifetime of active regions, 
were inferred from RV observations alone.



## Acknowledgements

This work was supported by Fundação para a Ciência e a Tecnologia (FCT)
through national funds and FEDER through COMPETE2020,
by the grants UID/FIS/04434/2013 & POCI-01-0145-FEDER-007672 
and PTDC/FIS-AST/1526/2014 & POCI-01-0145-FEDER-016886.
<!--  -->
J.P.F. acknowledges support from the fellowship SFRH/BD/93848/2013 
funded by FCT (Portugal) and POPH/FSE (EC).
N.C.S. and P.F. also acknowledge support from FCT through 
Investigador FCT contracts 
IF/00169/2012/CP0150/CT0002 and IF/01037/2013/CP1191/CT0001, respectively.
<!--  -->
B.J.B. acknowledges support from the Marsden Fund of the Royal Society of New Zealand.



![Results from a typical analysis with **kima**. 
  The top panel shows a simulated RV dataset with two injected "planets".
  The black curves represent ten samples from the posterior predictive distribution.
  On the top right, the posterior distribution for the number of planets $N_p$
  clearly favours the detection of the two planets
  (this parameter had a uniform prior between 0 and 2).
  The panels in the bottom row show the posterior distributions for 
  the orbital periods, the semi-amplitudes and the eccentricities of the two signals.](./joss_figure.png)


# References


<!-- Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.
 -->
<!-- citation examples: [see @Santos2011, pp. 33-35; also @Faria2016, ch. 1]. -->

<!-- citation examples: [@Santos2011, pp. 33-35, 38-39 and *passim*]. -->

<!-- citation examples: [@Faria2016; @Santos2011]. -->
