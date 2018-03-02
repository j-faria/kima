# CHANGELOG

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

We're still in 0.x.x so anything may change at any time. 
The API should not be considered stable.

## [Unreleased]


### [0.1.2] - 2018-03-02
#### Added
- Changelog!
- New methods `data.get_RV_std` and `data.get_RV_var`
  for standard deviation and variance of the RVs
- Reference time for slope is now middle of times t[0]+0.5*timespan  
  Thus in models with trend, Vsys is the systemic velocity a this time.
  A prior for Vsys between RVmin and RVmax is therefore appropriate.
  

#### Changed
- The prior for the fiber offset is now Gaussian(15, 3),
  based on a fit to the "standard" stars of Lo Curto et al. (2015)

#### Fixed
- Nasty bug in the perturb method. if(slope) should be if(trend)





### [0.0.0]

#### Added
#### Changed
#### Removed
#### Fixed

