# CHANGELOG

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

### [Unreleased]

#### Added

- ability to analyse data from multiple instruments!!
- new `kima-template` script to create a directory with all files necessary for a run
- new `get_timespan()` method of the `Data` class
- new `load`/`save` methods of `KimaResults` to load and save the model as a pickle file
- ctrl+c copy to all figures
- histogram of slope when present
- save log likelihoods in the file _posterior_sample_info.txt_
- read log likelihoods; new function to get the highest likelihood posterior sample
- kima tips 
- add "pickle" option to `kima-showresults`
- new helpful analysis functions

#### Changed

- fail when _kima_model_setup.txt_ is not present
- corner plot limits are adjusted if limits of the eta3 prior changed



### [1.0]  - 2018-06-19

#### Added 
- `save_setup()` function to save the current model settings for later analysis
- New example with data of 51 Peg
- Getting started guide, going over the 51 Peg example
- READMEs for each example
- New arguments for the `kima-showresults` script: 
  `rv`, `planets`,`orbital`, `gp`, `extra`, `diagnostic`.
  Numbered arguments for specific plots still work
- Titles and more labels to the DNest4 diagnostic plots and other plots

#### Changed
- The API for the RVModel constructor was simplified. One can now do
  `RVmodel::RVmodel():fix(false),npmax(1)` or even just `RVmodel::RVmodel()`
- `KimaResults` now reads a config file (created by `save_setup`)
  to get the model settings; much less error-prone than parsing the cpp files
- Changed the url for the Eigen submodule: https://github.com/eigenteam/eigen-git-mirror.git
- Easier to add new examples to the makefile
- Improve `kima-checkpriors` script
- Move utility functions from `display.py` to `utils.py`



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

