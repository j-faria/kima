### RV analysis with DNest4 and GPs

Clone with `git clone --recursive https://github.com/j-faria/kima.git` to get the submodules.

Running `make` will hopefully work. Need a fairly recent version of g++ (one that accepts `-std=c++11`)


### modes

`kima / ananás`:   
This version has a standard sum-of-Keplerians model. It's good, but pretty standard

`kima / maracujá`:   
This version uses GPs as models for stellar activity. It's as good as it gets

`kima / limão`:   
This version will use `celerite` to speed up the GP calculations. It has potential
