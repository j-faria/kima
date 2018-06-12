In this example, we analyse the simulated datasets created by 
[Balan & Lahav (2009)](https://academic.oup.com/mnras/article/394/4/1936/1202223).
The folder contains two datasets with 1 and 2 simulated planets, 
which were used to test the [ExoFit package](http://zuserver2.star.ucl.ac.uk/~lahav/exofit.html).

The `kima_setup.cpp` file sets the main options for the model: 
we use a standard sum-of-Keplerians model (no GP) 
without an offset due to the HARPS change of optical fibers, and no linear trend.  
The number of planets in the model is fixed to 1. 

Inside the RVmodel() constructor, we define the priors for the model parameters, 
the same as those used by Balan & Lahav (2009).

To compile and run, type

```
make
./run 
```