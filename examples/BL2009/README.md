In this example, we analyse the simulated datasets created by [Balan & Lahav
(2009)](https://academic.oup.com/mnras/article/394/4/1936/1202223). The folder
contains two datasets with 1 and 2 simulated planets, which were used to test
the [ExoFit package](http://zuserver2.star.ucl.ac.uk/~lahav/exofit.html).

The `kima_setup.cpp` file sets the main options for the model: we use a standard
sum-of-Keplerians model (no GP) and no linear trend.  
The number of planets in the model is fixed to 1 and, by default, we read the
dataset contaning only one planet.

Inside the `RVmodel()` constructor, we define the priors for the model
parameters to be the same as those used by Balan & Lahav (2009) -- see their
[Table 1](https://academic.oup.com/view-large/20641662).

To compile and run, type

```
kima-run
```

With the default options, this example could take around 10 minutes to finish.
