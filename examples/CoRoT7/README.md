This example uses HARPS radial velocity measurements of the active star CoRoT-7.
It reproduces the analysis done by [Faria et al. (2016)](https://www.aanda.org/articles/aa/abs/2016/04/aa27899-15/aa27899-15.html), 
where we successfully recovered both the orbits of CoRoT-7b and CoRot-7c, and the activity-induced signal.

In the `kima_setup.ccp` file, we set a GP model with hyperpriors and no linear trend.
The number of planets is free, with a uniform prior between 0 and 5.  
(**note:** in Faria et al. (2016) we considered a uniform prior for Np between 0 and 10.)

To compile and run, type

```
make
./run
```

This example takes a considerable time to run because of the GP model.
The analysis in the paper took around 4 days on a standard desktop computer.
