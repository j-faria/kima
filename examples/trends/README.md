This example showcases the linear, quadratic, and cubic trends in **kima**
using a simulated dataset with a known (quadratic) trend.

The `kima_setup.cpp` file sets the options for the model. Everything is default
except for 

```c++
const bool trend = true;
const int degree = 3;
```

which allows for up to a cubic trend in the model. The number of planets is
fixed to 1, and all priors take their default values.

The simulated parameters are recovered correctly

![coefficients](coefficients.png)


---

To compile and run, type

```
kima-run
```

With the default options, this example takes only a few seconds to finish.
