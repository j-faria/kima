This example uses the radial velocity measurements of the star HD10180, which is
the probable host of seven planets, as found by [Lovis et al.
(2011)](https://doi.org/10.1051/0004-6361/201015577).

To setup the model, we use a sum-of-Keplerians, with no linear trend. We
consider hyperpriors for the orbital periods and semi-amplitudes. The number of
planets is a free, with a uniform prior between 0 and 10.

To compile and run, type

```
kima-run
```

