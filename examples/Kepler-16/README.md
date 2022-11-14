This is an Example of the use of the binaries section of KIMA
This uses Kepler-16 data from SOPHIE 

This makes use of the KO mode giving it some extra, binary-specific, attributes
In the kima_setup.ccp file, we set a non-zero prior for the apsidal precession parameter
and turn the relativistic corrections on (but not the tidal correction)

To compile and run, type

```
kima-run
```

With the default options provided, this should take around 2.5 hours to finish.

The results should get a value for the precession parameter of around 280 +/- 90 (arcsec/year)
(in accordance with Baycroft et al 2022)