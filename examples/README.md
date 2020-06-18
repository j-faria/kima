This folder contains examples of how to run **kima**.  

This is also the test suite for the package. 
When you type `make test` in the kima directory, all the examples are executed with a given random seed 
and the output is checked against a previous run. **that's the goal, not yet happening!**

#### Use the examples

The easiest way to use these examples is to go in a given directory and type

```
kima-run
```

But in order to keep things organized, it is better to copy one of these folders
to some other directory. In this way you can use the example as a template and
modify it to your will.  
You only need to change:

- in the `Makefile`; change the `KIMA_DIR` definition to point to the kima directory
   
After this, you should be able to `kima-run` the copied example from the new directory location.

