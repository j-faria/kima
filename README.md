# bicerin
Working on the RV challenge in Torino


## Milestones

The goal of this project is to analyse the RV challenge data.  
Right now the repository has a template implementation of the RJ-DNest code, which includes a quasi-periodic GP to model the stellar activity RV signal. Ideally this can be easily applied to the RV challenge data, but some practicalities have to be taken into account, mainly linear trends and the fact that each data set has around 500 points. 

- [ ] implement a linear trend
- [ ] implement HODLR inversion (issue [#1](https://github.com/j-faria/bicerin/issues/1))


All data is in the `data` folder and some metadata is in the `docs` and `reports` folders. 

## To run

- Compile with `make`
- Change `OPTIONS`
- Run `./main`
- Analyse with `scripts/showresults.py`
