#include <iostream>
#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace std;
using namespace DNest4;

/* priors */
//  data-dependent priors should be defined in the RVmodel() 
//  constructor and use Data::get_instance() 
#include "default_GP_priors.h"
#include "default_hyper_priors.h"
#include "default_extra_priors.h"

Laplace *Pprior = new Laplace(0., 1.);
Exponential *Kprior = new Exponential(1.); // m/s

Uniform *eprior = new Uniform(0., 1.);
Uniform *phiprior = new Uniform(0.0, 2*M_PI);
Uniform *wprior = new Uniform(0.0, 2*M_PI);


const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = true;
const bool trend = false;


// options for the model
// 
RVmodel::RVmodel()
    :planets(5, 10, false, RVConditionalPrior())
    ,mu(Data::get_instance().N())
    ,C(Data::get_instance().N(), Data::get_instance().N())
{
    double ymin = Data::get_instance().get_y_min();
    double ymax = Data::get_instance().get_y_max();
    // can now use ymin and ymax in setting prior for vsys
    Cprior = new Uniform(ymin, ymax);

    save_setup();
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    // kima reads the first 3 columns into time, vrad and svrad
    char* datafile = "HD10180.kms.rv";

    // skip the first line in the file
    Data::get_instance().load(datafile, "kms", 1);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run();

    return 0;
}
