#include "kima.h"

using namespace DNest4;

#include "default_priors.h"
Laplace *Pprior = new Laplace(0., 1.);
Exponential *Kprior = new Exponential(1.); // m/s

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = true;
const bool trend = false;
const bool multi_instrument = false;

// options for the model
RVmodel::RVmodel():fix(false),npmax(10)
{}


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
