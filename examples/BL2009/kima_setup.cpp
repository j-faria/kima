#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace DNest4;

#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;

// options for the model
// 
RVmodel::RVmodel():fix(false),npmax(1)
{
    // priors as in Balan & Lahav (2009, DOI: 10.1111/j.1365-2966.2008.14385.x)
    Cprior = make_prior<Uniform>(-2000, 2000);
    Jprior = make_prior<ModifiedLogUniform>(1.0, 2000.); // additional white noise, m/s

    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(0.2, 15E3); // days
    conditional->Kprior = make_prior<ModifiedLogUniform>(1.0, 2E3); // m/s
    conditional->eprior = make_prior<Uniform>(0., 1.);
    conditional->phiprior = make_prior<Uniform>(0.0, 2*M_PI);
    conditional->wprior = make_prior<Uniform>(0.0, 2*M_PI);
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "BL2009_dataset1.kms.rv";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "kms", 0);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run();

    return 0;
}
