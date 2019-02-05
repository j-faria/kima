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

RVmodel::RVmodel():fix(false),npmax(1)
{
    Cprior = make_prior<Uniform>(-10, 10); // m/s
    Jprior = make_prior<ModifiedLogUniform>(1.0, 1000.); // m/s

    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(0.2, 2000); // days
    conditional->Kprior = make_prior<ModifiedLogUniform>(1.0, 1000); // m/s

    conditional->eprior = make_prior<Uniform>(0., 1.);
    conditional->phiprior = make_prior<Uniform>(0.0, 2*M_PI);
    conditional->wprior = make_prior<Uniform>(0.0, 2*M_PI);

    save_setup();
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "51Peg.rv";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "ms", 0);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
