#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;
const int n_known_object = 0;
const bool studentt = false;

RVmodel::RVmodel():fix(false),npmax(1)
{
    // priors as in Balan & Lahav (2009, DOI: 10.1111/j.1365-2966.2008.14385.x)
    Cprior = make_prior<Uniform>(-2000, 2000); // m/s
    Jprior = make_prior<ModifiedLogUniform>(1.0, 2000.); // m/s

    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(0.2, 15E3); // days
    conditional->Kprior = make_prior<ModifiedLogUniform>(1.0, 2E3); // m/s
    conditional->eprior = make_prior<Uniform>(0., 1.);
    conditional->phiprior = make_prior<Uniform>(0.0, 2*M_PI);
    conditional->wprior = make_prior<Uniform>(0.0, 2*M_PI);
}


int main(int argc, char** argv)
{
    datafile = "BL2009_dataset2.kms.rv";

    load(datafile, "kms", 0);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
