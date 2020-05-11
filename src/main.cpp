#include "kima.h"

using namespace std;

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;

RVmodel::RVmodel() : fix(true), npmax(1)
{
    // Cprior = make_prior<Uniform>(0, 1);
    auto c = planets.get_conditional_prior();
    // c->Pprior = make_prior<Gaussian>(0, 1);
}

int main(int argc, char** argv)
{
    /* set the RV data file */
    datafile = "examples/BL2009/BL2009_dataset1.kms.rv";

    /* load the file (RVs are in km/s) */
    /* don't skip any lines in the header */
    Data::get_instance().load(datafile, "kms", 0);

    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
