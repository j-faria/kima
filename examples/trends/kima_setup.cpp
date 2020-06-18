#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = true;
const int degree = 3;
const bool multi_instrument = false;
const bool known_object = false;
const int n_known_object = 0;
const bool studentt = false;

RVmodel::RVmodel():fix(true),npmax(0)
{
    // all default priors
}


int main(int argc, char** argv)
{
    datafile = "d1.txt";

    load(datafile, "ms", 2);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(100);

    return 0;
}
