#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = true;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;
const bool studentt = false;

RVmodel::RVmodel():fix(false),npmax(10)
{
    // use the default priors
}


int main(int argc, char** argv)
{
    datafile = "HD10180.kms.rv";

    load(datafile, "kms", 1);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
