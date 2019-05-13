#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = true;
const bool MA = false;
const bool hyperpriors = true;
const bool trend = false;
const bool multi_instrument = false;

RVmodel::RVmodel():fix(false),npmax(5)
{
    // use the default priors
}


int main(int argc, char** argv)
{
    char* datafile = "corot7.txt";

    load(datafile, "ms");
    
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
