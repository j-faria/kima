#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;

RVmodel::RVmodel():fix(true),npmax(1)
{}


int main(int argc, char** argv)
{
    datafile = "dummy_data.rv";

    load(datafile, "kms", 0);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(1000);

    return 0;
}
