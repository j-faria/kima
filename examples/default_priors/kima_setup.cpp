#include "kima.h"

const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;
const bool studentt = false;

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
