#include "kima.h"

const bool GP = true;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;
const int n_known_object = 0;
const bool studentt = false;

RVmodel::RVmodel():fix(false),npmax(0)
{
    // kernel = celerite;
    eta2_prior = make_prior<LogUniform>(20, 200);
    eta3_prior = make_prior<Uniform>(20, 60);
}


int main(int argc, char** argv)
{
    datafile = "HD128621_rhk.rdb";

    load(datafile, "arb", 2);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
