#include "kima.h"

const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = false;
const bool known_object = false;
const bool studentt = true;  // use a Student's t-distribution for the likelihood

RVmodel::RVmodel():fix(true),npmax(1)
{
    // all default priors
    // the default prior for the degrees of freedom (nu) of the t-distribution:
    // nu_prior = make_prior<LogUniform>(2, 1000);
}


int main(int argc, char** argv)
{
    datafile = "d1.txt";

    load(datafile, "ms", 2);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
