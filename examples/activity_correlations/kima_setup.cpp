#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;

RVmodel::RVmodel():fix(true),npmax(0)
{
    auto data = Data::get_data();
    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(1, 10*data.get_timespan()); // days
    conditional->Kprior = make_prior<ModifiedLogUniform>(1.0, 100); // m/s
    conditional->eprior = make_prior<Kumaraswamy>(0.867, 3.03);
}


int main(int argc, char** argv)
{
    datafile = "dummy2.txt";

    indicators = {"fwhm"};
    load(datafile, "ms", 0, indicators);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
