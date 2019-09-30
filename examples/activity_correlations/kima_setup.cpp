#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;
const bool known_object = false;

RVmodel::RVmodel():fix(false),npmax(2)
{
    auto data = Data::get_data();
    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(1, 10*data.get_timespan()); // days
    conditional->Kprior = make_prior<ModifiedLogUniform>(1.0, 100); // m/s
    conditional->eprior = make_prior<Kumaraswamy>(0.867, 3.03);
}


int main(int argc, char** argv)
{
    /* Four options to demonstrate the linear correlations with indicators */

    // Load the first file without any correlation (just reads time, RV, error)
    datafile = "simulated1.txt";
    load(datafile, "ms", 4);

    /* Load the first file with linear correlation with indicator "b", in the 4th column */
    // datafile = "simulated1.txt";
    // indicators = {"b"};
    // load(datafile, "ms", 4, indicators);

    /* Load the second file with linear correlation with indicator "b" only */
    // datafile = "simulated2.txt";
    // indicators = {"b"};
    // load(datafile, "ms", 4, indicators);

    /* Load the second file, with linear correlation with "b" and "c", skip 5th column */
    // datafile = "simulated2.txt";
    // indicators = {"b", "", "indc"};
    // load(datafile, "ms", 4, indicators);


    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
