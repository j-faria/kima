#include "kima.h"

const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const int degree = 0;
const bool multi_instrument = true;
const bool known_object = false;
const int n_known_object = 0;
const bool studentt = false;

RVmodel::RVmodel():fix(false),npmax(1)
{
  // use the default priors
}


int main(int argc, char** argv)
{
    // set the RV data files from multiple instruments
    datafiles = {"HD106252_ELODIE.txt",
                 "HD106252_HET.txt",
                 "HD106252_HJS.txt",
                 "HD106252_Lick.txt"
                };

    // note: all files should have the same structure, and be in the same units
    load_multi(datafiles, "ms", 2);

    // could also do (but not both!)
    // datafile = "HD106252_joined.txt";
    // load_multi(datafile, "ms", 1);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(100);

    return 0;
}
