#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = true;

RVmodel::RVmodel():fix(true),npmax(1)
{
  // use the default priors
}


int main(int argc, char** argv)
{
    // set the RV data files from multiple instruments
    std::vector<char*> datafiles = {"HD106252_ELODIE.txt",
                                     "HD106252_HET.txt",
                                     "HD106252_HJS.txt",
                                     "HD106252_Lick.txt"
                                   };

    // note: all files should have the same structure, and be in the same units
    load_multi(datafiles, "ms", 2);

    // could also do (but not both!)
    //char* datafile = "HD106252_joined.txt";
    // load_multi(datafile, "ms", 1);

    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(100);

    return 0;
}
