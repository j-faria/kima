#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace DNest4;

#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = true; // RVs come from multiple instruments

RVmodel::RVmodel():fix(true),npmax(1)
{
    auto data = Data::get_instance();
    double ymin = data.get_y_min();
    double ymax = data.get_y_max();
    double RVspan = data.get_RV_span();

    // set the prior for the systemic velocity
    Cprior = new Uniform(ymin, ymax);
    // set the prior for the between-instrument offsets
    offsets_prior = new Uniform(-RVspan, RVspan);

    save_setup();
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
    Data::get_instance().load_multi(datafiles, "ms", 2);

    // could also do (but not both!)
    //char* datafile = "HD106252_joined.txt";
    //Data::get_instance().load_multi(datafile, "ms", 1);

    // set the sampler and run it
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(100);

    return 0;
}
