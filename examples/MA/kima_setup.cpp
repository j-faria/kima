#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace DNest4;

#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool MA = true;
const bool hyperpriors = false;
const bool trend = true;

RVmodel::RVmodel():fix(false),npmax(1)
{
    auto data = Data::get_instance();
    double ymin = data.get_y_min();
    double ymax = data.get_y_max();

    // set the prior for the systemic velocity
    Cprior = new Uniform(ymin, ymax);

    save_setup();
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "ProxCen_harps_D2.rdb";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "ms", 2);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
