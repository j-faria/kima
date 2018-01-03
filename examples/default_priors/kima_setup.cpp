#include <iostream>
#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace std;
using namespace DNest4;

/* priors */
#include "default_priors.h"


const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;

// options for the model
RVmodel::RVmodel()
    :planets(5, 1, true, RVConditionalPrior())
    ,mu(Data::get_instance().N())
    ,C(Data::get_instance().N(), Data::get_instance().N())
{}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "dummy_data.rv";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "kms", 0);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run();

    return 0;
}
