#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace std;
using namespace DNest4;

#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;

RVmodel::RVmodel():fix(true),npmax(0)
{
    Cprior = new Uniform(-10, 10); // m/s
    Jprior = new ModifiedLogUniform(1.0, 1000.); // m/s

    Pprior = new LogUniform(0.2, 2000); // days
    Kprior = new ModifiedLogUniform(1.0, 1000); // m/s

    eprior = new Uniform(0., 1.);
    phiprior = new Uniform(0.0, 2*M_PI);
    wprior = new Uniform(0.0, 2*M_PI);

    save_setup();

}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "RVchallenge_system1.rdb";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "kms", 400, {"fwhm"});
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
