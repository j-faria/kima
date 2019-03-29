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

RVmodel::RVmodel():fix(false),npmax(5)
{
    auto data = Data::get_instance();
    Cprior = new Uniform(data.get_y_min(), data.get_y_max()); // m/s
    Jprior = new ModifiedLogUniform(1.0, 100.); // m/s

    Pprior = new LogUniform(0.2, 1000); // days
    Kprior = new ModifiedLogUniform(1.0, 10); // m/s

    eprior = new Uniform(0., 1.);
    phiprior = new Uniform(0.0, 2*M_PI);
    wprior = new Uniform(0.0, 2*M_PI);

    save_setup();

}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "RVchallenge_system2.rdb";
    // char* datafile = "dummy2.txt";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "kms", 2, {"fwhm"});
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
