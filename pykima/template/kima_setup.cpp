#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = false;
const bool multi_instrument = false;

RVmodel::RVmodel():fix(true),npmax(1)
{
    // auto data = Data::get_data();
    // double ymin = data.get_y_min();
    // double tspan = data.get_timespan();

    /// for example, set the prior for the systemic velocity
    // Cprior = make_prior<Uniform>(0, 1);
    /// other priors: Jprior, slope_prior, fiber_offset_prior, offsets_prior,
    ///               log_eta1_prior, eta2_prior, eta3_prior, log_eta4_prior


    /// for example, set the prior for some planet parameters
    // auto c = planets.get_conditional_prior();
    // c->Pprior = make_prior<Gaussian>(10, 1);
    /// other priors: Kprior, eprior, phiprior, wprior
}


int main(int argc, char** argv)
{
    /* only one instrument */
    char* datafile = "your data file here"; // set the RV data file

    /// load the file. the second argument sets the units of the RVs (can be 
    /// either "ms" or "kms") and the third (optional) argument, tells kima not 
    /// to skip any line in the header of the file
    Data::get_instance().load(datafile, "ms", 0);
    

    /* more than one instrument */
    // std::vector<char*> datafiles = {"data file 1", "data file 2"};

    /// load the files, same arguments as for load()
    // Data::get_instance().load_multi(datafiles, "ms", 0);



    /// set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50); // the optional argument to run() sets the print thining

    return 0;
}
