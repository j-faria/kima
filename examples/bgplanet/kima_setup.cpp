#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace DNest4;

// #include "default_priors.h"
Uniform *Cprior = new Uniform(-1000, 1000);
ModifiedLogUniform *Jprior = new ModifiedLogUniform(1.0, 99.); // additional white noise, m/s
Uniform *slope_prior = new Uniform(-10, 10);

LogUniform *Pprior = new LogUniform(1.0, 2E2); // days
ModifiedLogUniform *Kprior = new ModifiedLogUniform(1.0, 1E1); // m/s
Uniform *eprior = new Uniform(0., 1.);
Uniform *phiprior = new Uniform(0.0, 2*M_PI);
Uniform *wprior = new Uniform(0.0, 2*M_PI);

/* GP parameters */
Uniform *log_eta1_prior = new Uniform(-5, 5);
Uniform *log_eta2_prior = new Uniform(0, 5);
Uniform *eta3_prior = new Uniform(100., 140.);
Uniform *log_eta4_prior = new Uniform(-1, 1);

/* hyper parameters */
TruncatedCauchy *log_muP_prior = new TruncatedCauchy(log(365), 1., log(365)-21, log(365)+21);
Uniform *wP_prior = new Uniform(0.1, 3.);
TruncatedCauchy *log_muK_prior = new TruncatedCauchy(0., 1., 0.-21, 0.+21);



const bool obs_after_HARPS_fibers = false;
const bool GP = true;
const bool hyperpriors = false;
const bool trend = false;

// background planet
const bool bgplanet = true;

// Gaussian *bgplanet_Pprior = new Gaussian(0.85359165, 5.6e-7); // corot7
Gaussian *bgplanet_Pprior = new Gaussian(1.628930, 3.1e-5); // gj1132
Uniform *bgplanet_Kprior = new Uniform(1.0, 20.0); // m/s
Uniform *bgplanet_eprior = new Uniform(0., 1.);
Uniform *bgplanet_phiprior = new Uniform(0.0, 2*M_PI);
Uniform *bgplanet_wprior = new Uniform(0.0, 2*M_PI);


RVmodel::RVmodel():fix(false),npmax(4)
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
    char* datafile = "GJ1132_harps.txt";

    // the third (optional) argument, 
    // tells kima not to skip any line in the header of the file
    Data::get_instance().load(datafile, "kms", 0);
    
    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50);

    return 0;
}
