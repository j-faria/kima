#include "kima.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool MA = false;
const bool hyperpriors = false;
const bool trend = true;
const int degree = 1;
const bool multi_instrument = false;
const bool known_object = true;
const int n_known_object = 2;
const bool studentt = false;

RVmodel::RVmodel():fix(true),npmax(0)
{
    // periods = [20.8851, 42.3633]
    // period_errs = [0.0003, 0.0006]
    // t0s = [2072.7948, 2082.6251]
    // t0_errs = [0.0007, 0.0004]
    // Ks = [5.05069163 5.50983542] m/s

    // priors for first known_object
    KO_Pprior[0] = make_prior<Gaussian>(20.8851, 0.0003);
    KO_Kprior[0] = make_prior<LogUniform>(1, 20);
    // KO_eprior[0] = make_prior<Kumaraswamy>(0.867, 3.03);
    KO_eprior[0] = make_prior<Uniform>(0, 1);
    KO_wprior[0] = make_prior<Uniform>(-PI, PI);
    // KO_phiprior[0] = make_prior<Uniform>(0, 2*PI);
    KO_phiprior[0] = make_prior<Gaussian_from_Tc>(2072.7948, 0.0007, 20.8851, 0.0003);
    
    // // priors for first known_object
    KO_Pprior[1] = make_prior<Gaussian>(42.3633, 0.0006);
    KO_Kprior[1] = make_prior<LogUniform>(1, 20);
    // KO_eprior[1] = make_prior<Kumaraswamy>(0.867, 3.03);
    KO_eprior[1] = make_prior<Uniform>(0, 1);
    // KO_wprior[1] = make_prior<Uniform>(0, 2*PI);
    KO_wprior[1] = make_prior<Uniform>(-PI, PI);
    // KO_phiprior[1] = make_prior<Uniform>(0, 2*PI);
    KO_phiprior[1] = make_prior<Gaussian_from_Tc>(2082.6251, 0.0004, 42.3633, 0.0006);

}


int main(int argc, char** argv)
{
    datafile = "K2-24.rv"; // set the RV data file
    load(datafile, "ms", 1);

    /// set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run(50); // the optional argument to run() sets the print thining

    return 0;
}
