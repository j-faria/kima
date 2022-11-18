#include "kima.h"


BINARIESmodel::BINARIESmodel():fix(false),npmax(3)
{
    known_object = true;
    n_known_object = 1;

    relativistic_correction = true;
    tidal_correction = false;
    double_lined = false;

    Cprior = make_prior<Uniform>(-34300, -33300);         // m/s
    Jprior = make_prior<ModifiedLogUniform>(0.1, 100.0);  // m/s

    auto conditional = planets.get_conditional_prior();
    conditional->Pprior = make_prior<LogUniform>(160, 4510);    // days
    conditional->Kprior = make_prior<LogUniform>(0.1, 1000.0);  // m/s
    conditional->eprior = make_prior<Kumaraswamy>(0.867, 3.03);
    conditional->phiprior = make_prior<Uniform>(0.0, 2 * PI);
    conditional->wprior = make_prior<Uniform>(0.0, 2 * PI);
    conditional->wdotprior = make_prior<Gaussian>(0.0, 1e-10);

    KO_Pprior[0] = make_prior<Gaussian>(41.07922, 0.01);  // days
    KO_Kprior[0] = make_prior<Gaussian>(13600.0, 50.0);   // m/s
    KO_eprior[0] = make_prior<Gaussian>(0.15944, 0.01);   //
    KO_wprior[0] = make_prior<Uniform>(0.0, 2 * PI);      //
    KO_phiprior[0] = make_prior<Uniform>(0.0, 2 * PI);    //
    KO_wdotprior[0] = make_prior<Gaussian>(0.0, 1000.0);  // arcsec/yr

    star_mass = 0.654;
    star_radius = 0.77;
    binary_mass = 0.1964;
}


int main(int argc, char** argv)
{
    datafile = "Kepler-16.rv";
    load(datafile, "kms", 2);

    Sampler<BINARIESmodel> sampler = setup<BINARIESmodel>(argc, argv);
    sampler.run(100);

    return 0;
}
