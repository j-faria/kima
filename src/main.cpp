#include <iostream>
#include <typeinfo>
#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"
#include "RVConditionalPrior.h"

using namespace std;
using namespace DNest4;

/* edit from here on */
#include "default_priors.h"


const bool obs_after_HARPS_fibers = false; //if there are observations after the change in HARPS fibers
const bool GP = true; //the model includes a GP component
const bool RN = false; //the model its a GP regression network
const bool hyperpriors = false;
const bool trend = false; //the model includes a linear trend


RVmodel::RVmodel()
:planets(5, 0, true, RVConditionalPrior())
,mu(Data::get_instance().get_t().size())
,C(Data::get_instance().get_t().size(), Data::get_instance().get_t().size())
{
    auto data = Data::get_instance();
    double rvmin = data.get_rv_min();
    double rvmax = data.get_rv_max();
    double topslope = data.topslope();

    // set the prior for the systemic velocity
    Cprior = new Uniform(rvmin, rvmax);
    // and for the slope parameter
    if(trend)
        slope_prior = new Uniform(-topslope, topslope);
    
    // save the current model for further analysis
    save_setup();
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "corot7_harps.rdb";
    //char* datafile = "corot7.txt";

    /* load the file (RVs are in km/s) */
    /* don't skip any lines in the header */
    Data::get_instance().load(datafile, "kms", 2);


    // set the sampler and run it!
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run();


return 0;
}
