#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"
#include "GPRN.h"
//#include "Means.h"

using namespace std;
using namespace DNest4;

/* edit from here on */
#include "default_priors.h"
#include "default_GPRN_priors.h"

const bool obs_after_HARPS_fibers = false; //if there are observations after the change in HARPS fibers
const bool GP = true; //the model includes a GP component
const bool RN = true; //the model its a GP regression network
const bool hyperpriors = false;
const bool trend = false; //the model includes a linear trend


RVmodel::RVmodel():fix(true),npmax(0)
{
    auto data = Data::get_instance();
    double rvmin = data.get_rv_min();
    double rvmax = data.get_rv_max();
    double topslope = data.topslope();

    /* set the prior for the systemic velocity */
    Cprior = new Uniform(rvmin, rvmax);
    Jprior = new ModifiedLogUniform(0.01, 1.0);
    /* and for the slope parameter */
    if(trend)
        slope_prior = new Uniform(-topslope, topslope);

    //constant_weight = new Uniform(0, 100);

    /* save the current model for further analysis */
    save_setup();
}


GPRN::GPRN()
{
    /* Node functions of our GPRN */
    node = {"QP"};
    /* Weight funtion of our GPRN */
    weight = {"C"};

    /*  LIST OF AVAILABLE KERNELS
    C   = constant
    SE  = squared exponential
    P   = periodic
    QP  = quasi-periodic
    RQ  = rational quadratic
    COS = cosine
    EXP = exponential
    M32 = matern 3/2
    M52 = matern 5/2
    */
    
    /* Mean functions for our GPRN, except for the RVs*/
    mean = {"None", "None", "None"};
    
    /*  LIST OF AVAILABLE MEANS
    C   = constant
    L   = linear
    P   = parabolic
    CUB = cubic
    SIN = sinusoidal
    None = no mean
    */
}


int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "sampled_data_2.rdb";

    /* load the file (RVs are in km/s) */
    /* don't skip any lines in the header */
    Data::get_instance().load(datafile, "ms", 1);

    /* set the sampler and run it! */
    Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
    sampler.run();


return 0;
}
