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

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;

RVmodel::RVmodel()
:planets(5, 0, true, RVConditionalPrior())
,mu(Data::get_instance().get_t().size())
,C(Data::get_instance().get_t().size(), Data::get_instance().get_t().size())
{
    auto data = Data::get_instance();
    double ymin = data.get_y_min();
    double ymax = data.get_y_max();
    double topslope = data.topslope();

    // set the prior for the systemic velocity
    Cprior = new Uniform(ymin, ymax);
    // and for the slope parameter
    if(trend)
    	slope_prior = new Uniform(-topslope, topslope);
    
    // save the current model for further analysis
    save_setup();
}

int main(int argc, char** argv)
{
    /* set the RV data file */
    char* datafile = "examples/BL2009/BL2009_dataset1.kms.rv";

    /* load the file (RVs are in km/s) */
    /* don't skip any lines in the header */
	Data::get_instance().load(datafile, "kms", 0);

    // set the sampler and run it!
	Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
	sampler.run();

	return 0;
}
