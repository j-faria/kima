#include <iostream>
#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"

using namespace std;
using namespace DNest4;

/* priors */
//  data-dependent priors should be defined in the RVmodel() 
//  constructor and use Data::get_instance() 
#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;


// options for the model
// 
RVmodel::RVmodel()
	:objects(5, 1, false, RVConditionalPrior())
	,mu(Data::get_instance().N())
	,C(Data::get_instance().N(), Data::get_instance().N())
{
	double ymin = Data::get_instance().get_y_min();
    double ymax = Data::get_instance().get_y_max();
    // could now use ymin and ymax in setting prior for vsys
    Cprior = new Uniform(ymin, ymax);
}


int main(int argc, char** argv)
{
	/* set the RV data file */
	char* datafile = "BL2009_dataset1.kms.rv";

	// the third (optional) argument, 
	// tells kima not to skip any line in the header of the file
	Data::get_instance().load(datafile, "kms", 0);
	
	// set the sampler and run it!
	Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
	sampler.run();

	return 0;
}
