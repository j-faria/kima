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
const bool GP = true;
const bool hyperpriors = false;
const bool trend = false;

// options for the model
// 
RVmodel::RVmodel()
	:objects(5, 5, false, RVConditionalPrior())
	,mu(Data::get_instance().N())
	,C(Data::get_instance().N(), Data::get_instance().N())
{

}


int main(int argc, char** argv)
{
	/* set the RV data file */
	// kima skips the first 2 lines in the header
	// and reads the first 3 columns into time, vrad and svrad
	char* datafile = "corot7.txt";

	Data::get_instance().load(datafile, "ms");
	
	// set the sampler and run it!
	Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
	sampler.run();

	return 0;
}
