#include <iostream>
#include <typeinfo>
#include "DNest4.h"
#include "Data.h"
#include "RVmodel.h"
#include "RVConditionalPrior.h"

using namespace std;
using namespace DNest4;

#include "default_priors.h"

const bool obs_after_HARPS_fibers = false;
const bool GP = false;
const bool hyperpriors = false;
const bool trend = false;

RVmodel::RVmodel()
:objects(5, 1, true, RVConditionalPrior())
,mu(Data::get_instance().get_t().size())
,C(Data::get_instance().get_t().size(), Data::get_instance().get_t().size())
{
    double ymin = Data::get_instance().get_y_min();
    double ymax = Data::get_instance().get_y_max();
    double topslope = Data::get_instance().topslope();
    Cprior = new Uniform(ymin, ymax);
    if(trend)
    	slope_prior = new Uniform(-topslope, topslope);
}

int main(int argc, char** argv)
{
	Data::get_instance().load("data/data_to_test_priors.txt", "kms");

	//start<RVmodel>(argc, argv);
	Sampler<RVmodel> sampler = setup<RVmodel>(argc, argv);
	sampler.run();
	return 0;
}
