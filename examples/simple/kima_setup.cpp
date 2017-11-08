#include <iostream>
#include "DNest4.h"
#include "Data.h"
#include "MyModel.h"

using namespace std;
using namespace DNest4;

/* priors */
//  data-dependent priors should be defined in the MyModel() 
//  constructor and use Data::get_instance() 
#include "default_priors.h"


// options for the model
// 
MyModel::MyModel()
	:objects(5, 1, true, MyConditionalPrior())
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

	Data::get_instance().loadnew(datafile, "kms");
	
	// set the sampler and run it!
	Sampler<MyModel> sampler = setup<MyModel>(argc, argv);
	sampler.run();

	return 0;
}
