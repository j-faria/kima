#include <iostream>
#include "DNest4.h"
#include "Data.h"
#include "Start.h"
#include "MyModel.h"

using namespace std;
using namespace DNest4;

namespace priors
{
/* priors */
//  data-dependent priors should be defined in the MyModel() constructor

// Uniform Cprior(-1000., 1000.); // systematic velocity
ModifiedJeffreys Jprior(1.0, 99.); // additional white noise, m/s
}


// options for the model
// 
MyModel::MyModel()
	:objects(5, 1, true, MyConditionalPrior())
	,mu(Data::get_instance().N())
	,offsets(0)
	,C(Data::get_instance().N(), Data::get_instance().N())
{

}




int main(int argc, char** argv)
{
	/* set the RV data file */
	// kima skips the first 2 lines in the header
	// and reads the first 3 columns into time, vrad and svrad
	char* datafile = "corot7.txt";

	Data::get_instance().loadnew(datafile);
	start<MyModel>(argc, argv);
	return 0;
}
