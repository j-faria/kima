#include <iostream>
#include "Data.h"
//#include "MultiSite2.h"
#include "Start.h"
#include "MyModel.h"

using namespace std;
using namespace DNest4;

int main(int argc, char** argv)
{

	Data::get_instance().loadnew("/home/joao/phd/free-np-paper/data/HD10180.kms.rv", "kms");

	// start<MyModel>(argc, argv);
	// Sampler<MyModel> sampler = setup<MyModel>(argc, argv);
	// sampler.run();
	return 0;
}
