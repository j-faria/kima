#include <iostream>
#include <typeinfo>
#include "DNest4.h"
#include "Data.h"
// #include "Start.h"
#include "MyModel.h"

using namespace std;
using namespace DNest4;

Uniform *Cprior = new Uniform(-10, 10);
// Uniform *Cprior;

ModifiedJeffreys *Jprior = new ModifiedJeffreys(1.0, 99.); // additional white noise, m/s

Jeffreys *Pprior = new Jeffreys(1.0, 1E7); // days
ModifiedJeffreys *Kprior = new ModifiedJeffreys(10.0, 1E4); // m/s

DNest4::Uniform *eprior = new DNest4::Uniform(0., 1.);
DNest4::Uniform *phiprior = new DNest4::Uniform(0.0, 2*M_PI);
DNest4::Uniform *wprior = new DNest4::Uniform(0.0, 2*M_PI);

MyModel::MyModel()
:objects(5, 1, true, MyConditionalPrior())
,mu(Data::get_instance().get_t().size())
,C(Data::get_instance().get_t().size(), Data::get_instance().get_t().size())
{
	
    double ymin = Data::get_instance().get_y_min();
    double ymax = Data::get_instance().get_y_max();
    //double ptp = ymax - ymin;
    // Cprior = new Uniform(ymin, ymax);
}

int main(int argc, char** argv)
{

	Data::get_instance().loadnew("data/test_offsets.txt", "kms");

	// Uniform *Cprior = new Uniform(-10, 10);
	//start<MyModel>(argc, argv);
	Sampler<MyModel> sampler = setup<MyModel>(argc, argv);

	sampler.run();
	return 0;
}
