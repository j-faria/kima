#include "Data.h"
#include "DNest4.h"
#include "RNG.h"

#include "GPRN.h"
#include "Weights.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Weights Weights::instance;
Weights::Weights()
{}


Eigen::VectorXd Weights::constant(std::vector<double> vec)
// vec = [weight, constant]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
//cout << "size = " << C.size() << endl;
    for(size_t i=0; i<N; i++)
    {
        //cout << vec[0] << endl;
        C[i] = vec[0];
        //cout << C[i] << endl;
    }
return C;
}


Eigen::VectorXd Weights::squaredExponential(std::vector<double> vec)
// vec = [weigth, ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] * exp(-0.5 * pow((t[i])/vec[1], 2));
    }
return C;
}


Eigen::VectorXd Weights::periodic(std::vector<double> vec)
// vec = [weight, ell, P]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = exp(-2 * pow(sin(M_PI*abs(t[i])/vec[1])/vec[0], 2));
    }
return C;
}


Eigen::VectorXd Weights::quasiPeriodic(std::vector<double> vec)
// vec = [weight, ell_e, P, ell_p]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] * exp(-0.5*pow((t[i])/vec[1], 2) 
                    -2.0*pow(sin(M_PI*abs(t[i])/vec[2])/vec[3], 2));
    }
return C;
}


Eigen::VectorXd Weights::rationalQuadratic(std::vector<double> vec)
// vec = [weight, alpha, ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] / pow(1+ pow((t[i]), 2)/ (2* pow(vec[1]*vec[2],2)), vec[1]);
    }
return C;
}


Eigen::VectorXd Weights::cosine(std::vector<double> vec)
// vec = [weight, P]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] * cos(2 * M_PI * abs(t[i]) / vec[1]);
    }
return C;
}


Eigen::VectorXd Weights::exponential(std::vector<double> vec)
// vec = [weight, ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] * exp(-abs(t[i]) / vec[1]) ;
    }
return C;
}


Eigen::VectorXd Weights::matern32(std::vector<double> vec)
// vec = [ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] * (1.0 + sqrt(3.0)*abs(t[i])/vec[1]) 
                    *exp(sqrt(3.0)*abs(t[i])/ vec[1]);
    }
return C;
}


Eigen::VectorXd Weights::matern52(std::vector<double> vec)
// vec = [weight, ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::VectorXd C {N};
    for(size_t i=0; i<N; i++)
    {
        C[i] = vec[0] * vec[0] *(1.0 + (3*sqrt(5)*vec[1]*abs(t[i]) 
                    + 5*pow(abs(t[i]),2))/(3*pow(vec[1],2))) * exp(-sqrt(5.0)*abs(t[i])/vec[1]);
    }
return C;
}




