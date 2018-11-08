#include "Data.h"
#include "DNest4.h"
#include "RNG.h"

#include "GPRN.h"
#include "Nodes.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Nodes Nodes::instance;
Nodes::Nodes()
{
}

//just to compile for now
double extra_sigma;


Eigen::MatrixXd Nodes::constant(std::vector<double> vec, double extra_sigma)
// vec = [constant]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0];
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::squaredExponential(std::vector<double> vec, double extra_sigma)
// vec = [ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = exp(-0.5 * pow((t[i] - t[j])/vec[0], 2));
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::periodic(std::vector<double> vec, double extra_sigma)
// vec = [ell, P]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = exp(-2 * pow(sin(M_PI*abs(t[i] - t[j])/vec[1])/vec[0], 2));
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::quasiPeriodic(std::vector<double> vec, double extra_sigma)
// vec = [ell_e, P, ell_p]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = exp(-0.5*pow((t[i] - t[j])/vec[0], 2) 
                        -2.0*pow(sin(M_PI*abs(t[i] - t[j])/vec[1])/vec[2], 2));
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::rationalQuadratic(std::vector<double> vec, double extra_sigma)
// vec = [alpha, ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = 1 / pow(1+ pow((t[i] - t[j]), 2)/ (2* pow(vec[0]*vec[1],2)), vec[0]);
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::cosine(std::vector<double> vec, double extra_sigma)
// vec = [P]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = cos(2 * M_PI * abs(t[i] - t[j]) / vec[0]);
            if(i==j)
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::exponential(std::vector<double> vec, double extra_sigma)
// vec = [ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = exp(-abs(t[i] - t[j]) / vec[0]) ;
            if(i==j) 
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}


Eigen::MatrixXd Nodes::matern32(std::vector<double> vec, double extra_sigma)
// vec = [ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = (1.0 + sqrt(3.0)*abs(t[i] - t[j])/vec[0]) *exp(sqrt(3.0)*abs(t[i] - t[j])/ vec[0]);
            if(i==j) 
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}

Eigen::MatrixXd Nodes::matern52(std::vector<double> vec, double extra_sigma)
// vec = [ell]
{
const vector<double> t = Data::get_instance().get_t();
const vector<double> sig = Data::get_instance().get_sig();
const int N = Data::get_instance().get_t().size();
Eigen::MatrixXd C {N, N};
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = (1.0 + (3*sqrt(5)*vec[0]*abs(t[i] - t[j]) + 5*pow(abs(t[i] - t[j]),2))/(3*pow(vec[0],2))) * exp(-sqrt(5.0)*abs(t[i] - t[j])/vec[0]);
            if(i==j) 
                C(i, j) += sig[i]*sig[i] + extra_sigma*extra_sigma;
            else
                C(j, i) = C(i, j);
        }
    }
return C;
}




