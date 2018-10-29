#include "Data.h"
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

extern vector<double> t; //= Data::get_instance().get_t();
extern vector<double> sig; //= Data::get_instance().get_sig();
extern int N; //= Data::get_instance().get_t().size();

//just to compile for now
extern double extra_sigma;


Eigen::MatrixXd Weights::constant(std::vector<double> vec)
// vec = [weight, constant]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0];
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::squaredExponential(std::vector<double> vec)
// vec = [weigth, ell]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] * exp(-0.5 * pow((t[i] - t[j])/vec[1], 2));
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::periodic(std::vector<double> vec)
// vec = [weight, ell, P]
{
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


Eigen::MatrixXd Weights::quasiPeriodic(std::vector<double> vec)
// vec = [weight, ell_e, P, ell_p]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] * exp(-0.5*pow((t[i] - t[j])/vec[1], 2) 
                        -2.0*pow(sin(M_PI*abs(t[i] - t[j])/vec[2])/vec[3], 2));
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::rationalQuadratic(std::vector<double> vec)
// vec = [weight, alpha, ell]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] / pow(1+ pow((t[i] - t[j]), 2)/ (2* pow(vec[1]*vec[2],2)), vec[1]);
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::cosine(std::vector<double> vec)
// vec = [weight, P]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] * cos(2 * M_PI * abs(t[i] - t[j]) / vec[1]);
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::exponential(std::vector<double> vec)
// vec = [weight, ell]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] * exp(-abs(t[i] - t[j]) / vec[1]) ;
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::matern32(std::vector<double> vec)
// vec = [ell]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] * (1.0 + sqrt(3.0)*abs(t[i] - t[j])/vec[1]) 
                        *exp(sqrt(3.0)*abs(t[i] - t[j])/ vec[1]);
            C(j, i) = C(i, j);
        }
    }
    return C;
}


Eigen::MatrixXd Weights::matern52(std::vector<double> vec)
// vec = [weight, ell]
{
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            C(i, j) = vec[0] * vec[0] *(1.0 + (3*sqrt(5)*vec[1]*abs(t[i] - t[j]) 
                        + 5*pow(abs(t[i] - t[j]),2))/(3*pow(vec[1],2))) * exp(-sqrt(5.0)*abs(t[i] - t[j])/vec[1]);
            C(j, i) = C(i, j);
        }
    }
    return C;
}




