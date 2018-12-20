#include "Data.h"
#include "DNest4.h"
#include "RNG.h"

//#include "GPRN.h"
#include "Means.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

Means Means::instance;
Means::Means()
{
}




double Means::constant(std::vector<double> parameters, double time) const
/* Constant offset mean function given by
    m(t) = parameters[0]*time */
{
    double m = parameters[0] * time;
    return m;
}


double Means::linear(std::vector<double> parameters, double time) const
/* Linear mean function given by
    m(t) = parameters[0]*time + parameters[1]
where parameters[0] = slope and parameters[1] = intercept */
{
    double m = parameters[0] * time + parameters[1];
    return m;
}


double Means::parabolic(std::vector<double> parameters, double time) const
/* Parabolic mean function given by
    m(t) = parameters[0]*time*time + parameters[1]*time + parameters[2]
where parameters[0] = quadratic coefficient, parameters[1] = linear coefficient,
and parameters[2] = free term */
{
    double m = parameters[0]*time*time + parameters[1]*time + parameters[2];
    return m;
}


double Means::cubic(std::vector<double> parameters, double time) const
/* Cubic mean function given by
    m(t) = parameters[0]*time*time*time + parameters[1]*time*time 
            + parameters[2]*time + parameters[3]
where parameters[0] = cubic coefficient, parameters[1] = quadratic coefficient, 
parameters[2] = linear coefficient, and parameters[3] = free term */
{
    double m = parameters[0]*time*time*time + parameters[1]*time*time + 
                    parameters[2]*time + parameters[3];
    return m;
}


double Means::sinusoidal(std::vector<double> parameters, double time) const
/* Sinusoidal mean function given by
    m(t) = parameters[0] * sine(parameters[1]*time + parameters[2])
where parameters[0] = amplitude, parameters[1] = angular frequency, and 
parameters[2] = phase */
{
    double m = parameters[0] * sin(parameters[1]*time + parameters[2]);
    return m;
}

/* To check and calculate the mean values */
double Means::meanCalc(std::string check, std::vector<double> priors, double time)
{
    double mean_value;
    
    if(check == "C")
        mean_value = constant(priors, time);
    if(check == "L")
        mean_value = linear(priors, time);
    if(check == "P")
        mean_value = parabolic(priors, time);
    if(check == "CUB")
        mean_value = cubic(priors, time);
    if(check == "SIN")
        mean_value = sinusoidal(priors, time);
    if(check == "None")
        mean_value = 0;
    return mean_value;
}

