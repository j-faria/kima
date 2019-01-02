#ifndef MEANS_H
#define MEANS_H

#include "Data.h"
#include "DNest4.h"
#include "RNG.h"
#include "RVmodel.h"

#include <Eigen/Core>
#include <Eigen/Dense>


class Means
{
    public:
        Means();
        /* constant mean function */
        double constant(std::vector<double> parameters, double time);
        /* linear mean function */
        double linear(std::vector<double> parameters, double time);
        /* parabolic mean funtion */
        double parabolic(std::vector<double> parameters, double time);
        /* cubic mean function */
        double cubic(std::vector<double> parameters, double time);
        /* sinusoidal mean function*/
        double sinusoidal(std::vector<double> parameters, double time);
//        /* sc mean function*/
//        std::vector Means::sc(std::vector<double> parameters, double time);
        double meanCalc(std::string check, std::vector<double> priors, double time);



    private:

    /* Singleton */
    public:
        static Means& get_instance() {return instance ;}
    private:
        static Means instance;
};

#endif // MEANS_H
