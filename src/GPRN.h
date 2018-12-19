#ifndef GPRN_H
#define GPRN_H

#include "Data.h"
#include "DNest4.h"
#include "RNG.h"

#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

class RVmodel;
class GPRN
{
    public:
        GPRN();
        std::vector<Eigen::MatrixXd> matrixCalculation(std::vector<std::vector<double>> node_priors, 
                                                        std::vector<std::vector<double>> weight_priors,
                                                        std::vector<double> jitter, double extra_sigma);
        Eigen::MatrixXd nodeCheck(std::string check, std::vector<double> node_prior, double extra_sigma);
        Eigen::VectorXd weightCheck(std::string check, std::vector<double> weight_prior);
        /* comes from main.cpp */
        std::vector<std::string> node;
        std::vector<std::string> weight;
        std::vector<std::string> mean;


    private:
        /* number of nodes */
        int n_size;
        /* data size */
        int d_size;
        //covariance matrices
        Eigen::MatrixXd k; 
        /* to do math between weight and node */
        Eigen::MatrixXd wn;
        Eigen::MatrixXd wnw;
        /* vector with the 4 final matrices */
        std::vector<Eigen::MatrixXd> matrices_vector {4};

    /* Singleton */
    public:
        static GPRN& get_instance() {return instance; }
    private:
        static GPRN instance;
};

#endif // GPRN_H
