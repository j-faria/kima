#include "Data.h"
#include "DNest4.h"
#include "RNG.h"
#include "GPRN.h"
#include "Nodes.h"
#include "Weights.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
using namespace DNest4;

GPRN GPRN::instance;
/* Its already defined in main.cpp
GPRN::GPRN()
{}
*/

const vector<double>& t = Data::get_instance().get_t();
int N = Data::get_instance().get_t().size();


/* Construction of the covariance matrices */
std::vector<Eigen::MatrixXd> GPRN::matrixCalculation(std::vector<std::vector<double>> node_priors, 
                                                    std::vector<std::vector<double>> weight_priors,
                                                    std::vector<double> jitter_priors, double extra_sigma)
{

    /* node kernel */
    Eigen::MatrixXd nkernel;
    /* weight kernels */
    Eigen::VectorXd wkernel;
    /* measurements errors */
    const vector<double>& sig = Data::get_instance().get_sig();

    /* now we do math */
    int n_size = node.size();
    int d_size = Data::get_instance().get_t().size();
    for(int i=0; i<4; i++)
    {
        /* auxiliaty matrices */
        Eigen::MatrixXd wn = Eigen::MatrixXd::Zero(d_size, d_size);
        Eigen::MatrixXd wnw = Eigen::MatrixXd::Zero(d_size, d_size);
        for(int j=0; j < n_size; j++)
        {
            nkernel = nodeCheck(node[j], node_priors[j], extra_sigma);
            wkernel = weightCheck(weight[0], weight_priors[j + n_size*i]);
            wn = wkernel.asDiagonal() * nkernel;
            //wnw += nkernel * wkernel.asDiagonal();
            wnw += wn * wkernel.asDiagonal();
        }
        for(int j = i*d_size; j<(i+1)*d_size; j++)
        {
            wnw(j % d_size, j % d_size) += (sig[j]*sig[j]) 
                                            + (jitter_priors[i]*jitter_priors[i]);
        }

    matrices_vector[i] = wnw;
    }
return matrices_vector;
}


/* To check what type of kernel we have into the nodes */
Eigen::MatrixXd GPRN::nodeCheck(std::string check, std::vector<double> node_prior, double extra_sigma)
{
    Eigen::MatrixXd nkernel;
    //extra_sigma = 0;
    if(check == "C")
        nkernel = Nodes::get_instance().constant(node_prior, extra_sigma);
    if(check == "SE")
        nkernel = Nodes::get_instance().squaredExponential(node_prior, extra_sigma);
    if(check == "P")
        nkernel = Nodes::get_instance().periodic(node_prior, extra_sigma);
    if(check == "QP")
        nkernel = Nodes::get_instance().quasiPeriodic(node_prior, extra_sigma);
    if(check == "RQ")
        nkernel = Nodes::get_instance().rationalQuadratic(node_prior, extra_sigma);
    if(check == "COS")
        nkernel = Nodes::get_instance().cosine(node_prior, extra_sigma);
    if(check == "EXP")
        nkernel = Nodes::get_instance().exponential(node_prior, extra_sigma);
    if(check == "M32")
        nkernel = Nodes::get_instance().matern32(node_prior, extra_sigma);
    if(check == "M52")
        nkernel = Nodes::get_instance().matern52(node_prior, extra_sigma);
return nkernel;
}


/* To check what type of kernel we have into the weight */
Eigen::VectorXd GPRN::weightCheck(std::string check, std::vector<double> weight_prior)
{
    Eigen::VectorXd wkernel;

    if(check == "C")
        wkernel = Weights::get_instance().constant(weight_prior);
    if(check == "SE")
        wkernel = Weights::get_instance().squaredExponential(weight_prior);
    if(check == "P")
        wkernel = Weights::get_instance().periodic(weight_prior);
    if(check == "QP")
        wkernel = Weights::get_instance().quasiPeriodic(weight_prior);
    if(check == "RQ")
        wkernel = Weights::get_instance().rationalQuadratic(weight_prior);
    if(check == "COS")
        wkernel = Weights::get_instance().cosine(weight_prior);
    if(check == "EXP")
        wkernel = Weights::get_instance().exponential(weight_prior);
    if(check == "M32")
        wkernel = Weights::get_instance().matern32(weight_prior);
    if(check == "M52")
        wkernel = Weights::get_instance().matern52(weight_prior);
return wkernel;
}


