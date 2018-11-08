#include "Data.h"
#include "RVmodel.h"
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
const vector<double>& sig = Data::get_instance().get_sig();
int N = Data::get_instance().get_t().size();

//just to compile for now
//ModifiedLogUniform *Jprior = new ModifiedLogUniform(1.0, 99.); // additional white noise, m/s
//extra_sigma = Jprior;

//Construction of the covariance matrices
std::vector<Eigen::MatrixXd> GPRN::matrixCalculation(std::vector<double> vec1, std::vector<double> vec2)
{
    //number of nodes
    int n_size = node.size();
    //just to compile for now
    //extra_sigma = sigmaPrior.generate(DNest4::RNG rng);
    double extra_sigma = 0.01;
    
    //if we have n nodes, we will have 4n weigths
    weights = weight;
    for(int i=0; i<4*n_size; i++)
        {
            weights.insert(weights.end(), weight[0]);
        }

    //node kernel
    Eigen::MatrixXd nkernel;
    //weight kernels
    Eigen::VectorXd wkernel;
    //vector with the 4 matrices
    std::vector<Eigen::MatrixXd> matrices_vector {4};
    //now we do math
    for(int i=0; i<4; i++)
    {
    //block matrix to be built
    Eigen::MatrixXd k {Data::get_instance().get_t().size(), Data::get_instance().get_t().size()};
        for(int j=0; j <n_size; j++)
        {
            nkernel = nodeCheck(node[j], vec1, extra_sigma);
            wkernel = weightCheck(weights[j + n_size*i], vec2);
            Eigen::MatrixXd wn = wkernel.asDiagonal() * nkernel;
            Eigen::MatrixXd wnw = nkernel * wkernel.asDiagonal();
            k = k + wnw;
        }
    matrices_vector[i] = k;
    }
    
return matrices_vector;
}

// To check what type of kernel we have into the nodes
Eigen::MatrixXd GPRN::nodeCheck(std::string check, std::vector<double> vec1, double sigmaPrior)
{
    Eigen::MatrixXd nkernel;

    if(check == "C")
        nkernel = Nodes::get_instance().constant(vec1, sigmaPrior);
    if(check == "SE")
        nkernel = Nodes::get_instance().squaredExponential(vec1, sigmaPrior);
    if(check == "P")
        nkernel = Nodes::get_instance().periodic(vec1, sigmaPrior);
    if(check == "QP")
        nkernel = Nodes::get_instance().quasiPeriodic(vec1, sigmaPrior);
    if(check == "RQ")
        nkernel = Nodes::get_instance().rationalQuadratic(vec1, sigmaPrior);
    if(check == "COS")
        nkernel = Nodes::get_instance().cosine(vec1, sigmaPrior);
    if(check == "EXP")
        nkernel = Nodes::get_instance().exponential(vec1, sigmaPrior);
    if(check == "M32")
        nkernel = Nodes::get_instance().matern32(vec1, sigmaPrior);
    if(check == "M52")
        nkernel = Nodes::get_instance().matern52(vec1, sigmaPrior);
return nkernel;
}

// To check what type of kernel we have into the weight
Eigen::VectorXd GPRN::weightCheck(std::string check, std::vector<double> vec2)
{
    Eigen::VectorXd wkernel;

    if(check == "C")
        wkernel = Weights::get_instance().constant(vec2);
    if(check == "SE")
        wkernel = Weights::get_instance().squaredExponential(vec2);
    if(check == "P")
        wkernel = Weights::get_instance().periodic(vec2);
    if(check == "QP")
        wkernel = Weights::get_instance().quasiPeriodic(vec2);
    if(check == "RQ")
        wkernel = Weights::get_instance().rationalQuadratic(vec2);
    if(check == "COS")
        wkernel = Weights::get_instance().cosine(vec2);
    if(check == "EXP")
        wkernel = Weights::get_instance().exponential(vec2);
    if(check == "M32")
        wkernel = Weights::get_instance().matern32(vec2);
    if(check == "M52")
        wkernel = Weights::get_instance().matern52(vec2);
return wkernel;
}


