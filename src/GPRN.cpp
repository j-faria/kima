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

//Construction of the covariance matrix
Eigen::MatrixXd GPRN::matrixCalculation(std::vector<double> vec1, std::vector<double> vec2)
{
//    Eigen::MatrixXd w = Weights::get_instance().constant(vec1);
//    Eigen::MatrixXd n = Nodes::get_instance().squaredExponential(vec2);
//    Eigen::MatrixXd wn = w.cwiseProduct(n);

    //number of nodes
    int n_size = node.size();
    //just to compile for now
    //extra_sigma = sigmaPrior.generate(DNest4::RNG rng);
    double extra_sigma = 1;
    
    
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
    //final matrix will look like this
    Eigen::MatrixXd C {Data::get_instance().get_tt().size(), Data::get_instance().get_tt().size()};
    //now we do math
    for(int i=0; i<4; i++)
    {
    //block matrix to be built
    Eigen::MatrixXd k {dataset_size, dataset_size};
        for(int j=0; j <n_size; j++)
        {
            nkernel = nodeCheck(node[j], vec1, extra_sigma);
            wkernel = weightCheck(weights[j + n_size*i], vec2);
            Eigen::MatrixXd wn = wkernel.cwiseProduct(nkernel);
            Eigen::MatrixXd wnw = wn.cwiseProduct(wkernel.transpose());
            k += wnw;
        }

    C.block<i*Data::get_instance().get_tt().size(),i*Data::get_instance().get_tt().size()>(0,0) = k;
    }

    return C;
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
    else
        printf("Check your nodes!");
        //break();

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
    else
        printf("Check your weights!");
        //break();

    return wkernel;
}


//// In here we should count the nodes and put priors
//Eigen::MatrixXd GPRN::nodeBuilt(std::vector<double> vec1)
//{
//    auto nn = Nodes::get_instance();
//    int n_size = node.size();
//    
//    for(int i=0; i<n_size; i++)
//    {
//        nodeCheck(node, vec1);
//    }
//    printf("It is over ... for now ...");
//    //std::vector<Eigen::MatrixXd> = {n.nodes[0], n.nodes[1]};
//}


////In here we should count the weights and put priors
//Eigen::VectorXd GPRN::weightBuilt(std::vector<double> vec2)
//{
//    cout << "Selected weight: " << weight[0] << endl;
//    auto ww = Weights::get_instance();
//}





