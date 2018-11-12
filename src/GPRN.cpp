#include "Data.h"
//#include "RVmodel.h"
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
std::vector<Eigen::MatrixXd> GPRN::matrixCalculation(std::vector<std::vector<double>> vec1, 
                                                    std::vector<std::vector<double>> vec2)
{
//printf(" \n we are in  GPRN::matrixCalculation \n");
    //number of nodes
    //const int n_size = node.size();
    //just to compile for now
    //extra_sigma = sigmaPrior.generate(DNest4::RNG rng);
    double extra_sigma = 0.01;

    //cout << "\nnodes QP ----- " << vec1[0][0] << " "<< vec1[0][1] << " "<< vec1[0][2] << " "  << endl;
    //cout << "nodes P ----- " << vec1[1][0] << " "<< vec1[1][1] << " "  << endl;
    
    //node kernel
    Eigen::MatrixXd nkernel;
    //weight kernels
    Eigen::VectorXd wkernel;
    //vector with the 4 matrices
    //std::vector<Eigen::MatrixXd> matrices_vector {4};
    //now we do math
    printf("\n we have %i node and %i weights \n ", vec1.size(), vec2.size());
    int n_size = node.size();
    for(int i=0; i<4; i++)
    {
        printf(" \n %i", i);
        //block matrix to be built
        Eigen::MatrixXd k {Data::get_instance().get_t().size(), Data::get_instance().get_t().size()};
        //printf("are we breathing? \n");
        for(int j=0; j < n_size; j++)
        {
            printf("\n node j= %i, weight = %i \n", j, j+n_size*i);
            //cout << vec1 << endl;
            nkernel = nodeCheck(node[j], vec1[i], extra_sigma);
            //printf("\n 2 --- we got here --- \n");
            wkernel = weightCheck(weight[0], vec2[0]); //vec2[j + n_size*i]);
            printf("\n 1 --- we got here --- \n");
            Eigen::MatrixXd wn = wkernel.asDiagonal() * nkernel;
            printf("\n 2 --- we got here --- \n");
            Eigen::MatrixXd wnw = nkernel * wkernel.asDiagonal();
            printf("\n 3 --- we got here --- \n");
            //printf("Im gonna assume math was made before this point \n");
            k = k + wnw;
        }
//        printf("whyyyyyyyyyy? \n");
    //printf("\n 4 --- we got here tooo!--- \n");
    matrices_vector[i] = k;
    //printf("\n node = %i ", vec1[0][0]); 
    //printf("weight = %i ", vec2[0][0]);
    }
    
return matrices_vector;
}

// To check what type of kernel we have into the nodes
Eigen::MatrixXd GPRN::nodeCheck(std::string check, std::vector<double> vec1, double sigmaPrior)
{
//printf(" \n we are in  GPRN::nodeCheck \n");
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
//printf(" \n we are in  GPRN::weightCheck \n");
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


