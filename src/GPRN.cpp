#include "Data.h"
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


GPRN GPRN::instance;
/* Its already defined in main.cpp
GPRN::GPRN()
{}
*/

const vector<double>& t = Data::get_instance().get_t();
const vector<double>& sig = Data::get_instance().get_sig();
int N = Data::get_instance().get_t().size();

//just to compile for now
double extra_sigma = 1;


//Construction of the covariance matrix
Eigen::MatrixXd GPRN::matrixCalculation(std::vector<double> vec1, std::vector<double> vec2)
{
//    Eigen::MatrixXd w = Weights::get_instance().constant(vec1);
//    Eigen::MatrixXd n = Nodes::get_instance().squaredExponential(vec2);
//    Eigen::MatrixXd wn = w.cwiseProduct(n);

    //number of nodes
    int n_size = node.size();

    //if we have n nodes, we will have 4n weigths
    weights = weight;
    for(int i=0; i<4*n_size; i++)
        {
            weights.insert(weights.end(), weight[0]);
        }

    //node kernel
    Eigen::MatrixXd nkernel;
    //weight kernel
    Eigen::MatrixXd wkernel;
    //block matrices
    Eigen::MatrixXd k {Data::get_instance().get_t().size(), Data::get_instance().get_t().size()};
    //final matrix
    Eigen::MatrixXd C {Data::get_instance().get_tt().size(), Data::get_instance().get_tt().size()};
    //now we do math
    for(int i=0; i<4; i++)
    {
        for(int j=0; j <n_size; j++)
        {
            nkernel = nodeCheck(node[j], vec1);
            //weight_values[i-1 + self.q*(position_p-1)]
            wkernel = weightCheck(weights[j + n_size*(i)], vec2);
            k += wkernel.cwiseProduct(nkernel);
        }

    }

    return C;
}

// To check what type of kernel we have into the nodes
Eigen::MatrixXd GPRN::nodeCheck(std::string check, std::vector<double> vec1)
{
    Eigen::MatrixXd nkernel;

    if(check == "C")
        nkernel = Nodes::get_instance().constant(vec1);
    if(check == "SE")
        nkernel = Nodes::get_instance().squaredExponential(vec1);
    if(check == "P")
        nkernel = Nodes::get_instance().periodic(vec1);
    if(check == "QP")
        nkernel = Nodes::get_instance().quasiPeriodic(vec1);
    if(check == "RQ")
        nkernel = Nodes::get_instance().rationalQuadratic(vec1);
    if(check == "COS")
        nkernel = Nodes::get_instance().cosine(vec1);
    if(check == "EXP")
        nkernel = Nodes::get_instance().exponential(vec1);
    if(check == "M32")
        nkernel = Nodes::get_instance().matern32(vec1);
    if(check == "M52")
        nkernel = Nodes::get_instance().matern52(vec1);
    else
        {
        printf("Check your nodes!");
        //break();
        }
    return nkernel;
}

// To check what type of kernel we have into the weight
Eigen::MatrixXd GPRN::weightCheck(std::string check, std::vector<double> vec2)
{
    Eigen::MatrixXd wkernel;

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
    return nkernel;
    
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
//Eigen::MatrixXd GPRN::weightBuilt(std::vector<double> vec2)
//{
//    cout << "Selected weight: " << weight[0] << endl;
//    auto ww = Weights::get_instance();
//}





