#ifndef NODES_H
#define NODES_H

#include "Data.h"
#include "DNest4.h"
#include "RNG.h"
#include "RVmodel.h"

#include <Eigen/Core>
#include <Eigen/Dense>


class Nodes
{
    public:
        Nodes();
        //constant kernel
        Eigen::MatrixXd constant(std::vector<double> vec, double extra_sigma);
        //squared exponential kernel
        Eigen::MatrixXd squaredExponential(std::vector<double> vec, double extra_sigma);
        //periodic kernel
        Eigen::MatrixXd periodic(std::vector<double> vec, double extra_sigma);
        //quasi periodic kernel
        Eigen::MatrixXd quasiPeriodic(std::vector<double> vec, double extra_sigma);
        //rational quadratic kernel
        Eigen::MatrixXd rationalQuadratic(std::vector<double> vec, double extra_sigma);
        //cosine kernel
        Eigen::MatrixXd cosine(std::vector<double> vec, double extra_sigma);
        //exponential kernel
        Eigen::MatrixXd exponential(std::vector<double> vec, double extra_sigma);
        //matern 3/2 kernel
        Eigen::MatrixXd matern32(std::vector<double> vec, double extra_sigma);
        //matern 5/2 kernel
        Eigen::MatrixXd matern52(std::vector<double> vec, double extra_sigma);

    private:
        //double extra_sigma;
        Eigen::MatrixXd C {Data::get_instance().get_t().size(), Data::get_instance().get_t().size()};

    //Singleton
    public:
        static Nodes& get_instance() {return instance ;}
    private:
        static Nodes instance;
};

#endif // NODES_H
