#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "Data.h"
#include "RVmodel.h"
#include <Eigen/Core>
#include <Eigen/Dense>


class Weights
{
    public:
        Weights();
        //constant kernel
        Eigen::MatrixXd constant(std::vector<double> vec);
        //squared exponential kernel
        Eigen::MatrixXd squaredExponential(std::vector<double> vec);
        //periodic kernel
        Eigen::MatrixXd periodic(std::vector<double> vec);
        //quasi periodic kernel
        Eigen::MatrixXd quasiPeriodic(std::vector<double> vec);
        //rational quadratic kernel
        Eigen::MatrixXd rationalQuadratic(std::vector<double> vec);
        //cosine kernel
        Eigen::MatrixXd cosine(std::vector<double> vec);
        //exponential kernel
        Eigen::MatrixXd exponential(std::vector<double> vec);
        //matern 3/2 kernel
        Eigen::MatrixXd matern32(std::vector<double> vec);
        //matern 5/2 kernel
        Eigen::MatrixXd matern52(std::vector<double> vec);

    private:
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};

    //Singleton
    public:
        static Weights& get_instance() {return instance; }
    private:
        static Weights instance;
};

#endif // WEIGHTS_H
