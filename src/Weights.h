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
        /* constant kernel */
        Eigen::VectorXd constant(std::vector<double> vec);
        /* squared exponential kernel */
        Eigen::VectorXd squaredExponential(std::vector<double> vec);
        /* periodic kernel */
        Eigen::VectorXd periodic(std::vector<double> vec);
        /* quasi periodic kernel */
        Eigen::VectorXd quasiPeriodic(std::vector<double> vec);
        /* rational quadratic kernel */
        Eigen::VectorXd rationalQuadratic(std::vector<double> vec);
        /* cosine kernel */
        Eigen::VectorXd cosine(std::vector<double> vec);
        /* exponential kernel */
        Eigen::VectorXd exponential(std::vector<double> vec);
        /* matern 3/2 kernel */
        Eigen::VectorXd matern32(std::vector<double> vec);
        /* matern 5/2 kernel */
        Eigen::VectorXd matern52(std::vector<double> vec);

    private:
        Eigen::VectorXd C {Data::get_instance().get_t().size()};

    /* Singleton */
    public:
        static Weights& get_instance() {return instance; }
    private:
        static Weights instance;
};

#endif // WEIGHTS_H
