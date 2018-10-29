#ifndef GPRN_H
#define GPRN_H

#include "Data.h"
#include "RVmodel.h"
#include <Eigen/Core>
#include <Eigen/Dense>

class GPRN
{
    public:
        GPRN();
        Eigen::MatrixXd branch(std::vector<double> vec1, std::vector<double> vec2);

    private:
        Eigen::MatrixXd C {Data::get_instance().N(), Data::get_instance().N()};
    
    //Singleton
    public:
        static GPRN& get_instance() {return instance; }
    private:
        static GPRN instance;
};

#endif // GPRN_H
