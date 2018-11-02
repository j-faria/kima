#ifndef GPRN_H
#define GPRN_H

#include "Data.h"
#include "RVmodel.h"
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>


class GPRN
{
    public:
        GPRN();
        Eigen::MatrixXd matrixCalculation(std::vector<double> vec1, std::vector<double> vec2);
        Eigen::MatrixXd nodeCheck(std::string check, std::vector<double> vec1);
        Eigen::MatrixXd weightCheck(std::string check, std::vector<double> vec2);
        
        Eigen::MatrixXd nodeBuilt(std::vector<double> vec1);
        Eigen::MatrixXd weightBuilt(std::vector<double> vec2);


    private:
        //block matrix
        Eigen::MatrixXd k {Data::get_instance().get_t().size(), Data::get_instance().get_t().size()};
        //final matrix
        Eigen::MatrixXd C {Data::get_instance().get_tt().size(), Data::get_instance().get_tt().size()};
        //comes from main.cpp
        std::vector<std::string> node;
        //comes from main.cpp
        std::vector<std::string> weight;
        //might be smarter to put it in RVmodel.cpp
        std::vector<std::string> weights;
        //node we are working with
        Eigen::MatrixXd nkernel;
        //weight we are working with
        Eigen::MatrixXd wkernel;


    //Singleton
    public:
        static GPRN& get_instance() {return instance; }
    private:
        static GPRN instance;
};

#endif // GPRN_H
