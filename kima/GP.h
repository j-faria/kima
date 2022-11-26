#pragma once

// #include <vector>
#include "Data.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

using namespace kima;

namespace kima {

Eigen::MatrixXd calculate_C_quasiperiodic(const RVData& data, double eta1,
                                          double eta2, double eta3,
                                          double eta4);

Eigen::MatrixXd calculate_C_permatern32(const RVData& data, double eta1,
                                        double eta2, double eta3, double eta4);

Eigen::MatrixXd calculate_C_permatern52(const RVData& data, double eta1,
                                        double eta2, double eta3, double eta4);

Eigen::MatrixXd calculate_C_perrq(const RVData& data, double eta1, double eta2,
                                  double eta3, double eta4, double alpha);

Eigen::MatrixXd calculate_C_sqexp(const RVData& data, double eta1, double eta2);

Eigen::MatrixXd calculate_C_qpc(const RVData& data, double eta1, double eta2,
                                double eta3, double eta4, double eta5);

Eigen::MatrixXd calculate_C_periodic(const RVData& data, double eta1,
                                     double eta3, double eta4);

} // namespace kima