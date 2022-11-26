#include "GP.h"

namespace kima {

/* The "standard" quasi-periodic kernel, see R&W2006 */
Eigen::MatrixXd calculate_C_quasiperiodic(const RVData& data, double eta1,
                                          double eta2, double eta3, double eta4)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            K(i, j) = eta1 * eta1 *
                      exp(-0.5 * pow(r / eta2, 2) -
                          2.0 * pow(sin(M_PI * r / eta3) / eta4, 2));
        }
    }
    return K;
}

// This implements a quasi-periodic kernel built from the
// Matern 3/2 kernel, see R&W2006
Eigen::MatrixXd calculate_C_permatern32(const RVData& data, double eta1,
                                        double eta2, double eta3, double eta4)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            double s = 2 * abs(sin(M_PI * r / eta3));
            K(i, j) = eta1 * eta1 * exp(-0.5 * pow(r / eta2, 2)) *
                      (1 + sqrt(3) * s / eta4) * exp(-sqrt(3) * s / eta4);
        }
    }
    return K;
}

/* This implements a quasi-periodic kernel built from the
*  Matern 5/2 kernel, see R&W2006
*/
Eigen::MatrixXd calculate_C_permatern52(const RVData& data, double eta1,
                                        double eta2, double eta3, double eta4)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            double s = 2 * abs(sin(M_PI * r / eta3));
            K(i, j) = eta1 * eta1 * exp(-0.5 * pow(r / eta2, 2)) *
                      (1 + sqrt(5) * s / eta4 + 5 * s * s / (3 * eta4 * eta4)) *
                      exp(-sqrt(5) * s / eta4);
        }
    }
    return K;
}

/* This implements a quasi-periodic kernel built from the Rational Quadratic
*  kernel, see R&W2006 
*/
Eigen::MatrixXd calculate_C_perrq(const RVData& data, double eta1, double eta2,
                                  double eta3, double eta4, double alpha)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for(size_t i=0; i<N; i++)
    {
        for(size_t j=i; j<N; j++)
        {
            double r = t[i] - t[j];
            double s = abs(sin(M_PI * r / eta3));

            K(i, j) = eta1 * eta1 \
                        * exp(-0.5*pow(r/eta2, 2)) \
                        * pow(1 + 2*s*s/(alpha*eta4*eta4), -alpha);
        }
    }
    return K;
}

/* Squared-exponential kernel (or RBF), see R&W2006 */
Eigen::MatrixXd calculate_C_sqexp(const RVData& data, double eta1, double eta2)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            K(i, j) = eta1 * eta1 * exp(-0.5 * pow(r / eta2, 2));
        }
    }
    return K;
}

/* Quasi-periodic-cosine kernel from Perger+2020 */
Eigen::MatrixXd calculate_C_qpc(const RVData& data, double eta1, double eta2,
                                double eta3, double eta4, double eta5)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            K(i, j) =
                exp(-0.5 * pow(r / eta2, 2)) *
                (eta1 * eta1 * exp(-2.0 * pow(sin(M_PI * r / eta3) / eta4, 2)) +
                 eta5 * eta5 * cos(4 * M_PI * r / eta3));
        }
    }
    return K;
}

/* Periodic (or locally periodic) kernel, see R&W2006 */
Eigen::MatrixXd calculate_C_periodic(const RVData& data, double eta1,
                                     double eta3, double eta4)
{
    size_t N = data.N();
    auto t = data.get_t();
    Eigen::MatrixXd K(N, N);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double r = t[i] - t[j];
            K(i, j) =
                eta1 * eta1 * exp(-2.0 * pow(sin(M_PI * r / eta3) / eta4, 2));
        }
    }
    return K;
}

}  // namespace kima
