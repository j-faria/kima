#include "kepler.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;


PYBIND11_MODULE(pykepler, m) {
    py::options options;
    // options.disable_function_signatures();
    m.doc() = "Keplerian solvers wrapping the C++ implementations";

    m.def("murison_solver", 
          py::vectorize([](double M, double ecc) { return murison::solver(M, ecc); }),
          py::arg("M"), py::arg("ecc"),
          "A solver based on "
          "\"A Practical Method for Solving the Kepler Equation\", "
          "Marc A. Murison, 2006"
    );

    m.def("brandt_solver", 
          py::vectorize([](double M, double ecc) { 
              double sinE, cosE;
              return brandt::solver(M, ecc, &sinE, &cosE); }),
          py::arg("M"), py::arg("ecc"),
          "A solver based on "
          "Brandt et al (2021), AJ, 162, 186"
    );

    // m.def("keplerian", &brandt::keplerian, "A function");
    m.def("keplerian", 
          [](const std::vector<double> &t, const double P,
             const double K, const double ecc, const double w,
             const double M0, const double M0_epoch) {
                auto v = brandt::keplerian(t, P, K, ecc, w, M0, M0_epoch);
                return py::array(v.size(), v.data());
            },
            py::arg("t"), py::arg("P"), py::arg("K"), py::arg("ecc"),
            py::arg("w"), py::arg("M0"), py::arg("M0_epoch"),
           R"delim(
        Keplerian function, as implemented in C++
        
        Args:
            t (array): Times at which to calculate the Keplerian
            P (float): Orbital period
            K (float): Semi-amplitude
            e (float): Orbital eccentricity
            w (float): Argument of periastron
            M0 (float): Mean anomaly at the epoch
            M0_epoch (float): Epoch
        
        Returns:
            v (array): Keplerian function evaluated at `t`
        )delim"
    );
}