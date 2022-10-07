#include "kepler.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;


PYBIND11_MODULE(pykepler, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("murison_solver", 
          py::vectorize([](double M, double ecc) { return murison::solver(M, ecc); }),
          "Kepler solver - murison");

    m.def("brandt_solver", 
          py::vectorize([](double M, double ecc) { 
              double sinE, cosE;
              return brandt::solver(M, ecc, &sinE, &cosE); }),
          "Kepler solver - brandt");

    m.def("keplerian", &brandt::keplerian, "keplerian function"); 

}