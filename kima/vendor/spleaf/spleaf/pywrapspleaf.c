// Copyright 2019-2022 Jean-Baptiste Delisle
//
// This file is part of spleaf.
//
// spleaf is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// spleaf is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with spleaf.  If not, see <http://www.gnu.org/licenses/>.

#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION

#include "libspleaf.h"
#include <Python.h>
#include <numpy/arrayobject.h>

// Module docstring
static char module_docstring[] =
  "This module provides an interface for the C library libspleaf.";

// Methods docstrings
static char spleaf_cholesky_docstring[] =
  "Cholesky decomposition of the (n x n) symmetric S+LEAF\n"
  "(semiseparable + leaf) matrix C\n"
  "defined as\n"
  "C = A + Sep + F\n"
  "with\n"
  "* A: the diagonal part of C, stored as a vector of size n.\n"
  "* Sep: the symmetric semiseparable part of C.\n"
  "  For i > j,\n"
  "  Sep_{i,j} = Sep_{j,i}\n"
  "            = Sum_{s=0}^{r-1} U_{i,s} V_{j,s} Prod_{k=j}^{i-1} phi_{k,s}\n"
  "  where U, V are (n x r) matrices, and phi is a (n-1 x r) matrix,\n"
  "  all stored in row major order.\n"
  "  By definition Sep_{i,i} = 0.\n"
  "* F: the symmetric leaf part of C,\n"
  "  stored in strictly lower triangular form, and in row major order.\n"
  "  The i-th row of F is of size b[i], i.e., by definition,\n"
  "  F_{i,j} = 0 for j<i-b[i] and for j=i.\n"
  "  For i-b[i] <= j < i,\n"
  "  the non-zero value F_{i,j} is stored at index (offsetrow[i] + j)\n"
  "  (i.e. offsetrow should be defined as offsetrow = cumsum(b-1) + 1).\n"
  "\n"
  "The Cholesky decomposition of C reads\n"
  "C = L D L^T\n"
  "with\n"
  "L = Lsep + G\n"
  "and\n"
  "* D: diagonal part of the decomposition (vector of size n, like A).\n"
  "* Lsep: the strictly lower triangular semiseparable part of L.\n"
  "  For i > j,\n"
  "  Lsep_{i,j} = Sum_{s=0}^{r-1} U_{i,s} W_{j,s} Prod_{k=j}^{i-1} phi_{k,s},\n"
  "  where U and phi are left unchanged and W is a (n x r) matrix (like V).\n"
  "* G: the strictly lower triangular leaf part of L.\n"
  "  G is stored in the same way as F.\n";
static char spleaf_dotL_docstring[] =
  "Compute y = L x,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.\n";
static char spleaf_solveL_docstring[] =
  "Solve for x = L^-1 y,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.\n";
static char spleaf_dotLT_docstring[] =
  "Compute y = L^T x,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.\n";
static char spleaf_solveLT_docstring[] =
  "Solve for x = L^-T y,\n"
  "where L comes from the Cholesky decomposition C = L D L^T\n"
  "of a symmetric S+LEAF matrix C using spleaf_cholesky.\n";
static char spleaf_cholesky_back_docstring[] =
  "Backward propagation of the gradient for spleaf_cholesky.\n";
static char spleaf_dotL_back_docstring[] =
  "Backward propagation of the gradient for spleaf_dotL.\n";
static char spleaf_solveL_back_docstring[] =
  "Backward propagation of the gradient for spleaf_solveL.\n";
static char spleaf_dotLT_back_docstring[] =
  "Backward propagation of the gradient for spleaf_dotLT.\n";
static char spleaf_solveLT_back_docstring[] =
  "Backward propagation of the gradient for spleaf_solveLT.\n";
static char spleaf_expandsep_docstring[] =
  "Expand the semiseparable part of a symmetric S+LEAF matrix,\n"
  "or a subset of semiseparable terms,\n"
  "as a full (n x n) matrix.\n"
  "This is useful for the conditional covariance computation.\n";
static char spleaf_expandsepmixt_docstring[] =
  "Expand the semiseparable mixt part of a symmetric S+LEAF matrix,\n"
  "or a subset of semiseparable terms,\n"
  "as a full (n2 x n1) matrix.\n"
  "This is useful for the conditional covariance computation.\n";
static char spleaf_expandantisep_docstring[] =
  "Expand the semiseparable part of an anit-symmetric S+LEAF matrix,\n"
  "or a subset of semiseparable terms,\n"
  "as a full (n x n) matrix.\n"
  "This is useful for the conditional derivative covariance computation.\n";
static char spleaf_dotsep_docstring[] =
  "Compute y = K x,\n"
  "where K is the (n x n) semiseparable part of a symmetric S+LEAF matrix,\n"
  "or a subset of semiseparable terms.\n"
  "This is useful for the conditional mean computation.\n";
static char spleaf_dotsepmixt_docstring[] =
  "Compute y = Km x,\n"
  "where Km is the (n2 x n1) semiseparable mixt part\n"
  "of a symmetric S+LEAF matrix,\n"
  "or a subset of semiseparable terms.\n"
  "This is useful for the conditional mean computation.\n";
static char spleaf_dotantisep_docstring[] =
  "Compute y = K x,\n"
  "where K is the (n x n) semiseparable part of an anti-symmetric S+LEAF\n"
  "matrix, or a subset of semiseparable terms. This is useful for the\n"
  "conditional derivative mean computation.\n";

// Module methods
static PyObject *libspleaf_spleaf_cholesky(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotL(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveL(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotLT(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveLT(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_cholesky_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotL_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveL_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotLT_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_solveLT_back(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_expandsep(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_expandsepmixt(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_expandantisep(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotsep(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotsepmixt(PyObject *self, PyObject *args);
static PyObject *libspleaf_spleaf_dotantisep(PyObject *self, PyObject *args);
static PyMethodDef module_methods[] = {
  {"spleaf_cholesky", libspleaf_spleaf_cholesky, METH_VARARGS, spleaf_cholesky_docstring},
  {"spleaf_dotL", libspleaf_spleaf_dotL, METH_VARARGS, spleaf_dotL_docstring},
  {"spleaf_solveL", libspleaf_spleaf_solveL, METH_VARARGS, spleaf_solveL_docstring},
  {"spleaf_dotLT", libspleaf_spleaf_dotLT, METH_VARARGS, spleaf_dotLT_docstring},
  {"spleaf_solveLT", libspleaf_spleaf_solveLT, METH_VARARGS, spleaf_solveLT_docstring},
  {"spleaf_cholesky_back", libspleaf_spleaf_cholesky_back, METH_VARARGS, spleaf_cholesky_back_docstring},
  {"spleaf_dotL_back", libspleaf_spleaf_dotL_back, METH_VARARGS, spleaf_dotL_back_docstring},
  {"spleaf_solveL_back", libspleaf_spleaf_solveL_back, METH_VARARGS, spleaf_solveL_back_docstring},
  {"spleaf_dotLT_back", libspleaf_spleaf_dotLT_back, METH_VARARGS, spleaf_dotLT_back_docstring},
  {"spleaf_solveLT_back", libspleaf_spleaf_solveLT_back, METH_VARARGS, spleaf_solveLT_back_docstring},
  {"spleaf_expandsep", libspleaf_spleaf_expandsep, METH_VARARGS, spleaf_expandsep_docstring},
  {"spleaf_expandsepmixt", libspleaf_spleaf_expandsepmixt, METH_VARARGS, spleaf_expandsepmixt_docstring},
  {"spleaf_expandantisep", libspleaf_spleaf_expandantisep, METH_VARARGS, spleaf_expandantisep_docstring},
  {"spleaf_dotsep", libspleaf_spleaf_dotsep, METH_VARARGS, spleaf_dotsep_docstring},
  {"spleaf_dotsepmixt", libspleaf_spleaf_dotsepmixt, METH_VARARGS, spleaf_dotsepmixt_docstring},
  {"spleaf_dotantisep", libspleaf_spleaf_dotantisep, METH_VARARGS, spleaf_dotantisep_docstring},
  {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef myModule = {
  PyModuleDef_HEAD_INIT, "libspleaf", module_docstring, -1, module_methods};

// Module initialization
PyMODINIT_FUNC PyInit_libspleaf(void)
{
  // import numpy arrays
  import_array();
  return PyModule_Create(&myModule);
}

static PyObject *libspleaf_spleaf_cholesky(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_A;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_F;
  PyObject *obj_D;
  PyObject *obj_W;
  PyObject *obj_G;
  PyObject *obj_S;
  PyObject *obj_Z;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_A,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_F,
    &obj_D,
    &obj_W,
    &obj_G,
    &obj_S,
    &obj_Z))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_A = (PyArrayObject*) PyArray_FROM_OTF(obj_A, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V = (PyArrayObject*) PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_F = (PyArrayObject*) PyArray_FROM_OTF(obj_F, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_D = (PyArrayObject*) PyArray_FROM_OTF(obj_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_S = (PyArrayObject*) PyArray_FROM_OTF(obj_S, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_Z = (PyArrayObject*) PyArray_FROM_OTF(obj_Z, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_A == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_F == NULL ||
    arr_D == NULL ||
    arr_W == NULL ||
    arr_G == NULL ||
    arr_S == NULL ||
    arr_Z == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_A);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_F);
    Py_XDECREF(arr_D);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_S);
    Py_XDECREF(arr_Z);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *A = (double*)PyArray_DATA(arr_A);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *F = (double*)PyArray_DATA(arr_F);
  double *D = (double*)PyArray_DATA(arr_D);
  double *W = (double*)PyArray_DATA(arr_W);
  double *G = (double*)PyArray_DATA(arr_G);
  double *S = (double*)PyArray_DATA(arr_S);
  double *Z = (double*)PyArray_DATA(arr_Z);

  // Call the C function from libspleaf
  spleaf_cholesky(
    n,
    r,
    offsetrow,
    b,
    A,
    U,
    V,
    phi,
    F,
    D,
    W,
    G,
    S,
    Z);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_A);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_F);
  Py_XDECREF(arr_D);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_S);
  Py_XDECREF(arr_Z);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotL(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_y;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_y,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_f = (PyArrayObject*) PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_y == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_dotL(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    y,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveL(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_y;
  PyObject *obj_x;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_y,
    &obj_x,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_f = (PyArrayObject*) PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_y == NULL ||
    arr_x == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *y = (double*)PyArray_DATA(arr_y);
  double *x = (double*)PyArray_DATA(arr_x);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_solveL(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    y,
    x,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotLT(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_y;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_y,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_g = (PyArrayObject*) PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_y == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_dotLT(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    y,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveLT(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_y;
  PyObject *obj_x;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_y,
    &obj_x,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_g = (PyArrayObject*) PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_y == NULL ||
    arr_x == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_y);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *y = (double*)PyArray_DATA(arr_y);
  double *x = (double*)PyArray_DATA(arr_x);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_solveLT(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    y,
    x,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_y);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_cholesky_back(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_D;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_grad_D;
  PyObject *obj_grad_Ucho;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phicho;
  PyObject *obj_grad_G;
  PyObject *obj_grad_A;
  PyObject *obj_grad_U;
  PyObject *obj_grad_V;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_F;
  PyObject *obj_S;
  PyObject *obj_Z;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_D,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_grad_D,
    &obj_grad_Ucho,
    &obj_grad_W,
    &obj_grad_phicho,
    &obj_grad_G,
    &obj_grad_A,
    &obj_grad_U,
    &obj_grad_V,
    &obj_grad_phi,
    &obj_grad_F,
    &obj_S,
    &obj_Z))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_D = (PyArrayObject*) PyArray_FROM_OTF(obj_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_D = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_D, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_Ucho = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_Ucho, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_W = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phicho = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phicho, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_G = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_A = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_A, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_U = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_V = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_F = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_F, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_S = (PyArrayObject*) PyArray_FROM_OTF(obj_S, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_Z = (PyArrayObject*) PyArray_FROM_OTF(obj_Z, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_D == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_grad_D == NULL ||
    arr_grad_Ucho == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phicho == NULL ||
    arr_grad_G == NULL ||
    arr_grad_A == NULL ||
    arr_grad_U == NULL ||
    arr_grad_V == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_F == NULL ||
    arr_S == NULL ||
    arr_Z == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_D);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_grad_D);
    Py_XDECREF(arr_grad_Ucho);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phicho);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_A);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_V);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_F);
    Py_XDECREF(arr_S);
    Py_XDECREF(arr_Z);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *D = (double*)PyArray_DATA(arr_D);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *grad_D = (double*)PyArray_DATA(arr_grad_D);
  double *grad_Ucho = (double*)PyArray_DATA(arr_grad_Ucho);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phicho = (double*)PyArray_DATA(arr_grad_phicho);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_A = (double*)PyArray_DATA(arr_grad_A);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_V = (double*)PyArray_DATA(arr_grad_V);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_F = (double*)PyArray_DATA(arr_grad_F);
  double *S = (double*)PyArray_DATA(arr_S);
  double *Z = (double*)PyArray_DATA(arr_Z);

  // Call the C function from libspleaf
  spleaf_cholesky_back(
    n,
    r,
    offsetrow,
    b,
    D,
    U,
    W,
    phi,
    G,
    grad_D,
    grad_Ucho,
    grad_W,
    grad_phicho,
    grad_G,
    grad_A,
    grad_U,
    grad_V,
    grad_phi,
    grad_F,
    S,
    Z);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_D);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_grad_D);
  Py_XDECREF(arr_grad_Ucho);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phicho);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_A);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_V);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_F);
  Py_XDECREF(arr_S);
  Py_XDECREF(arr_Z);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotL_back(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_y;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_x;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_y,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_x,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_y = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_U = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_W = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_G = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_x = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_f = (PyArrayObject*) PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_y == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_x == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_dotL_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_y,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_x,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveL_back(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_x;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_y;
  PyObject *obj_f;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_x,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_y,
    &obj_f))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_x = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_U = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_W = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_G = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_y = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_f = (PyArrayObject*) PyArray_FROM_OTF(obj_f, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_x == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_y == NULL ||
    arr_f == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_f);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *f = (double*)PyArray_DATA(arr_f);

  // Call the C function from libspleaf
  spleaf_solveL_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_x,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_y,
    f);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_f);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotLT_back(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_y;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_x;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_y,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_x,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_y = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_U = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_W = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_G = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_x = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_g = (PyArrayObject*) PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_y == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_x == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_dotLT_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_y,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_x,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_solveLT_back(PyObject *self, PyObject *args)
{
  long n;
  long r;
  PyObject *obj_offsetrow;
  PyObject *obj_b;
  PyObject *obj_U;
  PyObject *obj_W;
  PyObject *obj_phi;
  PyObject *obj_G;
  PyObject *obj_x;
  PyObject *obj_grad_x;
  PyObject *obj_grad_U;
  PyObject *obj_grad_W;
  PyObject *obj_grad_phi;
  PyObject *obj_grad_G;
  PyObject *obj_grad_y;
  PyObject *obj_g;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llOOOOOOOOOOOOOO",
    &n,
    &r,
    &obj_offsetrow,
    &obj_b,
    &obj_U,
    &obj_W,
    &obj_phi,
    &obj_G,
    &obj_x,
    &obj_grad_x,
    &obj_grad_U,
    &obj_grad_W,
    &obj_grad_phi,
    &obj_grad_G,
    &obj_grad_y,
    &obj_g))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_offsetrow = (PyArrayObject*) PyArray_FROM_OTF(obj_offsetrow, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_b = (PyArrayObject*) PyArray_FROM_OTF(obj_b, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_W = (PyArrayObject*) PyArray_FROM_OTF(obj_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_G = (PyArrayObject*) PyArray_FROM_OTF(obj_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_x = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_U = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_W = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_W, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_G = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_G, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_grad_y = (PyArrayObject*) PyArray_FROM_OTF(obj_grad_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_g = (PyArrayObject*) PyArray_FROM_OTF(obj_g, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_offsetrow == NULL ||
    arr_b == NULL ||
    arr_U == NULL ||
    arr_W == NULL ||
    arr_phi == NULL ||
    arr_G == NULL ||
    arr_x == NULL ||
    arr_grad_x == NULL ||
    arr_grad_U == NULL ||
    arr_grad_W == NULL ||
    arr_grad_phi == NULL ||
    arr_grad_G == NULL ||
    arr_grad_y == NULL ||
    arr_g == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_offsetrow);
    Py_XDECREF(arr_b);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_W);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_G);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_grad_x);
    Py_XDECREF(arr_grad_U);
    Py_XDECREF(arr_grad_W);
    Py_XDECREF(arr_grad_phi);
    Py_XDECREF(arr_grad_G);
    Py_XDECREF(arr_grad_y);
    Py_XDECREF(arr_g);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *offsetrow = (long*)PyArray_DATA(arr_offsetrow);
  long *b = (long*)PyArray_DATA(arr_b);
  double *U = (double*)PyArray_DATA(arr_U);
  double *W = (double*)PyArray_DATA(arr_W);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *G = (double*)PyArray_DATA(arr_G);
  double *x = (double*)PyArray_DATA(arr_x);
  double *grad_x = (double*)PyArray_DATA(arr_grad_x);
  double *grad_U = (double*)PyArray_DATA(arr_grad_U);
  double *grad_W = (double*)PyArray_DATA(arr_grad_W);
  double *grad_phi = (double*)PyArray_DATA(arr_grad_phi);
  double *grad_G = (double*)PyArray_DATA(arr_grad_G);
  double *grad_y = (double*)PyArray_DATA(arr_grad_y);
  double *g = (double*)PyArray_DATA(arr_g);

  // Call the C function from libspleaf
  spleaf_solveLT_back(
    n,
    r,
    offsetrow,
    b,
    U,
    W,
    phi,
    G,
    x,
    grad_x,
    grad_U,
    grad_W,
    grad_phi,
    grad_G,
    grad_y,
    g);

  // Dereference arrays
  Py_XDECREF(arr_offsetrow);
  Py_XDECREF(arr_b);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_W);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_G);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_grad_x);
  Py_XDECREF(arr_grad_U);
  Py_XDECREF(arr_grad_W);
  Py_XDECREF(arr_grad_phi);
  Py_XDECREF(arr_grad_G);
  Py_XDECREF(arr_grad_y);
  Py_XDECREF(arr_g);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_expandsep(PyObject *self, PyObject *args)
{
  long n;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_K;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "lllOOOOO",
    &n,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_K))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V = (PyArrayObject*) PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_K = (PyArrayObject*) PyArray_FROM_OTF(obj_K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_K == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_K);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *K = (double*)PyArray_DATA(arr_K);

  // Call the C function from libspleaf
  spleaf_expandsep(
    n,
    r,
    rsi,
    sepindex,
    U,
    V,
    phi,
    K);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_K);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_expandsepmixt(PyObject *self, PyObject *args)
{
  long n1;
  long n2;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U1;
  PyObject *obj_V1;
  PyObject *obj_phi1;
  PyObject *obj_U2;
  PyObject *obj_V2;
  PyObject *obj_ref2left;
  PyObject *obj_phi2left;
  PyObject *obj_phi2right;
  PyObject *obj_Km;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llllOOOOOOOOOO",
    &n1,
    &n2,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U1,
    &obj_V1,
    &obj_phi1,
    &obj_U2,
    &obj_V2,
    &obj_ref2left,
    &obj_phi2left,
    &obj_phi2right,
    &obj_Km))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U1 = (PyArrayObject*) PyArray_FROM_OTF(obj_U1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V1 = (PyArrayObject*) PyArray_FROM_OTF(obj_V1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi1 = (PyArrayObject*) PyArray_FROM_OTF(obj_phi1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U2 = (PyArrayObject*) PyArray_FROM_OTF(obj_U2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V2 = (PyArrayObject*) PyArray_FROM_OTF(obj_V2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_ref2left = (PyArrayObject*) PyArray_FROM_OTF(obj_ref2left, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi2left = (PyArrayObject*) PyArray_FROM_OTF(obj_phi2left, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi2right = (PyArrayObject*) PyArray_FROM_OTF(obj_phi2right, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_Km = (PyArrayObject*) PyArray_FROM_OTF(obj_Km, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U1 == NULL ||
    arr_V1 == NULL ||
    arr_phi1 == NULL ||
    arr_U2 == NULL ||
    arr_V2 == NULL ||
    arr_ref2left == NULL ||
    arr_phi2left == NULL ||
    arr_phi2right == NULL ||
    arr_Km == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U1);
    Py_XDECREF(arr_V1);
    Py_XDECREF(arr_phi1);
    Py_XDECREF(arr_U2);
    Py_XDECREF(arr_V2);
    Py_XDECREF(arr_ref2left);
    Py_XDECREF(arr_phi2left);
    Py_XDECREF(arr_phi2right);
    Py_XDECREF(arr_Km);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U1 = (double*)PyArray_DATA(arr_U1);
  double *V1 = (double*)PyArray_DATA(arr_V1);
  double *phi1 = (double*)PyArray_DATA(arr_phi1);
  double *U2 = (double*)PyArray_DATA(arr_U2);
  double *V2 = (double*)PyArray_DATA(arr_V2);
  long *ref2left = (long*)PyArray_DATA(arr_ref2left);
  double *phi2left = (double*)PyArray_DATA(arr_phi2left);
  double *phi2right = (double*)PyArray_DATA(arr_phi2right);
  double *Km = (double*)PyArray_DATA(arr_Km);

  // Call the C function from libspleaf
  spleaf_expandsepmixt(
    n1,
    n2,
    r,
    rsi,
    sepindex,
    U1,
    V1,
    phi1,
    U2,
    V2,
    ref2left,
    phi2left,
    phi2right,
    Km);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U1);
  Py_XDECREF(arr_V1);
  Py_XDECREF(arr_phi1);
  Py_XDECREF(arr_U2);
  Py_XDECREF(arr_V2);
  Py_XDECREF(arr_ref2left);
  Py_XDECREF(arr_phi2left);
  Py_XDECREF(arr_phi2right);
  Py_XDECREF(arr_Km);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_expandantisep(PyObject *self, PyObject *args)
{
  long n;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_K;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "lllOOOOO",
    &n,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_K))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V = (PyArrayObject*) PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_K = (PyArrayObject*) PyArray_FROM_OTF(obj_K, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_K == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_K);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *K = (double*)PyArray_DATA(arr_K);

  // Call the C function from libspleaf
  spleaf_expandantisep(
    n,
    r,
    rsi,
    sepindex,
    U,
    V,
    phi,
    K);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_K);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotsep(PyObject *self, PyObject *args)
{
  long n;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_x;
  PyObject *obj_y;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "lllOOOOOO",
    &n,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_x,
    &obj_y))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V = (PyArrayObject*) PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_x == NULL ||
    arr_y == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);

  // Call the C function from libspleaf
  spleaf_dotsep(
    n,
    r,
    rsi,
    sepindex,
    U,
    V,
    phi,
    x,
    y);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotsepmixt(PyObject *self, PyObject *args)
{
  long n1;
  long n2;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U1;
  PyObject *obj_V1;
  PyObject *obj_phi1;
  PyObject *obj_U2;
  PyObject *obj_V2;
  PyObject *obj_ref2left;
  PyObject *obj_phi2left;
  PyObject *obj_phi2right;
  PyObject *obj_x;
  PyObject *obj_y;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "llllOOOOOOOOOOO",
    &n1,
    &n2,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U1,
    &obj_V1,
    &obj_phi1,
    &obj_U2,
    &obj_V2,
    &obj_ref2left,
    &obj_phi2left,
    &obj_phi2right,
    &obj_x,
    &obj_y))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U1 = (PyArrayObject*) PyArray_FROM_OTF(obj_U1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V1 = (PyArrayObject*) PyArray_FROM_OTF(obj_V1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi1 = (PyArrayObject*) PyArray_FROM_OTF(obj_phi1, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U2 = (PyArrayObject*) PyArray_FROM_OTF(obj_U2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V2 = (PyArrayObject*) PyArray_FROM_OTF(obj_V2, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_ref2left = (PyArrayObject*) PyArray_FROM_OTF(obj_ref2left, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi2left = (PyArrayObject*) PyArray_FROM_OTF(obj_phi2left, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi2right = (PyArrayObject*) PyArray_FROM_OTF(obj_phi2right, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U1 == NULL ||
    arr_V1 == NULL ||
    arr_phi1 == NULL ||
    arr_U2 == NULL ||
    arr_V2 == NULL ||
    arr_ref2left == NULL ||
    arr_phi2left == NULL ||
    arr_phi2right == NULL ||
    arr_x == NULL ||
    arr_y == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U1);
    Py_XDECREF(arr_V1);
    Py_XDECREF(arr_phi1);
    Py_XDECREF(arr_U2);
    Py_XDECREF(arr_V2);
    Py_XDECREF(arr_ref2left);
    Py_XDECREF(arr_phi2left);
    Py_XDECREF(arr_phi2right);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U1 = (double*)PyArray_DATA(arr_U1);
  double *V1 = (double*)PyArray_DATA(arr_V1);
  double *phi1 = (double*)PyArray_DATA(arr_phi1);
  double *U2 = (double*)PyArray_DATA(arr_U2);
  double *V2 = (double*)PyArray_DATA(arr_V2);
  long *ref2left = (long*)PyArray_DATA(arr_ref2left);
  double *phi2left = (double*)PyArray_DATA(arr_phi2left);
  double *phi2right = (double*)PyArray_DATA(arr_phi2right);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);

  // Call the C function from libspleaf
  spleaf_dotsepmixt(
    n1,
    n2,
    r,
    rsi,
    sepindex,
    U1,
    V1,
    phi1,
    U2,
    V2,
    ref2left,
    phi2left,
    phi2right,
    x,
    y);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U1);
  Py_XDECREF(arr_V1);
  Py_XDECREF(arr_phi1);
  Py_XDECREF(arr_U2);
  Py_XDECREF(arr_V2);
  Py_XDECREF(arr_ref2left);
  Py_XDECREF(arr_phi2left);
  Py_XDECREF(arr_phi2right);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);

  Py_RETURN_NONE;
}

static PyObject *libspleaf_spleaf_dotantisep(PyObject *self, PyObject *args)
{
  long n;
  long r;
  long rsi;
  PyObject *obj_sepindex;
  PyObject *obj_U;
  PyObject *obj_V;
  PyObject *obj_phi;
  PyObject *obj_x;
  PyObject *obj_y;

  // Parse input tuple
  if (!PyArg_ParseTuple(args, "lllOOOOOO",
    &n,
    &r,
    &rsi,
    &obj_sepindex,
    &obj_U,
    &obj_V,
    &obj_phi,
    &obj_x,
    &obj_y))
    return(NULL);

  // Interpret input objects as numpy arrays
  PyArrayObject *arr_sepindex = (PyArrayObject*) PyArray_FROM_OTF(obj_sepindex, NPY_LONG, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_U = (PyArrayObject*) PyArray_FROM_OTF(obj_U, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_V = (PyArrayObject*) PyArray_FROM_OTF(obj_V, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_phi = (PyArrayObject*) PyArray_FROM_OTF(obj_phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_x = (PyArrayObject*) PyArray_FROM_OTF(obj_x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *arr_y = (PyArrayObject*) PyArray_FROM_OTF(obj_y, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

  // Generate exception in case of failure
  if (
    arr_sepindex == NULL ||
    arr_U == NULL ||
    arr_V == NULL ||
    arr_phi == NULL ||
    arr_x == NULL ||
    arr_y == NULL) {
    // Dereference arrays
    Py_XDECREF(arr_sepindex);
    Py_XDECREF(arr_U);
    Py_XDECREF(arr_V);
    Py_XDECREF(arr_phi);
    Py_XDECREF(arr_x);
    Py_XDECREF(arr_y);
    return NULL;
  }

  // Get C-types pointers to numpy arrays
  long *sepindex = (long*)PyArray_DATA(arr_sepindex);
  double *U = (double*)PyArray_DATA(arr_U);
  double *V = (double*)PyArray_DATA(arr_V);
  double *phi = (double*)PyArray_DATA(arr_phi);
  double *x = (double*)PyArray_DATA(arr_x);
  double *y = (double*)PyArray_DATA(arr_y);

  // Call the C function from libspleaf
  spleaf_dotantisep(
    n,
    r,
    rsi,
    sepindex,
    U,
    V,
    phi,
    x,
    y);

  // Dereference arrays
  Py_XDECREF(arr_sepindex);
  Py_XDECREF(arr_U);
  Py_XDECREF(arr_V);
  Py_XDECREF(arr_phi);
  Py_XDECREF(arr_x);
  Py_XDECREF(arr_y);

  Py_RETURN_NONE;
}
