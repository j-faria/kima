/*################################################################################
  ##
  ##   Copyright (C) 2016-2022 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * CHANGED, October 2022, João Faria: instantiation for doubles and minor tweaks
 * 
*/

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

#include "incbeta.h"

#ifndef GCEM_INCML_BETA_INV_MAX_ITER
    #define GCEM_INCML_BETA_INV_MAX_ITER 35
#endif

double
incomplete_beta_inv_decision(const double value, const double a, const double b, const double p, const double direc,
                             const double lb_val, const int iter_count) noexcept;

//
// initial value for Halley

//
// a,b > 1 case

double
incomplete_beta_inv_initial_val_1_tval(const double p) noexcept;

double
incomplete_beta_inv_initial_val_1_int_begin(const double t_val) noexcept;

double
incomplete_beta_inv_initial_val_1_int_ab1(const double a, const double b) noexcept;

double
incomplete_beta_inv_initial_val_1_int_ab2(const double a, const double b) noexcept;

double
incomplete_beta_inv_initial_val_1_int_h(const double ab_term_1) noexcept;

double
incomplete_beta_inv_initial_val_1_int_w(const double value, const double ab_term_2, const double h_term) noexcept;

double
incomplete_beta_inv_initial_val_1_int_end(const double a, const double b, const double w_term) noexcept;

double
incomplete_beta_inv_initial_val_1(const double a, const double b, const double t_val, const double sgn_term) noexcept;

//
// a,b else

double
incomplete_beta_inv_initial_val_2_s1(const double a, const double b) noexcept;

double
incomplete_beta_inv_initial_val_2_s2(const double a, const double b) noexcept;

double
incomplete_beta_inv_initial_val_2(const double a, const double b, const double p, const double s_1, const double s_2) noexcept;

// initial value

double
incomplete_beta_inv_initial_val(const double a, const double b, const double p) noexcept;

//
// Halley recursion

double
incomplete_beta_inv_err_val(const double value, const double a, const double b, const double p) noexcept;

double
incomplete_beta_inv_deriv_1(const double value, const double a, const double b, const double lb) noexcept;

double
incomplete_beta_inv_deriv_2(const double value, const double a, const double b, const double deriv_1) noexcept;

double
incomplete_beta_inv_ratio_val_1(const double value, const double a, const double b, const double p, const double deriv_1) noexcept;

double
incomplete_beta_inv_ratio_val_2(const double value, const double a, const double b, const double deriv_1) noexcept;

double
incomplete_beta_inv_halley(const double ratio_val_1, const double ratio_val_2) noexcept;
double
incomplete_beta_inv_recur(const double value, const double a, const double b, const double p, const double deriv_1,
                          const double lb, const int iter_count) noexcept;

double
incomplete_beta_inv_decision(const double value, const double a, const double b, const double p, const double direc,
                             const double lb_val, const int iter_count) noexcept;

double
incomplete_beta_inv_begin(const double initial_val, const double a, const double b, const double p, const double lb_val) noexcept;

double
incomplete_beta_inv_check(const double a, const double b, const double p) noexcept;

/**
 * Compile-time inverse incomplete beta function
 *
 * @param a a real-valued, non-negative input.
 * @param b a real-valued, non-negative input.
 * @param p a real-valued input with values in the unit-interval.
 *
 * @return Computes the inverse incomplete beta function, a value \f$ x \f$ such that
 * \f[ f(x) := \frac{\text{B}(x;\alpha,\beta)}{\text{B}(\alpha,\beta)} - p \f]
 * equal to zero, for a given \c p.
 * GCE-Math finds this root using Halley's method:
 * \f[ x_{n+1} = x_n - \frac{f(x_n)/f'(x_n)}{1 - 0.5 \frac{f(x_n)}{f'(x_n)} \frac{f''(x_n)}{f'(x_n)} } \f]
 * where
 * \f[ \frac{\partial}{\partial x} \left(\frac{\text{B}(x;\alpha,\beta)}{\text{B}(\alpha,\beta)}\right) = \frac{1}{\text{B}(\alpha,\beta)} x^{\alpha-1} (1-x)^{\beta-1} \f]
 * \f[ \frac{\partial^2}{\partial x^2} \left(\frac{\text{B}(x;\alpha,\beta)}{\text{B}(\alpha,\beta)}\right) = \frac{1}{\text{B}(\alpha,\beta)} x^{\alpha-1} (1-x)^{\beta-1} \left( \frac{\alpha-1}{x} - \frac{\beta-1}{1 - x} \right) \f]
 */
double
incomplete_beta_inv(const double a, const double b, const double p) noexcept;
