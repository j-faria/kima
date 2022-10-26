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

#include "invincbeta.h"

//
// initial value for Halley

//
// a,b > 1 case

double
incomplete_beta_inv_initial_val_1_tval(const double p) noexcept
{ // a > 1.0
    return (p > 0.5 ? sqrt(-2 * log(1 - p)) : sqrt(-2 * log(p)));
}


double
incomplete_beta_inv_initial_val_1_int_begin(const double t_val) noexcept
{ // internal for a > 1.0
    return (t_val - (2.515517 + 0.802853 * t_val + 0.010328 * t_val * t_val) / (1 + 1.432788 * t_val + 0.189269 * t_val * t_val + 0.001308 * t_val * t_val * t_val));
}

double
incomplete_beta_inv_initial_val_1_int_ab1(const double a, const double b) noexcept
{
    return (1 / (2 * a - 1) + 1 / (2 * b - 1));
}

double
incomplete_beta_inv_initial_val_1_int_ab2(const double a, const double b) noexcept
{
    return (1 / (2 * b - 1) - 1 / (2 * a - 1));
}

double
incomplete_beta_inv_initial_val_1_int_h(const double ab_term_1) noexcept
{
    return (2 / ab_term_1);
}

double
incomplete_beta_inv_initial_val_1_int_w(const double value, const double ab_term_2, const double h_term) noexcept
{
    return (value * sqrt(h_term + (value * value - 3) / 6) / h_term - ab_term_2 * ((value * value - 3) / 6 + 5 / 6 - 2 / (3 * h_term)));
}

double
incomplete_beta_inv_initial_val_1_int_end(const double a, const double b, const double w_term) noexcept
{
    return (a / (a + b * exp(2 * w_term)));
}

double
incomplete_beta_inv_initial_val_1(const double a, const double b, const double t_val, const double sgn_term) noexcept
{ // a > 1.0
    return incomplete_beta_inv_initial_val_1_int_end(a, b,
                                                     incomplete_beta_inv_initial_val_1_int_w(
                                                         sgn_term * incomplete_beta_inv_initial_val_1_int_begin(t_val),
                                                         incomplete_beta_inv_initial_val_1_int_ab2(a, b),
                                                         incomplete_beta_inv_initial_val_1_int_h(
                                                             incomplete_beta_inv_initial_val_1_int_ab1(a, b))));
}

//
// a,b else

double
incomplete_beta_inv_initial_val_2_s1(const double a, const double b) noexcept
{
    return (pow(a / (a + b), a) / a);
}

double
incomplete_beta_inv_initial_val_2_s2(const double a, const double b) noexcept
{
    return (pow(b / (a + b), b) / b);
}

double
incomplete_beta_inv_initial_val_2(const double a, const double b, const double p, const double s_1, const double s_2) noexcept
{
    return (p <= s_1 / (s_1 + s_2) ? pow(p * (s_1 + s_2) * a, 1 / a) : 1 - pow(p * (s_1 + s_2) * b, 1 / b));
}

// initial value


double
incomplete_beta_inv_initial_val(const double a, const double b, const double p) noexcept
{

    if (a > 1.0 && b > 1.0)
        return incomplete_beta_inv_initial_val_1(a, b, incomplete_beta_inv_initial_val_1_tval(p), p < 0.5 ? 1 : -1);
    else {
        if (p > 0.5)
            return 1 - incomplete_beta_inv_initial_val_2(b, a, 1 - p,
                                                         incomplete_beta_inv_initial_val_2_s1(b, a),
                                                         incomplete_beta_inv_initial_val_2_s2(b, a));
        else
            return incomplete_beta_inv_initial_val_2(a, b, p,
                                                     incomplete_beta_inv_initial_val_2_s1(a, b),
                                                     incomplete_beta_inv_initial_val_2_s2(a, b));
    }
}

//
// Halley recursion


double
incomplete_beta_inv_err_val(const double value, const double a, const double b, const double p) noexcept
{ // err_val = f(x)
    return (incbeta(a, b, value) - p);
}


double
incomplete_beta_inv_deriv_1(const double value, const double a, const double b, const double lb) noexcept
{ // derivative of the incomplete beta function w.r.t. x
    if (std::numeric_limits<double>::min() > std::abs(value)) // indistinguishable from zero
        return 0.0;
    else {
        if (std::numeric_limits<double>::min() > std::abs(1.0 - value)) // indistinguishable from one
            return 0.0;
        return exp((a - 1.0) * log(value) + (b - 1.0) * log(1.0 - value) - lb);
    }
}


double
incomplete_beta_inv_deriv_2(const double value, const double a, const double b, const double deriv_1) noexcept
{ // second derivative of the incomplete beta function w.r.t. x
    return (deriv_1 * ((a - 1.0) / value - (b - 1.0) / (1.0 - value)));
}

double
incomplete_beta_inv_ratio_val_1(const double value, const double a, const double b, const double p, const double deriv_1) noexcept
{
    return (incomplete_beta_inv_err_val(value, a, b, p) / deriv_1);
}


double
incomplete_beta_inv_ratio_val_2(const double value, const double a, const double b, const double deriv_1) noexcept
{
    return (incomplete_beta_inv_deriv_2(value, a, b, deriv_1) / deriv_1);
}


double
incomplete_beta_inv_halley(const double ratio_val_1, const double ratio_val_2) noexcept
{
    return (ratio_val_1 / std::max(0.8, std::min(1.2, 1 - 0.5 * ratio_val_1 * ratio_val_2)));
}

double
incomplete_beta_inv_recur(const double value, const double a, const double b, const double p, const double deriv_1,
                          const double lb, const int iter_count) noexcept
{
    if (std::numeric_limits<double>::min() > std::abs(deriv_1)) // derivative = 0
        return incomplete_beta_inv_decision(value, a, b, p, 
                                            0.0, lb, GCEM_INCML_BETA_INV_MAX_ITER + 1);
    else
        return incomplete_beta_inv_decision(value, a, b, p,
                                            incomplete_beta_inv_halley(
                                                incomplete_beta_inv_ratio_val_1(value, a, b, p, deriv_1),
                                                incomplete_beta_inv_ratio_val_2(value, a, b, deriv_1)),
                                            lb, iter_count);
}

double
incomplete_beta_inv_decision(const double value, const double a, const double b, const double p, const double direc,
                             const double lb_val, const int iter_count) noexcept
{
    if (iter_count <= GCEM_INCML_BETA_INV_MAX_ITER)
        return incomplete_beta_inv_recur(value - direc, a, b, p,
                                         incomplete_beta_inv_deriv_1(value, a, b, lb_val),
                                         lb_val, iter_count + 1);
    else
        return value - direc;
}


double
incomplete_beta_inv_begin(const double initial_val, const double a, const double b, const double p, const double lb_val) noexcept
{
    return incomplete_beta_inv_recur(initial_val, a, b, p,
                                     incomplete_beta_inv_deriv_1(initial_val, a, b, lb_val),
                                     lb_val, 1);
}


double
incomplete_beta_inv_check(const double a, const double b, const double p) noexcept
{
    if (std::isnan(a) || std::isnan(b) || std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();
    else {
        if (std::numeric_limits<double>::min() > p)
            return 0.0;
        else if (std::numeric_limits<double>::min() > std::abs(1.0 - p))
            return 1.0;
        return incomplete_beta_inv_begin(incomplete_beta_inv_initial_val(a, b, p),
                                         a, b, p, log(std::beta(a, b)));
    }
}


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
incomplete_beta_inv(const double a, const double b, const double p) noexcept
{
    return incomplete_beta_inv_check(a, b, p);
}
