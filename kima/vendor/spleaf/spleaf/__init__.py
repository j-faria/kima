# -*- coding: utf-8 -*-

# Copyright 2019-2022 Jean-Baptiste Delisle
#
# This file is part of spleaf.
#
# spleaf is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# spleaf is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with spleaf.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ['Spleaf']

import numpy as np
from . import libspleaf
from .__info__ import __version__


class Spleaf():
  r"""
  Symmetric S+LEAF (semiseparable + leaf) matrix.

  A symmetric S+LEAF matrix :math:`C` is defined as

  .. math:: C = A + \mathrm{tril}(U V^T) + \mathrm{triu}(V U^T) + F,

  where
  :math:`A` is the diagonal part,
  the matrices :math:`U` and :math:`V` represent
  the symmetric semiseparable part,
  and :math:`F` is the symmetric leaf part of :math:`C`.

  A Cholesky decomposition of :math:`C` is performed

  .. math:: C = L D L^T,

  where
  :math:`D` is a diagonal matrix
  and :math:`L` is a lower triangular matrix which itself decomposes as

  .. math:: L = \mathbb{I} + \mathrm{tril}(U W^T) + G,

  with
  :math:`U` and :math:`W` representing the
  strictly lower triangular semiseparable part,
  and :math:`G` the strictly lower triangular leaf part of :math:`L`.

  Parameters
  ----------
  A : (n,) ndarray
    Diagonal of the matrix.
  U, V : (n, r) ndarrays
    Symmetric semiseparable part of the matrix,
    with preconditioning matrix `phi`.
    At row i, column j < i,
    the semiseparable part is
    ``np.sum(U[i] * V[j] * np.prod(phi[j:i-1], axis=0))``
  phi : (n-1, r) ndarray
    Preconditioning matrix for the semiseparable part.
  offsetrow : (n,) ndarray
    Offsets for the storage of each row in `F` (leaf part).
  b : (n,) ndarray
    Number of non-zero values left to the diagonal at each row of `F`.
  F : ndarray
    Symmetric leaf part of the matrix,
    stored in strictly lower triangular form,
    and in row major order.
    For ``i-b[i] <= j < i``, the non-zero value
    :math:`F_{i,j}` is stored at index ``offsetrow[i] + j`` in `F`.
    (i.e. offsetrow should be defined as ``offsetrow = np.cumsum(b-1) + 1``).
  copy : bool
    Whether to copy arrays.

  Attributes
  ----------
  n : int
    Size of the matrix.
  r : int
    Rank of the semiseparable part (number of components).
  A, U, V, phi, offsetrow, b, F : ndarrays
    See parameters.
  D : (n,) ndarray
    Diagonal part of the Cholesky decomposition of the matrix.
  W : (n,r) ndarray
    Semiseparable part of the Cholesky decomposition of the matrix.
  G : ndarray
    Leaf part of the Cholesky decomposition of the matrix
    (stored in the same way as `F`).
  """

  def __init__(self, A, U, V, phi, offsetrow, b, F, copy=False):
    self.A = self._copy(A, copy)
    self.U = self._copy(U, copy)
    self.V = self._copy(V, copy)
    self.phi = self._copy(phi, copy)
    self.offsetrow = self._copy(offsetrow, copy)
    self.b = self._copy(b, copy)
    self.F = self._copy(F, copy)

    self.n = A.size
    self.r = U.shape[1]

    # Cholesky decomposition
    self.D = np.empty_like(A)
    self.W = np.empty_like(V)
    self.G = np.empty_like(F)
    self._S = np.empty(self.n * self.r * self.r)
    self._Z = np.empty((self.F.size + self.n) * self.r)

    libspleaf.spleaf_cholesky(self.n, self.r, self.offsetrow, self.b, self.A,
      self.U, self.V, self.phi, self.F, self.D, self.W, self.G, self._S,
      self._Z)

    # Back propagation (no init)
    self._grad_A = None
    self._grad_U = None
    self._grad_V = None
    self._grad_phi = None
    self._grad_F = None

    self._grad_D = None
    self._grad_Ucho = None
    self._grad_W = None
    self._grad_phicho = None
    self._grad_G = None

    # Kernel derivative
    self._dU = None
    self._dV = None

    # Other
    self._logdet_value = None
    self._sqD_value = None
    self._x_dotL = None
    self._f_dotL = np.empty(self.n * self.r)
    self._x_solveL = None
    self._f_solveL = np.empty(self.n * self.r)
    self._x_dotLT = None
    self._g_dotLT = np.empty(self.n * self.r)
    self._x_solveLT = None
    self._g_solveLT = np.empty(self.n * self.r)

  def _copy(self, x, copy):
    r"""
    Perform a copy if required else pass the object.
    """
    if copy and isinstance(x, np.ndarray):
      return (x.copy())
    else:
      return (x)

  def set_param(self, A, U, V, phi, F, copy=False):
    r"""
    Update the initial parameters (`A`, `U`, `V`, `phi`, `F`)
    and recompute the Cholesky decomposition of the matrix.
    """

    self.A = self._copy(A, copy)
    self.U = self._copy(U, copy)
    self.V = self._copy(V, copy)
    self.phi = self._copy(phi, copy)
    self.F = self._copy(F, copy)

    # Cholesky decomposition
    libspleaf.spleaf_cholesky(self.n, self.r, self.offsetrow, self.b, self.A,
      self.U, self.V, self.phi, self.F, self.D, self.W, self.G, self._S,
      self._Z)

    # Re-init logdet/sqD
    self._logdet_value = None
    self._sqD_value = None

  def expand(self):
    r"""
    Expand the matrix as a full (n x n) matrix.

    Warnings
    --------
    This non-sparse representation has a
    :math:`\mathcal{O}(n^2)` cost and footprint.
    """

    C = np.diag(self.A)
    for i in range(self.n):
      for j in range(i - self.b[i], i):
        C[i, j] = self.F[self.offsetrow[i] + j]
      cumphi = np.ones(self.r)
      for j in range(i - 1, -1, -1):
        cumphi *= self.phi[j]
        C[i, j] += np.sum(cumphi * self.U[i] * self.V[j])
        C[j, i] = C[i, j]
    return (C)

  def expandL(self):
    r"""
    Expand :math:`L` as a full (n x n) matrix.

    Warnings
    --------
    This non-sparse representation has a
    :math:`\mathcal{O}(n^2)` cost and footprint.
    """

    L = np.identity(self.n)
    for i in range(self.n):
      for j in range(i - self.b[i], i):
        L[i, j] = self.G[self.offsetrow[i] + j]
      cumphi = np.ones(self.r)
      for j in range(i - 1, -1, -1):
        cumphi *= self.phi[j]
        L[i, j] += np.sum(cumphi * self.U[i] * self.W[j])
    return (L)

  def expandInv(self):
    r"""
    Expand the inverse of the matrix as a full (n x n) matrix.

    Warnings
    --------
    This non-sparse representation has a
    :math:`\mathcal{O}(n^2)` cost and footprint.
    """

    invC = np.empty((self.n, self.n))
    ei = np.zeros(self.n)
    for i in range(self.n):
      ei[i] = 1.0
      invC[:, i] = self.solveLT(self.solveL(ei) / self.D)
      ei[i] = 0.0
    return (invC)

  def expandInvL(self):
    r"""
    Expand :math:`L^{-1}` as a full (n x n) matrix.

    Warnings
    --------
    This non-sparse representation has a
    :math:`\mathcal{O}(n^2)` cost and footprint.
    """

    invL = np.empty((self.n, self.n))
    ei = np.zeros(self.n)
    for i in range(self.n):
      ei[i] = 1.0
      invL[:, i] = self.solveL(ei)
      ei[i] = 0.0
    return (invL)

  def logdet(self):
    r"""
    Compute the (natural) logarithm of the determinant of the matrix.

    Returns
    -------
    logdet : float
      The natural logarithm of the determinant.

    Notes
    -----
    This is a lazy computation (the result is stored for future calls).
    """

    if self._logdet_value is None:
      self._logdet_value = np.sum(np.log(self.D))
    return (self._logdet_value)

  def sqD(self):
    r"""
    Compute the square-root of `D`.

    Returns
    -------
    sqD : (n,) ndarray
      The square-root of `D`.

    Notes
    -----
    This is a lazy computation (the result is stored for future calls).
    """

    if self._sqD_value is None:
      self._sqD_value = np.sqrt(self.D)
    return (self._sqD_value)

  def dotL(self, x, copy=False):
    r"""
    Compute :math:`y = L x`.

    Parameters
    ----------
    x : (n,) ndarray
      The input vector :math:`x`.
    copy : bool
      Whether to copy arrays.

    Returns
    -------
    y : (n,) ndarray
      The dot product :math:`y = L x`.
    """

    self._x_dotL = self._copy(x, copy)
    y = np.empty_like(x)
    libspleaf.spleaf_dotL(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, x, y, self._f_dotL)
    return (y)

  def solveL(self, y, copy=False):
    r"""
    Solve for :math:`x = L^{-1} y`.

    Parameters
    ----------
    y : (n,) ndarray
      The righthand side vector :math:`y`.
    copy : bool
      Whether to copy arrays.

    Returns
    -------
    x : (n,) ndarray
      The solution :math:`x = L^{-1} y`.
    """

    self._x_solveL = np.empty_like(y)
    libspleaf.spleaf_solveL(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, y, self._x_solveL, self._f_solveL)
    return (self._copy(self._x_solveL, copy))

  def dotLT(self, x, copy=False):
    r"""
    Compute :math:`y = L^T x`.

    Parameters
    ----------
    x : (n,) ndarray
      The input vector :math:`x`.
    copy : bool
      Whether to copy arrays.

    Returns
    -------
    y : (n,) ndarray
      The dot product :math:`y = L^T x`.
    """

    self._x_dotLT = self._copy(x, copy)
    y = np.empty_like(x)
    libspleaf.spleaf_dotLT(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, x, y, self._g_dotLT)
    return (y)

  def solveLT(self, y, copy=False):
    r"""
    Solve for :math:`x = L^{-T} y`.

    Parameters
    ----------
    y : (n,) ndarray
      The righthand side vector :math:`y`.
    copy : bool
      Whether to copy arrays.

    Returns
    -------
    x : (n,) ndarray
      The solution :math:`x = L^{-T} y`.
    """

    self._x_solveLT = np.empty_like(y)
    libspleaf.spleaf_solveLT(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, y, self._x_solveLT, self._g_solveLT)
    return (self._copy(self._x_solveLT, copy))

  def init_grad(self):
    r"""
    Initialize (or reinitialize) the backward propagation of the gradient.
    """

    self._grad_A = np.zeros_like(self.A)
    self._grad_U = np.zeros_like(self.U)
    self._grad_V = np.zeros_like(self.V)
    self._grad_phi = np.zeros_like(self.phi)
    self._grad_F = np.zeros_like(self.F)

    self._grad_D = np.zeros_like(self.D)
    self._grad_Ucho = np.zeros_like(self.U)
    self._grad_W = np.zeros_like(self.W)
    self._grad_phicho = np.zeros_like(self.phi)
    self._grad_G = np.zeros_like(self.G)

  def cholesky_back(self):
    r"""
    Backward propagation of the gradient for the Cholesky decomposition.

    Propagate the gradient of a function with respect to `D`
    and to the components of :math:`L` (`U`, `W`, `phi`, `G`),
    to its gradient with respect to the initial components of the matrix
    (`A`, `U`, `V`, `phi`, `F`).

    Use :func:`grad_param` to get the results.
    """

    libspleaf.spleaf_cholesky_back(self.n, self.r, self.offsetrow, self.b,
      self.D, self.U, self.W, self.phi, self.G, self._grad_D, self._grad_Ucho,
      self._grad_W, self._grad_phicho, self._grad_G, self._grad_A,
      self._grad_U, self._grad_V, self._grad_phi, self._grad_F, self._S,
      self._Z)

  def dotL_back(self, grad_y):
    r"""
    Backward propagation of the gradient for :func:`dotL`.

    Propagate the gradient of a function with respect to `y`,
    to its gradient with respect to `x`
    and to the components of :math:`L`
    (`U`, `W`, `phi`, `G`).

    Use :func:`cholesky_back` to propagate the gradient
    to the initial components of the matrix
    (`A`, `U`, `V`, `phi`, `F`).

    Parameters
    ----------
    grad_y : (n,) ndarray
      Gradient of the function with respect to `y`.

    Returns
    -------
    grad_x : (n,) ndarray
      Gradient of the function with respect to `x`.
    """

    grad_x = np.empty_like(grad_y)
    libspleaf.spleaf_dotL_back(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, self._x_dotL, grad_y, self._grad_Ucho,
      self._grad_W, self._grad_phicho, self._grad_G, grad_x, self._f_dotL)
    return (grad_x)

  def solveL_back(self, grad_x):
    r"""
    Backward propagation of the gradient for :func:`solveL`.

    Propagate the gradient of a function with respect to `x`,
    to its gradient with respect to `y`
    and to the components of :math:`L`
    (`U`, `W`, `phi`, `G`).

    Use :func:`cholesky_back` to propagate the gradient
    to the initial components of the matrix
    (`A`, `U`, `V`, `phi`, `F`).

    Parameters
    ----------
    grad_x : (n,) ndarray
      Gradient of the function with respect to `x`.

    Returns
    -------
    grad_y : (n,) ndarray
      Gradient of the function with respect to `y`.
    """

    grad_y = np.empty_like(grad_x)
    libspleaf.spleaf_solveL_back(self.n, self.r, self.offsetrow, self.b,
      self.U, self.W, self.phi, self.G, self._x_solveL, grad_x,
      self._grad_Ucho, self._grad_W, self._grad_phicho, self._grad_G, grad_y,
      self._f_solveL)
    return (grad_y)

  def dotLT_back(self, grad_y):
    r"""
    Backward propagation of the gradient for :func:`dotLT`.

    Propagate the gradient of a function with respect to `y`,
    to its gradient with respect to `x`
    and to the components of :math:`L`
    (`U`, `W`, `phi`, `G`).

    Use :func:`cholesky_back` to propagate the gradient
    to the initial components of the matrix
    (`A`, `U`, `V`, `phi`, `F`).

    Parameters
    ----------
    grad_y : (n,) ndarray
      Gradient of the function with respect to `y`.

    Returns
    -------
    grad_x : (n,) ndarray
      Gradient of the function with respect to `x`.
    """

    grad_x = np.empty_like(grad_y)
    libspleaf.spleaf_dotLT_back(self.n, self.r, self.offsetrow, self.b, self.U,
      self.W, self.phi, self.G, self._x_dotLT, grad_y, self._grad_Ucho,
      self._grad_W, self._grad_phicho, self._grad_G, grad_x, self._g_dotLT)
    return (grad_x)

  def solveLT_back(self, grad_x):
    r"""
    Backward propagation of the gradient for :func:`solveLT`.

    Propagate the gradient of a function with respect to `x`,
    to its gradient with respect to `y`
    and to the components of :math:`L`
    (`U`, `W`, `phi`, `G`).

    Use :func:`cholesky_back` to propagate the gradient
    to the initial components of the matrix
    (`A`, `U`, `V`, `phi`, `F`).

    Parameters
    ----------
    grad_x : (n,) ndarray
      Gradient of the function with respect to `x`.

    Returns
    -------
    grad_y : (n,) ndarray
      Gradient of the function with respect to `y`.
    """

    grad_y = np.empty_like(grad_x)
    libspleaf.spleaf_solveLT_back(self.n, self.r, self.offsetrow, self.b,
      self.U, self.W, self.phi, self.G, self._x_solveLT, grad_x,
      self._grad_Ucho, self._grad_W, self._grad_phicho, self._grad_G, grad_y,
      self._g_solveLT)
    return (grad_y)

  def grad_param(self, *args, **kwargs):
    r"""
    Gradient of a function with respect to
    the initial parameters (`A`, `U`, `V`, `phi`, `F`),
    after a call to :func:`cholesky_back`.

    Returns
    -------
    grad_A : (n,) ndarray
      Gradient of the function with respect to `A`.
    grad_U : (n, r) ndarray
      Gradient of the function with respect to `U`.
    grad_V : (n, r) ndarray
      Gradient of the function with respect to `V`.
    grad_phi : (n-1, r) ndarray
      Gradient of the function with respect to `phi`.
    grad_F : ndarray
      Gradient of the function with respect to `F`.
    """

    return (self._grad_A, self._grad_U, self._grad_V, self._grad_phi,
      self._grad_F)

  def chi2(self, y):
    r"""
    Compute :math:`\chi^2 = y^T C^{-1} y`
    for a given vector of residuals :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of residuals :math:`y`.

    Returns
    -------
    chi2 : float
      The :math:`\chi^2`.
    """

    x = self.solveL(y)
    return (np.sum(x * x / self.D))

  def loglike(self, y):
    r"""
    Compute the (natural) logarithm of the likelihood
    for a given vector of residuals :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of residuals :math:`y`.

    Returns
    -------
    loglike : float
      The natural logarithm of the likelihood.
    """

    return (-0.5 *
      (self.chi2(y) + self.logdet() + self.n * np.log(2.0 * np.pi)))

  def chi2_grad(self, *args, **kwargs):
    r"""
    Compute the gradient of the :math:`\chi^2` (:func:`chi2`)
    with respect to the residuals and to the initial parameters
    (see :func:`grad_param`).

    Returns
    -------
    grad_y : (n,) ndarray
      Gradient of the :math:`\chi^2` with respect to the residuals :math:`y`.
    ...
      See :func:`grad_param`.
    """

    self.init_grad()
    xoD = self._x_solveL / self.D
    grad_x = 2.0 * xoD
    self._grad_D = -xoD * xoD
    grad_y = self.solveL_back(grad_x)
    self.cholesky_back()
    return (grad_y, self.grad_param(*args, **kwargs))

  def loglike_grad(self, *args, **kwargs):
    r"""
    Compute the gradient of the log-likelihood (:func:`loglike`)
    with respect to the residuals and to the initial parameters
    (see :func:`grad_param`).

    Returns
    -------
    grad_y : (n,) ndarray
      Gradient of the log-likelihood with respect to the residuals :math:`y`.
    ...
      See :func:`grad_param`.
    """

    self.init_grad()
    xoD = self._x_solveL / self.D
    grad_x = -xoD
    self._grad_D = 0.5 * (xoD * xoD - 1.0 / self.D)
    grad_y = self.solveL_back(grad_x)
    self.cholesky_back()
    return (grad_y, self.grad_param(*args, **kwargs))

  def self_conditional(self, y, calc_cov=False, index=None):
    r"""
    Compute the conditional mean and covariance
    of the Gaussian process corresponding to the semiseparable part
    of the covariance matrix, knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    index : (r',) ndarray or None
      Vector (of type int) giving the indices of semiseparable terms
      that should be considered for the Gaussian process.
      Other terms (semiseparable or leaf) are considered as noise.
      If index is None, all semiserable terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n,) ndarray
      The vector of conditional mean values.
    cov : (n, n) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the conditional mean scales as
    :math:`\mathcal{O}(n)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n^2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n^3)`.
    """

    if index is None:
      index = np.arange(self.r)
    ri = index.size

    u = self.solveLT(self.solveL(y) / self.D)
    y2 = np.empty(self.n)
    libspleaf.spleaf_dotsep(self.n, self.r, ri, index, self.U, self.V,
      self.phi, u, y2)

    if not calc_cov:
      return (y2)

    K = np.empty((self.n, self.n))
    libspleaf.spleaf_expandsep(self.n, self.r, ri, index, self.U, self.V,
      self.phi, K)
    H = np.array([self.solveL(Kk) / self.sqD() for Kk in K])

    if calc_cov == 'diag':
      return (y2, np.diag(K) - np.sum(H * H, axis=1))

    return (y2, K - H @ H.T)

  def conditional(self,
    y,
    U2,
    V2,
    phi2,
    ref2left,
    phi2left,
    phi2right,
    calc_cov=False,
    index=None):
    r"""
    Conditional mean and covariance at new abscissas
    of the Gaussian process corresponding to the semiseparable part
    of the covariance matrix, knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    U2, V2 : (n2, r) ndarrays
      Symmetric semiseparable part at new abscissas,
      with preconditioning matrix `phi2`.
    phi2 : (n2-1, r) ndarray
      Preconditioning matrix for the semiseparable part at new abscissas.
    ref2left : (n2,) ndarray
      Indices of the closest original abscissas to the left of new abscissas.
    phi2left : (n2, r) ndarray
      Preconditioning matrix linking new abscissas
      with their closest original abscissas to the left.
    phi2right : (n2, r) ndarray
      Preconditioning matrix linking new abscissas
      with their closest original abscissas to the right.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    index : (r',) ndarray or None
      Vector (of type int) giving the indices of semiseparable terms
      that should be considered for the Gaussian process.
      Other terms (semiseparable or leaf) are considered as noise.
      If index is None, all semiserable terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n2,) ndarray
      The vector of conditional mean values.
    cov : (n2, n2) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n2,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the conditional mean scales as
    :math:`\mathcal{O}(n+n_2)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n n_2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n n_2^2)`.
    """

    if index is None:
      index = np.arange(self.r)
    ri = index.size

    u = self.solveLT(self.solveL(y) / self.D)
    n2 = U2.shape[0]
    y2 = np.empty(n2)
    libspleaf.spleaf_dotsepmixt(self.n, n2, self.r, ri, index, self.U, self.V,
      self.phi, U2, V2, ref2left, phi2left, phi2right, u, y2)

    if not calc_cov:
      return (y2)

    Km = np.empty((n2, self.n))
    libspleaf.spleaf_expandsepmixt(self.n, n2, self.r, ri, index, self.U,
      self.V, self.phi, U2, V2, ref2left, phi2left, phi2right, Km)
    Hm = np.array([self.solveL(Kmk) / self.sqD() for Kmk in Km])

    if calc_cov == 'diag':
      return (y2,
        np.sum(U2[:, index] * V2[:, index], axis=1) - np.sum(Hm * Hm, axis=1))

    K = np.empty((n2, n2))
    libspleaf.spleaf_expandsep(n2, self.r, ri, index, U2, V2, phi2, K)
    return (y2, K - Hm @ Hm.T)

  def self_conditional_derivative(self,
    y,
    dU=None,
    dV=None,
    calc_cov=False,
    index=None):
    r"""
    Conditional mean and covariance
    of the derivative of the Gaussian process corresponding to the semiseparable part
    of the covariance matrix, knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    dU, dV : (n, r) ndarrays or None
      Semiseparable part of the derivatives.
      If None, the values from a previous call are kept.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    index : (r',) ndarray or None
      Vector (of type int) giving the indices of semiseparable terms
      that should be considered for the Gaussian process.
      Other terms (semiseparable or leaf) are considered as noise.
      If index is None, all semiserable terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n, ) ndarray
      The vector of derivative conditional mean values.
    cov : (n, n) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the derivative conditional mean scales as
    :math:`\mathcal{O}(n)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n^2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n^3)`.
    """

    if index is None:
      index = np.arange(self.r)
    ri = index.size

    if dU is not None:
      self._dU = dU
    if dV is not None:
      self._dV = dV

    u = self.solveLT(self.solveL(y) / self.D)
    dy = np.empty(self.n)

    libspleaf.spleaf_dotantisep(self.n, self.r, ri, index, self._dU, self.V,
      self.phi, u, dy)

    if not calc_cov:
      return (dy)

    dK = np.empty((self.n, self.n))
    d2K = np.empty((self.n, self.n))
    libspleaf.spleaf_expandantisep(self.n, self.r, ri, index, self._dU, self.V,
      self.phi, dK)
    libspleaf.spleaf_expandsep(self.n, self.r, ri, index, self._dU, self._dV,
      self.phi, d2K)
    H = np.array([self.solveL(dKk) / self.sqD() for dKk in dK])

    if calc_cov == 'diag':
      return (dy, np.diag(d2K) - np.sum(H * H, axis=1))

    return (dy, d2K - H @ H.T)

  def conditional_derivative(self,
    y,
    dU2,
    V2,
    dV2,
    phi2,
    ref2left,
    phi2left,
    phi2right,
    dU=None,
    calc_cov=False,
    index=None):
    r"""
    Conditional mean and covariance at new abscissas
    of the derivative of the Gaussian process corresponding to the semiseparable part
    of the covariance matrix, knowning the observed values :math:`y`.

    Parameters
    ----------
    y : (n,) ndarray
      The vector of observed values :math:`y`.
    dU2, V2, dV2 : (n2, r) ndarrays
      Semiseparable part at new abscissas,
      with preconditioning matrix `phi2`.
    phi2 : (n2-1, r) ndarray
      Preconditioning matrix for the semiseparable part at new abscissas.
    ref2left : (n2,) ndarray
      Indices of the closest original abscissas to the left of new abscissas.
    phi2left : (n2, r) ndarray
      Preconditioning matrix linking new abscissas
      with their closest original abscissas to the left.
    phi2right : (n2, r) ndarray
      Preconditioning matrix linking new abscissas
      with their closest original abscissas to the right.
    dU : (n, r) ndarray or None
      Semiseparable part of the derivative.
      If None, the value from a previous call is kept.
    calc_cov : False (default), True, or 'diag'
      Whether to output only the conditional mean (False),
      the mean and full covariance matrix (True),
      or the mean and main diagonal of the covariance matrix ('diag').
    index : (r',) ndarray or None
      Vector (of type int) giving the indices of semiseparable terms
      that should be considered for the Gaussian process.
      Other terms (semiseparable or leaf) are considered as noise.
      If index is None, all semiserable terms are considered for the
      Gaussian process.

    Returns
    -------
    mu : (n2,) ndarray
      The vector of derivative conditional mean values.
    cov : (n2, n2) ndarray
      Full covariance matrix (if calc_cov is True).
    var : (n2,) ndarray
      Main diagonal of the covariance matrix (if calc_cov is 'diag').

    Warnings
    --------
    While the computational cost of the derivative conditional mean scales as
    :math:`\mathcal{O}(n+n_2)`,
    the computational cost of the variance scales as
    :math:`\mathcal{O}(n n_2)`,
    and the computational cost of the full covariance scales as
    :math:`\mathcal{O}(n n_2^2)`.
    """

    if index is None:
      index = np.arange(self.r)
    ri = index.size

    if dU is not None:
      self._dU = dU

    u = self.solveLT(self.solveL(y) / self.D)
    n2 = dU2.shape[0]
    y2 = np.empty(n2)
    libspleaf.spleaf_dotsepmixt(self.n, n2, self.r, ri, index, -self._dU,
      self.V, self.phi, dU2, V2, ref2left, phi2left, phi2right, u, y2)

    if not calc_cov:
      return (y2)

    dKm = np.empty((n2, self.n))
    libspleaf.spleaf_expandsepmixt(self.n, n2, self.r, ri, index, -self._dU,
      self.V, self.phi, dU2, V2, ref2left, phi2left, phi2right, dKm)
    Hm = np.array([self.solveL(dKmk) / self.sqD() for dKmk in dKm])

    if calc_cov == 'diag':
      return (y2, np.sum(dU2[:, index] * dV2[:, index], axis=1) -
        np.sum(Hm * Hm, axis=1))

    d2K = np.empty((n2, n2))
    libspleaf.spleaf_expandsep(n2, self.r, ri, index, dU2, dV2, phi2, d2K)
    return (y2, d2K - Hm @ Hm.T)

  def sample(self, nreal=None):
    r"""
    Generate random samples from the covariance matrix.

    Parameters
    ----------
    nreal : int
      Number of realizations to generate.
      If None, only one realization is generated.

    Returns
    -------
    y : (nreal, n) or (n,) ndarray
      Matrix of the samples (or vector if nreal is None).
    """

    single = False
    if nreal is None:
      single = True
      nreal = 1
    u = np.random.normal(size=(nreal, self.n))
    y = np.array([self.dotL(self.sqD() * uk) for uk in u])
    if single:
      return (y[0])
    else:
      return (y)
