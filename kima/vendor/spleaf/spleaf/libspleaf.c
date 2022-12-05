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

#include "libspleaf.h"

void spleaf_cholesky(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *A, double *U, double *V, double *phi, double *F,
  // Output
  double *D, double *W, double *G,
  // Temporary
  double *S, double *Z)
{
  // Cholesky decomposition of the (n x n) symmetric S+LEAF
  // (semiseparable + leaf) matrix C
  // defined as
  // C = A + Sep + F
  // with
  // * A: the diagonal part of C, stored as a vector of size n.
  // * Sep: the symmetric semiseparable part of C.
  //   For i > j,
  //   Sep_{i,j} = Sep_{j,i}
  //             = Sum_{s=0}^{r-1} U_{i,s} V_{j,s} Prod_{k=j}^{i-1} phi_{k,s}
  //   where U, V are (n x r) matrices, and phi is a (n-1 x r) matrix,
  //   all stored in row major order.
  //   By definition Sep_{i,i} = 0.
  // * F: the symmetric leaf part of C,
  //   stored in strictly lower triangular form, and in row major order.
  //   The i-th row of F is of size b[i], i.e., by definition,
  //   F_{i,j} = 0 for j<i-b[i] and for j=i.
  //   For i-b[i] <= j < i,
  //   the non-zero value F_{i,j} is stored at index (offsetrow[i] + j)
  //   (i.e. offsetrow should be defined as offsetrow = cumsum(b-1) + 1).
  //
  // The Cholesky decomposition of C reads
  // C = L D L^T
  // with
  // L = Lsep + G
  // and
  // * D: diagonal part of the decomposition (vector of size n, like A).
  // * Lsep: the strictly lower triangular semiseparable part of L.
  //   For i > j,
  //   Lsep_{i,j} = Sum_{s=0}^{r-1} U_{i,s} W_{j,s} Prod_{k=j}^{i-1} phi_{k,s},
  //   where U and phi are left unchanged and W is a (n x r) matrix (like V).
  // * G: the strictly lower triangular leaf part of L.
  //   G is stored in the same way as F.

  long i, j, k, s, t;
  long r2 = r * r;
  double SU, GD;

  // Copy A -> D, V -> W, F -> G
  memcpy(D, A, n * sizeof(double));
  memcpy(W, V, n * r * sizeof(double));
  memcpy(G, F, (offsetrow[n - 1] + n - 1) * sizeof(double));

  // Case i = 0
  // Normalize W_{0,s}
  for (s = 0; s < r; s++) {
    W[s] /= D[0];
  }
  // Initialize S
  for (s = 0; s < r2; s++) {
    S[s] = 0.0;
  }

  for (i = 1; i < n; i++) {
    // Initialize Z
    for (s = 0; s < r; s++) {
      Z[r * (offsetrow[i] + 2 * i - b[i]) + s] = 0.0;
    }

    // Compute G_{i,j} (without normalizing by D_j yet)
    // Case j = i-b[i] -> nothing to do
    // Case j > i-b[i]
    for (j = i - b[i] + 1; j < i; j++) {
      // Purely leaf terms
      for (k = MAX(i - b[i], j - b[j]); k < j; k++) {
        // Note that G_{i,k} has not been normalized by D_k
        // but G_{j,k} is already normalized by D_k
        G[offsetrow[i] + j] -= G[offsetrow[i] + k] * G[offsetrow[j] + k];
      }
      // Mixed terms
      for (s = 0; s < r; s++) {
        // Update Z
        // Note that G_{i,j-1} has not been normalized by D_{j-1} yet
        Z[r * (offsetrow[i] + i + j) + s] = phi[r * (j - 1) + s] *
          (Z[r * (offsetrow[i] + i + j - 1) + s] +
           G[offsetrow[i] + j - 1] * W[r * (j - 1) + s]);
        G[offsetrow[i] + j] -= U[r * j + s] * Z[r * (offsetrow[i] + i + j) + s];
      }
    }

    // Compute D_i, W_{i,s} (without normalizing by D_i yet)
    for (s = 0; s < r; s++) {
      // Compute Z_{i,i,s}
      // Note that G_{i,i-1} has not been normalized by D_{i-1} yet
      if (b[i] > 0) {
        Z[r * (offsetrow[i] + 2 * i) + s] = phi[r * (i - 1) + s] *
          (Z[r * (offsetrow[i] + 2 * i - 1) + s] +
           G[offsetrow[i] + i - 1] * W[r * (i - 1) + s]);
      }
      // Compute S_{i,s,.} U_{i,.}
      SU = 0.0;
      for (t = 0; t < r; t++) {
        // Update S
        S[r2 * i + r * s + t] = phi[r * (i - 1) + s] * phi[r * (i - 1) + t] *
          (S[r2 * (i - 1) + r * s + t] +
           W[r * (i - 1) + s] * D[i - 1] * W[r * (i - 1) + t]);
        SU += S[r2 * i + r * s + t] * U[r * i + t];
      }
      // Compute D_i (omiting purely leaf terms)
      D[i] -= U[r * i + s] * (SU + 2.0 * Z[r * (offsetrow[i] + 2 * i) + s]);
      // Compute W_{i,s}
      W[r * i + s] -= SU + Z[r * (offsetrow[i] + 2 * i) + s];
    }

    // Normalize G_{i,j} and compute purely leaf terms in D_i
    for (j = i - b[i]; j < i; j++) {
      // Normalize G_{i,j} (+ store non-normalized value)
      GD = G[offsetrow[i] + j];
      G[offsetrow[i] + j] /= D[j];
      // Purely leaf terms in D_i
      D[i] -= GD * G[offsetrow[i] + j];
    }

    // Normalize W_{i,s}
    for (s = 0; s < r; s++) {
      W[r * i + s] /= D[i];
    }
  }
}

void spleaf_dotL(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x,
  // Output
  double *y,
  // Temporary
  double *f)
{
  // Compute y = L x,
  // where L comes from the Cholesky decomposition C = L D L^T
  // of a symmetric S+LEAF matrix C using spleaf_cholesky.

  long i, j, s;

  // Copy x -> y
  memcpy(y, x, n * sizeof(double));

  // Initialize f
  for (s = 0; s < r; s++) {
    f[s] = 0.0;
  }

  // Compute y = L x
  for (i = 1; i < n; i++) {
    // Semiseparable terms
    for (s = 0; s < r; s++) {
      // Update f
      f[r * i + s] = phi[r * (i - 1) + s] *
        (f[r * (i - 1) + s] + W[r * (i - 1) + s] * x[i - 1]);
      y[i] += U[r * i + s] * f[r * i + s];
    }
    // Leaf terms
    for (j = i - b[i]; j < i; j++) {
      y[i] += G[offsetrow[i] + j] * x[j];
    }
  }
}

void spleaf_solveL(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *y,
  // Output
  double *x,
  // Temporary
  double *f)
{
  // Solve for x = L^-1 y,
  // where L comes from the Cholesky decomposition C = L D L^T
  // of a symmetric S+LEAF matrix C using spleaf_cholesky.

  long i, j, s;

  // Copy y -> x
  memcpy(x, y, n * sizeof(double));

  // Initialize f
  for (s = 0; s < r; s++) {
    f[s] = 0.0;
  }

  // Solve for x = L^-1 y
  for (i = 1; i < n; i++) {
    // Semiseparable terms
    for (s = 0; s < r; s++) {
      // Update f
      f[r * i + s] = phi[r * (i - 1) + s] *
        (f[r * (i - 1) + s] + W[r * (i - 1) + s] * x[i - 1]);
      x[i] -= U[r * i + s] * f[r * i + s];
    }
    // Leaf terms
    for (j = i - b[i]; j < i; j++) {
      x[i] -= G[offsetrow[i] + j] * x[j];
    }
  }
}

void spleaf_dotLT(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x,
  // Output
  double *y,
  // Temporary
  double *g)
{
  // Compute y = L^T x,
  // where L comes from the Cholesky decomposition C = L D L^T
  // of a symmetric S+LEAF matrix C using spleaf_cholesky.

  long i, j, s;

  // Copy x -> y
  memcpy(y, x, n * sizeof(double));

  // Initialize g
  for (s = 0; s < r; s++) {
    g[(n - 1) * r + s] = 0.0;
  }

  // Leaf terms for n-1
  for (j = n - 1 - b[n - 1]; j < n - 1; j++) {
    y[j] += G[offsetrow[n - 1] + j] * x[n - 1];
  }

  // Loop for i < n-1
  for (i = n - 2; i >= 0; i--) {
    // Semiseparable terms
    for (s = 0; s < r; s++) {
      // Update g
      g[r * i + s] =
        phi[r * i + s] * (g[r * (i + 1) + s] + U[r * (i + 1) + s] * x[i + 1]);
      y[i] += W[r * i + s] * g[r * i + s];
    }
    // Leaf terms
    for (j = i - b[i]; j < i; j++) {
      y[j] += G[offsetrow[i] + j] * x[i];
    }
  }
}

void spleaf_solveLT(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *y,
  // Output
  double *x,
  // Temporary
  double *g)
{
  // Solve for x = L^-T y,
  // where L comes from the Cholesky decomposition C = L D L^T
  // of a symmetric S+LEAF matrix C using spleaf_cholesky.

  long i, j, s;

  // Copy y -> x
  memcpy(x, y, n * sizeof(double));

  // Initialize g
  for (s = 0; s < r; s++) {
    g[(n - 1) * r + s] = 0.0;
  }

  // Leaf terms for n-1
  for (j = n - 1 - b[n - 1]; j < n - 1; j++) {
    x[j] -= G[offsetrow[n - 1] + j] * x[n - 1];
  }

  // Loop for i < n-1
  for (i = n - 2; i >= 0; i--) {
    // Semiseparable terms
    for (s = 0; s < r; s++) {
      // Update g
      g[r * i + s] =
        phi[r * i + s] * (g[r * (i + 1) + s] + U[r * (i + 1) + s] * x[i + 1]);
      x[i] -= W[r * i + s] * g[r * i + s];
    }
    // Leaf terms
    for (j = i - b[i]; j < i; j++) {
      x[j] -= G[offsetrow[i] + j] * x[i];
    }
  }
}

void spleaf_cholesky_back(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *D, double *U, double *W, double *phi, double *G, double *grad_D,
  double *grad_Ucho, double *grad_W, double *grad_phicho, double *grad_G,
  // Output
  double *grad_A, double *grad_U, double *grad_V, double *grad_phi,
  double *grad_F,
  // Temporary
  double *S, double *Z)
{
  // Backward propagation of the gradient for spleaf_cholesky.

  long i, j, k, s, t;
  long r2 = r * r;
  double SU;
  double grad_SU, grad_GD;
  double *grad_S, *grad_Z;
  double grad_tmp;

  grad_S = (double *)malloc(r2 * sizeof(double));
  grad_Z = (double *)malloc(r * sizeof(double));

  // Copy grad_D -> grad_A, grad_Ucho -> grad_U, grad_W -> grad_V,
  // grad_phicho -> grad_phi, grad_G -> grad_F
  memcpy(grad_A, grad_D, n * sizeof(double));
  memcpy(grad_U, grad_Ucho, n * r * sizeof(double));
  memcpy(grad_V, grad_W, n * r * sizeof(double));
  memcpy(grad_phi, grad_phicho, (n - 1) * r * sizeof(double));
  memcpy(grad_F, grad_G, (offsetrow[n - 1] + n - 1) * sizeof(double));

  // Initialize grad_S
  for (s = 0; s < r2; s++) {
    grad_S[s] = 0.0;
  }

  // Reversed main loop
  for (i = n - 1; i > 0; i--) {
    for (s = 0; s < r; s++) {
      // W[r*i+s] /= D[i];
      grad_V[r * i + s] /= D[i];
      grad_A[i] -= W[r * i + s] * grad_V[r * i + s];
    }

    for (j = i - b[i]; j < i; j++) {
      // D[i] -= GD * G[offsetrow[i]+j];
      // G[offsetrow[i]+j] /= D[j];
      grad_GD = -G[offsetrow[i] + j] * grad_A[i];
      grad_F[offsetrow[i] + j] /= D[j];
      grad_F[offsetrow[i] + j] += grad_GD;
      grad_A[j] -= G[offsetrow[i] + j] * grad_F[offsetrow[i] + j];
      // GD = G[offsetrow[i]+j];
      grad_F[offsetrow[i] + j] += grad_GD;
    }

    for (s = 0; s < r; s++) {
      // Precompute SU
      SU = 0.0;
      for (t = 0; t < r; t++) {
        SU += S[r2 * i + r * s + t] * U[r * i + t];
      }
      // W[r*i+s] -= SU + Z[r*(offsetrow[i]+2*i)+s];
      grad_SU = -grad_V[r * i + s];
      grad_Z[s] = -grad_V[r * i + s];
      // D[i] -= U[r*i+s]*(SU + 2.0*Z[r*(offsetrow[i]+2*i)+s]);
      grad_U[r * i + s] -=
        (SU + 2.0 * Z[r * (offsetrow[i] + 2 * i) + s]) * grad_A[i];
      grad_SU -= U[r * i + s] * grad_A[i];
      grad_Z[s] -= 2.0 * U[r * i + s] * grad_A[i];
      for (t = 0; t < r; t++) {
        // SU += S[r2*i+r*s+t] * U[r*i+t];
        grad_S[r * s + t] += U[r * i + t] * grad_SU;
        grad_U[r * i + t] += S[r2 * i + r * s + t] * grad_SU;
        // S[r2*i+r*s+t] = phi[r*(i-1)+s] * phi[r*(i-1)+t] *
        // (S[r2*(i-1)+r*s+t] + W[r*(i-1)+s] * D[i-1] * W[r*(i-1)+t]);
        // Let us decompose it:
        // S[r2*i+r*s+t] *= phi[r*(i-1)+s] * phi[r*(i-1)+t];
        // We could do:
        // grad_phi[r*(i-1)+s] += S[r2*i+r*s+t]/phi[r*(i-1)+s] * grad_S[r*s+t];
        // grad_phi[r*(i-1)+t] += S[r2*i+r*s+t]/phi[r*(i-1)+t] * grad_S[r*s+t];
        // But unstable if phi << 1
        // Better to do:
        grad_tmp = (S[r2 * (i - 1) + r * s + t] +
                    W[r * (i - 1) + s] * D[i - 1] * W[r * (i - 1) + t]) *
          grad_S[r * s + t];
        grad_phi[r * (i - 1) + s] += phi[r * (i - 1) + t] * grad_tmp;
        grad_phi[r * (i - 1) + t] += phi[r * (i - 1) + s] * grad_tmp;
        grad_S[r * s + t] *= phi[r * (i - 1) + s] * phi[r * (i - 1) + t];
        // S[r2*i+r*s+t] = S[r2*(i-1)+r*s+t] +
        //   W[r*(i-1)+s] * D[i-1] * W[r*(i-1)+t];
        grad_V[r * (i - 1) + s] +=
          D[i - 1] * W[r * (i - 1) + t] * grad_S[r * s + t];
        grad_A[i - 1] +=
          W[r * (i - 1) + s] * W[r * (i - 1) + t] * grad_S[r * s + t];
        grad_V[r * (i - 1) + t] +=
          W[r * (i - 1) + s] * D[i - 1] * grad_S[r * s + t];
      }
      if (b[i] > 0) {
        // Z[r*(offsetrow[i]+2*i)+s] = phi[r*(i-1)+s] *
        //   (Z[r*(offsetrow[i]+2*i-1)+s] +
        //    G[offsetrow[i]+i-1] * W[r*(i-1)+s]);
        // Let us decompose it:
        // Z[r*(offsetrow[i]+2*i)+s] *= phi[r*(i-1)+s]
        // We could do:
        // grad_phi[r*(i-1)+s] += Z[r*(offsetrow[i]+2*i)+s]/phi[r*(i-1)+s] *
        //   grad_Z[s];
        // But unstable if phi << 1
        // Better to do:
        grad_phi[r * (i - 1) + s] +=
          (Z[r * (offsetrow[i] + 2 * i - 1) + s] +
           G[offsetrow[i] + i - 1] * D[i - 1] * W[r * (i - 1) + s]) *
          grad_Z[s];
        grad_Z[s] *= phi[r * (i - 1) + s];
        // Z[r*(offsetrow[i]+2*i)+s] = Z[r*(offsetrow[i]+2*i-1)+s] +
        //   G[offsetrow[i]+i-1] * W[r*(i-1)+s];
        grad_F[offsetrow[i] + i - 1] += W[r * (i - 1) + s] * grad_Z[s];
        grad_V[r * (i - 1) + s] +=
          G[offsetrow[i] + i - 1] * D[i - 1] * grad_Z[s];
      }
    }

    for (j = i - 1; j > i - b[i]; j--) {
      for (s = 0; s < r; s++) {
        // G[offsetrow[i]+j] -=  U[r*j+s] * Z[r*(offsetrow[i]+i+j)+s];
        grad_Z[s] -= U[r * j + s] * grad_F[offsetrow[i] + j];
        grad_U[r * j + s] -=
          Z[r * (offsetrow[i] + i + j) + s] * grad_F[offsetrow[i] + j];
        // Z[r*(offsetrow[i]+i+j)+s] = phi[r*(j-1)+s] *
        //   (Z[r*(offsetrow[i]+i+j-1)+s] +
        //    G[offsetrow[i]+j-1] * W[r*(j-1)+s]);
        // Let us decompose it:
        // Z[r*(offsetrow[i]+i+j)+s] *= phi[r*(j-1)+s]
        // We could do:
        // grad_phi[r*(j-1)+s] += Z[r*(offsetrow[i]+i+j)+s]/phi[r*(j-1)+s] *
        //   grad_Z[s];
        // But unstable if phi << 1
        // Better to do:
        grad_phi[r * (j - 1) + s] +=
          (Z[r * (offsetrow[i] + i + j - 1) + s] +
           G[offsetrow[i] + j - 1] * D[j - 1] * W[r * (j - 1) + s]) *
          grad_Z[s];
        grad_Z[s] *= phi[r * (j - 1) + s];
        // Z[r*(offsetrow[i]+i+j)+s] = Z[r*(offsetrow[i]+i+j-1)+s] +
        //   G[offsetrow[i]+j-1] * W[r*(j-1)+s];
        grad_F[offsetrow[i] + j - 1] += W[r * (j - 1) + s] * grad_Z[s];
        grad_V[r * (j - 1) + s] +=
          G[offsetrow[i] + j - 1] * D[j - 1] * grad_Z[s];
      }

      for (k = j - 1; k >= MAX(i - b[i], j - b[j]); k--) {
        // G[offsetrow[i]+j] -= G[offsetrow[i]+k] * G[offsetrow[j]+k];
        grad_F[offsetrow[i] + k] -=
          G[offsetrow[j] + k] * grad_F[offsetrow[i] + j];
        grad_F[offsetrow[j] + k] -=
          G[offsetrow[i] + k] * D[k] * grad_F[offsetrow[i] + j];
      }
    }
  }

  // i = 0
  for (s = 0; s < r; s++) {
    // W[s] /= D[0];
    grad_V[s] /= D[0];
    grad_A[0] -= W[s] * grad_V[s];
  }

  free(grad_S);
  free(grad_Z);
}

void spleaf_dotL_back(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x, double *grad_y,
  // Output
  double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
  double *grad_x,
  // Temporary
  double *f)
{
  // Backward propagation of the gradient for spleaf_dotL.

  long i, j, s;
  double *grad_f;

  grad_f = (double *)malloc(r * sizeof(double));

  // Copy grad_y -> grad_x
  memcpy(grad_x, grad_y, n * sizeof(double));

  // Initialize grad_f
  for (s = 0; s < r; s++) {
    grad_f[s] = 0.0;
  }

  // Reverse main loop
  for (i = n - 1; i > 0; i--) {
    for (j = i - b[i]; j < i; j++) {
      // y[i] += G[offsetrow[i]+j] * x[j];
      grad_G[offsetrow[i] + j] += x[j] * grad_y[i];
      grad_x[j] += G[offsetrow[i] + j] * grad_y[i];
    }
    for (s = 0; s < r; s++) {
      // y[i] += U[r*i+s] * f[r*i+s];
      grad_U[r * i + s] += f[r * i + s] * grad_y[i];
      grad_f[s] += U[r * i + s] * grad_y[i];
      // f[r*i+s] = phi[r*(i-1)+s] * (f[r*(i-1)+s] + W[r*(i-1)+s] * x[i-1]);
      // Let us decompose it:
      // f[r*i+s] *= phi[r*(i-1)+s];
      // We could do:
      // grad_phi[r*(i-1)+s] += f[r*i+s]/phi[r*(i-1)+s] * grad_f[s];
      // But unstable if phi << 1
      // Better to do:
      grad_phi[r * (i - 1) + s] +=
        (f[r * (i - 1) + s] + W[r * (i - 1) + s] * x[i - 1]) * grad_f[s];
      grad_f[s] *= phi[r * (i - 1) + s];
      // f[r*i+s] = f[r*(i-1)+s] + W[r*(i-1)+s] * x[i-1];
      grad_W[r * (i - 1) + s] += x[i - 1] * grad_f[s];
      grad_x[i - 1] += W[r * (i - 1) + s] * grad_f[s];
    }
  }

  free(grad_f);
}

void spleaf_solveL_back(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x, double *grad_x,
  // Output
  double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
  double *grad_y,
  // Temporary
  double *f)
{
  // Backward propagation of the gradient for spleaf_solveL.

  long i, j, s;
  double *grad_f;

  grad_f = (double *)malloc(r * sizeof(double));

  // Copy grad_x -> grad_y
  memcpy(grad_y, grad_x, n * sizeof(double));

  // Initialize grad_f
  for (s = 0; s < r; s++) {
    grad_f[s] = 0.0;
  }

  // Reverse main loop
  for (i = n - 1; i > 0; i--) {
    for (j = i - 1; j >= i - b[i]; j--) {
      // x[i] -= G[offsetrow[i]+j] * x[j];
      grad_G[offsetrow[i] + j] -= x[j] * grad_y[i];
      grad_y[j] -= G[offsetrow[i] + j] * grad_y[i];
    }
    for (s = 0; s < r; s++) {
      // x[i] -= U[r*i+s] * f[r*i+s];
      grad_U[r * i + s] -= f[r * i + s] * grad_y[i];
      grad_f[s] -= U[r * i + s] * grad_y[i];
      // f[r*i+s] = phi[r*(i-1)+s] * (f[r*(i-1)+s] + W[r*(i-1)+s] * x[i-1]);
      // Let us decompose it:
      // f[r*i+s] *= phi[r*(i-1)+s];
      // We could do:
      // grad_phi[r*(i-1)+s] += f[r*i+s]/phi[r*(i-1)+s] * grad_f[s];
      // But unstable if phi << 1
      // Better to do:
      grad_phi[r * (i - 1) + s] +=
        (f[r * (i - 1) + s] + W[r * (i - 1) + s] * x[i - 1]) * grad_f[s];
      grad_f[s] *= phi[r * (i - 1) + s];
      // f[r*i+s] = f[r*(i-1)+s] + W[r*(i-1)+s] * x[i-1];
      grad_W[r * (i - 1) + s] += x[i - 1] * grad_f[s];
      grad_y[i - 1] += W[r * (i - 1) + s] * grad_f[s];
    }
  }

  free(grad_f);
}

void spleaf_dotLT_back(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x, double *grad_y,
  // Output
  double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
  double *grad_x,
  // Temporary
  double *g)
{
  // Backward propagation of the gradient for spleaf_dotLT.

  long i, j, s;
  double *grad_g;

  grad_g = (double *)malloc(r * sizeof(double));

  // Copy grad_y -> grad_x
  memcpy(grad_x, grad_y, n * sizeof(double));

  // Initialize grad_g
  for (s = 0; s < r; s++) {
    grad_g[s] = 0.0;
  }

  // Reverse main loop
  for (i = 0; i < n - 1; i++) {
    for (j = i - b[i]; j < i; j++) {
      // y[j] += G[offsetrow[i]+j] * x[i];
      grad_G[offsetrow[i] + j] += x[i] * grad_y[j];
      grad_x[i] += G[offsetrow[i] + j] * grad_y[j];
    }
    for (s = 0; s < r; s++) {
      // y[i] += W[r*i+s] * g[r*i+s];
      grad_W[r * i + s] += g[r * i + s] * grad_y[i];
      grad_g[s] += W[r * i + s] * grad_y[i];

      // g[r*i+s] = phi[r*i+s] * (g[r*(i+1)+s] + U[r*(i+1)+s] * x[i+1]);
      // Let us decompose it:
      // g[r*i+s] *= phi[r*i+s];
      // We could do:
      // grad_phi[r*i+s] += g[r*i+s]/phi[r*i+s] * grad_g[s];
      // But unstable if phi << 1
      // Better to do:
      grad_phi[r * i + s] +=
        (g[r * (i + 1) + s] + U[r * (i + 1) + s] * x[i + 1]) * grad_g[s];
      grad_g[s] *= phi[r * i + s];
      // g[r*i+s] = g[r*(i+1)+s] + U[r*(i+1)+s] * x[i+1];
      grad_U[r * (i + 1) + s] += x[i + 1] * grad_g[s];
      grad_x[i + 1] += U[r * (i + 1) + s] * grad_g[s];
    }
  }

  // Leaf terms for n-1
  for (j = n - 1 - b[n - 1]; j < n - 1; j++) {
    // y[j] += G[offsetrow[n-1]+j] * x[n-1];
    grad_G[offsetrow[n - 1] + j] += x[n - 1] * grad_y[j];
    grad_x[n - 1] += G[offsetrow[n - 1] + j] * grad_y[j];
  }

  free(grad_g);
}

void spleaf_solveLT_back(
  // Shape
  long n, long r, long *offsetrow, long *b,
  // Input
  double *U, double *W, double *phi, double *G, double *x, double *grad_x,
  // Output
  double *grad_U, double *grad_W, double *grad_phi, double *grad_G,
  double *grad_y,
  // Temporary
  double *g)
{
  // Backward propagation of the gradient for spleaf_solveLT.

  long i, j, s;
  double *grad_g;

  grad_g = (double *)malloc(r * sizeof(double));

  // Copy grad_x -> grad_y
  memcpy(grad_y, grad_x, n * sizeof(double));

  // Initialize grad_g
  for (s = 0; s < r; s++) {
    grad_g[s] = 0.0;
  }

  // Reverse main loop
  for (i = 0; i < n - 1; i++) {
    for (j = i - b[i]; j < i; j++) {
      // x[j] -= G[offsetrow[i]+j] * x[i];
      grad_G[offsetrow[i] + j] -= x[i] * grad_y[j];
      grad_y[i] -= G[offsetrow[i] + j] * grad_y[j];
    }
    for (s = 0; s < r; s++) {
      // x[i] -= W[r*i+s] * g[r*i+s];
      grad_W[r * i + s] -= g[r * i + s] * grad_y[i];
      grad_g[s] -= W[r * i + s] * grad_y[i];

      // g[r*i+s] = phi[r*i+s] * (g[r*(i+1)+s] + U[r*(i+1)+s] * x[i+1]);
      // Let us decompose it:
      // g[r*i+s] *= phi[r*i+s];
      // We could do:
      // grad_phi[r*i+s] += g[r*i+s]/phi[r*i+s] * grad_g[s];
      // But unstable if phi << 1
      // Better to do:
      grad_phi[r * i + s] +=
        (g[r * (i + 1) + s] + U[r * (i + 1) + s] * x[i + 1]) * grad_g[s];
      grad_g[s] *= phi[r * i + s];
      // g[r*i+s] = g[r*(i+1)+s] + U[r*(i+1)+s] * x[i+1];
      grad_U[r * (i + 1) + s] += x[i + 1] * grad_g[s];
      grad_y[i + 1] += U[r * (i + 1) + s] * grad_g[s];
    }
  }

  // Leaf terms for n-1
  for (j = n - 1 - b[n - 1]; j < n - 1; j++) {
    // x[j] -= G[offsetrow[n-1]+j] * x[n-1];
    grad_G[offsetrow[n - 1] + j] -= x[n - 1] * grad_y[j];
    grad_y[n - 1] -= G[offsetrow[n - 1] + j] * grad_y[j];
  }

  free(grad_g);
}

void spleaf_expandsep(
  // Shape
  long n, long r, long rsi, long *sepindex,
  // Input
  double *U, double *V, double *phi,
  // Output
  double *K)
{
  // Expand the semiseparable part of a symmetric S+LEAF matrix,
  // or a subset of semiseparable terms,
  // as a full (n x n) matrix.
  // This is useful for the conditional covariance computation.

  long i, j, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  for (i = 0; i < n; i++) {
    K[(n + 1) * i] = 0.0;
    for (s = 0; s < rsi; s++) {
      K[(n + 1) * i] += U[r * i + sepindex[s]] * V[r * i + sepindex[s]];
      f[s] = 1.0;
    }
    for (j = i - 1; j >= 0; j--) {
      K[n * i + j] = 0.0;
      for (s = 0; s < rsi; s++) {
        f[s] *= phi[r * j + sepindex[s]];
        K[n * i + j] += f[s] * U[r * i + sepindex[s]] * V[r * j + sepindex[s]];
      }
      K[n * j + i] = K[n * i + j];
    }
  }

  free(f);
}

void spleaf_expandsepmixt(
  // Shape
  long n1, long n2, long r, long rsi, long *sepindex,
  // Input
  double *U1, double *V1, double *phi1, double *U2, double *V2, long *ref2left,
  double *phi2left, double *phi2right,
  // Output
  double *Km)
{
  // Expand the semiseparable mixt part of a symmetric S+LEAF matrix,
  // or a subset of semiseparable terms,
  // as a full (n2 x n1) matrix.
  // This is useful for the conditional covariance computation.

  long i1, i2, j1, j2, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  // Forward part (U2 V1^T)
  j2 = 0;
  for (j1 = 0; j1 <= ref2left[n2 - 1]; j1++) {
    for (s = 0; s < rsi; s++) {
      f[s] = V1[r * j1 + sepindex[s]];
    }
    while ((j2 < n2) && (ref2left[j2] < j1)) {
      j2++;
    }
    i1 = j1;
    i2 = j2;
    while (i2 < n2) {
      while (i1 < ref2left[i2]) {
        for (s = 0; s < rsi; s++) {
          f[s] *= phi1[r * i1 + sepindex[s]];
        }
        i1++;
      }
      while ((i2 < n2) && (ref2left[i2] == i1)) {
        Km[n1 * i2 + j1] = 0.0;
        for (s = 0; s < rsi; s++) {
          Km[n1 * i2 + j1] +=
            U2[r * i2 + sepindex[s]] * phi2left[r * i2 + sepindex[s]] * f[s];
        }
        i2++;
      }
    }
  }
  // Backward part (V2 U1^T)
  j2 = n2 - 1;
  for (j1 = n1 - 1; j1 > ref2left[0]; j1--) {
    for (s = 0; s < rsi; s++) {
      f[s] = U1[r * j1 + sepindex[s]];
    }
    while ((j2 >= 0) && (ref2left[j2] >= j1)) {
      j2--;
    }
    i1 = j1 - 1;
    i2 = j2;
    while (i2 >= 0) {
      while (i1 > ref2left[i2]) {
        for (s = 0; s < rsi; s++) {
          f[s] *= phi1[r * i1 + sepindex[s]];
        }
        i1--;
      }
      while ((i2 >= 0) && (ref2left[i2] == i1)) {
        Km[n1 * i2 + j1] = 0.0;
        for (s = 0; s < rsi; s++) {
          Km[n1 * i2 + j1] +=
            V2[r * i2 + sepindex[s]] * phi2right[r * i2 + sepindex[s]] * f[s];
        }
        i2--;
      }
    }
  }

  free(f);
}

void spleaf_expandantisep(
  // Shape
  long n, long r, long rsi, long *sepindex,
  // Input
  double *U, double *V, double *phi,
  // Output
  double *K)
{
  // Expand the semiseparable part of an anit-symmetric S+LEAF matrix,
  // or a subset of semiseparable terms,
  // as a full (n x n) matrix.
  // This is useful for the conditional derivative covariance computation.

  long i, j, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  for (i = 0; i < n; i++) {
    K[(n + 1) * i] = 0.0;
    for (s = 0; s < rsi; s++) {
      f[s] = 1.0;
    }
    for (j = i - 1; j >= 0; j--) {
      K[n * i + j] = 0.0;
      for (s = 0; s < rsi; s++) {
        f[s] *= phi[r * j + sepindex[s]];
        K[n * i + j] += f[s] * U[r * i + sepindex[s]] * V[r * j + sepindex[s]];
      }
      K[n * j + i] = -K[n * i + j];
    }
  }

  free(f);
}

void spleaf_dotsep(
  // Shape
  long n, long r, long rsi, long *sepindex,
  // Input
  double *U, double *V, double *phi, double *x,
  // Output
  double *y)
{
  // Compute y = K x,
  // where K is the (n x n) semiseparable part of a symmetric S+LEAF matrix,
  // or a subset of semiseparable terms.
  // This is useful for the conditional mean computation.

  long i, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  // Forward part (U V^T) + diagonal
  // Initialize f and y[0]
  y[0] = 0.0;
  for (s = 0; s < rsi; s++) {
    f[s] = V[sepindex[s]] * x[0];
    y[0] += U[sepindex[s]] * f[s];
  }
  for (i = 1; i < n; i++) {
    y[i] = 0.0;
    for (s = 0; s < rsi; s++) {
      // Update f
      f[s] =
        phi[r * (i - 1) + sepindex[s]] * f[s] + V[r * i + sepindex[s]] * x[i];
      y[i] += U[r * i + sepindex[s]] * f[s];
    }
  }
  // Backward part (V U^T)
  // Initialize f
  for (s = 0; s < rsi; s++) {
    f[s] = 0.0;
  }
  for (i = n - 2; i >= 0; i--) {
    for (s = 0; s < rsi; s++) {
      // Update f
      f[s] = phi[r * i + sepindex[s]] *
        (f[s] + U[r * (i + 1) + sepindex[s]] * x[i + 1]);
      y[i] += V[r * i + sepindex[s]] * f[s];
    }
  }

  free(f);
}

void spleaf_dotsepmixt(
  // Shape
  long n1, long n2, long r, long rsi, long *sepindex,
  // Input
  double *U1, double *V1, double *phi1, double *U2, double *V2, long *ref2left,
  double *phi2left, double *phi2right, double *x,
  // Output
  double *y)
{
  // Compute y = Km x,
  // where Km is the (n2 x n1) semiseparable mixt part
  // of a symmetric S+LEAF matrix,
  // or a subset of semiseparable terms.
  // This is useful for the conditional mean computation.

  long i, j, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  // Forward part (U2 V1^T)
  i = 0;
  while ((i < n2) && (ref2left[i] == -1)) {
    y[i] = 0.0;
    i++;
  }
  // Initialize f
  for (s = 0; s < rsi; s++) {
    f[s] = V1[sepindex[s]] * x[0];
  }
  j = 0;
  while (i < n2) {
    // Update f
    while (j < ref2left[i]) {
      for (s = 0; s < rsi; s++) {
        f[s] = phi1[r * j + sepindex[s]] * f[s] +
          V1[r * (j + 1) + sepindex[s]] * x[j + 1];
      }
      j++;
    }
    // Compute forward part of y[i]
    y[i] = 0.0;
    for (s = 0; s < rsi; s++) {
      y[i] += U2[r * i + sepindex[s]] * phi2left[r * i + sepindex[s]] * f[s];
    }
    i++;
  }
  // Backward part (V2 U1^T)
  i = n2 - 1;
  while ((i >= 0) && (ref2left[i] == n1 - 1)) {
    i--;
  }
  // Initialize f
  for (s = 0; s < rsi; s++) {
    f[s] = U1[r * (n1 - 1) + sepindex[s]] * x[n1 - 1];
  }
  j = n1 - 2;
  while (i >= 0) {
    // Update f
    while (j > ref2left[i]) {
      for (s = 0; s < rsi; s++) {
        f[s] =
          phi1[r * j + sepindex[s]] * f[s] + U1[r * j + sepindex[s]] * x[j];
      }
      j--;
    }
    // Compute backward part of y[i]
    for (s = 0; s < rsi; s++) {
      y[i] += V2[r * i + sepindex[s]] * phi2right[r * i + sepindex[s]] * f[s];
    }
    i--;
  }

  free(f);
}

void spleaf_dotantisep(
  // Shape
  long n, long r, long rsi, long *sepindex,
  // Input
  double *U, double *V, double *phi, double *x,
  // Output
  double *y)
{
  // Compute y = K x,
  // where K is the (n x n) semiseparable part of an anti-symmetric S+LEAF
  // matrix, or a subset of semiseparable terms. This is useful for the
  // conditional derivative mean computation.

  long i, s;
  double *f;

  f = (double *)malloc(rsi * sizeof(double));

  // Forward part (U V^T)
  // Initialize f and y[0]
  y[0] = 0.0;
  for (s = 0; s < rsi; s++) {
    f[s] = 0.0;
  }
  for (i = 1; i < n; i++) {
    y[i] = 0.0;
    for (s = 0; s < rsi; s++) {
      // Update f
      f[s] = phi[r * (i - 1) + sepindex[s]] *
        (f[s] + V[r * (i - 1) + sepindex[s]] * x[i - 1]);
      y[i] += U[r * i + sepindex[s]] * f[s];
    }
  }
  // Backward part (-V U^T)
  // Initialize f
  for (s = 0; s < rsi; s++) {
    f[s] = 0.0;
  }
  for (i = n - 2; i >= 0; i--) {
    for (s = 0; s < rsi; s++) {
      // Update f
      f[s] = phi[r * i + sepindex[s]] *
        (f[s] + U[r * (i + 1) + sepindex[s]] * x[i + 1]);
      y[i] -= V[r * i + sepindex[s]] * f[s];
    }
  }

  free(f);
}
