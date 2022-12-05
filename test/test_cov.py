import pytest
import numpy as np
from spleaf.cov import Cov
from spleaf.term import *

prec = 1e-12
n = 143
ninst = 3
calibmax = 12
calibprob = 0.8
nexp = 1
nqper = 1
nmat32 = 1
nmat52 = 1
nes = 1
nusho = 1
nosho = 1
nsho = 1
nmep = 1
nesp = 1
delta = 1e-6
coef_delta = -1.3
coef_num_err = 10


def _grad1d(x, f, k, deltaxk0, deltafmin, deltafmax, maxiter, absolute, args):
  xb = x.copy()
  deltaxk = deltaxk0
  fx = f(x, *args)
  if absolute:
    scale = 1.0
  else:
    scale = 1.0 / np.mean(np.abs(fx))
  xb[k] = x[k] + deltaxk
  fxb = f(xb, *args)
  for _ in range(maxiter):
    df = scale * np.mean(np.abs(fxb - fx))
    if df < deltafmin:
      deltaxk *= 2.0
    elif df > deltafmax:
      deltaxk /= 2.0
    else:
      break
    xb[k] = x[k] + deltaxk
    fxb = f(xb, *args)
  return ((fxb - fx) / deltaxk)


def grad(x,
  f,
  deltax0=1e-8,
  deltafmin=1e-8,
  deltafmax=5e-8,
  maxiter=50,
  absolute=False,
  args=()):
  if isinstance(deltax0, float):
    deltax = np.full_like(x, deltax0)
  else:
    deltax = deltax0
  return (np.array([
    _grad1d(x, f, k, deltax[k], deltafmin, deltafmax, maxiter, absolute, args)
    for k in range(x.size)
  ]))


def _generate_random_C(seed=0, deriv=False):
  np.random.seed(seed)
  t = np.cumsum(10**np.random.uniform(-2, 1.5, n))
  sig_err = np.random.uniform(0.5, 1.5, n)
  sig_jitter = np.random.uniform(0.5, 1.5)
  inst_id = np.random.randint(0, ninst, n)
  sig_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  calib_file = np.empty(n, dtype=object)
  sig_calib_meas = np.empty(n)
  lastfileinst = ["" for _ in range(ninst)]
  lastvarinst = [0 for _ in range(ninst)]
  nlastinst = [0 for _ in range(ninst)]
  for k in range(n):
    i = inst_id[k]
    if lastfileinst[i] == "" or nlastinst[i] == calibmax or np.random.rand(
    ) > calibprob:
      calib_file[k] = '{}'.format(k)
      sig_calib_meas[k] = np.random.uniform(0.5, 1.5)
      lastfileinst[i] = calib_file[k]
      lastvarinst[i] = sig_calib_meas[k]
      nlastinst[i] = 1
    else:
      calib_file[k] = lastfileinst[i]
      sig_calib_meas[k] = lastvarinst[i]
      nlastinst[i] += 1
  sig_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  if not deriv:
    a_exp = np.random.uniform(0.5, 1.5, nexp)
    la_exp = 10**np.random.uniform(-2, 2, nexp)
    a_qper = np.random.uniform(0.5, 1.5, nqper)
    b_qper = np.random.uniform(0.05, 0.15, nqper)
    la_qper = 10**np.random.uniform(-2, 2, nqper)
    nu_qper = 10**np.random.uniform(-2, 2, nqper)
  sig_mat32 = np.random.uniform(0.5, 1.5, nmat32)
  rho_mat32 = 10**np.random.uniform(-2, 2, nmat32)
  sig_mat52 = np.random.uniform(0.5, 1.5, nmat52)
  rho_mat52 = 10**np.random.uniform(-2, 2, nmat52)
  sig_es = np.random.uniform(0.5, 1.5, nes)
  rho_es = 10**np.random.uniform(-2, 2, nes)
  sig_usho = np.random.uniform(0.5, 1.5, nusho)
  P0_usho = 10**np.random.uniform(-2, 2, nusho)
  Q_usho = np.random.uniform(0.5, 20.0, nusho)
  sig_osho = np.random.uniform(0.5, 1.5, nosho)
  P0_osho = 10**np.random.uniform(-2, 2, nosho)
  Q_osho = np.random.uniform(0.01, 0.5, nosho)
  sig_sho = np.random.uniform(0.5, 1.5, nsho)
  P0_sho = 10**np.random.uniform(-2, 2, nsho)
  Q_sho = np.random.uniform(0.01, 2.0, nsho)
  sig_mep = np.random.uniform(0.5, 1.5, nmep)
  P_mep = 10**np.random.uniform(-2, 2, nmep)
  rho_mep = 10**np.random.uniform(-2, 2, nmep)
  eta_mep = 10**np.random.uniform(-2, 1, nmep)
  sig_esp = np.random.uniform(0.5, 1.5, nesp)
  P_esp = 10**np.random.uniform(-2, 2, nesp)
  rho_esp = 10**np.random.uniform(-2, 2, nesp)
  eta_esp = 10**np.random.uniform(-2, 1, nesp)

  if deriv:
    return (Cov(t,
      err=Error(sig_err),
      jit=Jitter(sig_jitter),
      **{
      f'insjit_{k}': InstrumentJitter(inst_id == k, sig_jitter_inst[k])
      for k in range(ninst)
      },
      calerr=CalibrationError(calib_file, sig_calib_meas),
      **{
      f'caljit_{k}': CalibrationJitter(inst_id == k, calib_file,
      sig_calib_inst[k])
      for k in range(ninst)
      },
      **{
      f'mat32_{k}': Matern32Kernel(sig_mat32[k], rho_mat32[k])
      for k in range(nmat32)
      },
      **{
      f'mat52_{k}': Matern52Kernel(sig_mat52[k], rho_mat52[k])
      for k in range(nmat52)
      },
      **{f'es_{k}': ESKernel(sig_es[k], rho_es[k])
      for k in range(nes)},
      **{
      f'usho_{k}': USHOKernel(sig_usho[k], P0_usho[k], Q_usho[k])
      for k in range(nusho)
      },
      **{
      f'osho_{k}': OSHOKernel(sig_osho[k], P0_osho[k], Q_osho[k])
      for k in range(nosho)
      },
      **{
      f'sho_{k}': SHOKernel(sig_sho[k], P0_sho[k], Q_sho[k])
      for k in range(nsho)
      },
      **{
      f'mep_{k}': MEPKernel(sig_mep[k], P_mep[k], rho_mep[k], eta_mep[k])
      for k in range(nmep)
      },
      **{
      f'esp_{k}': ESPKernel(sig_esp[k], P_esp[k], rho_esp[k], eta_esp[k])
      for k in range(nesp)
      }))
  else:
    return (Cov(t,
      err=Error(sig_err),
      jit=Jitter(sig_jitter),
      **{
      f'insjit_{k}': InstrumentJitter(inst_id == k, sig_jitter_inst[k])
      for k in range(ninst)
      },
      calerr=CalibrationError(calib_file, sig_calib_meas),
      **{
      f'caljit_{k}': CalibrationJitter(inst_id == k, calib_file,
      sig_calib_inst[k])
      for k in range(ninst)
      },
      **{
      f'exp_{k}': ExponentialKernel(a_exp[k], la_exp[k])
      for k in range(nexp)
      },
      **{
      f'qper_{k}': QuasiperiodicKernel(a_qper[k], b_qper[k], la_qper[k],
      nu_qper[k])
      for k in range(nqper)
      },
      **{
      f'mat32_{k}': Matern32Kernel(sig_mat32[k], rho_mat32[k])
      for k in range(nmat32)
      },
      **{
      f'mat52_{k}': Matern52Kernel(sig_mat52[k], rho_mat52[k])
      for k in range(nmat52)
      },
      **{f'es_{k}': ESKernel(sig_es[k], rho_es[k])
      for k in range(nes)},
      **{
      f'usho_{k}': USHOKernel(sig_usho[k], P0_usho[k], Q_usho[k])
      for k in range(nusho)
      },
      **{
      f'osho_{k}': OSHOKernel(sig_osho[k], P0_osho[k], Q_osho[k])
      for k in range(nosho)
      },
      **{
      f'sho_{k}': SHOKernel(sig_sho[k], P0_sho[k], Q_sho[k])
      for k in range(nsho)
      },
      **{
      f'mep_{k}': MEPKernel(sig_mep[k], P_mep[k], rho_mep[k], eta_mep[k])
      for k in range(nmep)
      },
      **{
      f'esp_{k}': ESPKernel(sig_esp[k], P_esp[k], rho_esp[k], eta_esp[k])
      for k in range(nesp)
      }))


def _generate_random_param(seed=1):
  np.random.seed(seed)
  sig_jitter = np.random.uniform(0.5, 1.5, 1)
  sig_jitter_inst = np.random.uniform(0.5, 1.5, ninst)
  sig_calib_inst = np.random.uniform(0.5, 1.5, ninst)
  a_exp = np.random.uniform(0.5, 1.5, nexp)
  la_exp = 10**np.random.uniform(-2, 2, nexp)
  a_qper = np.random.uniform(0.5, 1.5, nqper)
  b_qper = np.random.uniform(0.05, 0.15, nqper)
  la_qper = 10**np.random.uniform(-2, 2, nqper)
  nu_qper = 10**np.random.uniform(-2, 2, nqper)
  sig_mat32 = np.random.uniform(0.5, 1.5, nmat32)
  rho_mat32 = 10**np.random.uniform(-2, 2, nmat32)
  sig_mat52 = np.random.uniform(0.5, 1.5, nmat52)
  rho_mat52 = 10**np.random.uniform(-2, 2, nmat52)
  sig_es = np.random.uniform(0.5, 1.5, nes)
  rho_es = 10**np.random.uniform(-2, 2, nes)
  sig_usho = np.random.uniform(0.5, 1.5, nusho)
  P0_usho = 10**np.random.uniform(-2, 2, nusho)
  Q_usho = np.random.uniform(0.5, 20.0, nusho)
  sig_osho = np.random.uniform(0.5, 1.5, nosho)
  P0_osho = 10**np.random.uniform(-2, 2, nosho)
  Q_osho = np.random.uniform(0.01, 0.5, nosho)
  sig_sho = np.random.uniform(0.5, 1.5, nsho)
  P0_sho = 10**np.random.uniform(-2, 2, nsho)
  Q_sho = np.random.uniform(0.01, 2.0, nsho)
  sig_mep = np.random.uniform(0.5, 1.5, nmep)
  P_mep = 10**np.random.uniform(-2, 2, nmep)
  rho_mep = 10**np.random.uniform(-2, 2, nmep)
  eta_mep = 10**np.random.uniform(-2, 1, nmep)
  sig_esp = np.random.uniform(0.5, 1.5, nesp)
  P_esp = 10**np.random.uniform(-2, 2, nesp)
  rho_esp = 10**np.random.uniform(-2, 2, nesp)
  eta_esp = 10**np.random.uniform(-2, 1, nesp)

  return (sig_jitter, sig_jitter_inst, sig_calib_inst, a_exp, la_exp, a_qper,
    b_qper, la_qper, nu_qper, sig_mat32, rho_mat32, sig_mat52, rho_mat52,
    sig_es, rho_es, sig_usho, P0_usho, Q_usho, sig_osho, P0_osho, Q_osho,
    sig_sho, P0_sho, Q_sho, sig_mep, P_mep, rho_mep, eta_mep, sig_esp, P_esp,
    rho_esp, eta_esp)


def test_Cov():
  C = _generate_random_C()

  C_full = C.expand()
  L_full = C.expandL()
  D_full = np.diag(C.D)

  LDLt_full = L_full @ D_full @ L_full.T
  err = np.max(
    np.abs(C_full - LDLt_full)) / np.max(np.abs(C_full) + np.abs(LDLt_full))
  assert err < prec, ('Cholesky decomposition not working'
    ' at required precision ({} > {})').format(err, prec)


def test_set_param():
  C = _generate_random_C()
  param = list(_generate_random_param())
  Cb = Cov(C.t,
    err=Error(C.term['err']._sig),
    jit=Jitter(param[0][0]),
    **{
    f'insjit_{k}': InstrumentJitter(C.term[f'insjit_{k}']._indices,
    param[1][k])
    for k in range(ninst)
    },
    calerr=CalibrationError(C.term['calerr']._calib_id, C.term['calerr']._sig),
    **{
    f'caljit_{k}': CalibrationJitter(C.term[f'insjit_{k}']._indices,
    C.term['calerr']._calib_id, param[2][k])
    for k in range(ninst)
    },
    **{
    f'exp_{k}': ExponentialKernel(param[3][k], param[4][k])
    for k in range(nexp)
    },
    **{
    f'qper_{k}': QuasiperiodicKernel(param[5][k], param[6][k], param[7][k],
    param[8][k])
    for k in range(nqper)
    },
    **{
    f'mat32_{k}': Matern32Kernel(param[9][k], param[10][k])
    for k in range(nmat32)
    },
    **{
    f'mat52_{k}': Matern52Kernel(param[11][k], param[12][k])
    for k in range(nmat52)
    },
    **{f'es_{k}': ESKernel(param[13][k], param[14][k])
    for k in range(nes)},
    **{
    f'usho_{k}': USHOKernel(param[15][k], param[16][k], param[17][k])
    for k in range(nusho)
    },
    **{
    f'osho_{k}': OSHOKernel(param[18][k], param[19][k], param[20][k])
    for k in range(nosho)
    },
    **{
    f'sho_{k}': SHOKernel(param[21][k], param[22][k], param[23][k])
    for k in range(nsho)
    },
    **{
    f'mep_{k}': MEPKernel(param[24][k], param[25][k], param[26][k],
    param[27][k])
    for k in range(nmep)
    },
    **{
    f'esp_{k}': ESPKernel(param[28][k], param[29][k], param[30][k],
    param[31][k])
    for k in range(nesp)
    })

  C.set_param(np.concatenate(param),
    ['jit.sig'] + [f'insjit_{k}.sig'
    for k in range(ninst)] + [f'caljit_{k}.sig'
    for k in range(ninst)] + [f'exp_{k}.a'
    for k in range(nexp)] + [f'exp_{k}.la'
    for k in range(nexp)] + [f'qper_{k}.a'
    for k in range(nqper)] + [f'qper_{k}.b'
    for k in range(nqper)] + [f'qper_{k}.la'
    for k in range(nqper)] + [f'qper_{k}.nu'
    for k in range(nqper)] + [f'mat32_{k}.sig'
    for k in range(nmat32)] + [f'mat32_{k}.rho'
    for k in range(nmat32)] + [f'mat52_{k}.sig'
    for k in range(nmat52)] + [f'mat52_{k}.rho'
    for k in range(nmat52)] + [f'es_{k}.sig'
    for k in range(nes)] + [f'es_{k}.rho'
    for k in range(nes)] + [f'usho_{k}.sig'
    for k in range(nusho)] + [f'usho_{k}.P0'
    for k in range(nusho)] + [f'usho_{k}.Q'
    for k in range(nusho)] + [f'osho_{k}.sig'
    for k in range(nosho)] + [f'osho_{k}.P0'
    for k in range(nosho)] + [f'osho_{k}.Q'
    for k in range(nosho)] + [f'sho_{k}.sig'
    for k in range(nsho)] + [f'sho_{k}.P0'
    for k in range(nsho)] + [f'sho_{k}.Q'
    for k in range(nsho)] + [f'mep_{k}.sig'
    for k in range(nmep)] + [f'mep_{k}.P'
    for k in range(nmep)] + [f'mep_{k}.rho'
    for k in range(nmep)] + [f'mep_{k}.eta'
    for k in range(nmep)] + [f'esp_{k}.sig'
    for k in range(nesp)] + [f'esp_{k}.P'
    for k in range(nesp)] + [f'esp_{k}.rho'
    for k in range(nesp)] + [f'esp_{k}.eta' for k in range(nesp)])

  C_full = C.expand()
  Cb_full = Cb.expand()
  L_full = C.expandL()
  Lb_full = Cb.expandL()

  err = np.max(
    np.abs(C_full - Cb_full)) / np.max(np.abs(C_full) + np.abs(Cb_full))
  err = max(
    err,
    np.max(np.abs(L_full - Lb_full)) /
    np.max(np.abs(L_full) + np.abs(Lb_full)))
  err = max(err,
    np.max(np.abs(C.D - Cb.D)) / np.max(np.abs(C.D) + np.abs(Cb.D)))

  assert err < prec, ('set_param not working'
    ' at required precision ({} > {})').format(err, prec)


def test_inv():
  C = _generate_random_C()

  C_full = C.expand()
  invC_full = C.expandInv()

  CinvC_full = C_full @ invC_full
  err = np.max(np.abs(CinvC_full - np.identity(n)))
  assert err < prec, ('Inversion not working'
    ' at required precision ({} > {})').format(err, prec)


def test_invL():
  C = _generate_random_C()

  L_full = C.expandL()
  invL_full = C.expandInvL()

  LinvL_full = L_full @ invL_full
  err = np.max(np.abs(LinvL_full - np.identity(n)))
  assert err < prec, ('Cholesky inversion not working'
    ' at required precision ({} > {})').format(err, prec)


def test_logdet():
  C = _generate_random_C()

  logdet = C.logdet()

  C_full = C.expand()
  sign_full, logdet_full = np.linalg.slogdet(C_full)

  err = abs(logdet / logdet_full - 1)

  assert sign_full > 0, 'logdet is not positive'
  assert err < prec, ('logdet not working'
    ' at required precision ({} > {})').format(err, prec)


def test_dotL():
  C = _generate_random_C()
  x = np.random.normal(0.0, 1.0, C.n)

  y = C.dotL(x)

  L_full = C.expandL()
  y_full = L_full.dot(x)

  err = np.max(np.abs(y - y_full)) / np.max(np.abs(y) + np.abs(y_full))

  assert err < prec, ('dotL not working'
    ' at required precision ({} > {})').format(err, prec)


def test_solveL():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  x = C.solveL(y)

  L_full = C.expandL()
  x_full = np.linalg.solve(L_full, y)

  err = np.max(np.abs(x - x_full)) / np.max(np.abs(x) + np.abs(x_full))

  assert err < prec, ('solveL not working'
    ' at required precision ({} > {})').format(err, prec)


def test_chi2():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  chi2 = C.chi2(y)

  C_full = C.expand()
  invC_full = np.linalg.inv(C_full)
  chi2_full = y.T @ invC_full @ y

  err = abs(chi2 - chi2_full) / (abs(chi2) + abs(chi2_full))

  assert err < prec, ('chi2 not working'
    ' at required precision ({} > {})').format(err, prec)


def test_loglike():
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  loglike = C.loglike(y)

  C_full = C.expand()
  invC_full = np.linalg.inv(C_full)
  chi2_full = y.T @ invC_full @ y
  _, logdet_full = np.linalg.slogdet(C_full)
  loglike_full = -0.5 * (chi2_full + logdet_full + C.n * np.log(2.0 * np.pi))

  err = abs(loglike - loglike_full) / (abs(loglike) + abs(loglike_full))
  assert err < prec, ('loglike not working'
    ' at required precision ({} > {})').format(err, prec)


def _test_method_back(method):
  """
  Common code for testing dotL_back, solveL_back, dotLT_back, solveLT_back
  """
  C = _generate_random_C()
  a = np.random.normal(0.0, 5.0, C.n)
  grad_b = np.random.normal(0.0, 1.0, C.n)

  # Analytical grad
  func = getattr(C, method)
  _ = func(a)
  C.init_grad()
  grad_a = getattr(C, method + '_back')(grad_b)
  C.cholesky_back()
  grad_param = C.grad_param()

  # Numerical grad
  def func_param(x):
    C.set_param(x)
    return (func(a))

  grad_a_num = []
  grad_param_num = []
  for delta0 in [delta, coef_delta * delta]:
    grad_a_num.append(grad(a, func, deltax0=delta0) @ grad_b)
    grad_param_num.append(
      grad(C.get_param(), func_param, deltax0=delta0) @ grad_b)

  # Comparison
  err = np.max(
    np.abs(grad_a - np.mean(grad_a_num, axis=0)) /
    (np.abs(grad_a_num[1]) + np.abs(grad_a_num[0])))
  num_err = np.max(
    np.abs(grad_a_num[1] - grad_a_num[0]) /
    (np.abs(grad_a_num[1]) + np.abs(grad_a_num[0])))
  err = max(0.0, err - coef_num_err * num_err)
  assert err < prec, ('{}_back (a) not working'
    ' at required precision ({} > {})').format(method, err, prec)

  err = np.max(
    np.abs(grad_param - np.mean(grad_param_num, axis=0)) /
    (np.abs(grad_param_num[1]) + np.abs(grad_param_num[0])))
  num_err = np.max(
    np.abs(grad_param_num[1] - grad_param_num[0]) /
    (np.abs(grad_param_num[1]) + np.abs(grad_param_num[0])))
  err = max(0.0, err - coef_num_err * num_err)
  assert err < prec, ('{}_back (param) not working'
    ' at required precision ({} > {})').format(method, err, prec)


def test_dotL_back():
  _test_method_back('dotL')


def test_solveL_back():
  _test_method_back('solveL')


def test_dotLT_back():
  _test_method_back('dotLT')


def test_solveLT_back():
  _test_method_back('solveLT')


def _test_method_grad(method):
  """
  Common code for testing chi2_grad, loglike_grad
  """
  C = _generate_random_C()
  y = np.random.normal(0.0, 5.0, C.n)

  # Analytical grad
  func = getattr(C, method)
  _ = func(y)
  f_grad_res, f_grad_param = getattr(C, method + '_grad')()

  # Numerical grad
  def func_param(x):
    C.set_param(x)
    return (func(y))

  f_grad_res_num = []
  f_grad_param_num = []
  for delta0 in [delta, coef_delta * delta]:
    f_grad_res_num.append(grad(y, func, deltax0=delta0))
    f_grad_param_num.append(grad(C.get_param(), func_param, deltax0=delta0))

  # Comparison
  err = np.max(
    np.abs(f_grad_res - np.mean(f_grad_res_num, axis=0)) /
    (np.abs(f_grad_res_num[1]) + np.abs(f_grad_res_num[0])))
  num_err = np.max(
    np.abs(f_grad_res_num[1] - f_grad_res_num[0]) /
    (np.abs(f_grad_res_num[1]) + np.abs(f_grad_res_num[0])))
  err = max(0.0, err - coef_num_err * num_err)
  assert err < prec, ('{}_grad (y) not working'
    ' at required precision ({} > {})').format(method, err, prec)

  err = np.max(
    np.abs(f_grad_param - np.mean(f_grad_param_num, axis=0)) /
    (np.abs(f_grad_param_num[1]) + np.abs(f_grad_param_num[0])))
  num_err = np.max(
    np.abs(f_grad_param_num[1] - f_grad_param_num[0]) /
    (np.abs(f_grad_param_num[1]) + np.abs(f_grad_param_num[0])))
  err = max(0.0, err - coef_num_err * num_err)
  assert err < prec, ('{}_grad (param) not working'
    ' at required precision ({} > {})').format(method, err, prec)


def test_chi2_grad():
  _test_method_grad('chi2')


def test_loglike_grad():
  _test_method_grad('loglike')


def _test_self_conditional(kernel=None):
  print(kernel)
  C = _generate_random_C()
  y = C.sample()

  mu = C.self_conditional(y, kernel=kernel)
  muv, var = C.self_conditional(y, calc_cov='diag', kernel=kernel)
  muc, cov = C.self_conditional(y, calc_cov=True, kernel=kernel)

  invC_full = C.expandInv()
  invCy_full = invC_full.dot(y)
  term = {}
  if kernel is None:
    kernel = C.kernel
  for key in kernel:
    term[key] = C.kernel[key].__class__(
      *[getattr(C.kernel[key], f'_{param}') for param in C.kernel[key]._param])
  print(term)
  K = Cov(C.t, **term)
  K_full = K.expand()
  mu_full = K_full @ invCy_full
  cov_full = K_full - K_full @ invC_full @ K_full
  var_full = np.diag(cov_full)

  err = np.max(np.abs(mu - mu_full)) / np.max(np.abs(mu) + np.abs(mu_full))
  assert err < prec, ('self_conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muv - mu_full)) / np.max(np.abs(muv) + np.abs(mu_full))
  assert err < prec, ('self_conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muc - mu_full)) / np.max(np.abs(muc) + np.abs(mu_full))
  assert err < prec, ('self_conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(var - var_full)) / np.max(np.abs(var) + np.abs(var_full))
  assert err < prec, ('self_conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(cov - cov_full)) / np.max(np.abs(cov) + np.abs(cov_full))
  assert err < prec, ('self_conditional not working'
    ' at required precision ({} > {})').format(err, prec)


def _test_conditional(kernel=None):
  C = _generate_random_C()
  y = C.sample()

  n2 = 300
  Dt = C.t[-1] - C.t[0]
  margin = Dt / 10
  t2 = np.linspace(C.t[0] - margin, C.t[-1] + margin, n2)
  mu = C.conditional(y, t2, kernel=kernel)
  muv, var = C.conditional(y, t2, calc_cov='diag', kernel=kernel)
  muc, cov = C.conditional(y, t2, calc_cov=True, kernel=kernel)

  invC_full = C.expandInv()
  invCy_full = invC_full.dot(y)
  Km_full = C.eval(t2[:, None] - C.t[None, :], kernel=kernel)
  term = {}
  if kernel is None:
    kernel = C.kernel
  for key in kernel:
    term[key] = C.kernel[key].__class__(
      *[getattr(C.kernel[key], f'_{param}') for param in C.kernel[key]._param])
  K = Cov(t2, **term)
  K_full = K.expand()
  mu_full = Km_full @ invCy_full
  cov_full = K_full - Km_full @ invC_full @ Km_full.T
  var_full = np.diag(cov_full)

  err = np.max(np.abs(mu - mu_full)) / np.max(np.abs(mu) + np.abs(mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muv - mu_full)) / np.max(np.abs(muv) + np.abs(mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(muc - mu_full)) / np.max(np.abs(muc) + np.abs(mu_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(var - var_full)) / np.max(np.abs(var) + np.abs(var_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(cov - cov_full)) / np.max(np.abs(cov) + np.abs(cov_full))
  assert err < prec, ('conditional not working'
    ' at required precision ({} > {})').format(err, prec)


def _test_self_conditional_derivative(kernel=None):
  C = _generate_random_C(deriv=True)
  y = C.sample()

  # Analytical derivative
  dmu = C.self_conditional_derivative(y, kernel=kernel)
  dmuv, dvar = C.self_conditional_derivative(y, calc_cov='diag', kernel=kernel)
  dmuc, dcov = C.self_conditional_derivative(y, calc_cov=True, kernel=kernel)

  # Numerical derivative
  num_dmu = []
  num_dcov = []
  for dt in [delta, coef_delta * delta]:
    tfull = np.sort(np.concatenate((C.t, C.t + dt)))
    mu, cov = C.conditional(y, tfull, calc_cov=True, kernel=kernel)
    num_dmu.append((mu[1::2] - mu[::2]) / abs(dt))
    num_dcov.append(
      (cov[1::2, 1::2] + cov[::2, ::2] - cov[1::2, ::2] - cov[::2, 1::2]) /
      dt**2)

  num_dmu_mean = (num_dmu[0] + num_dmu[1]) / 2
  num_dmu_err = np.max(np.abs(num_dmu[0] -
    num_dmu[1])) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))

  num_dcov_mean = (num_dcov[0] + num_dcov[1]) / 2
  num_dcov_err = np.max(np.abs(num_dcov[0] -
    num_dcov[1])) / np.max(np.abs(num_dcov[0]) + np.abs(num_dcov[1]))

  num_dvar_mean = num_dcov_mean.diagonal()
  num_dvar_err = np.max(
    np.abs(num_dcov[0].diagonal() - num_dcov[1].diagonal())) / np.max(
    np.abs(num_dcov[0].diagonal()) + np.abs(num_dcov[1].diagonal()))

  err = np.max(np.abs(dmu -
    num_dmu_mean)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(
    np.abs(dmuv - num_dmu)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(
    np.abs(dmuc - num_dmu)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dvar - num_dvar_mean)) / np.max(
    np.abs(num_dcov[0].diagonal()) + np.abs(num_dcov[1].diagonal()))
  err = max(0.0, err - coef_num_err * num_dvar_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dcov -
    num_dcov_mean)) / np.max(np.abs(num_dcov[0]) + np.abs(num_dcov[1]))
  err = max(0.0, err - coef_num_err * num_dcov_err)
  assert err < prec, ('self_conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)


def _test_conditional_derivative(kernel=None):
  C = _generate_random_C(deriv=True)
  y = C.dotL(np.random.normal(0.0, C.sqD()))

  n2 = 1001
  Dt = C.t[-1] - C.t[0]
  margin = Dt / 10
  t2 = np.linspace(C.t[0] - margin, C.t[-1] + margin, n2)

  # Analytical derivative
  dmu = C.conditional_derivative(y, t2, kernel=kernel)
  dmuv, dvar = C.conditional_derivative(y, t2, calc_cov='diag', kernel=kernel)
  dmuc, dcov = C.conditional_derivative(y, t2, calc_cov=True, kernel=kernel)

  # Numerical derivative
  num_dmu = []
  num_dcov = []
  for dt in [delta, coef_delta * delta]:
    tfull = np.sort(np.concatenate((t2, t2 + dt)))
    mu, cov = C.conditional(y, tfull, calc_cov=True, kernel=kernel)
    num_dmu.append((mu[1::2] - mu[::2]) / abs(dt))
    num_dcov.append(
      (cov[1::2, 1::2] + cov[::2, ::2] - cov[1::2, ::2] - cov[::2, 1::2]) /
      dt**2)

  num_dmu_mean = (num_dmu[0] + num_dmu[1]) / 2
  num_dmu_err = np.max(np.abs(num_dmu[0] -
    num_dmu[1])) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))

  num_dcov_mean = (num_dcov[0] + num_dcov[1]) / 2
  num_dcov_err = np.max(np.abs(num_dcov[0] -
    num_dcov[1])) / np.max(np.abs(num_dcov[0]) + np.abs(num_dcov[1]))

  num_dvar_mean = num_dcov_mean.diagonal()
  num_dvar_err = np.max(
    np.abs(num_dcov[0].diagonal() - num_dcov[1].diagonal())) / np.max(
    np.abs(num_dcov[0].diagonal()) + np.abs(num_dcov[1].diagonal()))

  err = np.max(np.abs(dmu -
    num_dmu_mean)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(
    np.abs(dmuv - num_dmu)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(
    np.abs(dmuc - num_dmu)) / np.max(np.abs(num_dmu[0]) + np.abs(num_dmu[1]))
  err = max(0.0, err - coef_num_err * num_dmu_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dvar - num_dvar_mean)) / np.max(
    np.abs(num_dcov[0].diagonal()) + np.abs(num_dcov[1].diagonal()))
  err = max(0.0, err - coef_num_err * num_dvar_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)

  err = np.max(np.abs(dcov -
    num_dcov_mean)) / np.max(np.abs(num_dcov[0]) + np.abs(num_dcov[1]))
  err = max(0.0, err - coef_num_err * num_dcov_err)
  assert err < prec, ('conditional_derivative not working'
    ' at required precision ({} > {})').format(err, prec)


def test_all_conditional():
  for kernel in [
      None, ['mat32_0', 'usho_0'], ['osho_0'], ['mat52_0', 'sho_0'],
    ['es_0', 'sho_0'], ['mep_0'], ['esp_0']
  ]:
    _test_self_conditional(kernel)
    _test_conditional(kernel)
    _test_self_conditional_derivative(kernel)
    _test_conditional_derivative(kernel)
