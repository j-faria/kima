Modeling a Gaussian Process together with calibration noise
===========================================================

We consider a time series showing quasiperiodic oscillations
and affected by white noise as well as calibration noise.

The calibration noise is introduced
by the fact that the instrument is calibrated periodically (e.g. once a day) with some error.
Measurements using the same calibration share the same calibration error.

The aim is to use a gaussian process to fit the quasiperiodic oscillations
and be able to predict missing observations.

We first generate the simulated time series:

.. plot::
   :context: close-figs

   import numpy as np
   import matplotlib.pyplot as plt
   np.random.seed(0)

   # Generate calendar
   nt = 100
   tmax = 20
   t = np.sort(np.concatenate((
      np.random.uniform(0, tmax/3, nt//2),
      np.random.uniform(2*tmax/3, tmax, (nt+1)//2))))

   # Quasiperiodic signal
   amp = 3.0
   P0 = 5.2
   dP = 0.75
   P = P0 + dP*(t/tmax-1/2)
   y = amp*np.sin(2*np.pi*t/P)

   # Truth
   tsmooth = np.linspace(0, tmax, 2000)
   Psmooth = P0 + dP*(tsmooth/tmax-1/2)
   ysignal = amp*np.sin(2*np.pi*tsmooth/Psmooth)
   dysignal = amp*2*np.pi/Psmooth*(1-tsmooth*dP/(tmax*Psmooth))*np.cos(2*np.pi*tsmooth/Psmooth)

   # Measurement errors (white noise)
   yerr_meas = np.random.uniform(0.5, 1.5, nt)
   y = y + np.random.normal(0, yerr_meas)

   # Calibration errors (correlated noise)
   calib_id = (t//1).astype(int) # One calibration per day
   caliberr = np.random.uniform(0.5, 1.5, calib_id[-1]+1)
   yerr_calib = caliberr[calib_id]
   y += np.random.normal(0, caliberr)[calib_id]

   # Total errorbar
   yerr = np.sqrt(yerr_meas**2 + yerr_calib**2)

   # Plot
   plt.figure()
   plt.plot(tsmooth, ysignal, 'r', label='truth')
   plt.errorbar(t, y, yerr, fmt='.', color='k', label='meas.')
   plt.xlabel('t')
   plt.ylabel('y')
   plt.legend()

We now fit the data using S+LEAF.
As a first approach, we ignore the correlation induced by the calibration noise,
and treat it as if it was white noise:

.. plot::
   :context: close-figs

   from spleaf.cov import Cov
   from spleaf.term import *
   from scipy.optimize import fmin_l_bfgs_b

   # Initialize the S+LEAF model
   cov = Cov(t,
      err = Error(yerr),
      sho = SHOKernel(0.5, 5.0, 1.0))

   # We now fit the hyperparameters using the fmin_l_bfgs_b function from scipy.optimize.
   # Define the function to minimize
   def negloglike(x, y, cov):
      cov.set_param(x)
      nll = -cov.loglike(y)
      # gradient
      nll_grad = -cov.loglike_grad()[1]
      return(nll, nll_grad)

   # Fit
   xbest,_,_ = fmin_l_bfgs_b(negloglike, cov.get_param(), args=(y, cov))

   # We now use S+LEAF to predict the missing data
   cov.set_param(xbest)
   mu, var = cov.conditional(y, tsmooth, calc_cov='diag')

   # Plot
   plt.figure()
   plt.plot(tsmooth, ysignal, 'r', label='truth')
   plt.errorbar(t, y, yerr, fmt='.', color='k', label='meas.')
   plt.fill_between(tsmooth, mu-np.sqrt(var), mu+np.sqrt(var), color='g', alpha=0.5)
   plt.plot(tsmooth, mu, 'g', label='predict.')
   plt.xlabel('t')
   plt.ylabel('y')
   plt.legend()

We see that the gaussian process is not completely wrong but tend
to absorb the correlated noise due to the calibration.
The prediction in the gap is not very satisfying.

Let us now correctly model the calibration noise with S+LEAF:

.. plot::
   :context: close-figs

   # We define a new covariance matrix including calibration error
   cov = Cov(t,
      err = Error(yerr_meas),
      calerr = CalibrationError(calib_id, yerr_calib),
      sho = SHOKernel(0.5, 5.0, 1.0))

   # Fit
   xbest,_,_ = fmin_l_bfgs_b(negloglike, cov.get_param(), args=(y, cov))

   # Predict
   cov.set_param(xbest)
   mu, var = cov.conditional(y, tsmooth, calc_cov='diag')

   # Plot
   plt.figure()
   plt.plot(tsmooth, ysignal, 'r', label='truth')
   plt.errorbar(t, y, yerr, fmt='.', color='k', label='meas.')
   plt.fill_between(tsmooth, mu-np.sqrt(var), mu+np.sqrt(var), color='g', alpha=0.5)
   plt.plot(tsmooth, mu, 'g', label='predict.')
   plt.xlabel('t')
   plt.ylabel('y')
   plt.legend()

The results are indeed much better!

S+LEAF also allows to predict the derivative of the gaussian process:

.. plot::
   :context: close-figs

   # Predict derivative
   dmu, dvar = cov.conditional_derivative(y, tsmooth, calc_cov='diag')

   # Plot
   plt.figure()
   plt.plot(tsmooth, dysignal, 'r', label='truth')
   plt.fill_between(tsmooth, dmu-np.sqrt(dvar), dmu+np.sqrt(dvar), color='g', alpha=0.5)
   plt.plot(tsmooth, dmu, 'g', label='predict.')
   plt.xlabel('t')
   plt.ylabel('dy/dt')
   plt.legend()
