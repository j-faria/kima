from typing import Any

from scipy.stats import _continuous_distns as _c
from scipy.stats._distn_infrastructure import rv_frozen
# from scipy.stats import uniform as scipy_uniform, norm, loguniform
# from better_uniform import buniform as uniform
from loguniform import ModifiedLogUniform
from kumaraswamy import kumaraswamy


VARIABLE_NAMES = {
    # for all models?
    'C': 'Cprior',
    'J': 'Jprior',
    # for RVmodel, GPmodel, etc
    'P': 'Pprior',
    'K': 'Kprior',
    'e': 'eprior',
    'w': 'wprior',
    'phi': 'phiprior',
    # for BD model
    'L': 'Lprior',
    'tau1': 'tau1_prior',
    'tau2': 'tau2_prior',
    #
    'slope': 'slope_prior',
    'quadr': 'quadr_prior',
    'cubic': 'cubic_prior',
    #
    'eta1': 'eta1_prior',
    'eta2': 'eta2_prior',
    'eta3': 'eta3_prior',
    'eta4': 'eta4_prior',
    # for RVFWHMmodel
    'C2': 'C2prior',
    'J2': 'J2prior',
    'eta1_fwhm': 'eta1_fwhm_prior',
    'eta2_fwhm': 'eta2_fwhm_prior',
    'eta3_fwhm': 'eta3_fwhm_prior',
    'eta4_fwhm': 'eta4_fwhm_prior',
}


def wrong_npar(dist, expected, got):
    msg = f'Wrong number of parameters for {dist} distribution. '
    msg += f'Expected {expected}, got {got}'
    return msg


class Default:
    default = True
    conditional = False


class Fixed:
    def __init__(self, val):
        self.args = (val, )


class Prior:
    default: bool = False
    conditional: bool = False

    def __init__(self, dist='uniform', *args, conditional=False):
        if len(args) == 0:
            args = [0, 1]

        if isinstance(dist, rv_frozen):
            args = dist.args
            if args == ():
                args = (dist.kwds.get('loc', 0.0), dist.kwds.get('scale', 1.0))
            self.prior = (dist.dist.__class__, args)

        elif isinstance(dist, ModifiedLogUniform):
            self.prior = (ModifiedLogUniform, (dist.knee, dist.b))
        
        elif isinstance(dist, kumaraswamy):
            self.prior = (kumaraswamy, (dist.a, dist.b))

        elif isinstance(dist, Fixed):
            self.prior = (Fixed, (dist.args[0], None))

        elif dist.lower() in ('uniform', 'u'):
            assert len(args) == 2, wrong_npar('Uniform', 2, len(args))
            self.prior = (_c.uniform_gen, args)

        elif dist.lower() in ('normal', 'norm', 'gaussian', 'n'):
            assert len(args) == 2, wrong_npar('Gaussian', 2, len(args))
            self.prior = (_c.norm_gen, args)

        elif dist.lower() in ('rayleigh'):
            assert len(args) == 2, wrong_npar('Rayleigh', 2, len(args))
            self.prior = (_c.norm_gen, args)

        elif dist.lower() in ('loguniform', 'jeffeys', 'lu'):
            assert len(args) == 2, wrong_npar('LogUniform', 2, len(args))
            self.prior = (_c.reciprocal_gen, args)

        elif dist.lower() in ('modifiedloguniform', 'modjeffeys', 'mlu'):
            assert len(args) == 2, wrong_npar('ModifiedLogUniform', 2, len(args))
            self.prior = (ModifiedLogUniform, args)

        elif dist.lower() in ('kumaraswamy', 'kuma'):
            assert len(args) == 2, wrong_npar('Kumaraswamy', 2, len(args))
            self.prior = (kumaraswamy, args)

        elif dist.lower() in ('fix', 'fixed'):
            assert len(args) == 1, wrong_npar('Fixed', 1, len(args))
            self.prior = (Fixed, (args[0], None))

        else:
            raise ValueError(f'Cannot recognize distribution {dist}')

        self.conditional = conditional

    def to_kima(self):
        dist, args = self.prior
        # print(self.prior)
        reprs = {
            Fixed: f'<Fixed>({args[0]})',
            _c.uniform_gen: f'<Uniform>({args[0]}, {args[1]})',
            _c.reciprocal_gen: f'<LogUniform>({args[0]}, {args[1]})',
            ModifiedLogUniform: f'<ModifiedLogUniform>({args[0]}, {args[1]})',
            kumaraswamy: f'<Kumaraswamy>({args[0]}, {args[1]})',
            _c.norm_gen: f'<Gaussian>({args[0]}, {args[1]})',
            _c.rayleigh_gen: f'<Rayleigh>({args[1]})',
            _c.beta_gen: f'<Beta>({args[0]}, {args[1]})',
        }
        return f'make_prior{reprs[dist]}'

    def __repr__(self):
        dist, args = self.prior
        reprs = {
            Fixed: f'Fixed({args[0]})',
            _c.uniform_gen: f'Uniform({args[0]}, {args[1]})',
            _c.reciprocal_gen: f'LogUniform({args[0]}, {args[1]})',
            ModifiedLogUniform: f'ModifiedLogUniform({args[0]}, {args[1]})',
            kumaraswamy: f'Kumaraswamy({args[0]}, {args[1]})',
            _c.norm_gen: f'Gaussian({args[0]}, {args[1]})',
            _c.rayleigh_gen: f'Rayleigh({args[1]})',
            _c.beta_gen: f'Beta({args[0]}, {args[1]})',
        }
        return reprs[dist]


# _PRIORS = (Prior, Fixed, ModifiedLogUniform, kumaraswamy)


class PriorSet(dict):
    def __getattribute__(self, name: str):
        if name in self:
            return self[name]
        else:
            return super().__getattribute__(name)

    def setdefault(self, key, default=None, /) -> None:
        if key not in self:
            self[key] = default
            self[key].default = True
        return self[key]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self:
            old = self[name]
            if isinstance(value, Prior):
                self[name] = value
            else:
                self[name] = Prior(value)
            self[name].conditional = old.conditional

        elif not name.startswith('__'):
            print(f'No prior name "{name}" in this set')

    def to_kima(self, file=None, prefix=''):
        for name, prior in self.items():
            if prior.default:
                continue
            cond = 'conditional->' if prior.conditional else ''
            p = f'{prefix}{cond}{VARIABLE_NAMES[name]} = {prior.to_kima()};'
            if file is None:
                print(p)
            else:
                print(p, file=file)

