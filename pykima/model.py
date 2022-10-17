import os
import subprocess
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any
import numpy as np

from .showresults import showresults as pkshowresults
from .priors import Prior, PriorSet
from .utils import chdir


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


class data_from_arguments:
    def __repr__(self):
        return 'argv[1]'


class ModelContext(type):
    def __new__(cls, name, bases, attrs, **kwargs):
        def __enter__(self):
            return self

        def __exit__(self, typ, value, traceback):
            pass

        attrs[__enter__.__name__] = __enter__
        attrs[__exit__.__name__] = __exit__
        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, nmspc):
        super().__init__(name, bases, nmspc)


@dataclass
class RVmodel(metaclass=ModelContext):
    directory: str = field(default=os.getcwd(), init=False)

    _levels_hash: str = field(default='', init=False, repr=False)
    _OPTIONS_hash: int = field(default=0, init=False, repr=False)

    _GP: bool = field(default=False, init=False, repr=False)
    hyperpriors: bool = field(default=False, init=False, repr=False)
    _trend: bool = field(default=False, init=False, repr=False)
    _degree: int = field(default=0, init=False, repr=False)
    multi_instrument: bool = field(default=False, init=False, repr=False)
    known_object: bool = field(default=False, init=False, repr=False)
    n_known_object: int = field(default=0, init=False, repr=False)
    studentt: bool = field(default=False, init=False, repr=False)

    _data = None
    units: str = 'ms'
    skip: int = 0

    priors: PriorSet = field(default_factory=PriorSet, repr=False, init=False)
    OPTIONS: objdict = field(default_factory=objdict, repr=False, init=False)
    thinning: int = field(default=50, repr=False, init=False)

    fix: bool = True
    npmax: int = 1

    star_mass: float = 1.0
    enforce_stability: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        # default priors
        self.priors.setdefault('C', Prior('u', 'data.get_RV_min()', 'data.get_RV_max()'))
        self.priors.setdefault('J', Prior('mlu', 1.0, 'data.get_max_RV_span()'))

        # default OPTIONS
        self.OPTIONS.setdefault('particles', 2)
        self.OPTIONS.setdefault('new_level_interval', 5000)
        self.OPTIONS.setdefault('save_interval', 2000)
        self.OPTIONS.setdefault('thread_steps', 100)
        self.OPTIONS.setdefault('max_number_levels', 0)
        self.OPTIONS.setdefault('lambda', 10)
        self.OPTIONS.setdefault('beta', 100)
        self.OPTIONS.setdefault('max_saves', 1000)
        self.OPTIONS.setdefault('samples_file', '')
        self.OPTIONS.setdefault('sample_info_file', '')
        self.OPTIONS.setdefault('levels_file', '')

    def _get_fix(self):
        return self._fix

    def _set_fix(self, value: bool):
        self._fix = value

    def _get_npmax(self):
        return self._npmax

    def _set_npmax(self, value: int):
        self._npmax = value
        if value > 0:
            self.priors.setdefault('P', Prior('lu', 1, 1e5, conditional=True))
            self.priors.setdefault('K', Prior('mlu', 1, 1e3, conditional=True))
            self.priors.setdefault('e', Prior('u', 0, 1, conditional=True))
            self.priors.setdefault('w', Prior('u', 0, '2*PI', conditional=True))
            self.priors.setdefault('phi', Prior('u', 0, '2*PI', conditional=True))
        else:
            self.priors.pop('P', None)
            self.priors.pop('K', None)
            self.priors.pop('e', None)
            self.priors.pop('w', None)
            self.priors.pop('phi', None)

    # setter and getter for data ##############################################
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: Any):
        if isinstance(value, data_from_arguments):
            self._data = value
            self._data_arrays = False

        elif isinstance(value, str):
            self._filename = value
            assert os.path.exists(self._filename)
            self._data_arrays = False
            self._data = value

        elif isinstance(value, (tuple, list)):
            if all(isinstance(d, str) for d in value):
                self._filename = value
                _exist = all([os.path.exists(f) for f in self._filename])
                _exist_in_dir = all([os.path.exists(os.path.join(self.directory, f)) for f in self._filename])
                assert _exist or _exist_in_dir, f"Files {value} don't exist"

                self.multi_instrument = True
                self._data_arrays = False
            else:
                self._data = value
                self._data_arrays = True

    def define_data(self):
        if self._data_arrays:
            self._filename = os.path.join(self.directory, '_data.txt')
            np.savetxt(self._filename, np.c_[self.data])

    # setter and getter for GP ################################################

    @property
    def GP(self):
        return self._GP

    @GP.setter
    def GP(self, value: bool):
        self._GP = value
        if value is True:
            self.priors.setdefault('eta1', Prior('mlu', 1, 'data.get_RV_span()'))
            self.priors.setdefault('eta2', Prior('lu', 1, 'data.get_timespan()'))
            self.priors.setdefault('eta3', Prior('u', 10, 40))
            self.priors.setdefault('eta4', Prior('lu', 0.1, 10))
        else:
            self.priors.pop('eta1', None)
            self.priors.pop('eta2', None)
            self.priors.pop('eta3', None)
            self.priors.pop('eta4', None)

    # #########################################################################

    # setters and getters for trend and degree ################################

    @property
    def trend(self):
        return self._trend

    @trend.setter
    def trend(self, value: bool):
        self._trend = value
        if value is False:
            self.degree = 0

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value: int):
        self._degree = value
        if value == 0:
            self.priors.pop('slope', None)
            self.priors.pop('quadr', None)
            self.priors.pop('cubic', None)
        elif value == 1:
            self.priors.setdefault('slope', Prior('n', 0, '10**data.get_trend_magnitude(1)'))
            self.priors.pop('quadr', None)
            self.priors.pop('cubic', None)
        elif value == 2:
            self.priors.setdefault('slope', Prior('n', 0, '10**data.get_trend_magnitude(1)'))
            self.priors.setdefault('quadr', Prior('n', 0, '10**data.get_trend_magnitude(2)'))
            self.priors.pop('cubic', None)
        elif value == 3:
            self.priors.setdefault('slope', Prior('n', 0, '10**data.get_trend_magnitude(1)'))
            self.priors.setdefault('quadr', Prior('n', 0, '10**data.get_trend_magnitude(2)'))
            self.priors.setdefault('cubic', Prior('n', 0, '10**data.get_trend_magnitude(3)'))

    # #########################################################################


    def check(self):
        if self.trend:
            if self.degree == 0:
                raise ValueError('trend=True but degree=0')
        if self._GP and self.MA:
            msg = "GP and MA can't both be True at the same time"
            raise ValueError(msg)
        if self._GP and self.studentt:
            msg = "GP and studentt can't both be True at the same time"
            raise ValueError(msg)

    def save(self, really=False):
        """ Save this model to the OPTIONS file and the kima_setup file """
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self._save_OPTIONS()
        # self._check_makefile()
        if really:
            kima_setup_f = os.path.join(self.directory, 'kima_setup.cpp')
        else:
            kima_setup_f = os.path.join(self.directory, 'test.cpp')

        with open(kima_setup_f, 'w') as f:
            f.write('#include "kima.h"\n\n')
            self._write_constructor(f)
            self._inside_constructor(f)
            self._start_main(f)
            self._set_data(f)
            self._write_sampler(f)
            self._end_main(f)
            f.write('\n\n')

    # the following are helper methods to fill parts of the kima_setup file
    def _bool(self, val):
        return 'true' if val is True else 'false'

    def _write_constructor(self, file):
        """ fill in the beginning of the constructor """
        cons = 'RVmodel::RVmodel()'
        cons += f':fix({self._bool(self.fix)}),npmax({self.npmax})\n'
        file.write(cons)

    def _inside_constructor(self, file):
        """ fill in inside the constructor """
        file.write('{\n')

        indent = 4 * ' '

        if self.GP:
            file.write(f'{indent}GP = true;\n')
        #
        if self.trend:
            file.write(f'{indent}trend = true;\n')
            file.write(f'{indent}degree = {self.degree};\n')
        #
        if self.known_object:
            file.write(f'{indent}known_object = true;\n')
            file.write(f'{indent}n_known_object = {self.n_known_object};\n')
        #
        if self.studentt:
            file.write(f'{indent}studentt = true;\n')

        if self.star_mass != 1.0:
            file.write('\t' + f'star_mass = {self.star_mass};\n')

        if self.enforce_stability:
            file.write('\t' + 'enforce_stability = true;\n')

        # if self.GP and self.kernel is not None:
        #     file.write('\n\t' + f'kernel = {self.kernel};\n\n')

        cond = any([prior.conditional for prior in self.priors.values()])
        all_default = all([prior.default for prior in self.priors.values()])

        if cond and not all_default:
            file.write('    ')
            file.write('auto conditional = planets.get_conditional_prior();\n')

        self.priors.to_kima(file=file, prefix='    ')

        file.write('}\n')
        file.write('\n')

    def _write_sampler(self, file):
        mt = 'RVmodel'
        file.write(f'    Sampler<{mt}> sampler = setup<{mt}>(argc, argv);\n')
        file.write(f'    sampler.run({self.thinning});\n')

    def _set_data(self, file):
        T = '    '
        # if self.filename_from_argv:
        #     file.write(f'{T}datafile = argv[1];\n')
        #     file.write(f'{T}load(datafile, "{self.units}", {self.skip});\n')
        #     return
        # elif self.filename is None:
        #     file.write(f'{T}datafile = "";\n')
        #     file.write(f'{T}load(datafile, "{self.units}", {self.skip});\n')
        #     return

        if self.multi_instrument:
            files = [f'{T}{T}"{file}"' for file in self._filename]
            files = ',\n'.join(files)
            file.write(f'{T}datafiles = {{\n{files}\n{T}}};\n')
            file.write(
                f'{T}load_multi(datafiles, "{self.units}", {self.skip});\n')
        else:
            if self._data_arrays:
                self.define_data()
                file.write(f'{T}datafile = "_data.txt";\n')
                load = f'{T}load(datafile, "{self.units}", {self.skip});'
                file.write(load + '\n')
            else:
                if isinstance(self.data, str):
                    D = f'"{self.data}"'
                else:
                    D = self.data

                load = f'{T}load({D}, "{self.units}", {self.skip});'
                file.write(load + '\n')

    def _start_main(self, file):
        file.write('\n')
        file.write('int main(int argc, char** argv)\n')
        file.write('{\n')

    def _end_main(self, file):
        file.write('}\n')

    def _save_OPTIONS(self):
        """ Save sampler settings to a OPTIONS file """
        self._OPTIONS_hash = hash(frozenset(self.OPTIONS.items()))
        opt = list(self.OPTIONS.values())
        options_f = os.path.join(self.directory, 'OPTIONS')
        with open(options_f, 'w') as file:
            file.write('# File containing parameters for DNest4\n')
            file.write('# Put comments at the top, or at line ends\n')
            file.write(f'{opt[0]}\t# Number of particles\n')
            file.write(f'{opt[1]}\t# new level interval\n')
            file.write(f'{opt[2]}\t# save interval\n')
            file.write(
                f'{opt[3]}\t# threadSteps: number of steps each thread does independently before communication\n'
            )
            file.write(f'{opt[4]}\t# maximum number of levels\n')
            file.write(f'{opt[5]}\t# Backtracking scale length (lambda)\n')
            file.write(
                f'{opt[6]}\t# Strength of effect to force histogram to equal push (beta)\n'
            )
            file.write(f'{opt[7]}\t# Maximum number of saves (0 = infinite)\n')
            file.write('    # (optional) samples file\n')
            file.write('    # (optional) sample_info file\n')
            file.write('    # (optional) levels file\n')

    @property
    def results(self):
        _ = self.showresults()
        return self.res

    def showresults(self, force=False):
        # calculate the hash of the levels.txt file the first time
        # we create self.res to avoid calling showresults repeatedly
        levels_f = os.path.join(self.directory, 'levels.txt')
        h = md5(open(levels_f, 'rb').read()).hexdigest()

        output = None
        if h != self._levels_hash or force:
            self._levels_hash = h
            # with io.StringIO() as buf, contextlib.redirect_stdout(buf):  # redirect stdout
            with chdir(self.directory):
                self.res = pkshowresults(force_return=True, show_plots=False,
                                         verbose=False)
            # output = buf.getvalue()
        # self.res.return_figs = True
        return output


    def run(self, threads: int = 4, thinning: int = 50,
            verbose: bool = True) -> None:

        if hash(frozenset(self.OPTIONS.items())) != self._OPTIONS_hash:
            self._save_OPTIONS()

        cmd = f'kima-run {self.directory} -t {threads} '
        if not verbose:
            cmd += ' -q'
        _ = subprocess.check_call(cmd.split())


RVmodel.fix = property(RVmodel._get_fix, RVmodel._set_fix)
RVmodel.npmax = property(RVmodel._get_npmax, RVmodel._set_npmax)
