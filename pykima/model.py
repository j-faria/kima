import os, sys
import re
import io
import subprocess
from pprint import pformat, pprint
from hashlib import md5
import contextlib
import numpy as np
from .showresults import showresults as pkshowresults

thisdir = os.path.dirname(os.path.realpath(__file__))
kimadir = os.path.dirname(thisdir)

@contextlib.contextmanager
def chdir(dir):
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


# class to interface with a kima model and a KimaResults instance
class KimaModel:
    """ Create and run kima models from Python """
    def __init__(self):
        self.directory = os.getcwd()
        self.kima_setup = 'kima_setup.cpp'
        self.OPTIONS_file = 'OPTIONS'
        self._filename = None
        self.skip = 0
        self.units = 'kms'

        self._warnings = True
        self._levels_hash = ''
        self._loaded = False

        self.GP = False
        self.MA = False
        self.hyperpriors = False
        self._trend = False
        self.degree = 0
        self._known_object = False
        self._n_known_object = 0
        self.studentt = False

        self.fix_Np = True
        self.max_Np = 1

        self.set_priors('default')
        self._planet_priors = ('Pprior', 'Kprior', 'eprior', 'wprior', 'phiprior')

        self.priors_need_data = False

        self.threads = 1
        self.OPTIONS = {
            'particles': 2,
            'new_level_interval': 5000,
            'save_interval': 2000,
            'thread_steps': 100,
            'max_number_levels': 0,
            'lambda': 10,
            'beta': 100,
            'max_saves': 100,
            'samples_file': '',
            'sample_info_file': '',
            'levels_file': '',
        }

    def __repr__(self):
        r = f'kima model on directory: {self.directory}\n'
        return r

    def __str__(self):
        return f'kima model in {self.directory}'

    @property
    def multi_instrument(self):
        if self.filename is None:
            return False
        if len(self.filename) == 1:
            return False
        else:
            return True

    @property
    def trend(self):
        return self._trend
    @trend.setter
    def trend(self, b):
        if self._warnings and b and self.degree == 0:
            print("don't forget to set the degree")
        self._trend = b

    @property
    def known_object(self):
        return self._known_object
    @known_object.setter
    def known_object(self, b):
        if self._warnings and b:
            print("don't forget to set n_known_object and all respective priors")
        else:
            for i in range(self.n_known_object):
                self._priors.pop(f'KO_Pprior[{i}]')
                self._priors.pop(f'KO_Kprior[{i}]')
                self._priors.pop(f'KO_eprior[{i}]')
                self._priors.pop(f'KO_phiprior[{i}]')
                self._priors.pop(f'KO_wprior[{i}]')
            self.n_known_object = 0
        self._known_object = b

    @property
    def n_known_object(self):
        return self._n_known_object

    @n_known_object.setter
    def n_known_object(self, val):
        if val > 0:
            for i in range(val):
                self._priors.update({f'KO_Pprior[{i}]': ()})
                self._priors.update({f'KO_Kprior[{i}]': ()})
                self._priors.update({f'KO_eprior[{i}]': ()})
                self._priors.update({f'KO_phiprior[{i}]': ()})
                self._priors.update({f'KO_wprior[{i}]': ()})
                self._priors.update({f'separator{4+i}': ''})

        self._n_known_object = val

    @property
    def _default_priors(self):
        dp = {
            # name: (default, distribution, arg1, arg2)
            'Cprior': (True, 'Uniform', self.ymin, self.ymax),
            'Jprior': (True, 'ModifiedLogUniform', 1.0, 100.0),
            'slope_prior':
                (True, 'Uniform', -self.topslope if self.data else None, self.topslope),
            'offsets_prior':
                (True, 'Uniform', -self.yspan if self.data else None, self.yspan),
            #
            'separator1': '',
            #
            'log_eta1_prior': (True, 'Uniform', -5.0, 5.0),
            'eta2_prior': (True, 'LogUniform', 1.0, 100.0),
            'eta3_prior': (True, 'Uniform', 10.0, 40.0),
            'log_eta4_prior': (True, 'Uniform', -1.0, 1.0),
            #
            'separator2': '',
            #
            'Pprior': (True, 'LogUniform', 1.0, 100000.0),
            'Kprior': (True, 'ModifiedLogUniform', 1.0, 1000.0),
            'eprior': (True, 'Uniform', 0.0, 1.0),
            'wprior': (True, 'Uniform', -np.pi, np.pi),
            'phiprior': (True, 'Uniform', 0, np.pi),
            #
            'separator3': '',
            #
        }
        return dp

    @property
    def priors(self):
        """ 
        This dictionary holds the model priors in the form
            prior_name: (default?, distribution, parameter1, parameter2)
        when parameter1 or parameter2 are None, they depend on the data.

        To set a prior use
            set_priors(prior_name, distribution, paramemter1, <parameter2>)
        To set one specific prior to its default use
            set_prior_to_default(prior_name)
        or to set all priors to the defaults just use
            set_priors()
        """
        return self._priors

    def set_priors(self, which='default', *args):
        if len(args) > 0 and not isinstance(args[0], bool):
            args = [False, *args]
        if which == 'default':
            self._priors = self._default_priors
        else:
            if len(args) == 3:
                assert args[1] == 'Fixed'
            self._priors.update({which: args})

    def set_prior_to_default(self, which):
        default_prior = self._default_priors[which]
        self._priors.update({which: default_prior})

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, f):
        if f is None:
            self._filename = f
            return
        if isinstance(f, list):
            files = f
        else:
            files = [f]

        relative_files = []
        for i, file in enumerate(files):
            if os.path.dirname(file) != self.directory:
                relative_files.append(os.path.abspath(file))
            else:
                relative_files.append(os.path.basename(file))

        self._filename = relative_files
        self._filename_full_path = [os.path.abspath(f) for f in self._filename]
        self.data


    @property
    def data(self):
        if self.filename is None:
            return None

        d = dict(t=[], y=[], e=[])
        for f in self.filename:
            try:
                t, y, e = np.loadtxt(f, skiprows=self.skip, unpack=True,
                                     usecols=range(3))
            except Exception as e:
                print(f'cannot read from {f} (is skip correct?)')
                print(str(e))
                return None

            d['t'].append(t)
            d['y'].append(y)
            d['e'].append(e)

        d['t'] = np.concatenate(d['t'])
        d['y'] = np.concatenate(d['y'])
        d['e'] = np.concatenate(d['e'])
        return d

    @property
    def ymin(self):
        if self.data:
            return self.data['y'].min().round(3)

    @property
    def ymax(self):
        if self.data:
            return self.data['y'].max().round(3)

    @property
    def yspan(self):
        if self.data:
            return self.data['y'].ptp().round(3)

    @property
    def topslope(self):
        if self.data:
            val = np.abs(self.data['y'].ptp() / self.data['t'].ptp()).round(3)
            return val

    @property
    def results(self):
        self.showresults()
        return self.res

    def showresults(self, force=False):
        # calculate the hash of the levels.txt file the first time
        # we create self.res to avoid calling showresults repeatedly
        levels_f = os.path.join(self.directory, 'levels.txt')
        if not os.path.exists(levels_f):
            raise ValueError(
                'levels.txt does not exist, did you run the model?')
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

    def load(self):
        """
        Try to read and load a kima_setup file with the help of RegExp.
        Note that C++ is notoriously hard to parse, so for anything other than
        fairly standard kima_setup files, don't expect this function to work!
        """
        if not os.path.exists(self.kima_setup):
            raise FileNotFoundError(f'Cannot find "{self.kima_setup}"')

        setup = open(self.kima_setup).read()

        # find general model settings
        bools = (
            'GP',
            'MA',
            'hyperpriors',
            'trend',
            'known_object',
            'studentt'
        )
        for b in bools:
            pat = re.compile(f'const bool {b} = (\w+)')
            match = pat.findall(setup)
            if len(match) == 1:
                setattr(self, b, True if match[0] == 'true' else False)
            else:
                msg = f'Cannot find setting {b} in {self.kima_setup}'
                raise ValueError(msg)

        ints = (
            'degree',
            'n_known_object',
        )
        for i in ints:
            pat = re.compile(f'const int {i} = (\d+)')
            match = pat.findall(setup)
            if len(match) == 1:
                setattr(self, i, int(match[0]))
            else:
                msg = f'Cannot find setting {i} in {self.kima_setup}'
                raise ValueError(msg)

        # find fix Np
        pat = re.compile(r'fix\((\w+)\)')
        match = pat.findall(setup)
        if len(match) == 1:
            self.fix_Np = True if match[0] == 'true' else False
        else:
            msg = f'Cannot find option for fix in {self.kima_setup}'
            raise ValueError(msg)

        # find max Np
        pat = re.compile(r'npmax\(([-+]?[0-9]+)\)')
        match = pat.findall(setup)
        if len(match) == 1:
            if int(match[0]) < 0:
                raise ValueError('npmax must be >= 0')
            self.max_Np = int(match[0])
        else:
            msg = f'Cannot find option for npmax in {self.kima_setup}'
            raise ValueError(msg)

        # find priors (here be dragons!)
        number = '([-+]?[0-9]*.?[0-9]*[E0-9]*)'
        for prior in self._default_priors.keys():
            pat = re.compile(
                rf'{prior}\s?=\s?make_prior<(\w+)>\({number}\s?,\s?{number}\)')
            match = pat.findall(setup)
            if len(match) == 1:
                m = match[0]
                dist, arg1, arg2 = m[0], float(m[1]), float(m[2])
                self.set_priors(prior, False, dist, arg1, arg2)

        # find datafile(s)
        pat = re.compile(f'const bool multi_instrument = (\w+)')
        match = pat.findall(setup)
        multi_instrument = True if match[0] == 'true' else False

        if multi_instrument:
            pat = re.compile(r'datafiles\s?\=\s?\{(.*?)\}',
                             flags=re.MULTILINE | re.DOTALL)
            match_inside_braces = pat.findall(setup)
            inside_braces = match_inside_braces[0]

            pat = re.compile(r'"(.*?)"')
            match = pat.findall(inside_braces)
            self.filename = match
            #     msg = f'Cannot find datafiles in {self.kima_setup}'
            #     raise ValueError(msg)

            pat = re.compile(r'load_multi\(datafiles,\s*"(.*?)"\s*,\s*(\d)')
            match = pat.findall(setup)
            if len(match) == 1:
                units, skip = match[0]
                self.units = units
                self.skip = int(skip)
            else:
                msg = f'Cannot find units and skip in {self.kima_setup}'
                raise ValueError(msg)

        else:
            if 'datafile = ""' in setup:
                self.filename = None
                return

            pat = re.compile(r'^[^\/\/\n]*datafile\s?\=\s?"(.+?)"\s?;',
                             re.MULTILINE)
            match = pat.findall(setup)
            if len(match) == 1:
                self.filename = match
            else:
                msg = f'Cannot find unique datafile in {self.kima_setup}'
                raise ValueError(msg)

            pat = re.compile(r'load\(datafile,\s*"(.*?)"\s*,\s*(\d)')
            match = pat.findall(setup)
            if len(match) == 1:
                units, skip = match[0]
                self.units = units
                self.skip = int(skip)
            else:
                msg = f'Cannot find units and skip in {self.kima_setup}'
                raise ValueError(msg)

        # store that the model has been loaded
        self._loaded = True

    def load_OPTIONS(self):
        if not os.path.exists(self.OPTIONS_file):
            raise FileNotFoundError(f'Cannot find "{self.OPTIONS_file}"')

        options = open(self.OPTIONS_file).readlines()

        keys = list(self.OPTIONS.keys())
        i = 0
        for line in options:
            if line.strip().startswith('#'):
                continue
            val, _ = line.split('#')
            val = int(val)
            self.OPTIONS[keys[i]] = val
            i += 1

        try:
            with open('.KIMATHREADS') as f:
                self.threads = int(f.read().strip())
        except FileNotFoundError:
            pass

    def save(self):
        """ Save this model to the OPTIONS file and the kima_setup file """
        self._save_OPTIONS()
        self._check_makefile()

        kima_setup_f = os.path.join(self.directory, self.kima_setup)
        # if os.path.exists(kima_setup_f):
        #     print(f'File {kima_setup_f} exists in {self.directory}. ', end='')
        #     answer = input('Replace? (Y/n)')
        #     if answer.lower() != 'y':
        #         return

        with open(kima_setup_f, 'w') as f:
            f.write('#include "kima.h"\n\n')
            self._write_settings(f)
            self._write_constructor(f)
            self._inside_constructor(f)
            self._start_main(f)
            self._set_data(f)
            self._write_sampler(f)
            self._end_main(f)
            f.write('\n\n')

    # the following are helper methods to fill parts of the kima_setup file
    def _write_settings(self, file):
        """
        fill in the settings part of the file, with general model options
        """
        def r(val): return 'true' if val else 'false'
        cb = 'const bool'
        ci = 'const int'
        file.write(f'{cb} GP = {r(self.GP)};\n')
        file.write(f'{cb} MA = {r(self.MA)};\n')
        file.write(f'{cb} hyperpriors = {r(self.hyperpriors)};\n')
        file.write(f'{cb} trend = {r(self.trend)};\n')
        file.write(f'{ci} degree = {self.degree};\n')
        file.write(f'{cb} multi_instrument = {r(self.multi_instrument)};\n')
        file.write(f'{cb} known_object = {r(self.known_object)};\n')
        file.write(f'{ci} n_known_object = {self.n_known_object};\n')
        file.write(f'{cb} studentt = {r(self.studentt)};\n')
        file.write('\n')

    def _write_constructor(self, file):
        """ fill in the beginning of the RVmodel constructor """
        def r(val): return 'true' if val else 'false'
        file.write(
            f'RVmodel::RVmodel():fix({r(self.fix_Np)}),npmax({self.max_Np})\n')

    def _inside_constructor(self, file):
        """ fill in inside the RVmodel constructor """
        file.write('{\n')

        if self.priors_need_data:
            file.write('\t' + 'auto data = get_data();\n\n')

        def write_prior_n(name, sets, add_conditional=False):
            s = 'c->' if add_conditional else ''
            arguments = ", ".join(map(str, sets[2:]))
            return s + f'{name} = make_prior<{sets[1]}>({arguments});\n'

        def write_prior_2(name, sets, add_conditional=False):
            s = 'c->' if add_conditional else ''
            return s + f'{name} = make_prior<{sets[1]}>({sets[2]}, {sets[3]});\n'

        def write_prior_1(name, sets, add_conditional=False):
            s = 'c->' if add_conditional else ''
            return s + f'{name} = make_prior<{sets[1]}>({sets[2]});\n'

        def write_prior(name, sets, add_conditional):
            if len(sets) > 4:
                return write_prior_n(name, sets, add_conditional)
            elif sets[1] == 'Fixed':
                return write_prior_1(name, sets, add_conditional)
            else:
                return write_prior_2(name, sets, add_conditional)

        got_conditional = False
        wrote_something = False
        for name, sets in self._priors.items():

            if name.startswith('separator'):
                if wrote_something:
                    file.write('\n')
                continue

            # print(name, sets)
            if not sets[0]:  # if not default prior
                wrote_something = True
                if name in self._planet_priors:
                    if not got_conditional:
                        file.write(
                            '\t' +
                            f'auto c = planets.get_conditional_prior();\n')
                        got_conditional = True

                    file.write('\t' + write_prior(name, sets, True))
                else:
                    file.write('\t' + write_prior(name, sets, False))

        file.write('}\n')
        file.write('\n')

    def _write_sampler(self, file):
        file.write('\tSampler<RVmodel> sampler = setup<RVmodel>(argc, argv);\n')
        file.write('\tsampler.run(50);\n')

    def _set_data(self, file):
        if self.filename is None:
            file.write(f'\tdatafile = "";\n')
            file.write(f'\tload(datafile, "{self.units}", {self.skip});\n')
            return

        if self.multi_instrument:
            files = [f'"{datafile}"' for datafile in self.filename]
            files = ', '.join(files)
            file.write(f'\tdatafiles = {{ {files} }};\n')
            file.write(
                f'\tload_multi(datafiles, "{self.units}", {self.skip});\n')
        else:
            file.write(f'\tdatafile = "{self.filename[0]}";\n')
            file.write(f'\tload(datafile, "{self.units}", {self.skip});\n')

    def _start_main(self, file):
        file.write('int main(int argc, char** argv)\n')
        file.write('{\n')

    def _end_main(self, file):
        file.write('\treturn 0;\n')
        file.write('}\n')
    ##

    def _save_OPTIONS(self):
        """ Save sampler settings to a OPTIONS file """
        opt = list(self.OPTIONS.values())
        options_f = os.path.join(self.directory, self.OPTIONS_file)
        with open(options_f, 'w') as file:
            file.write('# File containing parameters for DNest4\n')
            file.write(
                '# Put comments at the top, or at the end of the line.\n')
            file.write(f'{opt[0]}\t# Number of particles\n')
            file.write(f'{opt[1]}\t# new level interval\n')
            file.write(f'{opt[2]}\t# save interval\n')
            file.write(
                f'{opt[3]}\t# threadSteps: number of steps each thread does independently before communication\n')
            file.write(f'{opt[4]}\t# maximum number of levels\n')
            file.write(f'{opt[5]}\t# Backtracking scale length (lambda)\n')
            file.write(
                f'{opt[6]}\t# Strength of effect to force histogram to equal push (beta)\n')
            file.write(f'{opt[7]}\t# Maximum number of saves (0 = infinite)\n')
            file.write('    # (optional) samples file\n')
            file.write('    # (optional) sample_info file\n')
            file.write('    # (optional) levels file\n')

    def _check_makefile(self):
        make_f = os.path.join(self.directory, 'Makefile')
        if not os.path.exists(make_f):
            with open(make_f, 'w') as f:
                print(f'KIMA_DIR = {kimadir}', file=f)
                print('include $(KIMA_DIR)/examples.mk', file=f)


    def run(self):
        if self.data is None:
            raise ValueError('Must set .filename before running')

        self.save()
        cmd = f'kima-run {self.directory}'
        out = subprocess.check_call(cmd.split())
