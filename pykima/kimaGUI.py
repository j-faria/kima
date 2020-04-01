import sys

try:
    import PyQt5
except ImportError:
    print('Sorry, the GUI requires the PyQT5 package...')
    sys.exit(1)

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QStatusBar, QInputDialog
from PyQt5.QtCore import QProcess, Qt

try:
    from .gui_helpers import *
except (ModuleNotFoundError, ImportError):
    from gui_helpers import *

import os
import re
import io
import shutil
import argparse
import contextlib
from contextlib import redirect_stdout
from collections import namedtuple
from hashlib import md5

import numpy as np
import matplotlib.pyplot as plt
import pykima as pk

# from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)


@contextlib.contextmanager
def chdir(dir):
    curdir = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(curdir)


def usage():
    u = "kima-gui [DIRECTORY]\n"
    u += "Create a kima template in DIRECTORY and run the GUI there.\n"
    return u


def _parse_args():
    parser = argparse.ArgumentParser(usage=usage())
    parser.add_argument('DIRECTORY', help='directory where to run GUI')
    parser.add_argument('--version', action='store_true',
                        help='show version and exit')
    parser.add_argument('--debug', action='store_true',
                        help='be more verbose, for debugging')
    return parser.parse_args()


# class to interface with the a kima model and a KimaResults instance
class KimaModel:
    def __init__(self):
        self.directory = 'some dir'
        self.kima_setup = 'kima_setup.cpp'
        self.OPTIONS_file = 'OPTIONS'
        self.filename = None
        self.skip = 0
        self.units = 'kms'

        self._levels_hash = ''
        self._loaded = False

        self.obs_after_HARPS_fibers = False
        self.GP = False
        self.MA = False
        self.hyperpriors = False
        self.trend = False
        self.known_object = False

        self.fix_Np = True
        self.max_Np = 1

        self.set_priors('default')
        self._planet_priors = ('Pprior', 'Kprior', 'eprior')

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

    @property
    def multi_instrument(self):
        if self.filename is None:
            return False
        if len(self.filename) == 1:
            return False
        else:
            return True

    @property
    def _default_priors(self):
        dp = {
            # name: (default, distribution, arg1, arg2)
            'Cprior': (True, 'Uniform', self.ymin, self.ymax),
            'Jprior': (True, 'ModifiedLogUniform', 1.0, 100.0),
            'slope_prior': (True, 'Uniform', -self.topslope if self.data else None, self.topslope),
            'offsets_prior': (True, 'Uniform', -self.yspan if self.data else None, self.yspan),
            #
            'log_eta1_prior': (True, 'Uniform', -5.0, 5.0),
            'eta2_prior': (True, 'LogUniform', 1.0, 100.0),
            'eta3_prior': (True, 'Uniform', 10.0, 40.0),
            'log_eta4_prior': (True, 'Uniform', -1.0, 1.0),
            #
            'Pprior': (True, 'LogUniform', 1.0, 100000.0),
            'Kprior': (True, 'ModifiedLogUniform', 1.0, 1000.0),
            'eprior': (True, 'Uniform', 0.0, 1.0),
        }
        return dp

    def set_priors(self, which='default', *args):
        if which == 'default':
            self.priors = self._default_priors
        else:
            assert len(args) == 4
            self.priors.update({which: args})

    def set_prior_to_default(self, which):
        default_prior = self._default_priors[which]
        self.priors.update({which: default_prior})

    @property
    def data(self):
        if self.filename is None:
            return None

        d = dict(t=[], y=[], e=[])
        for f in self.filename:
            try:
                t, y, e = np.loadtxt(f, skiprows=self.skip, usecols=range(3)).T
            except:
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

    def results(self, force=False):
        # calculate the hash of the levels.txt file the first time
        # we create self.res to avoid calling showresults repeatedly
        h = md5(open('levels.txt', 'rb').read()).hexdigest()

        output = None
        if h != self._levels_hash or force:
            self._levels_hash = h
            with io.StringIO() as buf, redirect_stdout(buf):  # redirect stdout
                self.res = pk.showresults(force_return=True, show_plots=False)
                output = buf.getvalue()

        self.res.return_figs = True
        plt.ioff()

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
            'obs_after_HARPS_fibers',
            'GP',
            'MA',
            'hyperpriors',
            'trend',
            'known_object',
        )
        for b in bools:
            pat = re.compile(f'const bool {b} = (\w+)')
            match = pat.findall(setup)
            if len(match) == 1:
                setattr(self, b, True if match[0] == 'true' else False)
            else:
                msg = f'Cannot find setting {b} in {self.kima_setup}'
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

            pat = re.compile(r'datafile\s?\=\s?"(.+?)"\s?;')
            match = pat.findall(setup)
            if len(match) == 1:
                self.filename = match
            else:
                msg = f'Cannot find datafile in {self.kima_setup}'
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

        with open(self.kima_setup, 'w') as f:
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
        def r(val): return 'true' if val else 'false'
        cb = 'const bool'
        file.write(
            f'{cb} obs_after_HARPS_fibers = {r(self.obs_after_HARPS_fibers)};\n')
        file.write(f'{cb} GP = {r(self.GP)};\n')
        file.write(f'{cb} MA = {r(self.MA)};\n')
        file.write(f'{cb} hyperpriors = {r(self.hyperpriors)};\n')
        file.write(f'{cb} trend = {r(self.trend)};\n')
        file.write(f'{cb} multi_instrument = {r(self.multi_instrument)};\n')
        file.write(f'{cb} known_object = {r(self.known_object)};\n')
        file.write('\n')

    def _write_constructor(self, file):
        def r(val): return 'true' if val else 'false'
        file.write(
            f'RVmodel::RVmodel():fix({r(self.fix_Np)}),npmax({self.max_Np})\n')

    def _inside_constructor(self, file):
        file.write('{\n')

        got_conditional = False
        for name, sets in self.priors.items():
            # print(name, sets)
            if not sets[0]:  # if not default prior
                if name in self._planet_priors:
                    if not got_conditional:
                        file.write(
                            f'auto c = planets.get_conditional_prior();\n')
                        got_conditional = True
                    file.write(
                        f'c->{name} = make_prior<{sets[1]}>({sets[2]}, {sets[3]});\n')
                else:
                    file.write(
                        f'{name} = make_prior<{sets[1]}>({sets[2]}, {sets[3]});\n')

        file.write('\n}\n')
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
        with open('OPTIONS', 'w') as file:
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


# PyQT class for the GUI
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, directory, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # load the UI Page
        uifile = os.path.join(os.path.dirname(__file__), 'kimaGUI.ui')
        uic.loadUi(uifile, self)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        plt.ioff()

        self.canvases, self.toolbars = {}, {}
        self.model = KimaModel()

        self.model.directory = directory
        os.chdir(directory)

        msg = ''
        if os.path.exists(self.model.kima_setup):
            msg += f'Loaded setup from {self.model.kima_setup}'
            self.model.load()
        if os.path.exists(self.model.OPTIONS_file):
            msg += '; Loaded OPTIONS'
            self.model.load_OPTIONS()
        self.statusMessage(msg)

        self.terminal_widget.push({"np": np, "plt": plt, "pk": pk})
        self.terminal_widget.execute_command('%matplotlib inline')

        self.output.setText('No output from run yet.')
        self.updateUI()
        self.toggleTrend(self.trend_check.isChecked())
        self.progressBar.setValue(0)

    def showKimaHelp(self):
        show_kima_help_window()

    def showGUIHelp(self):
        show_gui_help_window()

    def showAbout(self):
        show_about_window()

    def statusMessage(self, message):
        """ Print a message at the bottom of the window """
        self.statusBar.showMessage(message)

    def _error(self, message, title=''):
        """ Show an error message """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(f"Error\n{message}")
        msg.setWindowTitle(title)
        msg.exec_()

    def updateUI(self):
        """
        Updates the widgets whenever an interaction happens. Typically some
        interaction takes place, the UI responds, and informs the model of the
        change. Then this method is called, pulling from the model information
        that is updated in the GUI.
        """
        self.lineEdit_2.setText(self.model.directory)
        if self.model.filename is None:
            files = ''
        else:
            files = [os.path.basename(f) for f in self.model.filename]
            files = ', '.join(files)
        self.lineEdit.setText(files)

        unit = {'kms': 'km/s', 'ms': 'm/s'}[self.model.units]
        index = self.units.findText(unit, Qt.MatchFixedString)
        if index >= 0:
            self.units.setCurrentIndex(index)
        self.header_skip.setValue(self.model.skip)

        # general model settings
        self.obs_after_HARPS_fibers_check.setChecked(
            self.model.obs_after_HARPS_fibers)
        self.GP_check.setChecked(self.model.GP)
        self.MA_check.setChecked(self.model.MA)
        self.hyperpriors_check.setChecked(self.model.hyperpriors)
        self.trend_check.setChecked(self.model.trend)
        self.known_object_check.setChecked(self.model.known_object)

        self.fix_Np_check.setChecked(self.model.fix_Np)
        self.max_Np.setValue(self.model.max_Np)

        # grey-out plot tabs
        self.tabWidget.setTabEnabled(3, self.model.GP)
        self.tabWidget.setTabEnabled(4, self.model.GP)
        # self.tabWidget.setCurrentIndex(8)
        self.tabWidget_2.setTabEnabled(2, self.model.GP)

        # priors!
        self.toggleMultiInstrument(self.model.multi_instrument)

        for prior_name, prior in self.model.priors.items():
            # print(prior_name, prior)
            # each prior's comboBox
            dist = getattr(self, 'comboBox_' + prior_name)
            # find the index of this prior's distribution name
            index = dist.findText(prior[1], Qt.MatchFixedString)
            if index >= 0:
                dist.setCurrentIndex(index)
            # set the two prior arguments
            arg1 = getattr(self, 'lineEdit_' + prior_name + '_arg1')
            if prior[2] is None:
                arg1.setText('None')
            else:
                arg1.setText(str(round(prior[2], 3)))

            arg2 = getattr(self, 'lineEdit_' + prior_name + '_arg2')
            if prior[3] is None:
                arg2.setText('None')
            else:
                arg2.setText(str(round(prior[3], 3)))

            # check the "is default" radio button
            radio = getattr(self, 'is_default_' + prior_name)
            radio.setChecked(prior[0])

        # sampler options
        self.new_level_interval.setValue(
            self.model.OPTIONS['new_level_interval'])
        self.save_interval.setValue(self.model.OPTIONS['save_interval'])
        self.max_saves.setValue(self.model.OPTIONS['max_saves'])
        self.threads.setValue(self.model.threads)

    def shutdown_kernel(self):
        """ Shutdown the embeded IPython kernel, but quietly """
        self.terminal_widget.kernel_client.stop_channels()
        self.terminal_widget.kernel_manager.shutdown_kernel()

    def addmpl(self, tab, fig):
        self.canvases[tab] = FigureCanvas(fig)
        getattr(self, tab+'_plot').addWidget(self.canvases[tab])
        self.canvases[tab].draw()
        self.toolbars[tab] = NavigationToolbar(self.canvases[tab], self)
        self.toolbars[tab].setMaximumHeight(30)
        self.toolbars[tab].setStyleSheet("QToolBar { border: 0px }")
        getattr(self, tab+'_plot').addWidget(self.toolbars[tab])

    def rmmpl(self, tab):
        getattr(self, tab+'_plot').removeWidget(self.canvases[tab])
        self.canvases[tab].close()
        getattr(self, tab+'_plot').removeWidget(self.toolbars[tab])
        self.toolbars[tab].close()

    def setmpl(self, tab, fig):
        if tab in self.canvases:
            self.rmmpl(tab)
        self.addmpl(tab, fig)

    def slot1(self):
        """ Called when the user presses the Browse button """
        self.statusMessage("Browse button pressed")
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "Select data file(s)",
            "",
            "All Files (*);;RDB Files (*.rdb);;Text files (*.txt)",
            options=options)
        if files:
            n = len(files)
            if n > 1:
                self.statusMessage(f"Loading {n} files")
            else:
                self.statusMessage(f"Loading file {files[0]}")

            if self.copy_locally_check.isChecked():
                newfiles = []
                for file in files:
                    bname = os.path.basename(file)
                    if not os.path.exists(bname):
                        shutil.copy(file, bname)
                    newfiles.append(bname)
                files = newfiles

            self.model.filename = files
            for prior, sets in self.model.priors.items():
                if sets[0]:
                    self.model.set_prior_to_default(prior)

            self.updateUI()

    # def makePlot1(self):
    #     out = self.model.results()
    #     fig = self.model.res.make_plot1()
    #     self.setmpl('tab_1', fig)

    # def makePlot2(self):
    #     out = self.model.results()
    #     fig = self.model.res.make_plot2()
    #     self.setmpl('tab_2', fig)

    # def makePlot6(self):
    #     out = self.model.results()
    #     fig = self.model.res.plot_random_planets(show_vsys=True, show_trend=True)
    #     self.setmpl('tab_6', fig)
    #     self.tabWidget.setCurrentIndex(5)

    def makePlotsAll(self):
        out = self.model.results()
        if out:
            self.results_output.setText(out)
        self.terminal_widget.push({'res': self.model.res})
        plt.close('all')

        button = self.sender().objectName()

        if '6' in button:
            try:
                fig = self.model.res.plot_random_planets(show_vsys=True,
                                                         show_trend=True)
            except ValueError as e:
                self._error('Something went wrong creating this plot.',
                            'This is awkward...')
                return

            self.setmpl('tab_6', fig)
            self.tabWidget.setCurrentIndex(5)
            plt.close()

        elif '7' in button:
            fig71, fig72 = self.model.res.hist_vsys(show_offsets=True)
            self.setmpl('tab_7_1', fig71)
            if fig72:
                self.setmpl('tab_7_2', fig72)
            plt.close()

            fig73 = self.model.res.hist_extra_sigma()
            self.setmpl('tab_7_3', fig73)
            plt.close()

            self.tabWidget.setCurrentIndex(6)

        elif '8' in button:
            plt.ion()
            pk.classic.postprocess(plot=True)
            # for i in plt.get_fignums()[:1]:
            #     fig = plt.figure(i)
            # self.addmpl('tab_diagnostic', fig)
            # self.tabWidget.setCurrentIndex(0)
            plt.ioff()

        else:
            p = button[-1]
            method_name = 'make_plot' + p
            method = getattr(self.model.res, method_name)
            fig = method()
            if fig is None:
                self.statusMessage(f'plot number {p} cannot be created')
                return
            self.setmpl('tab_' + p, fig)
            self.tabWidget.setCurrentIndex(int(p)-1)

    def reloadResults(self):
        out = self.model.results(force=True)
        if out:
            self.results_output.setText(out)
        self.terminal_widget.push({'res': self.model.res})

    def savePickle(self):
        try:
            self.model.res
        except AttributeError:
            self._error('No results to save.', 'Error')
            return

        filename, ok = QInputDialog.getText(self, 'Save results',
                                            'Enter the file name:')
        if ok:
            if not filename.endswith('.pickle'):
                filename += '.pickle'

            if os.path.exists(filename):
                self.statusMessage(f'{filename} already exists!')
                return

            self.model.res.save_pickle(filename)
            self.statusMessage(f'Saved {filename}')

    def setDefaultPrior(self, *args):
        # get the name of the button which was pressed and from that which prior
        # is supposed to be made default
        button = self.sender()
        which = button.objectName().replace('makeDefault_', '')
        self.model.set_prior_to_default(which)
        self.updateUI()

    def setPrior(self, prior_name):
        names = {
            'Cprior': 'systemic velocity',
            'Jprior': 'jitter',
            'slope_prior': 'trend slope',
        }
        dist = getattr(self, 'comboBox_' + prior_name).currentText()
        arg1 = getattr(self, 'lineEdit_' + prior_name + '_arg1').text()
        arg2 = getattr(self, 'lineEdit_' + prior_name + '_arg2').text()
        if arg1 == '' or arg2 == '':
            name = names[prior_name]
            self._error(
                f'Please set the arguments in the prior for {name}',
                'Setting prior')
            raise ValueError

        if arg1 == 'None' or arg2 == 'None':
            return

        h1 = hash(self.model._default_priors[prior_name])
        h2 = hash((True, dist, float(arg1), float(arg2)))
        if h1 != h2:
            if prior_name == 'eprior' and dist == 'Uniform':
                a1, a2 = float(arg1), float(arg2)
                if a1 < 0 or a1 > 1 or a2 < 0 or a2 > 1 or a1 > a2:
                    self._error(
                        'Eccentricity prior must have 0 < arg1 < arg2 < 1',
                        'Prior limits')
                    return

            self.model.set_priors(prior_name, False, dist,
                                  float(arg1), float(arg2))

    def toggleTrend(self, toggled):
        self.model.trend = toggled
        self.label_slope_prior.setEnabled(toggled)
        self.comboBox_slope_prior.setEnabled(toggled)
        self.lineEdit_slope_prior_arg1.setEnabled(toggled)
        self.lineEdit_slope_prior_arg2.setEnabled(toggled)
        self.makeDefault_slope_prior.setEnabled(toggled)

    def toggleGP(self, toggled):
        self.model.GP = toggled
        self.tabWidget_2.setTabEnabled(2, self.model.GP)

    def togglePlanets(self, val):
        self.tabWidget_2.setTabEnabled(1, val != 0)

    def toggleMultiInstrument(self, toggled):
        self.label_offsets_prior.setEnabled(toggled)
        self.comboBox_offsets_prior.setEnabled(toggled)
        self.lineEdit_offsets_prior_arg1.setEnabled(toggled)
        self.lineEdit_offsets_prior_arg2.setEnabled(toggled)
        self.makeDefault_offsets_prior.setEnabled(toggled)

    def saveModel(self):
        # if model has been loaded from an existing kima_setup file, warn the
        # user that this action might destroy that file
        if self.model._loaded:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Warning\nThe model has been set using a custom "
                        "kima_setup file. Saving the model might destroy "
                        "information in that file "
                        "(e.g. comments, extra code, etc)")
            msg.setWindowTitle("Replacing kima_setup file")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ignore
                                   | QtWidgets.QMessageBox.Cancel)
            pressed = msg.exec_()
            if pressed == QtWidgets.QMessageBox.Cancel:
                return

            # not anymore
            self.model._loaded = False

        self.model.obs_after_HARPS_fibers = self.obs_after_HARPS_fibers_check.isChecked()
        self.model.GP = self.GP_check.isChecked()
        self.model.MA = self.MA_check.isChecked()
        self.model.hyperpriors = self.hyperpriors_check.isChecked()
        self.model.trend = self.trend_check.isChecked()
        self.model.known_object = self.known_object_check.isChecked()

        self.model.fix_Np = self.fix_Np_check.isChecked()
        self.model.max_Np = self.max_Np.value()

        self.model.skip = self.header_skip.value()
        self.model.units = {'km/s': 'kms',
                            'm/s': 'ms'}[self.units.currentText()]
        self.model.data  # try loading the data

        self.setOPTIONS()
        for prior in self.model.priors.keys():
            self.setPrior(prior)

        self.updateUI()
        self.model.save()
        self.statusMessage('Saved model')

    def setOPTIONS(self):
        self.model.OPTIONS['new_level_interval'] = int(
            self.new_level_interval.value())
        self.model.OPTIONS['save_interval'] = int(self.save_interval.value())
        self.model.OPTIONS['max_saves'] = int(self.max_saves.value())

    def stop(self):
        try:
            self.process
        except AttributeError:
            return
        if self.process.state() == QProcess.Running:
            self.process.terminate()

    def run(self):
        if self.model.filename is None:
            self._error('Please set the data file(s)', 'No data file')
            return

        threads = self.threads.value()
        self.model.threads = threads
        with open('.KIMATHREADS', 'w') as f:
            f.write(str(threads) + '\n')

        self.saveModel()

        # Clear the output panel
        self.output.setText('')

        # QProcess object for external app
        self.process = QProcess(self)

        # Should it run in the background?
        bg = self.run_in_bg.isChecked()

        if not bg:
            # QProcess emits `readyRead` when there is data to be read
            self.process.readyRead.connect(self.printProcessOutput)

            # To prevent accidentally running multiple times disable the "Run"
            # button when process starts, and enable it when it finishes
            self.process.started.connect(
                lambda: self.runButton.setEnabled(False))
            self.process.finished.connect(
                lambda: self.runButton.setEnabled(True))

        self.callKima(threads, bg)

    def printProcessOutput(self):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAll(), 'utf-8'))
        self.output.ensureCursorVisible()

        try:
            samples = sum(1 for i in open('sample.txt', 'rb')) - 1
            total = self.model.OPTIONS['max_saves']
        except FileNotFoundError:
            samples = 0
            total = 1
        self.progressBar.setValue(100 * samples / total)

    def callKima(self, threads, bg=False):
        # run the process (in the background and close window if bg=True)
        # arguments are the executable and a list of arguments
        executable, args = 'kima-run', ['-t', str(threads), '--no-colors']
        if bg:
            args.append('--background')
            args.append('--quiet')
            self.process.startDetached(executable, args)
            self.close()
        else:
            self.process.start(executable, args)


def main(args=None, tests=False):
    if not tests:
        from .make_template import main as kima_template
        if args is None:
            args = _parse_args()
        if isinstance(args, str):
            Args = namedtuple('Args', 'version debug DIRECTORY')
            args = Args(version=False, debug=False, DIRECTORY=args)
        # print(args)

        if args.version:
            version_file = os.path.join(
                os.path.dirname(__file__), '../VERSION')
            print('kima', open(version_file).read().strip())  # same as kima
            sys.exit(0)

        dst = args.DIRECTORY
        kima_template(dst, stopIfNoReplace=False)
    else:
        dst = '.'

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(os.path.abspath(dst))
    main.show()
    app.aboutToQuit.connect(main.shutdown_kernel)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(tests=True)
