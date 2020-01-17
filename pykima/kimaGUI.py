from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtCore import QProcess

import pyqtgraph as pg
import sys, os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pykima as pk

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

from hashlib import md5
from collections import namedtuple
import contextlib


@contextlib.contextmanager
def chdir(dir):
    curdir= os.getcwd()
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


class KimaModel:
    def __init__(self):
        self.directory = 'some dir'
        self.kima_setup = 'kima_setup.cpp'
        self.filename = 'no data file'
        self.skip = 0
        self.units = 'kms'

        self._levels_hash = ''
        self.obs_after_HARPS_fibers = False
        self.GP = False
        self.MA = False
        self.hyperpriors = False
        self.trend = False
        self.known_object = False

        self.fix_Np = True
        self.max_Np = 1

        self.priors = {
            # name: (default, distribution, arg1, arg2)
            'Cprior': [True, 'Uniform', 0.0, 0.0],
            'Jprior': [True, 'LogUniform', 0.0, 0.0],
            'slope_prior': [True, 'Uniform', 0.0, 0.0],
        }

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
        if self.filename == 'no data file':
            return False
        if len(self.filename) == 1:
            return False
        else:
            return True

    def results(self):
        # calculate the hash of the levels.txt file the first time
        # we create self.res to avoid recalculations
        h = md5(open('levels.txt', 'rb').read()).hexdigest()
        if h != self._levels_hash:
            self._levels_hash = h
            self.res = pk.showresults(force_return=True)

        self.res.return_figs = True
        plt.ioff()


    def save(self):
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

    def _write_settings(self, file):
        r = lambda val: 'true' if val else 'false'
        cb = 'const bool'
        file.write(f'{cb} obs_after_HARPS_fibers = {r(self.obs_after_HARPS_fibers)};\n')
        file.write(f'{cb} GP = {r(self.GP)};\n')
        file.write(f'{cb} MA = {r(self.MA)};\n')
        file.write(f'{cb} hyperpriors = {r(self.hyperpriors)};\n')
        file.write(f'{cb} trend = {r(self.trend)};\n')
        file.write(f'{cb} multi_instrument = {r(self.multi_instrument)};\n')
        file.write(f'{cb} known_object = {r(self.known_object)};\n')
        file.write('\n')

    def _write_constructor(self, file):
        r = lambda val: 'true' if val else 'false'
        file.write(f'RVmodel::RVmodel():fix({r(self.fix_Np)}),npmax({self.max_Np})\n')

    def _inside_constructor(self, file):
        file.write('{\n')

        for name, sets in self.priors.items():
            if not sets[0]: # if not default prior
                file.write(f'{name} = make_prior<{sets[1]}>({sets[2]}, {sets[3]});\n')

        file.write('\n}\n')
        file.write('\n')


    def _write_sampler(self, file):
        file.write('\tSampler<RVmodel> sampler = setup<RVmodel>(argc, argv);\n')
        file.write('\tsampler.run(50);\n')

    def _set_data(self, file):
        if self.multi_instrument:
            files = [f'"{datafile}"' for datafile in self.filename]
            files = ', '.join(files)
            file.write(f'\tdatafiles = {{ {files} }};\n')
            file.write(f'\tload_multi(datafiles, "{self.units}", {self.skip});\n')
        else:
            file.write(f'\tdatafile = "{self.filename[0]}";\n')
            file.write(f'\tload(datafile, "{self.units}", {self.skip});\n')

    def _start_main(self, file):
        file.write('int main(int argc, char** argv)\n')
        file.write('{\n')

    def _end_main(self, file):
        file.write('\treturn 0;\n')
        file.write('}\n')

    def _save_OPTIONS(self):
        opt = list(self.OPTIONS.values())
        with open('OPTIONS', 'w') as file:
            file.write('# File containing parameters for DNest4\n')
            file.write('# Put comments at the top, or at the end of the line.\n')
            file.write(f'{opt[0]}\t# Number of particles\n')
            file.write(f'{opt[1]}\t# new level interval\n')
            file.write(f'{opt[2]}\t# save interval\n')
            file.write(f'{opt[3]}\t# threadSteps: number of steps each thread does independently before communication\n')
            file.write(f'{opt[4]}\t# maximum number of levels\n')
            file.write(f'{opt[5]}\t# Backtracking scale length (lambda)\n')
            file.write(f'{opt[6]}\t# Strength of effect to force histogram to equal push (beta)\n')
            file.write(f'{opt[7]}\t# Maximum number of saves (0 = infinite)\n')
            file.write('    # (optional) samples file\n')
            file.write('    # (optional) sample_info file\n')
            file.write('    # (optional) levels file\n')




class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, directory, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # load the UI Page
        uifile = os.path.join(os.path.dirname(__file__), 'kimaGUI.ui')
        uic.loadUi(uifile, self)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.canvases = {}
        self.model = KimaModel()

        self.model.directory = directory
        os.chdir(directory)

        terminal_msg = "This is a Python terminal.\n" \
                       "numpy and pykima have been loaded as 'np' and 'pk'\n"

        self.terminal_widget.push({"np": np, "pk": pk})


        self.refreshAll()
        self.toggleTrend(self.trend_check.isChecked())
        self.progressBar.setValue(0)


    def statusMessage(self, msg):
        """ Print the message at the bottom of the window """
        self.statusBar.showMessage(msg)

    def refreshAll(self):
        """
        Updates the widgets whenever an interaction happens.
        Typically some interaction takes place, the UI responds,
        and informs the model of the change.  Then this method
        is called, pulling from the model information that is
        updated in the GUI.
        """
        self.lineEdit_2.setText(self.model.directory)
        if isinstance(self.model.filename, str):
            files = self.model.filename
        else:
            files = [os.path.basename(f) for f in self.model.filename]
            files = ', '.join(files)
        self.lineEdit.setText(files)

        # grey-out plot tabs
        self.tabWidget.setTabEnabled(3, self.model.GP)
        self.tabWidget.setTabEnabled(4, self.model.GP)

    def shutdown_kernel(self):
        """ Shutdown the embeded IPython kernel, but quietly """
        self.terminal_widget.kernel_client.stop_channels()
        self.terminal_widget.kernel_manager.shutdown_kernel()


    def addmpl(self, tab, fig):
        self.canvases[tab] = FigureCanvas(fig)
        getattr(self, tab+'_plot').addWidget(self.canvases[tab])
        self.canvases[tab].draw()

    def rmmpl(self, tab):
        getattr(self, tab+'_plot').removeWidget(self.canvases[tab])
        self.canvases[tab].close()

    def setmpl(self, tab, fig):
        if tab in self.canvases:
            self.rmmpl(tab)
        self.addmpl(tab, fig)


    def slot1(self):
        """ Called when the user presses the Browse button """
        self.statusMessage( "Browse button pressed" )
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
            self.model.filename = files
            self.refreshAll()

    def makePlot1(self):
        self.model.results()
        fig = self.model.res.make_plot1()
        self.setmpl('tab_1', fig)
    def makePlot2(self):
        self.model.results()
        fig = self.model.res.make_plot2()
        self.setmpl('tab_2', fig)
    def makePlot6(self):
        self.model.results()
        fig = self.model.res.plot_random_planets(show_vsys=True, show_trend=True)
        self.setmpl('tab_6', fig)

    def makePlotsAll(self):
        self.model.results()
        for p in range(1, 3):
            method_name = 'make_plot%d' % p
            method = getattr(self.model.res, method_name)
            fig = method()
            self.setmpl('tab_%d' % p, fig)

    def setDefaultPrior(self, *args):
        # get the name of the button which was pressed and from that which prior
        # is supposed to be made default
        button = self.sender()
        which = button.objectName().replace('makeDefault_', '')
        self.model.priors[which][0] = True
        getattr(self, f'lineEdit_{which}_arg1').setText('')
        getattr(self, f'lineEdit_{which}_arg2').setText('')

    def toggleTrend(self, toggled):
        self.label_slope_prior.setEnabled(toggled)
        self.comboBox_slope_prior.setEnabled(toggled)
        self.lineEdit_slope_prior_arg1.setEnabled(toggled)
        self.lineEdit_slope_prior_arg2.setEnabled(toggled)
        self.makeDefault_slope_prior.setEnabled(toggled)

    def saveModel(self):
        self.model.obs_after_HARPS_fibers = self.obs_after_HARPS_fibers_check.isChecked()
        self.model.GP = self.GP_check.isChecked()
        self.model.MA = self.MA_check.isChecked()
        self.model.hyperpriors = self.hyperpriors_check.isChecked()
        self.model.trend = self.trend_check.isChecked()
        self.model.known_object = self.known_object_check.isChecked()

        self.model.fix_Np = self.fix_Np_check.isChecked()
        self.model.max_Np = self.max_Np.value()

        self.model.skip = self.header_skip.value()
        self.model.units = {'km/s':'kms', 'm/s':'ms'}[self.units.currentText()]

        self.setOPTIONS()

        self.model.save()
        self.statusMessage('Saved model')


    def setOPTIONS(self):
        self.model.OPTIONS['new_level_interval'] = int(self.new_level_interval.value())
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
        self.saveModel()
        threads = self.threads.value()

        # Clear the output panel
        self.output.setText('')

        # QProcess object for external app
        self.process = QProcess(self)
        # QProcess emits `readyRead` when there is data to be read
        self.process.readyRead.connect(self.printProcessOutput)

        # To prevent accidentally running multiple times
        # disable the "Run" button when process starts, and enable it when it finishes
        self.process.started.connect(lambda: self.runButton.setEnabled(False))
        self.process.finished.connect(lambda: self.runButton.setEnabled(True))

        self.callKima(threads)


    def printProcessOutput(self):
        cursor = self.output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAll(), 'utf-8'))
        self.output.ensureCursorVisible()
        samples = sum(1 for i in open('sample.txt', 'rb')) - 1
        self.progressBar.setValue(100 * samples / 50)


    def callKima(self, threads):
        # run the process
        # `start` takes the exec and a list of arguments
        self.process.start('kima-run', ['-t', str(threads), '--timeout', '10', '--no-colors'])


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
            version_file = os.path.join(os.path.dirname(__file__), '../VERSION')
            print('kima', open(version_file).read().strip())  # same as kima
            sys.exit(0)

        dst = args.DIRECTORY
        kima_template(dst, stopIfNoReplace=True)
    else:
        dst = '.'

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow(os.path.abspath(dst))
    main.show()
    app.aboutToQuit.connect(main.shutdown_kernel)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(tests=True)
