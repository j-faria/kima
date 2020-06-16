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
import functools
from contextlib import redirect_stdout
from collections import namedtuple
from hashlib import md5

import numpy as np
import matplotlib.pyplot as plt
import pykima as pk
from .model import KimaModel

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

def no_exception(function):
    """ A decorator that wraps a function and ignores all exceptions """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # do nothing
            print(str(e))

    return wrapper



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

    # @no_exception
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
