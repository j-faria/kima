from PyQt5 import QtWidgets
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont

kima = "<strong>kima</strong>"

font = QFont()
font.setFamily("Arial")
font.setPointSize(12)

text_gui_help = f"""
<p>
    Most options in the GUI correspond directly 
    to those you would write in a {kima} setup file.
<br>
    Also, many components of the GUI will show a small help text on mouse over
    for more detailed information.
</p>

<br>

<p>
    The <i>Data</i> section defines the RV data file(s).
    If multiple files are chosen, containing data from different instruments,
    this will trigger the multi_instrument mode in {kima}.
</p>

<br>

<p>
    The number of planets can be fixed to the value set in 'max'
    or free with a uniform prior between 0 and 'max'.
</p>

<br>

<p>
    In the <i>Priors</i> section, clicking the "D" buttons
    will assign default priors for that parameter.
    Other distributions may also be chosen, with different parameters.
</p>

<br>

<p>
    The model can be started and stopped at any time
    (any new changes will be saved),
    with information displayed in the "output" tab. 
    The results and plots can be obtained 
    after or <i>while</i> running the model,
    and will be shown in the different numbered tabs.
</p>

"""
def show_gui_help_window():
    win = QtWidgets.QMessageBox()
    win.setTextFormat(Qt.RichText)
    win.setFont(font)
    win.setIcon(QtWidgets.QMessageBox.Question)
    win.setText(text_gui_help)
    win.setWindowTitle('Help')
    win.exec_()


text_kima_help = f"""
Detailed documentation for {kima} can be found
<a href="https://github.com/j-faria/kima/wiki">here</a>. <br>
"""

def show_kima_help_window():
    win = QtWidgets.QMessageBox()
    win.setTextFormat(Qt.RichText)
    win.setFont(font)
    win.setIcon(QtWidgets.QMessageBox.Question)
    win.setText(text_kima_help)
    win.setWindowTitle('Help')
    win.exec_()


text_about = """
This GUI provides a front-end to <strong>kima</strong> <br>
for easy analysis of radial velocity data.

<br><br>
It was created with ❤️ by João Faria (joao.faria@astro.up.pt). <br>
Please report any issues 
<a href="https://github.com/j-faria/kima/issues">here</a>. <br>
"""

def show_about_window():
    win = QtWidgets.QMessageBox()
    win.setTextFormat(Qt.RichText)
    win.setFont(font)
    # win.setIcon(QtWidgets.QMessageBox.Critical)
    win.setText(text_about)
    win.setWindowTitle('About')
    win.exec_()



