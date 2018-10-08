from .classic import postprocess
from .display import KimaResults
from .showresults import showresults


## add Ctrl+C copy to matplotlib figures
import io
import matplotlib.pyplot as plt
replace_figure = True
try:
    from PySide.QtGui import QApplication, QImage
except ImportError:
    try:
        from PyQt4.QtGui import QApplication, QImage
    except ImportError:
        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QImage
        except ImportError:
            replace_figure = False
    

def add_clipboard_to_figures():
    # replace the original plt.figure() function with one that supports 
    # clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)
        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf)
                QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
                buf.close()
                print('Ctrl+C pressed: image is now in the clipboard')

        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig

    plt.figure = newfig

if replace_figure: add_clipboard_to_figures()