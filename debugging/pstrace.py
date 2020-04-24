import sys
import time
import argparse

import psutil                                                                                            

import numpy as np

from matplotlib import style, rc
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5

from PyQt5.QtWidgets import QGridLayout, QWidget, QDesktopWidget
from PyQt5.QtCore import QCoreApplication, Qt

if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure


class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self, pid, delay):

        self.process = psutil.Process(6704)
        self.stats = {
            'memory': [],
            'cpu_percent': []
        }
        self.ma = {k: [] for k in self.stats.keys()}

        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        layout = QtWidgets.QVBoxLayout(self._main)

        # Center window
        self.setFixedSize(800, 600)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        self.setWindowTitle(f"pstrace : pid={pid} ({self.process.name()})")

        dynamic_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(dynamic_canvas)
        # self.addToolBar(QtCore.Qt.BottomToolBarArea, NavigationToolbar(dynamic_canvas, self))

        self._dynamic_ax = dynamic_canvas.figure.subplots(2, sharex=True)
        self._timer = dynamic_canvas.new_timer(
            delay*1000, [(self._update_canvas, (), {})])
        self._figure = dynamic_canvas.figure
        self._timer.start()

    def _update_canvas(self):
        self.stats['memory'].append(self.process.memory_info()[0]/1e9)
        self.stats['cpu_percent'].append(self.process.cpu_percent())

        alpha = 0.1
        for k in self.stats.keys():
            if len(self.stats[k]) > 1:
                self.ma[k].append(alpha * self.stats[k][-1] + (1 - alpha) * self.ma[k][-1])
            else:
                self.ma[k].append(self.stats[k][-1])

        for i, (ax, stat) in enumerate(zip(self._dynamic_ax, self.stats.keys())):
            ax.clear()
            ax.plot(self.stats[stat], '.-')
            ax.plot(self.ma[stat], '-')
            max_val = max(self.stats[stat])
            ax.plot([0, len(self.stats[stat])], [max_val, max_val], ':')
            if i == len(self.stats) -1:
                ax.set_xlabel('samples')
            ax.set_ylabel(stat)
            ax.figure.canvas.draw()
        self._figure.tight_layout()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot changes in process stats over time')

    group = parser.add_argument_group('General')
    group.add_argument('-p', '--pid', dest='pid', action='store', type=int,
                        help='process ID')
    group.add_argument('-d', '--delay', dest='delay', action='store', type=float, default=0.5,
                        help='sampling delay')

    args = parser.parse_args()

    if args.pid is None:
        parser.usage()
        sys.exit(1)

    style.use('seaborn-pastel')
    rc('font', **{'family' : 'normal', 'size' : 8})

    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow(args.pid, args.delay)
    app.show()
    qapp.exec_()