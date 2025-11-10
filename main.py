# -*- coding: utf-8 -*-
# @Author  : LJQ
import os

from PyQt5 import QtWidgets
from widgets.mainwindow import MainWindow
import sys




if __name__ == '__main__':

    app = QtWidgets.QApplication([''])
    mainwindow = MainWindow()
    mainwindow.restore_window_state()
    mainwindow.show()
    app.aboutToQuit.connect(mainwindow.save_window_state)

    sys.exit(app.exec_())


