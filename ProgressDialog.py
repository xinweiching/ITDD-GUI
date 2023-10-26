from ProgressDialog_UI import Ui_Dialog
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QDialog


class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super(ProgressDialog, self).__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)