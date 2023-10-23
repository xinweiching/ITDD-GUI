import sys
from PyQt5 import QtWidgets
from main_Window import MainWindow
from ITDD import ITDD

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    itdd = ITDD(main_window)
    sys.exit(app.exec_())