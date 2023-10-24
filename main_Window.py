import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QFileDialog, QMessageBox, QTableWidgetItem
from PIL import Image
from main_UI import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.uiMain = Ui_MainWindow()
        self.uiMain.setupUi(self)
        self.initialise()
        self.resize(1200, 700)
        self.centerOnScreen()
        self.setMaximumSize(1200, 800)

        self.currentImage_path = None
        self.currentImage = None
        self.currentImgProcessed_path = 'assets/processed_image.jpg'
        self.currentImgProcessed = None
        self.predictedImage = None
        self.predict_data = None

    def centerOnScreen (self):
        resolution = QDesktopWidget().screenGeometry()
        self.move((resolution.width() // 2) - (self.frameSize().width() // 2),
                  (resolution.height() // 2) - int(round(self.frameSize().height() / 1.67, 0)))

    def clear_resultTable(self):
        self.uiMain.result_table.clear()

    def initialise(self):
        # set up sliders
        self.uiMain.conf_Slider.valueChanged.connect(self.slide_conf)
        self.uiMain.IoU_Slider.valueChanged.connect(self.slide_iou)
        self.uiMain.conf_Slider.setValue(50)
        self.uiMain.IoU_Slider.setValue(70)

        # set up display photo
        self.uiMain.photo_label.setPixmap(QtGui.QPixmap("assets/placeholder.jpg").scaled(800, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.uiMain.video_label.setPixmap(QtGui.QPixmap("assets/placeholder.jpg").scaled(800, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        # set up default checkbox
        self.uiMain.preprocess_checkBox.setChecked(True)
        self.uiMain.hideBoxes_checkBox.stateChanged.connect(self.update_hideBoxes)

        # default photo
        self.uiMain.stackedWidget.setCurrentIndex(1)

    def get_conf(self): 
        return float(self.uiMain.confValue_label.text()) / 100

    def get_iou(self):
        return float(self.uiMain.iouValue_label.text()) / 100

    def get_preprocess(self):
        return self.uiMain.preprocess_checkBox.isChecked()
    
    def get_hideLabels(self):
        return not self.uiMain.hideLabels_checkBox.isChecked()
    
    def get_hideConf(self):
        return not self.uiMain.hideConf_checkBox.isChecked()
    
    def get_hideBoxes(self):
        return not self.uiMain.hideBoxes_checkBox.isChecked()
    
    def get_imageSize(self):
        return self.uiMain.imageSize_spinBox.value()

    def get_maxDet(self):
        return self.uiMain.maxDet_spinBox.value()

    def make_bold(self, text):
        return f"<span style=\" font-weight:600;\">{text}</span>"
    
    def open_file(self):
        img_path, file_type = QFileDialog.getOpenFileName(self, 'Open File', '/home', 'Image Files (*.jpg *.jpeg *.png)')
        
        if img_path == '':
            print("Load cancelled.")
            return False
        
        else:
            # update display
            img = Image.open(img_path)
            width, height = img.size
            self.set_imageSizeInfo(width, height)
            self.set_image(img)

            self.currentImage_path = img_path
            self.currentImage = img

            self.clear_resultTable()
            return True
        
    def populate_resultTable(self, classes_ls, conf_ls):
        # display table results
        self.uiMain.result_table.setRowCount(len(classes_ls))
        self.uiMain.result_table.setColumnCount(2)
        self.uiMain.result_table.setColumnWidth(0, 150)
        self.uiMain.result_table.setColumnWidth(1, 80)
        self.uiMain.result_table.setHorizontalHeaderLabels(["class", "conf"])

        row = 0
        for cls, conf in zip(classes_ls, conf_ls):
            self.uiMain.result_table.setItem(row, 0, QTableWidgetItem(self.classes[cls]))
            self.uiMain.result_table.setItem(row, 1, QTableWidgetItem(str(round(conf, 3))))
            row += 1
    
    def save_file(self):
        # show warning if no predicted img to save
        if self.predictedImage == None:
            self.noImage_dialog(1)
            return False, ""
        
        # saves image
        image_name = self.currentImage_path.split('/')[-1].split('.')[0]
        save_location, save_type = QFileDialog.getSaveFileName(self, 'Save File', f'/home/{image_name}.jpg', 'Image Files (*.jpg *.jpeg *.png)')
        try:
            self.predictedImage.save(save_location)
            save_success = True
        except:
            print("Saving image cancelled.")
            save_success = False
        return save_success, save_location
    
    def save_table(self):
        # show warning if no predicted result to save
        if self.predictedImage == None:
            self.noImage_dialog(1)
            return False, ""
    
        image_name = self.currentImage_path.split('/')[-1].split('.')[0]
        save_location, save_type = QFileDialog.getSaveFileName(self, "Save Table", f'/home/{image_name}.csv', "CSV Files (*.csv)")
        try:
            columns = range(self.uiMain.result_table.columnCount())
            headers = [self.uiMain.result_table.horizontalHeaderItem(column).text() for column in columns]
            with open(save_location, 'w') as csvfile:
                writer = csv.writer(csvfile, dialect='excel', lineterminator='\n')
                writer.writerow(headers)
                for row in range(self.uiMain.result_table.rowCount()):
                    writer.writerow(self.uiMain.result_table.item(row, column).text() for column in columns)
            save_success = True
        except:
            print("Saving table cancelled.")
            save_success = False
        
        return save_success, save_location
    
    def set_image(self, image):
        image = image.convert("RGBA")
        data = image.tobytes("raw", "BGRA")
        q_image = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_ARGB32)
        self.uiMain.photo_label.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(800, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    
    def set_image_from_path(self, image_path):
        self.uiMain.photo_label.setPixmap(QtGui.QPixmap(image_path).scaled(800, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    
    def set_imageSizeInfo(self, width, height):
        self.uiMain.imageSizeInfo_label.setText(self.make_bold(f"{width} x {height} px"))

    def slide_conf(self):
        self.uiMain.confValue_label.setText(str(self.uiMain.conf_Slider.value()))

    def slide_iou(self):
        self.uiMain.iouValue_label.setText(str(self.uiMain.IoU_Slider.value()))

    def update_hideBoxes(self):
        if self.uiMain.hideBoxes_checkBox.isChecked():
            # check other hide checkboxes & disable
            self.uiMain.hideLabels_checkBox.setChecked(True)
            self.uiMain.hideConf_checkBox.setChecked(True)
            self.uiMain.hideLabels_checkBox.setEnabled(False)
            self.uiMain.hideConf_checkBox.setEnabled(False)
        else:
            # uncheck other hide checkboxes & enable
            self.uiMain.hideLabels_checkBox.setChecked(False)
            self.uiMain.hideConf_checkBox.setChecked(False)
            self.uiMain.hideLabels_checkBox.setEnabled(True)
            self.uiMain.hideConf_checkBox.setEnabled(True)

    def noImage_dialog(self, code):
        if code == 0:
            error_text = "No image uploaded.\n\nPlease upload an image for prediction."
        elif code == 1:
            error_text = "No image predicted.\n\nPlease upload an image for prediction."
        error_MsgBox = QMessageBox(self)
        error_MsgBox.setWindowTitle("ERROR: No Image")
        error_MsgBox.setIcon(QMessageBox.Warning)
        error_MsgBox.setText(error_text)
        error_MsgBox.setStandardButtons(QMessageBox.Ok)
        return_value = error_MsgBox.exec_()
        if return_value == QMessageBox.Ok:
            pass
        pass