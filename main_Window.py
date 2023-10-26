import csv
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QFileDialog, QMessageBox, QTableWidgetItem
from PIL import Image
from main_UI import Ui_MainWindow
from ProgressDialog import ProgressDialog


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.uiMain = Ui_MainWindow()
        self.uiMain.setupUi(self)
        self.resize(1400, 800)
        self.centerOnScreen()
        self.setMaximumSize(1600, 800)

        # Image data
        self.currentImage_path = None
        self.currentImage = None
        self.currentImgProcessed_path = 'assets/processed_image.jpg'
        self.currentImgProcessed = None
        self.predictedImage = None
        self.predict_data = None

        # Extract Video
        self.srcVideo_path = ""
        self.outputFolder_path = ""
        self.progress_dialog = None

        self.initialise()

    def centerOnScreen (self):
        resolution = QDesktopWidget().screenGeometry()
        self.move((resolution.width() // 2) - (self.frameSize().width() // 2),
                  (resolution.height() // 2) - int(round(self.frameSize().height() / 1.67, 0)))

    def change_conf(self):
        self.uiMain.conf_Slider.setValue(self.uiMain.confValue_spinBox.value())
    
    def change_iou(self):
        self.uiMain.IoU_Slider.setValue(self.uiMain.iouValue_spinBox.value())

    def change_model(self):
        # checks it back if unchecked
        if not self.uiMain.modelV1_radioButton.isChecked():
            self.uiMain.modelV1_radioButton.setChecked(True)

    def clear_resultTable(self):
        self.uiMain.result_table.clear()

    def initialise(self):
        # model select (just one for now lol)
        self.uiMain.modelV1_radioButton.toggled.connect(self.change_model)

        # set up sliders
        self.uiMain.conf_Slider.valueChanged.connect(self.slide_conf)
        self.uiMain.IoU_Slider.valueChanged.connect(self.slide_iou)
        self.uiMain.conf_Slider.setValue(50)
        self.uiMain.IoU_Slider.setValue(70)
        self.uiMain.confValue_spinBox.valueChanged.connect(self.change_conf)
        self.uiMain.iouValue_spinBox.valueChanged.connect(self.change_iou)

        # set up display photo
        self.set_image_from_path("assets/placeholder.jpg")

        # set up default checkbox
        self.uiMain.preprocess_checkBox.setChecked(True)
        self.uiMain.hideBoxes_checkBox.stateChanged.connect(self.update_hideCheckBoxes)
        self.uiMain.hideLabels_checkBox.stateChanged.connect(self.update_hideCheckBoxes)

        self.uiMain.tabWidget.setCurrentIndex(0) # image segmentation page
        self.uiMain.stackedWidget.setCurrentIndex(1) # photo mode

        # set up extract video page
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\assets/openfile.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.uiMain.openSrcVideo_Button.setIcon(icon)
        self.uiMain.selectOutputFolder_Button.setIcon(icon)
        self.uiMain.srcVideoPath_label.setText(self.srcVideo_path)
        self.uiMain.selectOutputFolder_label.setText(self.outputFolder_path)

    def get_conf(self): 
        return float(self.uiMain.confValue_spinBox.text()) / 100

    def get_iou(self):
        return float(self.uiMain.iouValue_spinBox.text()) / 100

    def get_preprocess(self):
        return self.uiMain.preprocess_checkBox.isChecked()
    
    def get_preprocessVid(self):
        return self.uiMain.preprocessFrames_checkBox.isChecked()

    def get_resize(self):
        return self.uiMain.resize_checkBox.isChecked()
    
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
    
    def make_align(self, text, align=None):
        if align in ['left', 'right', 'center']:
            return f"<html><head/><body><p align=\"{align}\"><span>" + text + "</span></p></body></html>"
    
    def open_imgfile(self):
        dir = self.currentImage_path if self.currentImage_path != None else "/home"
        try:
            img_path, file_type = QFileDialog.getOpenFileName(self, 'Open File', dir, 'Image Files (*.jpg *.jpeg *.png)')
        except:
            img_path, file_type = QFileDialog.getOpenFileName(self, 'Open File', "/home", 'Image Files (*.jpg *.jpeg *.png)')

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
            
            # delete prev input results
            self.currentImgProcessed = None
            self.predictedImage = None
            self.predict_data = None

            self.clear_resultTable()
            return True
    
    def open_vidfile(self):
        dir = self.srcVideo_path if self.srcVideo_path != "" else "/home"
        try:
            new_vid_path, file_type = QFileDialog.getOpenFileName(self, 'Open File', dir, 'Video Files (*.mp4 *.avi)')
        except:
            new_vid_path, file_type = QFileDialog.getOpenFileName(self, 'Open File', "/home", 'Video Files (*.mp4 *.avi)')

        if new_vid_path == '': # if no path chosen
            print("Load cancelled.")
            self.uiMain.srcVideoPath_label.setText(self.make_align(self.srcVideo_path, 'right'))
            return False
        
        else: # if valid path chosen
            self.srcVideo_path = new_vid_path
            self.uiMain.srcVideoPath_label.setText(self.make_align(self.srcVideo_path, 'right'))
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
    
    def save_image(self):
        # show warning if no predicted img to save
        if self.predictedImage == None:
            self.dialog_noImage(1)
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
            self.dialog_noImage(1)
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
    
    def select_output_folder(self):
        dir = self.outputFolder_path if self.outputFolder_path != "" else "/home"
        try:
            new_output_path = QFileDialog.getExistingDirectory(self, 'Open File', dir)
        except:
            new_output_path = QFileDialog.getExistingDirectory(self, 'Open File', '/home')
        
        if new_output_path == '':
            print("Load cancelled.")
            self.uiMain.selectOutputFolder_label.setText(self.make_align(self.outputFolder_path, 'right'))
            return False
        else: # if valid path chosen
            self.outputFolder_path = new_output_path
            self.uiMain.selectOutputFolder_label.setText(self.make_align(self.outputFolder_path, 'right'))
            return True

    def set_image(self, image):
        image = image.convert("RGBA")
        data = image.tobytes("raw", "BGRA")
        q_image = QtGui.QImage(data, image.size[0], image.size[1], QtGui.QImage.Format_ARGB32)
        self.uiMain.photo_label.setPixmap(QtGui.QPixmap.fromImage(q_image).scaled(1200, 600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    
    def set_image_from_path(self, image_path):
        self.uiMain.photo_label.setPixmap(QtGui.QPixmap(image_path).scaled(1200, 600, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    
    def set_imageSizeInfo(self, width, height):
        self.uiMain.imageSizeInfo_label.setText(self.make_bold(f"{width} x {height} px"))

    def slide_conf(self):
        self.uiMain.confValue_spinBox.setValue(self.uiMain.conf_Slider.value())

    def slide_iou(self):
        self.uiMain.iouValue_spinBox.setValue(self.uiMain.IoU_Slider.value())

    def update_hideCheckBoxes(self):

        if self.uiMain.hideBoxes_checkBox.isChecked():
            # check and disable hide labels and conf
            self.uiMain.hideLabels_checkBox.setChecked(True)
            self.uiMain.hideConf_checkBox.setChecked(True)
            self.uiMain.hideLabels_checkBox.setEnabled(False)
            self.uiMain.hideConf_checkBox.setEnabled(False)

        elif not self.uiMain.hideBoxes_checkBox.isChecked() and not self.uiMain.hideLabels_checkBox.isEnabled() and not self.uiMain.hideConf_checkBox.isEnabled():
            # uncheck and enable the two hide checkboxes below
            self.uiMain.hideLabels_checkBox.setChecked(False)
            self.uiMain.hideConf_checkBox.setChecked(False)
            self.uiMain.hideLabels_checkBox.setEnabled(True)
            self.uiMain.hideConf_checkBox.setEnabled(True)

        elif self.uiMain.hideLabels_checkBox.isChecked() and self.uiMain.hideLabels_checkBox.isEnabled():
            # check and disable hide conf
            self.uiMain.hideConf_checkBox.setChecked(True)
            self.uiMain.hideConf_checkBox.setEnabled(False)

        else:
            # uncheck other hide checkboxes & enable
            self.uiMain.hideLabels_checkBox.setChecked(False)
            self.uiMain.hideConf_checkBox.setChecked(False)
            self.uiMain.hideLabels_checkBox.setEnabled(True)
            self.uiMain.hideConf_checkBox.setEnabled(True)
        

    def dialog_confirm_extract(self, total_frames):
        response = QMessageBox.question(self, "Confirm Extraction", 
                                        f"{total_frames} images will be extracted. This may require a large amount of space and time to process.\n\nAre you sure?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes: 
            return True
        else: 
            return False

    def dialog_missingPaths(self, code):
        if code == 0:
            error_text = "No video was selected.\n\nPlease choose a video for extraction."
        elif code == 1:
            error_text = "No output folder was selected.\n\nPlease choose an output folder for frames to be extracted to."
        error_MsgBox = QMessageBox(self)
        error_MsgBox.setWindowTitle("ERROR: Missing Paths")
        error_MsgBox.setIcon(QMessageBox.Warning)
        error_MsgBox.setText(error_text)
        error_MsgBox.setStandardButtons(QMessageBox.Ok)
        return_value = error_MsgBox.exec_()
        if return_value == QMessageBox.Ok:
            pass
    
    def dialog_noImage(self, code):
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
    
    def dialog_progress_init(self):
        self.progress_dialog = ProgressDialog(self)

    def dialog_progress_start(self):
        if self.progress_dialog != None:
            self.progress_dialog.ui.extract_progressBar.setValue(0)
            self.progress_dialog.show()
            print("self.progress_dialog is executed.")
        else:
            print("self.progress_dialog is None!")
