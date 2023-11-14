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

        # Batch predict
        self.batchIn_path = ""
        self.batchOut_path = ""

        # Extract Video
        self.srcVideo_path = ""
        self.outputFolder_path = ""

        # Create Video from Photos
        self.srcPhotos_path = ""
        self.outputVideo_path = ""

        self.progress_dialog = None

        self.initialise()

    def centerOnScreen (self):
        resolution = QDesktopWidget().screenGeometry()
        self.move((resolution.width() // 2) - (self.frameSize().width() // 2),
                  (resolution.height() // 2) - int(round(self.frameSize().height() / 1.67, 0)))

    def change_conf(self, i):
        if i == 1:
            self.uiMain.conf_Slider.setValue(self.uiMain.confValue_spinBox.value())
        elif i == 2:
            self.uiMain.conf_Slider_2.setValue(self.uiMain.confValue_spinBox_2.value())

    def change_iou(self, i):
        if i == 1:
            self.uiMain.IoU_Slider.setValue(self.uiMain.iouValue_spinBox.value())
        elif i == 2:
            self.uiMain.IoU_Slider_2.setValue(self.uiMain.iouValue_spinBox_2.value())

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
        self.uiMain.conf_Slider.valueChanged.connect(lambda: self.slide_conf(1))
        self.uiMain.IoU_Slider.valueChanged.connect(lambda: self.slide_iou(1))
        self.uiMain.conf_Slider.setValue(50)
        self.uiMain.IoU_Slider.setValue(70)
        self.uiMain.confValue_spinBox.valueChanged.connect(lambda: self.change_conf(1))
        self.uiMain.iouValue_spinBox.valueChanged.connect(lambda: self.change_iou(1))

        self.uiMain.conf_Slider_2.valueChanged.connect(lambda: self.slide_conf(2))
        self.uiMain.IoU_Slider_2.valueChanged.connect(lambda: self.slide_iou(2))
        self.uiMain.conf_Slider_2.setValue(50)
        self.uiMain.IoU_Slider_2.setValue(70)
        self.uiMain.confValue_spinBox_2.valueChanged.connect(lambda: self.change_conf(2))
        self.uiMain.iouValue_spinBox_2.valueChanged.connect(lambda: self.change_iou(2))

        # set up display photo
        self.set_image_from_path("assets/placeholder.jpg")

        # set up default checkbox
        self.uiMain.preprocess_checkBox.setChecked(True)
        self.uiMain.hideBoxes_checkBox.stateChanged.connect(lambda: self.update_hideCheckBoxes(1))
        self.uiMain.hideLabels_checkBox.stateChanged.connect(lambda: self.update_hideCheckBoxes(1))
        self.uiMain.preprocess_checkBox_2.setChecked(True)
        self.uiMain.hideBoxes_checkBox_2.stateChanged.connect(lambda: self.update_hideCheckBoxes(2))
        self.uiMain.hideLabels_checkBox_2.stateChanged.connect(lambda: self.update_hideCheckBoxes(2))

        self.uiMain.stackedWidget.setCurrentIndex(1) # photo mode

        # set up extract video page
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\assets/openfile.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.uiMain.openSrcVideo_Button.setIcon(icon)
        self.uiMain.selectOutputFolder_Button.setIcon(icon)
        self.uiMain.srcVideoPath_label.setText(self.srcVideo_path)
        self.uiMain.selectOutputFolder_label.setText(self.outputFolder_path)

        # set up create video page
        self.uiMain.openSrcPhotosPath_Button.setIcon(icon)
        self.uiMain.selectVidOutFolder_Button.setIcon(icon)
        self.uiMain.srcPhotosPath_label.setText(self.srcPhotos_path)
        self.uiMain.selectVidOutFolder_label.setText(self.outputVideo_path)
        regex = QtCore.QRegExp("[a-z-A-Z-0-9_]+")
        validator = QtGui.QRegExpValidator(regex)
        self.uiMain.videoName_lineEdit.setValidator(validator)

        # show image segment page first
        self.uiMain.tabWidget.setCurrentIndex(0)

    def get_conf(self): 
        if self.uiMain.tabWidget.currentIndex() == 0:
            return float(self.uiMain.confValue_spinBox.text()) / 100
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return float(self.uiMain.confValue_spinBox.text()) / 100

    def get_iou(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return float(self.uiMain.iouValue_spinBox.text()) / 100
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return float(self.uiMain.iouValue_spinBox_2.text()) / 100

    def get_preprocess(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return self.uiMain.preprocess_checkBox.isChecked()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return self.uiMain.preprocess_checkBox_2.isChecked()
    
    def get_preprocessVid(self):
        return self.uiMain.preprocessFrames_checkBox.isChecked()

    def get_resize(self):
        return self.uiMain.resize_checkBox.isChecked()
    
    def get_hideLabels(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return not self.uiMain.hideLabels_checkBox.isChecked()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return not self.uiMain.hideLabels_checkBox_2.isChecked()
    
    def get_hideConf(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return not self.uiMain.hideConf_checkBox.isChecked()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return not self.uiMain.hideConf_checkBox_2.isChecked()
    
    def get_hideBoxes(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return not self.uiMain.hideBoxes_checkBox.isChecked()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return not self.uiMain.hideBoxes_checkBox_2.isChecked()
    
    def get_imageSize(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return self.uiMain.imageSize_spinBox.value()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return self.uiMain.imageSize_spinBox_2.value()

    def get_maxDet(self):
        if self.uiMain.tabWidget.currentIndex() == 0:
            return self.uiMain.maxDet_spinBox.value()
        elif self.uiMain.tabWidget.currentIndex() == 1:
            return self.uiMain.maxDet_spinBox_2.value()
    
    def get_videoname(self):
        return self.uiMain.videoName_lineEdit.text()

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
    
    def select_batchIn_path(self):
        dir = self.batchIn_path if self.batchIn_path != "" else "/home"
        try:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', dir)
        except:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', '/home')
        
        if new_input_path == '':
            print("Load cancelled.")
            self.uiMain.batchSrc_label.setText(self.make_align(self.batchIn_path, 'right'))
            return False
        else: # if valid path chosen
            self.batchIn_path = new_input_path
            self.uiMain.batchSrc_label.setText(self.make_align(self.batchIn_path, 'right'))
            return True
        
    def select_batchOut_path(self):
        dir = self.batchOut_path if self.batchOut_path != "" else "/home"
        try:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', dir)
        except:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', '/home')
        
        if new_input_path == '':
            print("Load cancelled.")
            self.uiMain.batchOut_label.setText(self.make_align(self.batchOut_path, 'right'))
            return False
        else: # if valid path chosen
            self.batchOut_path = new_input_path
            self.uiMain.batchOut_label.setText(self.make_align(self.batchOut_path, 'right'))
            return True
        
    def select_input_folder(self):
        dir = self.srcPhotos_path if self.srcPhotos_path != "" else "/home"
        try:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', dir)
        except:
            new_input_path = QFileDialog.getExistingDirectory(self, 'Open File', '/home')
        
        if new_input_path == '':
            print("Load cancelled.")
            self.uiMain.srcPhotosPath_label.setText(self.make_align(self.srcPhotos_path, 'right'))
            return False
        else: # if valid path chosen
            self.srcPhotos_path = new_input_path
            self.uiMain.srcPhotosPath_label.setText(self.make_align(self.srcPhotos_path, 'right'))
            return True
        
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
        
    def select_outputvid_folder(self):
        dir = self.outputVideo_path if self.outputVideo_path != "" else "/home"
        try:
            new_output_path = QFileDialog.getExistingDirectory(self, 'Open File', dir)
        except:
            new_output_path = QFileDialog.getExistingDirectory(self, 'Open File', '/home')
        
        if new_output_path == '':
            print("Load cancelled.")
            self.uiMain.selectVidOutFolder_label.setText(self.make_align(self.outputVideo_path, 'right'))
            return False
        else: # if valid path chosen
            self.outputVideo_path = new_output_path
            self.uiMain.selectVidOutFolder_label.setText(self.make_align(self.outputVideo_path, 'right'))
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

    def slide_conf(self, i):
        if i == 1:
            self.uiMain.confValue_spinBox.setValue(self.uiMain.conf_Slider.value())
        elif i == 2:
            self.uiMain.confValue_spinBox_2.setValue(self.uiMain.conf_Slider_2.value())

    def slide_iou(self, i):
        if i == 1:
            self.uiMain.iouValue_spinBox.setValue(self.uiMain.IoU_Slider.value())
        elif i == 2:
            self.uiMain.iouValue_spinBox_2.setValue(self.uiMain.IoU_Slider_2.value())

    def update_hideCheckBoxes(self, i):
        if i == 1:
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

        elif i == 2:
            if self.uiMain.hideBoxes_checkBox_2.isChecked():
                # check and disable hide labels and conf
                self.uiMain.hideLabels_checkBox_2.setChecked(True)
                self.uiMain.hideConf_checkBox_2.setChecked(True)
                self.uiMain.hideLabels_checkBox_2.setEnabled(False)
                self.uiMain.hideConf_checkBox_2.setEnabled(False)

            elif not self.uiMain.hideBoxes_checkBox_2.isChecked() and not self.uiMain.hideLabels_checkBox_2.isEnabled() and not self.uiMain.hideConf_checkBox_2.isEnabled():
                # uncheck and enable the two hide checkboxes below
                self.uiMain.hideLabels_checkBox_2.setChecked(False)
                self.uiMain.hideConf_checkBox_2.setChecked(False)
                self.uiMain.hideLabels_checkBox_2.setEnabled(True)
                self.uiMain.hideConf_checkBox_2.setEnabled(True)

            elif self.uiMain.hideLabels_checkBox_2.isChecked() and self.uiMain.hideLabels_checkBox_2.isEnabled():
                # check and disable hide conf
                self.uiMain.hideConf_checkBox_2.setChecked(True)
                self.uiMain.hideConf_checkBox_2.setEnabled(False)

            else:
                # uncheck other hide checkboxes & enable
                self.uiMain.hideLabels_checkBox_2.setChecked(False)
                self.uiMain.hideConf_checkBox_2.setChecked(False)
                self.uiMain.hideLabels_checkBox_2.setEnabled(True)
                self.uiMain.hideConf_checkBox_2.setEnabled(True)
        

    def dialog_confirm_create(self, total_frames):
        response = QMessageBox.question(self, "Confirm Video Creation", 
                                        f"Video is to be created with {total_frames} images. This may require some time to process.\n\nAre you sure?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes: 
            return True
        else: 
            return False
        
    def dialog_confirm_extract(self, total_frames):
        response = QMessageBox.question(self, "Confirm Extraction", 
                                        f"{total_frames} images will be extracted. This may require a large amount of space and time to process.\n\nAre you sure?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes: 
            return True
        else: 
            return False
        
    def dialog_confirm_predict(self, total_frames):
        response = QMessageBox.question(self, "Confirm Prediction", 
                                        f"{total_frames} images will be predicted. This may require a large amount of space and time to process.\n\nAre you sure?",
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
        elif code == 2:
            error_text = "No input folder was selected.\n\nPlease choose a folder of images to create video from."
        elif code == 3:
            error_text = "No output folder was selected.\n\nPlease choose an output folder for video to be saved."
        elif code == 4:
            error_text = "Please enter a video name."
        elif code == 5:
            error_text = "No input folder was selected.\n\nPlease choose a folder of images to predict."
        elif code == 6:
            error_text = "No output folder was selected.\n\nPlease choose an output folder."
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
        else:
            print("self.progress_dialog is None!")
