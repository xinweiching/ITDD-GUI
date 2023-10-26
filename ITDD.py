import math
from main_Window import MainWindow
from PIL import Image
from ultralytics import YOLO
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from cv_functions import incr_contrast, Extractor
from time import sleep


class ITDD(MainWindow):
    def __init__(self, parent):
        super(ITDD, self).__init__(parent)

        # Buttons
        self.uiMain.extract_Button.clicked.connect(self.extractButton_clicked)
        self.uiMain.openFile_Button.clicked.connect(self.openFileButton_clicked)
        self.uiMain.openSrcVideo_Button.clicked.connect(self.openSrcVideoButton_clicked)
        self.uiMain.predict_Button.clicked.connect(self.predictButton_clicked)
        self.uiMain.selectOutputFolder_Button.clicked.connect(self.selectOutputFolder_clicked)
        self.uiMain.saveFile_Button.clicked.connect(self.saveFileButton_clicked)
        self.uiMain.saveTable_Button.clicked.connect(self.saveTableButton_clicked)
        self.uiMain.sortClass_Button.clicked.connect(self.sortClassButton_clicked)
        self.uiMain.sortConf_Button.clicked.connect(self.sortConfButton_clicked)
        self.uiMain.viewOriginal_Button.setCheckable(True)
        self.uiMain.viewOriginal_Button.clicked.connect(self.viewOGButton_clicked)

        # Prediction
        self.model_path = "models/ITDD_v1.pt"
        self.model = YOLO(self.model_path)
        self.classes = {
            0: 'asphalts', 
            1: 'interlocking-tiles', 
            2: 'manhole', 
            3: 'repair-tiles', 
            4: 'road marking -fade-', 
            5: 'road marking -good-'
            }

        self.show()

    #------------ BUTTON CLICKS ------------ #

    def extractButton_clicked(self):
        # extract video frames if input and output paths are not empty
        if self.srcVideo_path != "" and self.outputFolder_path != "":
            # extract_frames(video_path=self.srcVideo_path, output_folder=self.outputFolder_path, preprocess=self.get_preprocessVid(), resize_pred=self.get_resize())
            self.extract_video_frames()
        if self.srcVideo_path == "":
            self.dialog_missingPaths(0)
        if self.outputFolder_path == "":
            self.dialog_missingPaths(1)
        # self.uiMain.extract_Button.setEnabled(True)
    
    def openFileButton_clicked(self):
        self.uiMain.openFile_Button.setEnabled(False)

        success = self.open_imgfile()
        if success:
            self.uiMain.statusbar.showMessage("Load image successful.")
        else:
            self.uiMain.statusbar.showMessage("Load image cancelled.")

        self.uiMain.openFile_Button.setEnabled(True)
    
    def openSrcVideoButton_clicked(self):
        self.uiMain.openSrcVideo_Button.setEnabled(False)
        self.open_vidfile()
        self.uiMain.openSrcVideo_Button.setEnabled(True)
    
    def saveFileButton_clicked(self):
        self.uiMain.saveFile_Button.setEnabled(False)
        # saves image
        success, location = self.save_image()
        if success:
            self.uiMain.statusbar.showMessage(f"Saved image at {location}.")
        else:
            self.uiMain.statusbar.showMessage("Saving cancelled.")

        self.uiMain.saveFile_Button.setEnabled(True)
    
    def saveTableButton_clicked(self):
        self.uiMain.saveTable_Button.setEnabled(False)
        # saves table
        success, location = self.save_table()
        if success:
            self.uiMain.statusbar.showMessage(f"Saved table at {location}.")
        else:
            self.uiMain.statusbar.showMessage("Saving cancelled.")

        self.uiMain.saveTable_Button.setEnabled(True)
    
    def selectOutputFolder_clicked(self):
        self.uiMain.selectOutputFolder_Button.setEnabled(False)
        self.select_output_folder()
        self.uiMain.selectOutputFolder_Button.setEnabled(True)

    def sortClassButton_clicked(self):
        self.uiMain.sortClass_Button.setEnabled(False)

        self.uiMain.result_table.sortByColumn(0, QtCore.Qt.AscendingOrder)

        self.uiMain.sortClass_Button.setEnabled(True)
    
    def sortConfButton_clicked(self):
        self.uiMain.sortConf_Button.setEnabled(False)

        self.uiMain.result_table.sortByColumn(1, QtCore.Qt.DescendingOrder)

        self.uiMain.sortConf_Button.setEnabled(True)

    def viewOGButton_clicked(self):
        # to display original/predicted image when toggled
        if self.uiMain.viewOriginal_Button.isChecked() and self.currentImage != None:
            self.set_image(self.currentImage)
        else:
            if self.predictedImage != None:
                self.set_image(self.predictedImage)
    
    def predictButton_clicked(self):
        self.uiMain.predict_Button.setEnabled(False)
        sleep(0.2)
        self.uiMain.statusbar.showMessage("Predicting...")

        if self.currentImage_path != None:
            if self.get_preprocess():
                self.currentImgProcessed = incr_contrast(self.currentImage, clip_limit=5)
                self.currentImgProcessed.save(self.currentImgProcessed_path)
                result, classes_ls, conf_ls = self.predict_image(preprocess=True)
            else:
                result, classes_ls, conf_ls = self.predict_image()

            # display table results
            self.populate_resultTable(classes_ls, conf_ls)

            # set predicted image
            array = result.plot(conf=self.get_hideConf(),
                                boxes=self.get_hideBoxes(),
                                labels=self.get_hideLabels())
            array = array[:, :, ::-1]
            self.predictedImage = Image.fromarray(array)
            self.set_image(self.predictedImage)

        else:
            self.noImage_dialog(0)

        self.uiMain.statusbar.showMessage("Predictions done.")
        self.uiMain.predict_Button.setEnabled(True)

    # ------------- FUNCTIONS ------------- #
    
    def predict_image(self, preprocess=False):
        if preprocess: 
            path = self.currentImgProcessed_path
        else:
            path = self.currentImage_path

        results = self.model(source=path, 
                             conf=self.get_conf(), 
                             iou=self.get_iou(), 
                             imgsz=self.get_imageSize(),
                             max_det=self.get_maxDet())
                             
        for result in results:
            cls = result.boxes.cls
            conf = result.boxes.conf

        # get list of classes and conf and set them
        classes_ls = [int(i) for i in cls]
        conf_ls = [float(j) for j in conf]
        return result, classes_ls, conf_ls
    
    def extract_video_frames(self):
        self.thread = QThread()
        self.worker = Extractor(video_path=self.srcVideo_path, output_folder_path=self.outputFolder_path, preprocess=self.get_preprocessVid(), resize_pred=self.get_resize())
        self.worker.moveToThread(self.thread)

        self.total_frames_no = self.worker.total_frames_count
        # give warning for video with many frames 
        proceed = self.dialog_confirm_extract(self.total_frames_no)
        
        if proceed:
            self.dialog_progress_init() # init progress dialog
            # connect signals 
            self.thread.started.connect(self.dialog_progress_start)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.report_progress)
            self.progress_dialog.ui.cancelExtract_Button.clicked.connect(lambda: self.worker.stop_run(True))

            # start thread
            self.thread.start()
            self.uiMain.extract_Button.setEnabled(False)
            # self.dialog_progress_start()

            # when complete
            self.worker.finished.connect(self.report_extract_result)
            self.thread.finished.connect(lambda: self.uiMain.extract_Button.setEnabled(True))
            self.thread.finished.connect(self.progress_dialog.accept)
    
    def report_progress(self, count):
        self.progress_dialog.ui.extract_progressBar.setValue(int(math.ceil(count/self.total_frames_no * 100)))
        self.progress_dialog.ui.extractProgress_label.setText(f"Extracting frames ({count}/{self.total_frames_no + 1})...")

    def report_extract_result(self, code):
        if code == 1:
            self.uiMain.statusbar.showMessage(f"Extraction complete.")
        elif code == 2:
            self.uiMain.statusbar.showMessage(f"Extraction cancelled by user.")
        else:
            self.uiMain.statusbar.showMessage(f"Extraction incomplete/failed.")
