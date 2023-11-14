import cv2, os
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from copy import deepcopy

def incr_contrast(img, clip_limit=2):
    # input is PIL img
    cv_img = np.array(img)[:, :, ::-1].copy()

    # converting to LAB color space
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert result to PIL img
    color_converted = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB) 
    result_image = Image.fromarray(color_converted) 

    return result_image

def incr_contrast_cv(cv_img, clip_limit=2):
    # input is cv img

    # converting to LAB color space
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img

class Extractor(QObject):
    finished = pyqtSignal(int)
    progress = pyqtSignal(int)
    def __init__(self, video_path, output_folder_path, preprocess=False, resize_pred=False, frame_count = -1):
        super().__init__()
        self.count = 0
        self.video_path = video_path
        self.output_path = output_folder_path
        self.preprocess_bool = preprocess
        self.resize_bool = resize_pred
        self.frame_count = -1

        # count the number of frames
        vidObj = cv2.VideoCapture(self.video_path)
        self.fps = vidObj.get(cv2.CAP_PROP_FPS)
        self.total_frames_count = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        self.durationInSeconds = self.total_frames_count // self.fps
        vidObj.release()

    def run(self):
        self.run_bool = True
        vidObj = cv2.VideoCapture(self.video_path)
        if not vidObj.isOpened():
            print("Error: Could not open file:", self.video_path)
            self.finished.emit(0)
        
        image_name = self.video_path.split('/')[-1].split('.')[0]
        success = 1
        too_large = None
        read_fail_limit = 4096
        read_fail_count = 0

        while self.run_bool:
            read_success, image = vidObj.read()
            if not read_success:
                read_fail_count += 1
                if read_fail_count > read_fail_limit: 
                    self.finished.emit(0)
                    break

            try:
                # preprocess
                if self.preprocess_bool:
                    enhanced_img = incr_contrast_cv(cv_img=image, clip_limit=5)
                else:
                    enhanced_img = image
                
                # check if too large if have not
                if too_large == None:
                    h, w = enhanced_img.shape[0], enhanced_img.shape[1] # (h, w, c)
                    longer = h if h>w else w
                    shorter = h if h<w else w
                    too_large = True if longer > 1280 or shorter > 960 else False

                # resize if checked and larger than needed
                if self.resize_bool and too_large:
                    dim = (1280, 960) if longer == w else (960, 1280) # (w, h)
                    sized_img = cv2.resize(enhanced_img, dim) 
                else:
                    sized_img = enhanced_img

                # save in output folder
                cv2.imwrite(f"{self.output_path}/{image_name}_frame{self.count_to_string()}.jpg", sized_img)
                self.count += 1
                self.progress.emit(self.count)

            except Exception as error: # skip frames that has unidentified error
                print("An exception occurred:", error)
                self.count += 1


            if self.count > self.total_frames_count:
                break
        
        vidObj.release()
        if not self.run_bool: # if cancelled by user
            self.finished.emit(2)
        else: # else it is complete
            self.finished.emit(1)

    def count_to_string(self):
        l = 8
        return "0"*(l-len(str(self.count))) + str(self.count)
    
    def stop_run(self, bool):
        self.run_bool = not bool

class VideoCreator(QObject):
    finished = pyqtSignal(int)
    progress = pyqtSignal(int)
    def __init__(self, input_folder_path, output_folder_path, video_name):
        super().__init__()
        self.count = 0
        self.input_path = input_folder_path
        self.output_path = output_folder_path
        self.video_name = video_name

        self.supported_formats = ['.jpg', '.bmp', '.jpeg']
        self.current_format = ''
        self.image_count = len(os.listdir(self.input_path))

    def get_file_format(self, file_name):
        return '.' + file_name.split('.')[-1].lower()
    
    def get_img_size(self, file_name):
        im = cv2.imread(f"{self.input_path}/{file_name}")
        return im.shape

    def run(self):
        self.run_bool = True
        # check folder contents
        image_list = os.listdir(self.input_path)
        self.current_format = self.get_file_format(image_list[0])
        self.current_shape = self.get_img_size(image_list[0])

        for img in image_list:
            # if current img format is not supported
            if not self.get_file_format(img) in self.supported_formats:
                print(f"Invalid file format found in source folder: {self.input_path}/{img}.")
                self.finished.emit(4)
            # if current img format is not same as others
            elif self.get_file_format(img) != self.current_format:
                print(f"Different file format found in source folder: {self.input_path}/{img}.")
                self.finished.emit(5)
            # if current img size is not same as others
            elif self.get_img_size(img) != self.current_shape:
                print(f"Different image size found in source folder: {self.input_path}/{img}.")
                self.finished.emit(6)
        
        # write into video if passes above check
        try:
            video = cv2.VideoWriter(f"{self.output_path}/{self.video_name}.mp4", 0, 30, (800, 800)) 
            self.count = 0
            while self.run_bool:
                video.write(cv2.imread(self.input_path+f"/{image_list[self.count]}"))
                self.count += 1
                self.progress.emit(self.count)

                if self.count >= self.image_count: 
                    break

            video.release()

        except Exception as error:
            print("An exception occurred:", error)
            self.finished.emit(3)

        if not self.run_bool: # if cancelled by user
            self.finished.emit(2)
        else: # else it is complete
            self.finished.emit(1)
    
    def stop_run(self, bool):
        self.run_bool = not bool

# TODO
class BatchPredictor(QObject):
    finished = pyqtSignal(int)
    progress = pyqtSignal(int)
    def __init__(self, input_folder_path, output_folder_path, 
                 model, conf, iou, imgsz, max_det, preprocess, 
                 hide_conf, hide_boxes, hide_labels):
        super().__init__()
        self.count = 0
        self.input_path = input_folder_path
        self.output_path = output_folder_path
        
        # prediction params
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.preprocess = preprocess

        # plotting params
        self.hide_conf = hide_conf
        self.hide_boxes = hide_boxes
        self.hide_labels = hide_labels

        self.supported_formats = ['.jpg', '.png', '.jpeg']
        self.currentProcessedPath = 'assets/processed_image_batch.jpg'
        self.valid_images_ls = self.get_valid_images()
        self.image_count = len(self.valid_images_ls)

    def get_file_format(self, file_name):
        return '.' + file_name.split('.')[-1].lower()
    
    def get_img_size(self, file_name):
        im = cv2.imread(f"{self.input_path}/{file_name}")
        return im.shape
    
    def get_valid_images(self):
        valid_images = []
        # check folder contents
        image_list = os.listdir(self.input_path)
        # go through every image in folder
        for image_name in image_list:
            if self.get_file_format(image_name) in self.supported_formats:
                valid_images.append(image_name)
        return valid_images
    
    def predict(self):
        if self.preprocess: 
            path = self.currentProcessedPath
        else:
            path = self.currentImagePath

        results = self.model(source=path, 
                             conf=self.conf, 
                             iou=self.iou, 
                             imgsz=self.imgsz,
                             max_det=self.max_det)
                             
        for result in results:
            cls = result.boxes.cls
            conf = result.boxes.conf

        # get list of classes and conf and set them
        classes_ls = [int(i) for i in cls]
        conf_ls = [float(j) for j in conf]
        return result, classes_ls, conf_ls

    def run(self):
        self.run_bool = True
        # go through every image 
        for image_name in self.valid_images_ls:
            try:
                print("hello.")
                self.currentImagePath = f"{self.input_path}/{image_name}"
                img = Image.open(self.currentImagePath)
                self.currentImage = img
                print("image opened.")
                self.currentImageProcessed = incr_contrast(self.currentImage, clip_limit=5)
                self.currentImageProcessed.save(self.currentProcessedPath)
                result, classes_ls, conf_ls = self.predict()
                print("about to plot.")

                # plot predictions on image
                array = result.plot(conf=self.hide_conf,
                                boxes=self.hide_boxes,
                                labels=self.hide_labels)
                array = array[:, :, ::-1]
                self.predictedImage = Image.fromarray(array)

                # save predicted image 
                self.predictedImage.save(f"{self.output_path}/{image_name}")
            except Exception as e:
                print(e)
                print(f"Exception occurred when processing image {image_name}.")

            self.count += 1
            self.progress.emit(self.count)

        if not self.run_bool: # if cancelled by user
            self.finished.emit(2)
        else: # else it is complete
            self.finished.emit(1)
    
    def stop_run(self, bool):
        self.run_bool = not bool