import cv2
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