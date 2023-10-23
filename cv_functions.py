import cv2
import os
import sys
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
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


########################################## EDGE DETECTORS ##########################################

def edge_detector_view(image_path, threshold_1=100, threshold_2=300):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply Canny edge detection
    edges = cv2.Canny(image=image_rgb, threshold1=threshold_1, threshold2=threshold_2)

    fig, axs = plt.subplots(1, 2, figsize=(7,4))
    axs[0].imshow(image_rgb)
    axs[0].set_title("Original Image")

    axs[1].imshow(edges)
    axs[1].set_title("Image edges")

    for ax in axs: 
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def edge_detector_save(image_path, output_folder=None, threshold_1=100, threshold_2=300):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    name = image_path.split('/')[-1].split('.')[0]  

    # apply Canny edge detection
    edges = cv2.Canny(image=image_rgb, threshold1=threshold_1, threshold2=threshold_2)

    if output_folder == None:
        cv2.imwrite(f'output/{name}_edges.jpg', edges)
    else:
        if output_folder[-1] != '/':
            cv2.imwrite(output_folder + f'/{name}_edges.jpg', edges)
        else:
            cv2.imwrite(output_folder + f'{name}_edges.jpg', edges)

########################################## GETTERS ##########################################

def get_image_dimensions(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    return height, width

def get_d32(old_height, old_width, new_height=None, new_width=None):
    if new_width != None and new_height == None:
        new_w = new_width
        new_h = int(round((old_height / old_width) * new_width, 0))
    else:
        print("Write this code pls.")
        return 
    print(f'New dimensions in multiple of 32 is ({new_w}, {new_h}).')

########################################## RESIZE ##########################################

def resize_save(image_path, output_folder, new_width, new_height):
    image = cv2.imread(image_path)
    new_image = cv2.resize(image, (new_width, new_height))
    name = image_path.split('/')[-1].split('.')[0]
    if output_folder[-1] != '/':
        cv2.imwrite(output_folder + f'/{name}.jpg', new_image)
    else:
        cv2.imwrite(output_folder + f'{name}.jpg', new_image)

def resize_byFactor_save(image_path, output_folder=None, fx=0.5, fy=0.5):
    image = cv2.imread(image_path)
    new_image = cv2.resize(image, (0,0), fx=fx, fy=fy)
    name = image_path.split('/')[-1].split('.')[0]  
    if output_folder == None:
        cv2.imwrite(output_folder + f"/{name}_scaled.jpg", new_image)
    else:
        if output_folder[-1] != '/':
            cv2.imwrite(output_folder + f'/{name}_scaled.jpg', new_image)
        else:
            cv2.imwrite(output_folder + f'{name}_scaled.jpg', new_image)

def resize_byFactor_all(input_folder, output_folder=None, fx=1, fy=1):
    for image in os.listdir(input_folder):
        resize_byFactor_save(input_folder + f"/{image}", output_folder, 0.4, 0.4)
        print(image)

########################################## SPLIT ##########################################

def split_image_in_4(image_path, output_folder):
    image = cv2.imread(image_path)
    old_h, old_w = get_image_dimensions(image_path)
    new_name = '_'.join(image_path.split('/')[-1].split('.')[0].split('_')[:-1])

    crop_1 = image[:old_h//2, :old_w//2]
    crop_2 = image[:old_h//2, old_w//2:]
    crop_3 = image[old_h//2:, :old_w//2]
    crop_4 = image[old_h//2:, old_w//2:]

    if output_folder[-1] != '/':
        cv2.imwrite(output_folder + f'/{new_name}_1.jpg', crop_1)
        cv2.imwrite(output_folder + f'/{new_name}_2.jpg', crop_2)
        cv2.imwrite(output_folder + f'/{new_name}_3.jpg', crop_3)
        cv2.imwrite(output_folder + f'/{new_name}_4.jpg', crop_4)
    else:
        cv2.imwrite(output_folder + f'{new_name}_1.jpg', crop_1)
        cv2.imwrite(output_folder + f'{new_name}_2.jpg', crop_2)
        cv2.imwrite(output_folder + f'{new_name}_3.jpg', crop_3)
        cv2.imwrite(output_folder + f'{new_name}_4.jpg', crop_4)


########################################## VIDEOS ##########################################

def preprocess_frames(video_path, output_folder, frame_count=-1):
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()

        try:
            # print(f"count:{count} , success:{success}")

            # preprocess
            enhanced_img = incr_contrast(image_path=image, clip_limit=5, save=False)
            sized_img = cv2.resize(enhanced_img, (800, 800))

            # save in output folder
            cv2.imwrite(f"{output_folder}/frame{count_to_string(count)}.jpg", sized_img)
            count += 1
        except:
            success = 1
            count += 1

        if frame_count != -1 and count >= frame_count:
            break

def count_to_string(count):
    l = 8
    return "0"*(l-len(str(count))) + str(count)