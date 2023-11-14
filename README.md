# ITDD-GUI

## Overview
GUI for Interlocking Tiles Defect Detection. Made with PyQt5.
- Displays image segmentation prediction from a pre-trained model.
    - Detects the following 6 classes:
        - Interlocking tiles, Repair Tiles, Asphalt
        - Manhole, Road paint (fade), Road paint (good)
    - Trained with YOLOv8, 1498 images, imgsz=800
### Other Features
- Performs predictions on all images in folder
- Extract frames from video w/ preprocess and resizing options
- Create video from images
 
 <br>
 
![Screenshot 2023-11-14 121929](https://github.com/pseuds/ITDD-GUI/assets/112696906/3c73cc68-4abd-4e1a-9ba0-bc8a7e00448c)

### Installation
Git clone this repo and run:
```
pip install -r requirements.txt
```

### Running the program
To run, execute in terminal:
```
python3.11 main.py
```
