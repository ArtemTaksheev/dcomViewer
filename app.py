import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QFileDialog
from design import design

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import cv2 as cv
import pydicom

# test_datapath = "e://data/Шапка Богдана/23072418"
test_datapath = "E://data/Brain_marks_AEvit/23090311"

def mask(image,mask):
    result = image.copy()
    for i in range(len(image)):
                    for j in range(len(image)):
                        if  mask[i][j] != 0:
                            result[i][j] = 0
    return result

def get_coordinates(image,z_coord):
    result = []
    #  for i in range(len(image)):
    #                 for j in range(len(image)):
    #                     if  image[i][j] != 0:
    #                         result.append([i,j])

    contours, hierarchies = cv.findContours(
    image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(image.shape[:2],
                 dtype='uint8')
 
    cv.drawContours(blank, contours, -1,
                    (255, 0, 0), 1)
    
    # cv.imshow("Contours", blank)
    for i in contours:
        M = cv.moments(i)
        cx = None
        cy = None
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.drawContours(blank, [i], -1, (255, 255, 255), 2)
            cv.circle(blank, (cx, cy), 7, (255, 255, 255), -1)
            cv.putText(blank, "center", (cx - 20, cy - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv.imshow("Contours", blank)
            print(f"x: {cx} y: {cy} z: {z_coord}")
            result.append([cx,cy,z_coord])
    return result

def find_points(image,z_coord,show=True):
    # treshold(image,200)

    blank_image = image.copy()
    blank_image = cv.normalize(blank_image, blank_image, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    blank_image = cv.medianBlur(blank_image, 5)
    # if show:
        # cv.imshow("original",blank_image)

    _, img_thresh = cv.threshold(blank_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    kernel = np.ones((11, 11), np.uint8)
    img_open = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    img_open = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # img_open = cv.erode(img_open, cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations=1)
    cv.imshow("img_open",img_open)

    height = blank_image.shape[0]
    width = blank_image.shape[1]
    center = (int(height/2),int(width/2))
    # hard mask for images (expected similar mri analysis)
    img_open = cv.rectangle(img_open, (0,int(height/3)), (int(width/2),height), (255,255,255), -1)
    img_open = cv.rectangle(img_open, (0,height), (width,int(2*height/3)), (255,255,255), -1)
    img_open = cv.circle(img_open, center,int(height/4), (255,255,255), -1)
    cv.imshow("Hard_img_open",img_open)
    contours, _ = cv.findContours(img_open, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    
    if contours:
        cnt = max(contours, key=cv.contourArea)

        # Create mask
        my_mask = np.zeros_like(blank_image)
        cv.drawContours(my_mask, [cnt], -1, (255), thickness=cv.FILLED)
    
    # my_mask = cv.dilate(my_mask,cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5)), iterations=1)
    cv.imshow("mask",my_mask)
    # thresh = cv.dilate(thresh, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)

    # treshold(blank_image,7000)
    # blank_image = blank_image.astype("uint8")
    # # treshold(grad,4000)
    # cv.imshow("thresh_grad",blank_image)
    # cut = cut + blank_image
    cv.imshow('thresh', img_thresh)

    cut = mask(img_thresh, my_mask)
    cut = cv.erode(cut, cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)), iterations=1)
    cv.imshow("cut",cut)
    
    return cut

class DataProvider:

    def __init__(self, folder_path=None) -> None:
        if folder_path:
            self.folder_path = folder_path
        else:
            self.folder_path = test_datapath
        

    def get_images(self,path):
        slices = self.load_scan(path)
        return [s.pixel_array for s in slices]
    
    def load_scan(self,path):
        slices = [pydicom.dcmread(path + "/" + s) for s in os.listdir(path)]
        # slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices
    
def treshold(image,value):
    for i in range(len(image)):
                    # print(i)
                    for j in range(len(image)):
                        if  image[i][j] < value:
                            image[i][j] = 0

class Window(QWidget, design):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        self.path = ""
        self.provider = DataProvider()
        self.slices = []
        self.connectActions()
        
        self.coords = []
        self.path = test_datapath
        
        self.slices = self.provider.get_images(self.path)
        # print(self.slices)
        self.slicesSlider.setMaximum(len(self.slices) - 1)
        self.tresholdSlider.setMaximum(self.slices[0].max())


    def connectActions(self):
        self.slicesSlider.valueChanged.connect(self.updatePicture)
        self.tresholdSlider.valueChanged.connect(self.updatePicture)
        self.loadButton.clicked.connect(self.loadDcom)

    def loadDcom(self):
        fileName = QFileDialog.getExistingDirectory(self, 'Select Folder with DCOM')
        if fileName:
            self.path = fileName
            print(fileName)
        self.slices = self.provider.get_images(self.path)
        # print(self.slices)
        self.slicesSlider.setMaximum(len(self.slices) - 1)
        self.tresholdSlider.setMaximum(self.slices[0].max())
    
    def updatePicture(self):
        if self.slices:
            image = self.slices[self.slicesSlider.value()].copy()
            self.tresholdSlider.setMaximum(self.slices[self.slicesSlider.value()].max())
            # print(image)
            # print(len(image))
            treshold(image,self.tresholdSlider.value())
                # print(i)
            cut = find_points(image,self.slicesSlider.value())

            self.coords = self.coords + get_coordinates(cut,self.slicesSlider.value())
            # print(self.coords)
            self.plot(image)
            # print(image)

    def plot(self, image):
        self.figure.clear()
        plt.imshow(image, cmap=plt.cm.gray)

        self.canvas.draw()

def main():
    app = QApplication(sys.argv)

    window = Window()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

