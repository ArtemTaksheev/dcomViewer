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

test_datapath = "e://data/Шапка Богдана/23072418"

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
    treshold(image,1000)

    blank_image = image.copy()
    if show:
        cv.imshow("original",blank_image)
    grad_x = cv.Scharr(blank_image, cv.CV_16S, 1, 0,  scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y

    grad_y = cv.Scharr(blank_image, cv.CV_16S, 0, 1,  scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    if show:
        cv.imshow("original_grad", grad)

    
    erode_img = cv.dilate(grad,cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)
    erode_img = cv.erode(erode_img, cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)), iterations=1)

    erode_img = cv.dilate(erode_img,cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)), iterations=1)
    erode_img = cv.dilate(erode_img,cv.getStructuringElement(cv.MORPH_RECT,(5,5)), iterations=2)
    # erode_img = cv.dilate(erode_img,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)), iterations=2)
    erode_img = ~erode_img

    # hard mask for images (expected similar mri analysis)
    erode_img = cv.rectangle(erode_img, (0,100), (180,255), (0,0,0), -1)
    erode_img = cv.rectangle(erode_img, (0,255), (255,200), (0,0,0), -1)
    erode_img = cv.circle(erode_img, (150,100),50, (0,0,0), -1)


    for i in range(len(erode_img)):
                    for j in range(len(erode_img)):
                        if  erode_img[i][j] != 255:
                            erode_img[i][j] = 0
    cut = mask(grad, ~erode_img)
    if show:
        cv.imshow("erode", ~erode_img)
        cv.imshow("cut",cut)

    # cut = cv.erode(cut,cv.getStructuringElement(cv.MORPH_CROSS,(3,3)),iterations=1)

    
    thresh = cv.threshold(grad, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    thresh = cv.dilate(thresh, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)


    # cv.imshow('thresh', thresh)
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
        self.path = "e://data/Шапка Богдана/23072418"
        
        self.slices = self.provider.get_images(self.path)
        # print(self.slices)
        self.slicesSlider.setMaximum(len(self.slices) - 1)
        self.tresholdSlider.setMaximum(self.slices[0].max())


    def connectActions(self):
        self.slicesSlider.valueChanged.connect(self.updatePicture)
        self.tresholdSlider.valueChanged.connect(self.updatePicture)
        self.loadButton.clicked.connect(self.loadDcom)

    def loadDcom(self,path="e://data/Шапка Богдана/23072418"):
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

