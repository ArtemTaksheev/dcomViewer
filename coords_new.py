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
 
    # cv.drawContours(blank, contours, -1,
    #                 (255, 0, 0), 1)
    
    # cv.imshow("Contours", blank)
    for i in contours:
        M = cv.moments(i)
        cx = None
        cy = None
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # cv.drawContours(blank, [i], -1, (255, 255, 255), 2)
            # cv.circle(blank, (cx, cy), 7, (255, 255, 255), -1)
            # cv.putText(blank, "center", (cx - 20, cy - 20),
            #         cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # cv.imshow("Contours", blank)
        # print(f"x: {cx} y: {cy} z: {z_coord}")
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

class Points():
    def __init__(self, parent=None) -> None:
        self.path = ""
        self.provider = DataProvider()
        self.slices = []
        
        self.coords = []
        self.path = test_datapath
        
        # self.slices = self.provider.get_images(self.path)
        self.slices = self.provider.load_scan(self.path)
        # print(self.slices)
        self.slicesMax = (len(self.slices) - 1)
        # self.tresholdSlider.setMaximum(self.slices[0].max())


def group_coords(data,maxgap):
    data.sort(key=lambda x:x[2])
    # print(data)
    data.sort(key=lambda x:x[1])
    # print(data)
    groups = [[data[0]]]
    for x in data[1:]:
        # print(x)
        # print(np.array(x) - np.array(groups[-1]))
        # print("group[-1]",groups[-1])
        # print(abs(np.array(x) - np.array(groups[-1]))[0])
        condition = abs(np.array(x) - np.array(groups[-1]))
        # print(condition[0])
        if (condition[0][0] <= maxgap) and (condition[0][1] <= maxgap) and (condition[0][2] <= maxgap):
            groups[-1].append(x)
        else:
            groups.append([x])
    print(groups)
    result = []
    for x in groups:
         x_s,y_s,z_s = 0,0,0
         for i in x:  
            x_s = x_s+i[0]
            y_s = y_s+i[1]  
            z_s = z_s+i[2]

        #  print(x_s,y_s,z_s)
         n = len(x)
        #  print(x_s/n,y_s/n,z_s/n)
         result.append([int(x_s/n),int(y_s/n),int(z_s/n)])
        #  print(x.mean())

    return result
def convert_coords(coords,x_axis,y_axis,z_axis):
    coords[0] = x_axis[coords[0]]
    coords[1] = y_axis[coords[1]]
    coords[2] = z_axis[coords[2]]
    #  return [x_axis[coords[0]],y_axis[coords[1]],z_axis[coords[2]]]

def main():

    print("Start")
    point = Points()
    rows = point.slices[16].Rows
    columns = point.slices[16].Columns
    pixel_spacing = point.slices[16].PixelSpacing
    image_position = point.slices[16].ImagePositionPatient
    ss = point.slices[16].SliceThickness
    print("Rows: ", rows," Columns: ", columns," Pixel_spacing: ", pixel_spacing," image_position: ",image_position, " SliceThickness: ",ss)
    x_axis = np.arange(columns)*pixel_spacing[0]
    y_axis = np.arange(rows)*pixel_spacing[1]
    # z_axis = np.arange(0,len(point.slices),ss)
    # z_axis = np.arange(len(point.slices))*ss
    z_axis = np.arange(279)*ss # HOW TO GET COUNT OF NEEDED SLICES?
    # print("x_axis: ",x_axis, "x len: ",len(x_axis), "\ny_axis: ", y_axis, " \nz_axis: ",z_axis," z len:", len(z_axis))
    print("Got data")
    count = 0
    for x in range(10,len(point.slices)):
        image = point.slices[x].pixel_array.copy()
        # self.tresholdSlider.setMaximum(x[self.slicesSlider.value()].max())
        # print(image)
        # print(len(image))
        # treshold(image,self.tresholdSlider.value())
            # print(i)
        
        cut = find_points(image,x)
        print("filtered ",x," slice")
        point.coords = point.coords + get_coordinates(cut,x-10) #because z start from 16
        print("got coordinates of ",x," slice")
    # print(point.coords)

    result = group_coords(point.coords,20)

    file = open('result_px.txt', 'w')
    for i in result:
         i.insert(0,result.index(i))
         print(i)
         file.writelines("%s\n" % str(i))
    # file.write('whatever')
    file.close()

    result = []
    result = group_coords(point.coords,20)
    result.sort(key=lambda x:x[2])

    file = open('result_xyz.txt', 'w')
    for i in result:
        convert_coords(i,x_axis,y_axis,z_axis)
        i.insert(0,result.index(i))
        print(i)
        file.writelines("%s\n" % str(i))
if __name__ == "__main__":
    main()

