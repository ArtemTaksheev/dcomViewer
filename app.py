import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QFileDialog
from design import design

from dataProvider import DataProvider
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import cv2 as cv
import pydicom

from analizeDCOM import AnalizeDICOM

# test_datapath = "e://data/Шапка Богдана/23072418"
test_datapath = "E://data/Brain_marks_AEvit/23090311"


class Window(QWidget, design):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setupUi(self)
        # self.path = ""
        # self.provider = DataProvider()
        self.analizer = AnalizeDICOM(test_datapath)
        # self.slices = []
        self.connectActions()
        
        # self.coords = []
        # self.path = test_datapath
        
        # self.slices = self.provider.get_images(self.path)
        # print(self.slices)
        self.slicesSlider.setMaximum(len(self.analizer.slices) - 1)
        self.tresholdSlider.setMaximum(self.analizer.slices[0].pixel_array.max())
        self.plot(self.analizer.slices[0].pixel_array)

    def connectActions(self):
        self.slicesSlider.valueChanged.connect(self.updatePicture)
        self.tresholdSlider.valueChanged.connect(self.updatePicture)
        self.loadButton.clicked.connect(self.loadDcom)
        self.getMarkers.clicked.connect(self.exportCoords)
        self.savePoint.clicked.connect(self.exportRoi)
        self.canvas.mpl_connect('button_press_event', self.mouse_event)

    def mouse_event(self,event):
        if(self.analizer.slices):
            print('x: {} y: {} z: {} in pixel'.format(event.xdata, event.ydata,self.slicesSlider.value()-10))
            coords = [int(event.xdata), int(event.ydata),self.slicesSlider.value()-10] #HOW TO GET STARTING POINT
            self.analizer.convert_coords(coords,self.analizer.x_axis,self.analizer.y_axis,self.analizer.z_axis)
            print('x: {} y: {} z: {} in xyz'.format(coords[0],coords[1],coords[2]))
            self.coords.setText('[{}, {}, {}]'.format(coords[0],coords[1],coords[2]))
    def exportRoi(self):
        file = open('point.txt', 'w')
        file.writelines("%s\n" % self.coords.text())
        file.close()

    def loadDcom(self):
        fileName = QFileDialog.getExistingDirectory(self, 'Select Folder with DCOM')
        if fileName:
            self.analizer.path = fileName
            print(fileName)
        self.analizer.slices = self.analizer.provider.get_images(self.analizer.path)
        # print(self.slices)
        self.slicesSlider.setMaximum(len(self.analizer.slices) - 1)
        self.tresholdSlider.setMaximum(self.analizer.slices[0].max())
    
    def updatePicture(self):
        if self.analizer.slices:
            image = self.analizer.slices[self.slicesSlider.value()].pixel_array.copy()
            self.tresholdSlider.setMaximum(self.analizer.slices[self.slicesSlider.value()].pixel_array.max())

            self.analizer.treshold(image,self.tresholdSlider.value())

            
            # print(self.coords)
            self.plot(image)
            # print(image)

    def plot(self, image):
        self.figure.clear()
        plt.imshow(image, cmap=plt.cm.gray)

        self.canvas.draw()
    def exportCoords(self):
        print("Starting to calculate coords")
        self.analizer.coords = []
        for x in range(10,len(self.analizer.slices)):
            image = self.analizer.slices[x].pixel_array.copy()
            cut = self.analizer.find_points(image,show = False)
            print("filtered ",x-10," slice")
            self.analizer.coords = self.analizer.coords + self.analizer.get_coordinates(cut,x-10)

        result = self.analizer.group_coords(self.analizer.coords,20)

        print("Export result_px")
        file = open('result_px.txt', 'w')
        for i in result:
            i.insert(0,result.index(i))
            print(i)
            file.writelines("%s\n" % str(i))
        # file.write('whatever')
        file.close()

        result = []
        result = self.analizer.group_coords(self.analizer.coords,20)
        result.sort(key=lambda x:x[2])


        print("Export result_xyz")
        file = open('result_xyz.txt', 'w')
        for i in result:
            self.analizer.convert_coords(i,self.analizer.x_axis,self.analizer.y_axis,self.analizer.z_axis)
            i.insert(0,result.index(i))
            print(i)
            file.writelines("%s\n" % str(i))
        file.close()

def main():
    app = QApplication(sys.argv)

    window = Window()

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

