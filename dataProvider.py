import matplotlib.pyplot as plt
import pydicom
# from pydicom import dcmread
import os


test_datapath = "e://data/Шапка Богдана/23072418"
test_filename = "00000016.dcm"





class DataProvider:
    def __init__(self, datafolder=test_datapath) -> None:
        self.data_folder = datafolder
        self.slices = self.load_scan(datafolder)

    def get_slices(self):
        return self.slices
    
    def load_scan(self,path):
        slices = [pydicom.dcmread(path + "/" + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        return slices


def main():
    # for item in load_scan(test_datapath):
    #     plt.imshow(item.pixel_array, cmap=plt.cm.gray)
    #     plt.show()
    print("nothing")


if __name__ == "__main__":
    main()