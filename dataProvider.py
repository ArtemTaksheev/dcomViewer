import matplotlib.pyplot as plt
import pydicom
# from pydicom import dcmread
import os


test_datapath = "e://data/Шапка Богдана/23072418"
test_filename = "00000016.dcm"





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


def main():
    # for item in load_scan(test_datapath):
    #     plt.imshow(item.pixel_array, cmap=plt.cm.gray)
    #     plt.show()
    print("nothing")


if __name__ == "__main__":
    main()