import numpy as np
import cv2


class Videoto3D:

    def __init__(self, width, height, depth):#32，32，10
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color=False, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        framearray = []

        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            frame = cv2.resize(frame, (self.height, self.width))
            if color:
                framearray.append(frame)
            else:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        cap.release()
        return np.array(framearray)

    def image3d(self,name):
        frame = cv2.imread(name)
        frame = cv2.resize(frame, (self.height, self.width))
        return frame

    def get_UCF_classname(self, filename):
        #return filename[filename.find('_') + 1:filename.find('_', 2)]
        return filename[0:filename.find('_', 1)]

    def get_classname(self, filename):
        #return filename[filename.find('_') + 1:filename.find('_', 2)]
        return filename[filename.find('_', 1)+1:]

    def get_test_classname(self,filename):
        return filename[filename.rindex('_')+1:]
