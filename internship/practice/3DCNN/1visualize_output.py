from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np
import os
import videoto3d

rows = 32
cols = 32
depth_t = 100
def read_image(file_folder,color=True):
    files=os.listdir(file_folder)
    images=[]
    for file in files:
        path = os.path.join(file_folder,file)
        frame=cv2.imread(path)
        frame = cv2.resize(frame, (rows, cols))
        if color:
           images.append(frame)
    return np.array(images)
def testdata_preprocess2(imagedir,model, color=True, skip=True):
    i=0
    vid = videoto3d.Videoto3D(32,32,10)
    test_images = os.listdir(imagedir)
    test_data = []
    labels=[]
    sum=0.0
    for file in test_images:
        i=i+1
        if i==2:
           break
        label=vid.get_test_classname(file)
        #通过测试图片所在的文件夹的名称得到图片序列的标签
        #labels.append(label)
        image_folder_name = os.path.join(imagedir,file)
        print(image_folder_name)
        arr=read_image(image_folder_name)
        arr=np.expand_dims(arr, axis=0)
        prediction=model.predict_classes(np.transpose(arr,axes=(0,2,3,4,1)))#prediction数值为0/1
        #prediction=model.predict(np.array(arr).transpose((0,2,3,4,1)))
        #print(prediction)
    
def main():
    model = model_from_json((open('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.json').read()))
    model.load_weights('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.h5')
    testdata_preprocess2('/media/f404/新加卷/yixuan/allvideo/test_video',model)

if __name__ == '__main__':
    main()
