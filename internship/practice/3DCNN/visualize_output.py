from keras.models import model_from_json
from keras.utils import plot_model
from keras.models import model_from_json
import cv2
import cv2
import numpy as np
import os
import videoto3d
from keras.models import Model
from matplotlib import pyplot as plt


rows = 256
cols = 256
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
def testdata_preprocess2(imagedir, model, color=True, skip=True):
    vid = videoto3d.Videoto3D(256, 256, 10)
    test_images = os.listdir(imagedir)
    test_data = []
    labels = []
    sum = 0.0
    for file in test_images:
        # 通过测试图片所在的文件夹的名称得到图片序列的标签
        # labels.append(label)
        image_folder_name = os.path.join(imagedir, file)
        print(image_folder_name)
        arr = read_image(image_folder_name)
        arr = np.expand_dims(arr, axis=0)
        # prediction=model.predict_classes(np.transpose(arr,axes=(0,2,3,4,1)))#prediction数值为0/1
        prediction = model.predict(np.array(arr).transpose((0, 2, 3, 4, 1)))
        print(prediction)

def main():
    model = model_from_json((open('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.json').read()))
    model.load_weights('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.h5')
    # testdata_preprocess2('F:\code\Github\\3DCNN\\testdata',model)
    layer1=model.get_layer(name='activation_1')
    layer2 = model.get_layer(name='activation_2')
    layer3 = model.get_layer(name='activation_3')
    layer4 = model.get_layer(name='activation_4')
    activation_1_output = Model(inputs=model.input,outputs=layer1.output)
    activation_2_output = Model(inputs=model.input, outputs=layer2.output)
    activation_3_output = Model(inputs=model.input, outputs=layer3.output)
    activation_4_output = Model(inputs=model.input, outputs=layer4.output)
    arr = read_image('/media/f404/新加卷/yixuan/allvideo/test_video/the-1493th-mp4-day_rain_68-day_rain_6-day_rain')
    arr = np.expand_dims(arr, axis=0)
    o1=activation_1_output.predict(np.array(arr).transpose((0, 2, 3, 4, 1)))
    print(o1.shape)
    for o in o1:
        for i in range(32):
            plt.imshow(o[:,:,1,i])
            plt.show()
if __name__ == '__main__':
        main()
