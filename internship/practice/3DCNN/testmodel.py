from keras.models import load_model
from keras.models import model_from_json
import cv2
import numpy as np
import os
import videoto3d

rows = 256
cols = 256
depth_t = 100
def read_vedio(filename, color=True, skip=True):
    cap = cv2.VideoCapture(filename)
    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if skip:
        frames = [x * nframe / depth_t for x in range(depth_t)]
    else:
        frames = [x for x in range(depth_t)]
    framearray = []
    for i in range(depth_t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        frame = cv2.resize(frame, (rows, cols))
        if color:
            framearray.append(frame)
        else:
            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.array(framearray)
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
def testdata_preprocess(videodir, color=False, skip=True):
    test_videos = os.listdir(videodir)
    test_data = []
    for file in test_videos:
        video_name = os.path.join(videodir,file)
        print(video_name)
        test_data.append(list(read_vedio(video_name)))
    return np.array(test_data).transpose((0,2,3,4,1))

def testdata_preprocess2(imagedir,model, color=True, skip=True):
    vid = videoto3d.Videoto3D(256,256,10)
    test_images = os.listdir(imagedir)
    test_data = []
    labels=[]
    sum=0.0
    for file in test_images:
        label=vid.get_test_classname(file)
        #通过测试图片所在的文件夹的名称得到图片序列的标签
        #labels.append(label)
        image_folder_name = os.path.join(imagedir,file)
        print(image_folder_name)
        arr=read_image(image_folder_name)
        arr=np.expand_dims(arr, axis=0)
        #prediction=model.predict_classes(np.transpose(arr,axes=(0,2,3,4,1)))#prediction数值为0/1
        prediction=model.predict_classes(np.array(arr).transpose((0,2,3,4,1)))
        print(prediction)
        print(label)
        if prediction==1:
            if label=='norain':
                sum+=1
        else:
            print('aaaa')
            if label=='rain':
                sum+=1
        test_data.append(list(arr))
        print('sum={}'.format(sum))
    acc=sum/len(test_images)
    print('test_acc={}'.format(acc))
    print('len(test_dataset)',len(test_images))
    #return np.array(test_data).transpose((0,2,3,4,1)),labels
    
    return acc,sum,len(test_images)

def main():
    model = model_from_json((open('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.json').read()))
    model.load_weights('/home/f404/derain/3DCNN/3dcnnresult/ucf101_3dcnnmodel.h5')
    #testdata = testdata_preprocess('/home/f404/derain/EMR/EMR')  
    testdata,labels = testdata_preprocess2('/media/f404/新加卷/yixuan/allvideo/test_video',model)
    '''print(testdata.shape())
    prediction=model.predict_classes(testdata)
    print(prediction,labels)
    sum=0.0
    print('length={}'.format(len(prediction)))
    for i in range(len(prediction)):
        if prediction[i]:
           if labels[i]=='rain':
              sum+=1
        else:
           if labels[i]=='norain':
              sum+=1
        print('sum={}'.format(sum))
    acc=sum/len(prediction)
    print('test_acc={}'.format(acc))'''
if __name__ == '__main__':
    main()
