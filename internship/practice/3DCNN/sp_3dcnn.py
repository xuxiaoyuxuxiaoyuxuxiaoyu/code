'''
数据预处理：
将视频切割成2s的片段，每个片段抽取40帧
--------------------------------------
需要设置的参数：
|是否插值reshape序列尺寸-ifreshape
|数据量-|500|1000|2000|4000|8000|16000|
|网络深度-|2|3|5|8|13|21|
|特征个数-|4|8|16|32|64|128|
--------------------------------------
添加正则化：
|L2正则化
|DROPOUT
--------------------------------------
需要保存的数据：
train_loss|train_acc|val_loss|val_acc|
test_loss|test_acc|test_acc vs data|
模型文件|
'''
import argparse
import videoto3d
from tqdm import tqdm
import os
from keras.models import model_from_json
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,
                          MaxPooling3D)
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras import regularizers

#def get_class(file_foleder_name,label_list):


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))

def loaddata_video(video_dir, vid3d, nclass, result_dir, color=False, skip=True):
    i=0
    file_folders = os.listdir(video_dir)
    X = []
    labels = []
    labellist = []
    pbar = tqdm(total=len(file_folders))
    for file_oneclass in file_folders:
        #i += 1
        pbar.update(1)
        files_dir = os.path.join(video_dir, file_oneclass)
        files = os.listdir(files_dir)
        pbar2 = tqdm(total=len(files))
        for filename in files:
            pbar2.update(1)
            if filename == '.DS_Store':
                continue
            name = os.path.join(files_dir, filename)
            label = vid3d.get_UCF_classname(filename)
            if label not in labellist:
                if len(labellist) >= nclass:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(vid3d.video3d(name, color=color, skip=skip))
        #if i==1:break
    pbar.close()
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{}\n'.format(labellist[i]))

    for num, label in enumerate(labellist):#enumrate能同时获得索引和索引对应的值
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels

def loaddata_image(image_dir, vid3d, nclass, result_dir, color=False, skip=True):
    #i=0
    file_folders = os.listdir(image_dir)
    X = []
    labels = []
    labellist = []
    pbar = tqdm(total=len(file_folders),desc='reading filefolders')
    for file_oneclass in file_folders:
        x_ = []
        #i += 1
        pbar.update(1)
        files_dir = os.path.join(image_dir, file_oneclass)
        files = os.listdir(files_dir)
        label = vid3d.get_classname(file_oneclass)
        pbar2 = tqdm(total=len(files),desc='reading images')
        for filename in files:
            pbar2.update(1)
            name = os.path.join(files_dir, filename)
            x_.append(vid3d.image3d(name))
        X.append(x_)
        labels.append(label)
        pbar2.close()
        #+if i==1000:break
    pbar.close()
    for it in labels:
        if it not in labellist:
            if len(labellist)>=nclass:
                continue
            labellist.append(it)
    for num, label in enumerate(labellist):#enumrate能同时获得索引和索引对应的值
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
    if color:
        return np.array(X).transpose((0, 2, 3, 4, 1)), labels
    else:
        return np.array(X).transpose((0, 2, 3, 1)), labels

def param_set():
    parser = argparse.ArgumentParser(
        description='simple 3D convolution for action recognition')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--videos', type=str, default='/media/f404/新加卷/yixuan/allvideo/train_video',
                        help='directory where videos are stored')
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--output', type=str, required=True,default='./3dcnnresult')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--skip', type=bool, default=True)
    parser.add_argument('--depth', type=int, default=100)
    ##########################################################################
    parser.add_argument('--size', type=list, default=[256,256])
    args = parser.parse_args()
    return args

def data_preprocess(args):
    assert type(args) == argparse.Namespace, 'args type error'
    vid3d = videoto3d.Videoto3D(args.size[0],args.size[1],args.depth)
    nb_classes = args.nclass
    channel = 3 if args.color else 1

    fname_npz = 'dataset_{}_{}_{}.npz'.format(#npz类型为numpy数组保存的文件名，
        args.nclass, args.depth, args.skip)
    if os.path.exists(fname_npz):
        loadeddata = np.load(fname_npz)
        train_data, train_label = loadeddata["train_data"], loadeddata["train_label"]
    else:
        train_data,labels = loaddata_image(args.videos, vid3d, args.nclass,
                        args.output, args.color, args.skip)
        train_label = np_utils.to_categorical(labels, nb_classes)
        train_data = train_data.astype('float32')
        np.savez(fname_npz, train_data=train_data, train_label=train_label)
        print('Saved dataset to dataset.npz.')
        print('data_shape:{}\nlabel_shape:{}'.format(train_data.shape, train_label.shape))
    return train_data,train_label,nb_classes


def model_depth_2(inputshape,numlabel):
    model = Sequential()
    #输入model_depth_2的train_data.shape[1:]=(256, 256, 3, 10)
    model.add(Conv3D(4, kernel_size=(3, 3, 3), input_shape=
        inputshape,kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(8, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01),border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01),border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01),border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(numlabel, activation='sigmoid'))
    return model
def model_depth_3(inputshape,numlabel):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=
        inputshape,kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    #model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(numlabel, activation='sigmoid'))
    return model
def model_depth_5(inputshape,numlabel):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=
        inputshape,kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    #model.add(Dropout(0.25))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=
        inputshape,kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(128, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), input_shape=
        inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3),kernel_regularizer=regularizers.l2(0.01), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(numlabel, activation='sigmoid'))
    return model
def model_depth_8(inputshape,numlabel):
    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    # model.add(Dropout(0.25))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(512, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(512, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(1024, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(1024, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(numlabel, activation='sigmoid'))
    return model
def model_depth_13(inputshape,numlabel):
    model = Sequential()
    model.add(Conv3D(4, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(4, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))
    # model.add(Dropout(0.25))

    model.add(Conv3D(8, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(256, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(256, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(512, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(512, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(1024, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(1024, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Conv3D(2048, kernel_size=(3, 3, 3), input_shape=
    inputshape, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv3D(2048, kernel_size=(3, 3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), border_mode='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(numlabel, activation='sigmoid'))

    return model

def train(model,train_data,train_label,args):
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    print(len(train_data),len(train_label))
    X_train, X_test, Y_train, Y_test = train_test_split(
        train_data, train_label, test_size=0.2, random_state=43)

    ############################################################
    best_weights_filepath = './best_we'
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss',
                                             patience=50, verbose=1, mode='auto')
    saveBestModel = callbacks.ModelCheckpoint(best_weights_filepath,
                                               monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             #factor=0.1, verbose=1, patience=20, min_lr=0.1)
    tensorboard = callbacks.TensorBoard(log_dir='./log ',histogram_freq=1,write_graph=True,write_images=True)
    ############################################################
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=args.batch,
                        epochs=args.epoch, verbose=1,callbacks=[earlyStopping,saveBestModel,tensorboard], shuffle=True)
    model.evaluate(X_test, Y_test, verbose=0)

    model_json = model.to_json()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    with open(os.path.join(args.output, 'ucf101_3dcnnmodel.json'), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output, 'ucf101_3dcnnmodel.h5'))

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    plot_history(history, args.output)
    save_history(history, args.output)

def main():
    args = param_set()
    train_data, train_label, nb_classes = data_preprocess(args)
    print('train_data.shape',train_data.shape)
    #train_data.shape (3051, 256, 256, 3, 10)
    model = model_depth_2(train_data.shape[1:],nb_cl asses)
    #输入model_depth_2的train_data.shape[1:]=(256, 256, 3, 10)
    train(model, train_data, train_label, args)
if __name__ == '__main__':
     main()
