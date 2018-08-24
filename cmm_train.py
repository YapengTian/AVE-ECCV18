from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input,  Lambda
from keras.layers import Dense, Average, Activation, add, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import  LeakyReLU
from keras.optimizers import Adam
from keras import backend as K

import h5py
import math
import random
random.seed(3344)

def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 5
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


##data loader
#load train data
with h5py.File('data/labels_closs.h5', 'r') as hf:
    closs_labels = hf['avadataset'][:]

with h5py.File('data/visual_feature_vec.h5', 'r') as hf:
    video_features = hf['avadataset'][:]
with h5py.File('data/audio_feature.h5', 'r') as hf:
    audio_features = hf['avadataset'][:]
with h5py.File('data/train_order_match.h5', 'r') as hf:
    train_l = hf['order'][:]
with h5py.File('data/val_order_match.h5', 'r') as hf:
    val_l = hf['order'][:]
with h5py.File('data/test_order_match.h5', 'r') as hf:
    test_l = hf['order'][:]

closs_labels = np.array(closs_labels)
audio_features = np.array(audio_features)
video_features = np.array(video_features)
closs_labels = closs_labels.astype("float32")
audio_features = audio_features.astype("float32")
video_features = video_features.astype("float32")
##
x_audio_train = np.zeros((len(train_l)*10, 128))
x_video_train = np.zeros((len(train_l)*10, 512))
x_audio_val = np.zeros((len(val_l)*10, 128))
x_video_val = np.zeros((len(val_l)*10, 512))
x_audio_test = np.zeros((len(test_l)*10, 128))
x_video_test = np.zeros((len(test_l)*10, 512))
y_train      = np.zeros((len(train_l)*10))
y_val        = np.zeros((len(val_l)*10))
y_test       = np.zeros((len(test_l)*10))
##
for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        x_audio_train[10*i + j, :] = audio_features[id, j, :]
        x_video_train[10*i + j, :] = video_features[id, j, :]
        y_train[10*i + j] = closs_labels[id, j]

for i in range(len(val_l)):
    id = val_l[i]
    for j in range(10):
        x_audio_val[10 * i + j, :] = audio_features[id, j, :]
        x_video_val[10 * i + j, :] = video_features[id, j, :]
        y_val[10 * i + j] = closs_labels[id, j]

for i in range(len(test_l)):
    id = test_l[i]
    for j in range(10):
        x_audio_test[10 * i + j, :] = audio_features[id, j, :]
        x_video_test[10 * i + j, :] = video_features[id, j, :]
        y_test[10 * i + j] = closs_labels[id, j]
print("data loading finished!")

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 2.0
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    preds = predictions.ravel() < 0.5
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)

def scmm_net(input_audio_shape, input_video_shape):
    video_input = Input(shape=(512,)) #512
    video       = Dense(128)(video_input)
    video       = LeakyReLU(alpha=0.3)(video)
    video       = Dense(64)(video)


    audio_input = Input(shape=(128,)) #128
    audio       = Dense(128)(audio_input)
    audio       = LeakyReLU(alpha=0.3)(audio)
    audio       = Dense(64)(audio)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([video, audio])
    model = Model([video_input, audio_input], distance)
    return model

'''parameters'''
batch_size = 8
nb_epoch = 20

output_dim = 29#9
input_video_shape = x_video_train.shape[1]
input_audio_shape = x_audio_train.shape[1]
# network definition
model = scmm_net(input_video_shape, input_audio_shape)


# train
adam = Adam()
model.compile(loss=contrastive_loss, optimizer=adam)
model.fit([x_video_train, x_audio_train], y_train,
          batch_size=8,
          epochs=nb_epoch,
          validation_data=([x_video_val, x_audio_val], y_val))

json_string = model.to_json()
open('model/cmm_model.json','w').write(json_string)
model.save_weights('model/cmm_model_weights.h5')

