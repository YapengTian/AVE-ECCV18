from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Lambda
from keras.layers import Dense, Average, Activation, add, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import average_precision_score
import h5py
import math
import random

random.seed(3344)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


##data loader
# load train data
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
x_audio_train = np.zeros((len(train_l) * 10, 128))
x_video_train = np.zeros((len(train_l) * 10, 512))
x_audio_val = np.zeros((len(val_l) * 10, 128))
x_video_val = np.zeros((len(val_l) * 10, 512))
x_audio_test = np.zeros((len(test_l) * 10, 128))
x_video_test = np.zeros((len(test_l) * 10, 512))
y_train = np.zeros((len(train_l) * 10))
y_val = np.zeros((len(val_l) * 10))
y_test = np.zeros((len(test_l) * 10))
##
for i in range(len(train_l)):
    id = train_l[i]
    for j in range(10):
        x_audio_train[10 * i + j, :] = audio_features[id, j, :]
        x_video_train[10 * i + j, :] = video_features[id, j, :]
        y_train[10 * i + j] = closs_labels[id, j]

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
    margin = 1.0
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            c += 1
    return c / len(predictions)


def compute_precision(predictions, labels):
    c = 0
    for i in range(len(predictions)):
        if predictions[i] == 1 and predictions[i] == labels[i]:
            c += 1
    return c


def scmm_net(input_audio_shape, input_video_shape):
    video_input = Input(shape=(512,))  # 512
    video = Dense(128)(video_input)
    video = LeakyReLU(alpha=0.3)(video)
    video = Dense(64)(video)
    # video       = LeakyReLU(alpha=0.3)(video)

    audio_input = Input(shape=(128,))  # 128
    audio = Dense(128)(audio_input)
    audio = LeakyReLU(alpha=0.3)(audio)
    audio = Dense(64)(audio)
    # audio       = LeakyReLU(alpha=0.3)(audio)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([video, audio])
    model = Model([video_input, audio_input], distance)
    return model


'''parameters'''
batch_size = 8
nb_epoch = 20

output_dim = 29
input_video_shape = x_video_train.shape[1]
input_audio_shape = x_audio_train.shape[1]
# network definition
model = scmm_net(input_video_shape, input_audio_shape)
model.load_weights("model/cmm_model_weights.h5")
# train
adam = Adam()
model.compile(loss=contrastive_loss, optimizer=adam)

N = y_test.shape[0]
s = 200
e = s + 10
count_num = 0
audio_count = 0
video_count = 0
video_acc = 0
audio_acc = 0
pos_len = 0

f = open("data/Annotations.txt", 'r')
dataset = f.readlines()  # all good videos and the duration is 10s
xx = 0
for video_id in range(int(N / 10)):
    s = video_id * 10
    e = s + 10
    x_test = y_test[s: e]
    if np.sum(x_test) == 10:
        continue
    count_num += 1
    xx += np.sum(x_test)
    nb = np.argwhere(x_test == 1)

    seg = np.zeros(len(nb)).astype('int8')
    for i in range(len(nb)):
        seg[i] = nb[i][0]

    l = len(seg)
    x_test_video_feature = x_video_test[s:e, :]
    x_test_audio_feautre = x_audio_test[s:e, :]

    # given audio clip
    score = []
    for nn in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += model.predict([x_test_video_feature[nn + i:nn + i + 1, :], x_test_audio_feautre[seg[i:i + 1], :]])
        score.append(s)
    score = np.array(score).astype('float32')
    id = int(np.argmin(score))
    pred_vid = np.zeros(10)
    for i in range(id, id + int(l)):
        pred_vid[i] = 1

    if np.argmin(score) == seg[0]:
        audio_count += 1
    video_acc += compute_precision(x_test, pred_vid)
    # calculate single accuracy
    ind = np.where(x_test - pred_vid == 0)[0]
    acc_v = len(ind)

    # given video clip
    score = []
    for nn in range(10 - l + 1):
        s = 0
        for i in range(l):
            s += model.predict([x_test_video_feature[seg[i:i + 1], :], x_test_audio_feautre[nn + i:nn + i + 1, :]])
        score.append(s)
    score = np.array(score).astype('float32')

    if np.argmin(score) == seg[0]:
        video_count += 1
    pred_aid = np.zeros(10)
    id = int(np.argmin(score))
    for i in range(id, id + int(l)):
        pred_aid[i] = 1
    audio_acc += compute_precision(x_test, pred_aid)
    pos_len += len(seg)

    # calculate single accuracy
    ind = np.where(x_test - pred_aid == 0)[0]
    acc_a = len(ind)
    print('num:{}, {}'.format(video_id, dataset[test_l[video_id]].rstrip('\n')))
    print('vid_input: ', x_test, 'pred:', pred_vid, 'correct: ', acc_v)
    print('aud_input: ', x_test, 'pred:', pred_aid, 'correct: ', acc_a)

print(video_count * 100 / count_num, audio_count * 100 / count_num)