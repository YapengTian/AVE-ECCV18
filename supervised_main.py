## ECCV-2018-Audio-Visual Event Localization in Unconstrained Videos
## https://arxiv.org/abs/1803.08842
## supervised audio-visual event localization with feature fusion and audio-guided visual attention

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
from dataloader import *
import random
from models_fusion import *
from models import *
random.seed(3344)
import time
import warnings
warnings.filterwarnings("ignore") 
import argparse

parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--model_name', type=str, default='AV_att',
                    help='model name')

parser.add_argument('--dir_video', type=str, default="data/visual_feature.h5",
                    help='visual features')
parser.add_argument('--dir_audio', type=str,
                    default='data/audio_feature.h5',
                    help='audio features')
parser.add_argument('--dir_labels', type=str, default='data/labels.h5',
                    help='labels of AVE dataset')

parser.add_argument('--dir_order_train', type=str, default='data/train_order.h5',
                    help='indices of training samples')
parser.add_argument('--dir_order_val', type=str, default='data/val_order.h5',
                    help='indices of validation samples')
parser.add_argument('--dir_order_test', type=str, default='data/test_order.h5',
                    help='indices of testing samples')

parser.add_argument('--nb_epoch', type=int, default=300,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')

parser.add_argument('--train', action='store_true', default=False,
                    help='train a new model')
args = parser.parse_args()


# model
model_name = args.model_name
if model_name == 'AV_att': # corresponding to A+V-att model in the paper
    net_model = att_Net(128, 128, 512, 29)
elif model_name == 'DMRN': # corresponding to DMRN. The pre-trained DMRN.pt was trained by fine-tuning the AV_att model.
    net_model = TBMRF_Net(128, 128, 512, 29, 1)

net_model.cuda()

loss_function = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)

def compute_acc(labels, x_labels, nb_batch):
    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]): 
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))

    return accuracy_score(real_labels, pre_labels)


def train(args):
    AVEData = AVEDataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                         order_dir=args.dir_order_train, batch_size=args.batch_size)
    nb_batch = AVEData.__len__() // args.batch_size
    epoch_l = []
    best_val_acc = 0
    for epoch in range(args.nb_epoch):
        epoch_loss = 0
        n = 0
        start = time.time()
        for i in range(nb_batch):
            audio_inputs, video_inputs, labels = AVEData.get_batch(i)

            audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
            video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
            labels = Variable(labels.cuda(), requires_grad=False)
            net_model.zero_grad()
            scores = net_model(audio_inputs, video_inputs)
            loss = loss_function(scores, labels)
            epoch_loss += loss.cpu().data.numpy()
            loss.backward()
            scheduler.step()
            optimizer.step()
            n = n + 1

        end = time.time()
        epoch_l.append(epoch_loss)
        print("=== Epoch {%s}   Loss: {%.4f}  Running time: {%4f}" % (str(epoch), (epoch_loss) / n, end - start))
        if epoch % 5 == 0:
            val_acc = val(args)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(net_model, 'model/' + model_name + ".pt")

def val(args):
    net_model.eval()
    AVEData = AVEDataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                         order_dir=args.dir_order_test, batch_size=402)
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
    video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    labels = labels.numpy()
    x_labels = net_model(audio_inputs, video_inputs)
    x_labels = x_labels.cpu().data.numpy()

    acc = compute_acc(labels, x_labels, nb_batch)
    print(acc)
    return acc


def test(args):

    model = torch.load('model/' + model_name  + ".pt")
    model.eval()
    AVEData = AVEDataset(video_dir=args.dir_video, audio_dir=args.dir_audio, label_dir=args.dir_labels,
                         order_dir=args.dir_order_test, batch_size=402)
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
    video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    labels = labels.numpy()
    x_labels = model(audio_inputs, video_inputs)
    x_labels = x_labels.cpu().data.numpy()
    acc = compute_acc(labels, x_labels, nb_batch)
    print(acc)
    return acc


# training and testing
if args.train:
    train(args)
else:
    test(args)