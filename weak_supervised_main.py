from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, classification_report
from dataloader import *
import random
from models_weakly import *
import warnings
warnings.filterwarnings("ignore")

random.seed(3344)
import time
import argparse

parser = argparse.ArgumentParser(description='AVE')

# Data specifications
parser.add_argument('--dir_video', type=str, default="data/visual_feature.h5",
                    help='dataset directory')
parser.add_argument('--dir_video_bg', type=str, default="data/visual_feature_noisy.h5",
                    help='dataset directory')

parser.add_argument('--dir_audio', type=str,
                    default='data/audio_feature.h5',
                    help='dataset directory')

parser.add_argument('--dir_audio_bg', type=str,
                    default='data/audio_feature_noisy.h5',
                    help='dataset directory')

parser.add_argument('--dir_labels', type=str, default='data/mil_labels.h5',
                    help='dataset directory')
parser.add_argument('--dir_labels_bg', type=str, default='data/labels_noisy.h5',
                    help='dataset directory')
parser.add_argument('--dir_labels_gt', type=str, default='data/labels.h5',
                    help='dataset directory')

parser.add_argument('--dir_order_train', type=str, default='data/train_order.h5',
                    help='dataset directory')

parser.add_argument('--dir_order_val', type=str, default='data/val_order.h5',
                    help='dataset directory')
parser.add_argument('--dir_order_test', type=str, default='data/test_order.h5',
                    help='dataset directory')

parser.add_argument('--nb_epoch', type=int, default=250,
                    help='number of epoch')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--train', action='store_true', default=False,
                    help='train a new model')

args = parser.parse_args()

# model
model_name = 'AV_att_weak'
net_model = att_Net(128, 128, 512, 29)
net_model.cuda()

net_model.cuda()
loss_function = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(net_model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=15000, gamma=0.1)


def train(args):
    AVEData = AVE_weak_Dataset(video_dir=args.dir_video, video_dir_bg=args.dir_video_bg, audio_dir=args.dir_audio,
                         audio_dir_bg=args.dir_audio_bg, label_dir=args.dir_labels,label_dir_bg=args.dir_labels_bg,
                         label_dir_gt = args.dir_labels_gt,
                         order_dir=args.dir_order_train, batch_size=args.batch_size, status = "train")
    nb_batch = AVEData.__len__() // args.batch_size
    print(AVEData.__len__())
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
            scores, _ = net_model(audio_inputs, video_inputs)
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
                torch.save(net_model, 'saved_models/' + model_name + ".pt")



def val(args):
    net_model.eval()
    AVEData = AVE_weak_Dataset(video_dir=args.dir_video, video_dir_bg=args.dir_video_bg, audio_dir=args.dir_audio,
                         audio_dir_bg=args.dir_audio_bg, label_dir=args.dir_labels, label_dir_bg=args.dir_labels_bg,
                         label_dir_gt = args.dir_labels_gt, order_dir=args.dir_order_val, batch_size=402, status="val")
    nb_batch = AVEData.__len__()
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
    video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    labels = labels.numpy()
    _, x_labels = net_model(audio_inputs, video_inputs)
    #print(x_labels)
    x_labels = x_labels.cpu().data.numpy()

    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]):  # 10
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))
    print(accuracy_score(real_labels, pre_labels))
    return accuracy_score(real_labels, pre_labels)


def test(args):
    model = torch.load('model/' + model_name + ".pt")
    model.eval()
    AVEData = AVE_weak_Dataset(video_dir=args.dir_video, video_dir_bg=args.dir_video_bg, audio_dir=args.dir_audio,
                         audio_dir_bg=args.dir_audio_bg, label_dir=args.dir_labels, label_dir_bg=args.dir_labels_bg,
                         label_dir_gt=args.dir_labels_gt,
                         order_dir=args.dir_order_test, batch_size=402, status="test")
    nb_batch = AVEData.__len__()
    print(nb_batch)
    audio_inputs, video_inputs, labels = AVEData.get_batch(0)
    audio_inputs = Variable(audio_inputs.cuda(), requires_grad=False)
    video_inputs = Variable(video_inputs.cuda(), requires_grad=False)
    labels = labels.numpy()
    _, x_labels = model(audio_inputs, video_inputs)
    x_labels = x_labels.cpu().data.numpy()

    N = int(nb_batch * 10)
    pre_labels = np.zeros(N)
    real_labels = np.zeros(N)
    c = 0
    for i in range(nb_batch):
        for j in range(x_labels.shape[1]):  # 10
            pre_labels[c] = np.argmax(x_labels[i, j, :])
            real_labels[c] = np.argmax(labels[i, j, :])
            # print(pre_labels[c], real_labels[c])
            c += 1
    target_names = []
    for i in range(29):
        target_names.append("class" + str(i))
    print(accuracy_score(real_labels, pre_labels))


if args.train:
    train(args)
else:
    test(args)
