"""AVE dataset"""
import numpy as np
import torch
import h5py

class AVEDataset(object):

    def __init__(self, video_dir, audio_dir, label_dir, order_dir, batch_size):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.batch_size = batch_size

        with h5py.File(order_dir, 'r') as hf:
            order = hf['order'][:]
        self.lis = order

        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]

        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        self.label_batch = np.float32(np.zeros([self.batch_size, 10, 29]))

    def __len__(self):
        return len(self.lis)

    def get_batch(self, idx):

        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[self.lis[id], :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[self.lis[id], :, :]
            self.label_batch[i, :, :] = self.labels[self.lis[id], :, :]

        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()

class AVE_weak_Dataset(object):
    def __init__(self, video_dir, video_dir_bg, audio_dir , audio_dir_bg, label_dir, label_dir_bg, label_dir_gt, order_dir, batch_size, status):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.video_dir_bg = video_dir_bg
        self.audio_dir_bg = audio_dir_bg

        self.status = status
        # self.lis_video = os.listdir(video_dir)
        self.batch_size = batch_size
        with h5py.File(order_dir, 'r') as hf:
            train_l = hf['order'][:]
        self.lis = train_l
        with h5py.File(audio_dir, 'r') as hf:
            self.audio_features = hf['avadataset'][:]
        with h5py.File(label_dir, 'r') as hf:
            self.labels = hf['avadataset'][:]
        with h5py.File(video_dir, 'r') as hf:
            self.video_features = hf['avadataset'][:]
        self.audio_features = self.audio_features[train_l, :, :]
        self.video_features = self.video_features[train_l, :, :]
        self.labels = self.labels[train_l, :]

        if status == "train":
            with h5py.File(label_dir_bg, 'r') as hf:
                self.negative_labels = hf['avadataset'][:]

            with h5py.File(audio_dir_bg, 'r') as hf:
                self.negative_audio_features = hf['avadataset'][:]
            with h5py.File(video_dir_bg, 'r') as hf:
                self.negative_video_features = hf['avadataset'][:]

            size = self.audio_features.shape[0] + self.negative_audio_features.shape[0]
            audio_train_new = np.zeros((size, self.audio_features.shape[1], self.audio_features.shape[2]))
            audio_train_new[0:self.audio_features.shape[0], :, :] = self.audio_features
            audio_train_new[self.audio_features.shape[0]:size, :, :] = self.negative_audio_features
            self.audio_features = audio_train_new

            video_train_new = np.zeros((size, 10, 7, 7, 512))
            video_train_new[0:self.video_features.shape[0], :, :] = self.video_features
            video_train_new[self.video_features.shape[0]:size, :, :] = self.negative_video_features
            self.video_features = video_train_new

            y_train_new = np.zeros((size, 29))
            y_train_new[0:self.labels.shape[0], :] = self.labels
            y_train_new[self.labels.shape[0]:size, :] = self.negative_labels
            self.labels = y_train_new
        else:
            with h5py.File(label_dir_gt, 'r') as hf:
                self.labels = hf['avadataset'][:]
                self.labels = self.labels[train_l, :, :]



        self.video_batch = np.float32(np.zeros([self.batch_size, 10, 7, 7, 512]))
        self.audio_batch = np.float32(np.zeros([self.batch_size, 10, 128]))
        if status == "train":
            self.label_batch = np.float32(np.zeros([self.batch_size, 29]))
        else:
            self.label_batch = np.float32(np.zeros([self.batch_size,10, 29]))

    def __len__(self):
        return len(self.labels)

    def get_batch(self, idx):
        for i in range(self.batch_size):
            id = idx * self.batch_size + i

            self.video_batch[i, :, :, :, :] = self.video_features[id, :, :, :, :]
            self.audio_batch[i, :, :] = self.audio_features[id, :, :]
            if self.status == "train":
                self.label_batch[i, :] = self.labels[id, :]
            else:
                self.label_batch[i, :, :] = self.labels[id, :, :]
        return torch.from_numpy(self.audio_batch).float(), torch.from_numpy(self.video_batch).float(), torch.from_numpy(
            self.label_batch).float()
