import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

class att_Net(nn.Module):
    '''
    audio-visual event localization with audio-guided visual attention and audio-visual fusion
    '''
    def __init__(self, embedding_dim, hidden_dim, hidden_size, tagset_size):
        super(att_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.affine_audio = nn.Linear(128, hidden_size)
        self.affine_video = nn.Linear(512, hidden_size)
        self.affine_v = nn.Linear(hidden_size, 49, bias=False)
        self.affine_g = nn.Linear(hidden_size, 49, bias=False)
        self.affine_h = nn.Linear(49, 1, bias=False)

        self.L1 = nn.Linear(hidden_dim * 4, 64)
        self.L2 = nn.Linear(64, tagset_size)

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.affine_v.weight)
        init.xavier_uniform(self.affine_g.weight)
        init.xavier_uniform(self.affine_h.weight)

        init.xavier_uniform(self.L1.weight)
        init.xavier_uniform(self.L2.weight)
        init.xavier_uniform(self.affine_audio.weight)
        init.xavier_uniform(self.affine_video.weight)

    def forward(self, audio, video):

        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t

        # Audio-guided visual attention
        v_t = self.relu(self.affine_video(v_t))
        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))
        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2)

        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1)) # attention map
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        video_t = c_t.view(video.size(0), -1, 512) #attended visual features

        # Bi-LSTM for temporal modeling
        hidden1 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                   autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
        hidden2 = (autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()),
                   autograd.Variable(torch.zeros(2, audio.size(0), self.hidden_dim).cuda()))
        self.lstm_video.flatten_parameters()
        self.lstm_audio.flatten_parameters()
        lstm_audio, hidden1 = self.lstm_audio(
            audio.view(len(audio), 10, -1), hidden1)
        lstm_video, hidden2 = self.lstm_video(
            video_t.view(len(video), 10, -1), hidden2)

        # concatenation and prediction
        output = torch.cat((lstm_audio, lstm_video), -1)
        output = self.relu(output)
        out = self.L1(output)
        out = self.relu(out)
        out = self.L2(out)
        out = F.softmax(out, dim=-1)


        return out