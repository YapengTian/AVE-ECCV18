import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init

# two-branch/dual muli-modal residual fusion
class TBMRF_Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, hidden_size, tagset_size, nb_block):
        super(TBMRF_Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        self.lstm_audio = nn.LSTM(128, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.lstm_video = nn.LSTM(512, hidden_dim, 1, batch_first=True, bidirectional=True)
        self.affine_audio = nn.Linear(128, hidden_size)  
        self.affine_video = nn.Linear(512, hidden_size)  
        self.affine_v = nn.Linear(hidden_size, 49, bias=False) 
        self.affine_g = nn.Linear(hidden_size, 49, bias=False) 
        self.affine_h = nn.Linear(49, 1, bias=False)

        # fusion transformation functions
        self.nb_block = nb_block
        
        self.U_v  = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        self.U_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

       
        self.L2 = nn.Linear(hidden_dim*2, tagset_size)

        self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def TBMRF_block(self, audio, video, nb_block):

        for i in range(nb_block):
            video_residual = video
            v = self.U_v(video)
            audio_residual = audio
            a = self.U_a(audio)
            merged = torch.mul(v + a, 0.5) 

            a_trans = audio_residual
            v_trans = video_residual

            video = self.tanh(a_trans + merged)
            audio = self.tanh(v_trans + merged)

        fusion = torch.mul(video + audio, 0.5)#
        return fusion

    def init_weights(self):
        """Initialize the weights."""
        init.xavier_uniform(self.L2.weight)

    def forward(self, audio, video):

        v_t = video.view(video.size(0) * video.size(1), -1, 512)
        V = v_t
        v_t = self.relu(self.affine_video(v_t))

        a_t = audio.view(-1, audio.size(-1))
        a_t = self.relu(self.affine_audio(a_t))


        content_v = self.affine_v(v_t) \
                    + self.affine_g(a_t).unsqueeze(2)
        z_t = self.affine_h((F.tanh(content_v))).squeeze(2)
        alpha_t = F.softmax(z_t, dim=-1).view(z_t.size(0), -1, z_t.size(1))
        # Construct c_t: B x seq x hidden_size
        c_t = torch.bmm(alpha_t, V).view(-1, 512)
        # SElayer
        video_t = c_t.view(video.size(0), -1, 512)

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
        #print(lstm_video.size())
        output = self.TBMRF_block(lstm_audio, lstm_video, self.nb_block)
        out = self.L2(output)
        out = F.softmax(out, dim=-1)

        return out


