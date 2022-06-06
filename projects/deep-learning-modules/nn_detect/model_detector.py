import numpy as np
import os 
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn import functional as F
from torch import distributions


class Detect(nn.Module):
    def __init__(self, n_filters, spike_size, channel_index, CONFIG):
        super(Detect, self).__init__()
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
        torch.cuda.set_device(CONFIG.resources.gpu_id)

        self.spike_size = spike_size
        self.channel_index = channel_index
        n_neigh = self.channel_index.shape[1]
        
        feat1, feat2, feat3 = n_filters

        self.temporal_filter1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=feat1,            # n_filters
                kernel_size=[spike_size, 1],              # filter size
                stride=1,                   # filter movement/step
                padding=[(self.spike_size-1)//2, 0],                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )

        self.temporal_filter2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(feat1, feat2, [1, 1], 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )
        
        #self.spatial_filter = nn.Sequential(         # input shape (16, 14, 14)
        #    nn.Conv1d(feat2, feat3, [1, n_neigh], 1, 0),     # output shape (32, 14, 14)
        #    nn.ReLU(),                      # activation
        #)
        self.out = nn.Linear(feat2*n_neigh, 1)

    def forward(self, x):

        x = x[:, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)[:, :, 0]
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        output = torch.sigmoid(x)

        return output, x   # return x for visualization
    
    def forward_recording(self, recording_tensor):

        x = recording_tensor[None, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)

        zero_buff = torch.zeros(
            [1, x.shape[1], x.shape[2], 1]).to(x.device)
        x = torch.cat((x, zero_buff), 3)[0]
        x = x[:, :, self.channel_index].permute(1, 2, 0, 3)
        x = self.out(x.reshape(
            recording_tensor.shape[0]*recording_tensor.shape[1], -1))
        x = x.reshape(recording_tensor.shape[0],
                      recording_tensor.shape[1])
        
        return x

    def get_spike_times(self, recording_tensor, max_window=5, threshold=0.5, buffer=None):
        
        probs = self.forward_recording(recording_tensor)
        
        maxpool = torch.nn.MaxPool2d(kernel_size=[max_window, 1], stride=1, padding=[(max_window-1)//2, 0])
        temporal_max = maxpool(probs[None])[0] - 1e-8

        spike_index_torch = torch.nonzero(
            (probs >= temporal_max) & (probs > np.log(threshold / (1 - threshold))))
        
        # remove edge spikes
        if buffer is None:
            buffer = self.spike_size//2

        spike_index_torch = spike_index_torch[
            (spike_index_torch[:, 0] > buffer) & 
            (spike_index_torch[:, 0] < recording_tensor.shape[0] - buffer)]

        wf_t_range = torch.arange(
            -(self.spike_size//2), self.spike_size//2+1).to(spike_index_torch.device)
        time_index = spike_index_torch[:, 0][:, None] + wf_t_range
        channel_index = spike_index_torch[:, 1][:,None].repeat((1, self.spike_size))
        wf = recording_tensor[time_index, channel_index]

        return spike_index_torch, wf