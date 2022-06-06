import numpy as np
import os
import torch
from torch import nn, optim
import torch.utils.data as Data
from torch.nn import functional as F
from torch import distributions

class Denoise(nn.Module):
    def __init__(self, n_filters, filter_sizes, spike_size, CONFIG):
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.resources.gpu_id)
        torch.cuda.set_device(CONFIG.resources.gpu_id)
        self.CONFIG = CONFIG

        super(Denoise, self).__init__()
        
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=feat1,            # n_filters
                kernel_size=size1,              # filter size
                stride=1,                   # filter movement/step
                padding=0,                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
        )

        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(feat1, feat2, size2, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )

        self.conv3 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv1d(feat2, feat3, size3, 1, 0),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
        )

        #n_input_feat = feat3*(61-size1-size2-size3+3)
        n_input_feat = feat2*(spike_size-size1-size2+2)
        self.out = nn.Linear(n_input_feat, spike_size)
        
        #self.counter=0
        
    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        #print (x.shape)
        #print (self.out(x).shape)
        #np.save('/home/cat/temp/'+str(self.counter)+'.npy', x.cpu().data.numpy())
        output = self.out(x)
        #self.counter+=1
        return output, x   # return x for visualization
