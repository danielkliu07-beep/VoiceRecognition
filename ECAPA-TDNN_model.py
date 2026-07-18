import torch
import torch.nn as nn
from torch.nn import functional as F


class Res2DilatedConv1D(nn.Module):

    def __init__(self, channels, kernel_size = 3, dilation = 1, scale = 8): #Scale = 8 -> Split input into 8 chunks

        #kernel size = 3, dilation = 1, scale = 8

        super().__init__()

        assert channels % scale == 0

        self.width = channels // scale #Width - how much input and output channels each of the 8 chunks have 
        self.scale = scale

        padding = dilation * (kernel_size - 1) // 2 #Padding is calculated to ensure temporal dimension (T) remains unchanged

        self.convolutions = nn.ModuleList([nn.Conv1d(in_features = self.width, out_features = self.width, kernel_size = kernel_size, dilation = dilation, padding = padding) for _ in range(self.scale - 1)])

    def forward(self, x): #x shape = (batch, channel, time)

        chunks = torch.split(x, self.width, dim = 1) #Splits the input vector into 8 different chunks of dimension (batch, time)
        out = []

        for i in range(self.scale):
            if i == 0:
                out.append(chunks[i])
            elif i == 1:
                out.append(self.convolutions[i - 1](chunks[i]))
            else:
                residual_connection = chunks[i] + out[i - 1]
                out.append(self.convolutions[i - 1](residual_connection))

        return torch.cat(out, dim = 1) #Adds back the channel dimension 



class SE_Block(nn.Module):

    #1-Dimensional Squeeze-Excitation Res2Blocks

    def __init__(self, reduced_dimension, channels):

        super().__init__()

        #Note that in_features = col, out_features = row

        self.W1 = nn.Linear(in_features = channels, out_features = reduced_dimension)
        self.W2 = nn.Linear(in_features = reduced_dimension, out_features = channels)

    
    def forward(self, x): #x shape = (batch, channel, time)

        z = torch.mean(x, dim = 2) #z shape = (batch, channel)

        s = self.W1(z)
        s = F.relu(s)

        s = self.W2(s)

        s = F.sigmoid(s)

        s = s.unsqueeze(2) #Turns s back into a (batch, channel, time) shape so we can multiply it with the orignal input
        out = s @ x

        return out


class SE_Res2Block(nn.Module):

    def __init__(self, channels, temporal_dim):
        
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_features = temporal_dim, out_features = channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block2 = nn.Sequential(
            Res2DilatedConv1D(channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(in_features = temporal_dim, out_features = channels),
            nn.ReLU(),
            nn.BatchNorm1d(num_features = channels)
        )

        self.block4 = SE_Block(temporal_dim, channels)


    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        out = out + x

        return out

class AttentiveStatPooling(nn.Module):

    def __init__(self, channels):

        super().__init__()


class ECAPA_TDNN(nn.Module):

    def __init__(self, channel, T):
        
        #Conv1D, ReLu, BN, kernel size = 5, dilation = 1, input = 80 * T, output = C * T
        first_block = nn.Sequential(
            nn.Conv1d(in_channels = 80 * T, out_channels = T * channel, kernel_size = 5, dilation = 1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=C)
        )
