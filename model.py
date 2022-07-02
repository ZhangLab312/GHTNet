import torch.nn as nn
import copy
from convolutions import Convolution, Transformation
from attention import MultiHeadAttention
from otherlayer import PositionalEncoding, PositionwiseFeedForward
from encoderblock import Encoder, EncoderLayer
import torch
import collections

import numpy as np
# from einops import rearrange, repeat
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
        A standard Transformer-Encoder architecture
        Base for this and many other models.
    """

    def __init__(self, convolution, encoder, dim, motifscanner):
        super(TransformerEncoder, self).__init__()
        self.convolution = convolution
        self.encoder = encoder
        self.dim = dim
        self.position = nn.Parameter(torch.zeros(1,101,self.dim ))
        self.deep = motifscanner

        self.cs = nn.Sequential(
            nn.Conv1d(18,32, kernel_size=8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(32,64, kernel_size=8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(7520, 925),
            nn.Dropout(p=0.2),
            nn.LayerNorm(925),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(925, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, src, src_mask,data):
        """
            Take in and process masked src sequences
        """
        x = self.convolution(torch.transpose(src,1,2))
        b, n, _ = x.shape
        x = x+self.position#
        x = self.encode(x, src_mask)
        if data == None:
            cs = self.cs(torch.transpose(data, 1, 2))
            cs = cs.view(-1,20*64)
            x = self.deep(torch.transpose(x, 1, 2))
            x = torch.cat([x, cs], 1)
            x = self.fc(x)
        return x

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

class MotifScanner(nn.Module):
    def __init__(self, TFs_cell_line_pair=2):
        super(MotifScanner, self).__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=160, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=160)
        )
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=(1))
        self.Conv2 = nn.Sequential(
            nn.Conv1d(in_channels=160, out_channels=320, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=320)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=480)
        )

        self.MaxPool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)

        self.Linear1 = nn.Sequential(
            nn.Linear(13 * 480, 925),
            nn.ReLU(inplace=True)
        )
        self.Linear2 = nn.Linear(925, TFs_cell_line_pair)

    def forward(self, input):
        x = self.Conv1(input)
        x = self.MaxPool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = self.MaxPool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = self.Drop2(x)
        x = x.view(-1, 13 * 480)
        return x


def make_model(NConv, NTrans, ms_num, d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    ms = Transformation(size=ms_num, dropout=dropout)
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = TransformerEncoder(convolution=Convolution(c(ms), N=NConv, dropout=dropout),
                               encoder=Encoder(layer=EncoderLayer(d_model, c(attn), c(ff), dropout), N=NTrans),
                               dim=ms_num, motifscanner=MotifScanner())
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
