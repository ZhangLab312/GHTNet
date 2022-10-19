import torch.nn as nn
import copy
from convolutions import Convolution, MotifScanner
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
        注：
        DNA中各个核苷酸位点 按着自己的路径单独经过 Encoder中两个 layer
        Attention中这些路径间存在依赖关系  但是 FFN中不具有这样的依赖关系 因此各路径可在经过FFN时同时做运算
        在模型处理每个核苷酸时（DNA中的每个位置），Self-Attention让其可以查看input的DNA中的其它位置，以寻找思路来更好的对该位置Encode
    """
    """
        A standard Transformer-Encoder architecture
        Base for this and many other models.
    """

    def __init__(self, convolution, encoder, pos_embed, generator, dim, d_model, cnn):
        super(TransformerEncoder, self).__init__()
        self.convolution = convolution
        self.encoder = encoder
        self.pos_embed = pos_embed
        self.generator = generator
        self.dim = dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_latent = nn.Identity()
        self.position = nn.Parameter(torch.zeros(1,101,self.dim ))
        self.p = nn.Parameter(torch.zeros(1, 4, 101))
        self.deep = cnn
        self.cs = nn.Sequential(
            nn.Conv1d(18,32, kernel_size=8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
            nn.Conv1d(32,64, kernel_size=8),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),
        )

        self.fc = nn.Sequential(

            nn.Linear(7520, 925),#7520
            nn.Dropout(p=0.2),
            nn.LayerNorm(925),
            # nn.BatchNorm1d(num_features=925),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(925, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, src, src_mask,data):
        """
            Take in and process masked src sequences
        """

        x = self.convolution(torch.transpose(src,1,2))#)
        b, n, _ = x.shape
        x = x+self.position#
        x = self.encode(x, src_mask)
        if data != None:
            cs = self.cs(torch.transpose(data, 1, 2))
            cs = cs.view(-1,20*64)
            x = self.deep(torch.transpose(x, 1, 2))
            x = torch.cat([x, cs], 1)
            x = self.fc(x)


        return x#self.generator(x)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)#self.pos_embed(

class CNN(nn.Module):
    def __init__(self, TFs_cell_line_pair=2):
        super(CNN, self).__init__()

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
        """
            shape:
            [batch_size, 4, 101] - > [batch_size, 160, 94]
        """
        x = self.Conv1(input)
        # pool_seq_1 = self.max_pool(x).squeeze()
        # n = torch.argmax(pool_seq_1, 1).cpu()
        # a = n.tolist()
        # dic = collections.Counter(a)
        # # Counter({'Cuckoo': 4, 'Albatross': 3, 'Yellow_breasted_Chat': 3})
        # print(dic)
        # for key in dic:
        #     print(key, dic[key])
        # """
        #     shape:
        #     [batch_size, 160, 94] - > [batch_size, 160, 47]
        # """
        # conv_seq_1 = x[:, 50]
        # n = torch.argmax(conv_seq_1, 1).cpu().tolist()
        # f = open('position.txt', 'a')
        # for position in n:
        #     print(position)
        #
        #     f.write(str(position) + "\n")
        # f.close()
        # print(n)
        #
        # # pool_seq_1 = self.max_pool(conv_seq_1).squeeze()
        # sequence = pool_seq_1[:, 113]
        # average = torch.mean(pool_seq_1, 0)
        # n = torch.argmax(pool_seq_1, 1).cpu()
        # average = average[50]
        # # p = pool_seq_1[:,33]
        # a = n.tolist()
        # b = []
        # f = open('sequen1.txt', 'a')
        # for i in range(len(sequence)):
        #     if sequence[i] >= average * 0.1:
        #         f.write(str(i) + "\n")
        # f.close()
        x = self.MaxPool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = self.MaxPool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = self.Drop2(x)
        x = x.view(-1, 13 * 480)

        return x


class Generator(nn.Module):
    """
        The output layer
    """

    def __init__(self, d_model, TFs_cell_line_pair, dim):
        super(Generator, self).__init__()
        """
          
        """
        self.proj = nn.Sequential(
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(d_model,  2),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        """
            x -> [batch_size, TFs_cell_line_pair]
        """
        x = torch.flatten(x, start_dim=1)
        return self.proj(x)


def make_model(TFs_cell_line_pair, NConv, NTrans, ms_num, d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    ms = MotifScanner(size=ms_num, dropout=dropout)
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(1, dropout)
    model = TransformerEncoder(convolution=Convolution(c(ms), N=NConv, dropout=dropout),
                               encoder=Encoder(layer=EncoderLayer(d_model, c(attn), c(ff), dropout), N=NTrans),
                               pos_embed=c(position),
                               generator=Generator(d_model=d_model, TFs_cell_line_pair=TFs_cell_line_pair, dim=ms_num),
                               dim=ms_num, d_model=d_model, cnn=CNN())
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
