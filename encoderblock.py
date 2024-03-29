import torch.nn as nn
from subutils import clones, LayerNorm, SublayerConnection

"""
    Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
"""


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        i=0
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask,i)
            i+=1
        if x.size(-1)!=1:
            x = self.norm(x)
        return x


"""
    EncoderLayer(d_model, c(attn), c(ff), dropout)
"""


class EncoderLayer(nn.Module):
    """
        Encoder is made up of self-attn and feed forward
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask,i):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, i,mask))
        return self.sublayer[1](x, self.feed_forward)
