import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from subutils import clones
import matplotlib.pylab as plt
"""
    Scaled Dot product attention
"""



def attention(query, key, value,i, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    #visulize


    return torch.matmul(p_attn, value), p_attn#!!
"""
    Multi-Head Attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value,i ,mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to module h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do module the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on module the projected vectors in batch.
        x, self.attn = attention(query, key, value,i, mask=mask,
                                 dropout=self.dropout)
        #visualize
        # if i == 1:
        #     xx = x.mean(-1).view(1024,8,1,101)
        #     for n in range(self.attn.shape[0]):
        #         for h in range(self.attn.shape[1]):
        #             fig = plt.figure(figsize = (5,5))
        #             # 定义画布为1*1个划分，并在第1个位置上进行作图
        #             ax = fig.add_subplot(211)
        #             # 定义横纵坐标的刻度
        #             ax.set_yticks(range(101))
        #             # ax.set_yticklabels(yLabel)
        #             ax.set_xticks(range(101))
        #             # ax.set_xticklabels(xLabel)
        #             # 作图并选择热图的颜色填充风格，这里选择hot
        #             im = ax.imshow(self.attn[346, h, :, :].cpu().detach().numpy(), cmap=plt.cm.hot_r)
        #             # 增加右侧的颜色刻度条
        #             plt.colorbar(im)
        #             # 增加标题
        #             plt.title("n=" + str(n) + ",h=" + str(h))
        #             # show
        #             # plt.show()
        #
        #             # fig = plt.figure()
        #             # 定义画布为1*1个划分，并在第1个位置上进行作图
        #             ax = fig.add_subplot(212)
        #             # 定义横纵坐标的刻度
        #             ax.set_yticks(range(1))
        #             # ax.set_yticklabels(yLabel)
        #             ax.set_xticks(range(101))
        #
        #             # ax.set_xticklabels(xLabel)
        #             # 作图并选择热图的颜色填充风格，这里选择hot
        #             im = ax.imshow(xx[346, h, :,44:56].cpu().detach().numpy(), cmap=plt.cm.hot_r)
        #             # 增加右侧的颜色刻度条
        #             plt.colorbar(im)
        #             # 增加标题
        #             # plt.title("n=" + str(n) + ",h=" + str(h))
        #             # show
        #             plt.show()

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


