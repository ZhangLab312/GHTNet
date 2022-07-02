import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

"""
  C-FeedForward-Network 
"""


class PositionwiseFeedForward(nn.Module):
    """
        Implements C-FFN
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
            [d_model, d_ff]
        """
        self.w_1 = nn.Linear(d_model, d_ff)
        self.conv = nn.Conv1d(in_channels=d_ff,out_channels=2*d_ff,kernel_size=3,padding=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2,stride=2)
        """
            [d_ff, d_model]
        """
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.maxpool(torch.transpose(self.dropout(self.conv(torch.transpose(self.dropout(F.relu(self.w_1(x))),1,2))),1,2)))


