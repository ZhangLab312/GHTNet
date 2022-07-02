import torch.nn as nn
import torch


class Convolution(nn.Module):
    def __init__(self, ms_layer, dropout, N):
        super(Convolution, self).__init__()
        self.layer_num = N

        self.conv_layers = nn.ModuleList()

        self.size = 0

        if self.layer_num == 1:
            self.ms_layer = ms_layer
            self.size = ms_layer.size
        else:
            self.ms_layer = ms_layer

            self.size = self.ms_layer.size * 2

            for i in range(self.layer_num - 1):
                self.conv_layers.append(ConvolutionLayer(size=self.size, dropout=dropout))
                self.size = self.size * 2
            self.size = self.size / 2

    def forward(self, x):
        """
            ms_layer:
            [batch_size, 4, seq_len] -> [batch_size, ms_layer.size, seq_len]
        """
        if self.layer_num == 1:
            x = self.ms_layer(x)
        else:
            x = self.ms_layer(x)
            for layer in self.conv_layers:
                x = layer(x)
        """
            [batch_size, seq_len, layer.size]
        """
        return x


class Transformation(nn.Module):
    def __init__(self, size, dropout):
        super(Transformation, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size,size ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(size, size),
            nn.ReLU(),
        )
    def forward(self, x):
        '''
            Linear transformation
        '''
        x = self.fc(x)
        return x


class ConvolutionLayer(nn.Module):
    def __init__(self, size, dropout):

        super(ConvolutionLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=size // 2, out_channels=size, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=size),
            nn.Dropout(p=dropout)
        )
        self.size = size

    def forward(self, x):
        return self.conv(x)
