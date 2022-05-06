import torch.nn.functional as F
import torch
from torch import nn
import numpy as np
from wildfire_forecasting.models.modules.convlstm import ConvLSTM
from torchvision.models.resnet import resnet18
from fastai.vision.models import unet

np.seterr(divide='ignore', invalid='ignore')

class SimpleConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        dropout = hparams['dropout']
        kernel_size = 3
        self.ln1 = torch.nn.LayerNorm(input_dim)
        # clstm part
        self.convlstm = ConvLSTM(input_dim,
                                 hidden_size,
                                 (kernel_size, kernel_size),
                                 lstm_layers,
                                 True,
                                 True,
                                 False, dilation=1)

        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(kernel_size, kernel_size), stride=(1, 1),
                               padding=(1, 1))
        # fully-connected part
        self.fc1 = nn.Linear((25 // 2) * (25 // 2) * hidden_size, 2 * hidden_size)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor):
        # (b x t x c x h x w) -> (b x t x h x w x c) -> (b x t x c x h x w)
        x = self.ln1(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        # cnn
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # fully-connected
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop2(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        if hparams['clc'] == 'vec':
            input_dim += 10
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        dropout = hparams['dropout']
        self.ln1 = torch.nn.LayerNorm(input_dim)
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        # fully-connected part
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop1 = torch.nn.Dropout(dropout)

        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.drop2 = torch.nn.Dropout(dropout)

        self.fc3 = torch.nn.Linear(hidden_size // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.drop1,
            self.fc2,
            self.relu,
            self.drop2,
            self.fc3
        )

    def forward(self, x: torch.Tensor):
        x = self.ln1(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)