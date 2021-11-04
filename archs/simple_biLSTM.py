import torch.nn as nn
from helpers.methods import Swish

act_function = Swish()


class LSTM_archi(nn.Module):
    def __init__(
            self,
            # nb of expected features
            input_dim,
            lstm_dim,
            dense_dim,
            logit_dim,
            num_classes=1,
    ):
        super().__init__()
        self.classic_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=2 * (dense_dim // 3)),
            nn.ReLU(),
            # act_function,
        )

        self.upscale_layer = nn.Sequential(
            nn.Linear(in_features=2 * (dense_dim // 3), out_features=dense_dim),
            # act_function,
            nn.ReLU(),
        )

        self.LSTM_layer = nn.LSTM(
            input_size=dense_dim,
            hidden_size=lstm_dim,
            bidirectional=True,
            num_layers=4,
            # then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature).
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=dense_dim * 2, out_features=logit_dim),
            # act_function,
            # nn.ReLU(),
            nn.Linear(in_features=logit_dim, out_features=num_classes)
        )

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.classic_layer(x)
        # x = self.dropout(x)
        x = self.upscale_layer(x)
        # x = self.dropout(x)
        x, _ = self.LSTM_layer(x)
        # x = self.dropout(x)
        pred = self.output_layer(x)
        return pred
