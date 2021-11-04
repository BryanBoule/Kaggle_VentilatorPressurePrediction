import torch.nn as nn
from helpers.methods import Swish

act_function = Swish()


class GRU_archi(nn.Module):
    def __init__(
            self,
            input_dim,
            lstm_dim,
            dense_dim,
            logit_dim,
            num_classes=1,
    ):
        super().__init__()

        self.classic_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=dense_dim),
            act_function,
        )

        self.GRU_layer_1 = nn.GRU(
            input_size=dense_dim,
            hidden_size=lstm_dim // 2,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        self.GRU_layer_2 = nn.GRU(
            input_size=lstm_dim,
            hidden_size=lstm_dim // 4,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        self.GRU_layer_3 = nn.GRU(
            input_size=lstm_dim // 2,
            hidden_size=lstm_dim // 8,
            bidirectional=True,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(in_features=lstm_dim // 4, out_features=logit_dim),
            nn.Linear(in_features=logit_dim, out_features=num_classes)
        )

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.classic_layer(x)
        x, _ = self.GRU_layer_1(x)
        x, _ = self.GRU_layer_2(x)
        x, _ = self.GRU_layer_3(x)
        pred = self.output_layer(x)
        return pred
