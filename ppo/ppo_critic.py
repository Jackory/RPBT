import torch
import torch.nn as nn

from util.util_mlp import MLPBase, MLPLayer
from util.util_gru import GRULayer
from util.util_act import ACTLayer
from util.util_util import check


class PPOCritic(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(PPOCritic, self).__init__()
        # network config
        self.hidden_size = args.hidden_size
        self.act_hidden_size = args.act_hidden_size
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_hidden_size = args.recurrent_hidden_size
        self.recurrent_hidden_layers = args.recurrent_hidden_layers
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.use_cnn = args.use_cnn
        # (1) feature extraction module
        if self.use_cnn: # just hard code obs_dim 4 * 40 * 40
            self.base = nn.Sequential(
            nn.Conv2d(4, 8, 4, stride=2), # (40+2-4)/2+1 = 20
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(8, 16, kernel_size=2, stride=1), # (20-2)/2 + 1 = 10
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Flatten(),
            nn.Linear(16*7*7, 128),
            nn.ReLU()
        )
            input_size = 128
        else:
            self.base = MLPBase(obs_space, self.hidden_size, self.activation_id, self.use_feature_normalization)
            input_size = self.base.output_size
        # (2) rnn module
        if self.use_recurrent_policy:
            self.rnn = GRULayer(input_size, self.recurrent_hidden_size, self.recurrent_hidden_layers)
            input_size = self.rnn.output_size
        # (3) value module
        if len(self.act_hidden_size) > 0:
            self.mlp = MLPLayer(input_size, self.act_hidden_size, self.activation_id)
        self.value_out = nn.Linear(input_size, 1)

        self.to(device)

    def forward(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(obs)

        if self.use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        if len(self.act_hidden_size) > 0:
            critic_features = self.mlp(critic_features)

        values = self.value_out(critic_features)

        return values, rnn_states
