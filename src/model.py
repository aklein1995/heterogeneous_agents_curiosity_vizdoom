import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# feature_output = 1152
feature_output = 288
class Flatten(nn.Module):
    def forward(self, input):
        # print('inp shape:',input.shape)
        # print('sale:',input.view(input.size(0), -1).shape)
        return input.view(input.size(0), -1)

def CNN(input_size):
    """
        CNN design help:
        https://discuss.pytorch.org/t/runtimeerror-calculated-padded-input-size-per-channel-2-x-18-kernel-size-3-x-3-kernel-size-cant-be-greater-than-actual-input-size/55978
        https://cs231n.github.io/convolutional-networks/
        Produces a volume of size W2×H2×D2 where:
        -W2=(W1−F+2P)/S+1
        -H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
        -D2=K
    """
    return nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            Flatten(),
        )

class Critic_LSTM(nn.Module):
    def __init__(self, input_size, output_size,
                use_lstm = True, hidden_size=256, bidirectional = False,
                use_role_as_input = False, use_action_as_input = False,
                device = 'cpu'):
        super(Critic_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.use_lstm = use_lstm
        self.bidirectional = bidirectional
        self.action_size = output_size
        self.device = device

        self.use_action_as_input = use_action_as_input
        self.use_role_as_input = use_role_as_input

        self.additional_neurons = 0
        if self.use_action_as_input:
            self.additional_neurons += 5 # max of 5 actions cosidered
        if self.use_role_as_input:
            self.additional_neurons += 2 # two agent type considered

        self.feature = CNN(input_size)
        self.fc = nn.Sequential(nn.Linear(feature_output, 256), nn.ELU())

        # output of feature extractor
        output_feature_extractor = 256+self.additional_neurons

        # LSTM
        self.lstm = nn.LSTM(output_feature_extractor, self.hidden_size, bidirectional = self.bidirectional)
        self.hidden_lstm_state = (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
                                  torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))

        #  Additional Dense layer
        input_neurons_extra_layer = hidden_size*2 if self.bidirectional else hidden_size
        self.extra_layer = nn.Sequential(nn.Linear(input_neurons_extra_layer, 256),nn.ELU())

        # Critic heads
        self.critic_ext = nn.Linear(256, self.action_size)
        self.critic_int = nn.Linear(256, self.action_size)

        # ***Weights initialization
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                try:
                    p.bias.data.zero_()
                except AttributeError:
                    pass

    def _hotencode(self,aux_list,neuron):
        identity = np.identity(neuron)
        hot_encoded = torch.from_numpy(np.asarray([identity[a] for a in aux_list]))
        return hot_encoded.to(self.device).float()

    def forward(self, state, hidden, action=[],role=[],memory_attention=[]):
        # Feature extractor layers
        x = self.feature(state)
        x = self.fc(x)

        # concat if necessary additional features after features
        if self.use_action_as_input and self.use_role_as_input:
            action = self._hotencode(action,5)
            r = self._hotencode(role,2)
            x = torch.cat((x,action),dim=1)
            x = torch.cat((x,r),dim=1)
        elif self.use_action_as_input:
            action = self._hotencode(action,5)
            x = torch.cat((x,action),dim=1)
        elif self.use_role_as_input:
            role = self._hotencode(role,2)
            x = torch.cat((x,role),dim=1)

        #LSTM -- input of shape (seq_len, batch, input_size)
        x, self.hidden_lstm_state = self.lstm(x.view(-1, 1, 256+self.additional_neurons), hidden)

        # Final Fully connected
        value_ext = self.critic_ext(self.extra_layer(x))
        value_ext = value_ext.squeeze(1).squeeze(1)

        value_int = self.critic_int(self.extra_layer(x))
        value_int = value_int.squeeze(1).squeeze(1)

        return value_ext, value_int, self.hidden_lstm_state

    def getLSTMHiddenState(self):
        return self.hidden_lstm_state

    def init_hidden(self,batch_size=1):
        self.hidden_lstm_state = (torch.zeros(1 + int(self.bidirectional), batch_size, self.hidden_size),
                                  torch.zeros(1 + int(self.bidirectional), batch_size, self.hidden_size))
        return self.hidden_lstm_state

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        # =============================================================================
        # Feature extractor layers
        # =============================================================================
        self.feature = CNN(input_size)
        self.fc = nn.Sequential(nn.Linear(feature_output, 256), nn.ELU())

        # Actor network part
        self.actor = nn.Sequential(
            nn.Linear(256, output_size),
            nn.Softmax(dim=1)
        )

        # ***Weights initialization
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward(self, state):
        x = self.feature(state)
        x = self.fc(x)
        policy = self.actor(x)
        return policy

class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNDModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2,padding=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(feature_output, 512),
        )

        # ***Weights initialization
        for p in self.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
