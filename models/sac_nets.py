import torch
import numpy as np
from torch import nn

from tianshou.data import to_torch


class Actor(nn.Module):

    def __init__(self, layer_num, state_shape, action_shape,
                 softmax=True, hidden_units=128, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear(np.prod(state_shape), hidden_units),
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, np.prod(action_shape))]
        if softmax:
            self.model += [nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*self.model)
        self.initialize()

    def forward(self, s, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits, None

    def initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)


class Critic(Actor):

    def __init__(self, layer_num, state_shape, action_shape=0,
                 hidden_units=128, device='cpu'):
        # Same as Actor but without the softmax.
        super().__init__(layer_num, state_shape, action_shape,
                         softmax=False, hidden_units=hidden_units,
                         device=device)

    def forward(self, s, **kwargs):
        return super().forward(s, **kwargs)[0]


class CentralizedCritic(nn.Module):
    def __init__(self, layer_num, state_shape, action_shape, num_agts,
                 hidden_units=128, device='cpu'):
        super().__init__()
        self.device = device
        self.model = [
            nn.Linear((np.prod(state_shape) + 1) * num_agts, hidden_units),
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, np.prod(action_shape))]
        self.model = nn.Sequential(*self.model)
        self.initialize()

    def forward(self, s, a, **kwargs):
        batch = s.shape[0]
        s = to_torch(s, device=self.device, dtype=torch.float).view(batch, -1)
        a = to_torch(a, device=self.device, dtype=torch.float).view(batch, -1)
        s = torch.cat((s, a), dim=-1)
        logits = self.model(s)
        return logits

    def initialize(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)

class ConditionalActor(Actor):

    def __init__(self, layer_num, state_shape, strategy_num,
                 action_shape, softmax=True, hidden_units=128,
                 device='cpu'):
        super().__init__(
                layer_num, np.prod(state_shape) + strategy_num,
                action_shape, softmax=softmax, hidden_units=hidden_units,
                device=device)
        self.strategy_num = strategy_num

    def forward(self, s, z, **kwargs):
        batch = s.shape[0]
        b = np.zeros((batch, self.strategy_num))
        b[np.arange(batch), z] = 1
        s = np.concatenate([s, b], axis=1)
        output = super().forward(s, **kwargs)
        return output


class ConditionalCritic(ConditionalActor):

    def __init__(self, layer_num, state_shape, strategy_num,
                 action_shape, hidden_units=128, device='cpu'):
        # Same as ConditionalActor but without the softmax.
        super().__init__(layer_num, state_shape, strategy_num,
                         action_shape, softmax=False,
                         hidden_units=hidden_units,
                         device=device)

    def forward(self, s, z, **kwargs):
        return super().forward(s, z, **kwargs)[0]
