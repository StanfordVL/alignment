import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict
from torch import nn
from torch.optim import Adam
from tianshou.data import to_torch, Batch

class WorldModel(nn.Module):
    def __init__(self, num_agent, layer_num, state_shape, hidden_units=128, device='cpu', wm_noise_level=0.0):
        super().__init__()
        self.device = device
        # plus one for the action
        self.model = [
            nn.Linear(np.prod(state_shape) + 1, hidden_units),
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, np.prod(state_shape))]
        self.num_agent = num_agent
        self.model = nn.Sequential(*self.model)
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.wm_noise_level = wm_noise_level

    def forward(self, s, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        if self.wm_noise_level != 0.0:
            logits += torch.normal(torch.zeros(logits.size()), self.wm_noise_level).to(logits.device)
        return logits

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        total_loss = np.zeros(self.num_agent)
        for i in range(self.num_agent):
            # concatenating each agent's observation and action
            inputs = np.concatenate((batch.obs[:, i], np.expand_dims(batch.act[:, i], axis=-1)), axis=1)
            next_obs_pred = self.model(to_torch(inputs, device=self.device, dtype=torch.float))
            true_next_obs = to_torch(batch.obs_next[:, i], device=self.device, dtype=torch.float)
            loss = F.mse_loss(next_obs_pred, true_next_obs)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss[i] = loss.item()
        # pass in dummy state and action
        output = {}
        for i in range(self.num_agent):
            output[f'models/actor_{i}'] = total_loss[i]
        output[f'loss/world_model'] = total_loss.sum()
        return output



