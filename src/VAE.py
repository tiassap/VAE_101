import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def load_data(self):
        pass

    def encode(self):
        pass

    def decode(self):
        pass

    def train(self):
        pass

    def calc_loss(self):
        pass

    def save_model(self):
        pass

    def plot_score(self):
        pass

    def visualize(self):
        pass


class Encoder(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        encoder = None
    
    def reparametrization(self):
        pass

    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass