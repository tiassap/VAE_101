import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def reparametrization(self):
        pass

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
    def __init__(self):
        pass

    def forward(self):
        pass

class Decoder(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass