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

    def encode(self, input_vector):
        return self.encoder(input_vector)

    def decode(self, latent_vector):
        return self.decoder(latent_vector)

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
        self.initialize_weight()
    
    def reparametrization(self):
        pass

    def forward(self):
        pass

    def initialize_weight(self):
        pass

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = None
        self.initialize_weight()

    def forward(self):
        pass

    def initialize_weight(self):
        pass