import torch
import torch.nn as nn
import torch.distributions as ptd

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.config = config

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def load_data(self):
        pass

    def encode(self, input_vector):
        return self.encoder(input_vector)

    def decode(self, latent_vector):
        return self.decoder(latent_vector)

    def train(self):
        for epoc in self.config["hyperparameters"]["num_epoch"]:
            for data, _ in train.dataLoader:
                pass

    def calc_loss(self, x, x_hat, mean, log_var):
        bce_loss = nn.BCELoss(reduction='sum')
        reconstruction_loss = bce_loss(x_hat, x)
        kl_loss = 0.5* torch.sum(torch.exp(log_var) + torch.pow(mean,2) - 1 - log_var)
        return reconstruction_loss, kl_loss

    def save_model(self):
        pass

    def plot_score(self):
        pass

    def visualize(self):
        pass


class Encoder(nn.Module, VAE):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_layer = None
        self.mean_layer = None
        self.log_var_layer = None
        self.initialize_weight()
    
    def reparametrization(self, mean, std):
        dist = ptd.Normal(mean, std)
        return dist.sample()

    def forward(self, x):
        x = self.hidden_layer(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        std = torch.sqrt(torch.exp(log_var))

        return self.reparametrization(mean, std), mean, log_var

    def initialize_weight(self):
        pass

class Decoder(nn.Module, VAE):
    def __init__(self):
        super().__init__()
        self.decoder = None
        self.initialize_weight()

    def forward(self, x):
        return self.decoder(x)

    def initialize_weight(self):
        pass