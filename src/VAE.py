import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from utils.general import get_data_loaders, device
import tqdm

class VAE():
    def __init__(self, config):
        in_feature_dim = config["hyperparameters"]["in_feature_dim"]
        hidden_dim = config["hyperparameters"]["hidden_dim"]
        latent_dim = config["hyperparameters"]["latent_dim"]
        self.encoder = Encoder(in_feature_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(in_feature_dim, hidden_dim, latent_dim)
        self.config = config
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.optimizer = torch.optim.Adam(
            nn.ModuleList(self.encoder.parameters(), self.decoder.parameters()),
            lr=config["hyperparameters"]["learning_rate"]
            )

    def load_data(self):
        pass

    def encode(self, input_vector):
        return self.encoder(input_vector)

    def decode(self, latent_vector):
        return self.decoder(latent_vector)

    def train(self):
        train_loader = get_data_loaders(
            data_root=None,
            batch_size=None,
            num_workers=None,
        )
        epoch_train_loss = np.array([])
        best_loss = torch.tensor(np.inf)

        for epoch in tqdm(range(self.config["hyperparameters"]["num_epoch"])):
            train_loss = self.train_step(train_loader)
            epoch_train_loss = np.append(epoch_train_loss, [train_loss])
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model()

            

    def train_step(self, train_loader):
        self.encoder.train()
        self.decoder.train()
        batch_loss = np.array([])
        for data, _ in train_loader:
            data = data.view(self.config["hyperparameters"]["batch_size"], -1)
            data = data.to(device)
            self.optimizer.zero_Grad()
            x_hat, mean, log_var = self.decoder(self.encoder(data))
            reconstruction_loss, kl_loss = self.calc_loss(data, x_hat, mean, log_var)
            loss = reconstruction_loss + kl_loss
            loss.backward()
            self.optimizer.step()
            batch_loss = np.rappend(batch_loss, [loss.item()])
        
        epoch_loss = batch_loss.mean()

        return epoch_loss


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


class Encoder(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_layer = nn.Sequantial(
            nn.Linear(in_feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.mean_layer = nn.Linear(self.hidden_dim,self.latent_dim)
        self.log_var_layer = nn.Linear(self.hidden_dim,self.latent_dim)
        self.apply(self._init_weights)
    
    def reparametrization(self, mean, std):
        dist = ptd.Normal(mean, std)
        return dist.sample()

    def forward(self, x):
        x = self.hidden_layer(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        std = torch.sqrt(torch.exp(log_var))

        return self.reparametrization(mean, std), mean, log_var

    def _init_weights(self, module):
        pass

class Decoder(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace= True),
            nn.Linear(hidden_dim, in_feature_dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.decoder(x)

    def _init_weights(self, module):
        pass