import torch
import torch.nn as nn
import torch.distributions as ptd
from torch.optim import Adam 
import numpy as np
from utils.general import *
import time


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
        self.optimizer = Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=config["hyperparameters"]["learning_rate"]
            )
        self.logger = get_logger(config["output"]["log_path"])

    def load_data(self):
        pass

    def encode(self, input_vector):
        return self.encoder(input_vector)

    def decode(self, latent_vector):
        return self.decoder(latent_vector)

    def train(self):
        self.logger.info("Training start.. Using {}".format(device))
        time_start = time.time()
        train_loader = get_data_loaders(
            data_root = "./dataset/",
            batch_size = self.config["hyperparameters"]["batch_size"],
            num_workers = 4,
        )
        self.epoch_train_loss = np.array([])
        best_loss = torch.tensor(np.inf)

        for epoch in range(self.config["hyperparameters"]["num_epoch"]):
            train_loss = self.train_step(train_loader)
            self.epoch_train_loss = np.append(self.epoch_train_loss, [train_loss])
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model()
                self.logger.info("Saving model..")
            h, m, s = time_message(time.time() - time_start)
            self.logger.info("Epoch {} | Training loss: {:.2f} | Elapsed time: {:02d}:{:02d}:{:02d}".format(epoch, train_loss, h, m, s))
        h, m, s = time_message(time.time() - time_start)
        self.logger.info("Training finished. Total training time: {:02d}:{:02d}:{:02d}".format(h, m, s) + "\n")
        export_plot(self.epoch_train_loss, "Training loss", self.config["name"]+self.config["config_no"], self.config["output"]["plot_output"])
        self.visualize()
        
    def train_step(self, train_loader):
        self.encoder.train()
        self.decoder.train()
        batch_loss = np.array([])
        for data, _ in train_loader:
            data = data.view(self.config["hyperparameters"]["batch_size"], -1)
            data = data.to(device)
            self.optimizer.zero_grad()
            latent_vec, mean, log_var = self.encoder(data)
            x_hat = self.decoder(latent_vec)
            reconstruction_loss, kl_loss = self.calc_loss(data, x_hat, mean, log_var)
            loss = reconstruction_loss + kl_loss
            loss.backward()
            self.optimizer.step()
            batch_loss = np.append(batch_loss, [loss.item()])
        return batch_loss.mean()


    def calc_loss(self, x, x_hat, mean, log_var):
        bce_loss = nn.BCELoss(reduction='sum')
        reconstruction_loss = bce_loss(x_hat, x)
        kl_loss = 0.5* torch.sum(torch.exp(log_var) + torch.pow(mean,2) - 1 - log_var)
        return reconstruction_loss, kl_loss

    def save_model(self):
        torch.save(self.encoder.state_dict(), self.config["output"]["encoder_model"])
        torch.save(self.decoder.state_dict(), self.config["output"]["decoder_model"])
        np.save(self.config["output"]["scores_output"], self.epoch_train_loss)

    def plot_score(self):
        pass

    def visualize(self):
        n = 16
        x_axis = np.linspace(-3, 3, n)
        y_axis = np.linspace(-3, 3, n)

        canvas = np.empty((28 * n, 28 * n))
        for i, yi in enumerate(x_axis):
            for j, xi in enumerate(y_axis):
                z_mu = np.array([[xi, yi]])
                z_mu = torch.tensor(z_mu, device=device).float()
                out = self.decoder(z_mu)
                out = out.view(28,28).detach().cpu().numpy()
                canvas[i*28:(i+1)*28,j*28:(j+1)*28] = out
                
        plt.figure(figsize=(10, 10))
        plt.title('Effect of latent variable change.')
        plt.imshow(canvas, cmap="gray")
        plt.savefig(self.config["output"]["figure_output"])
        # plt.show()


class Encoder(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_var_layer = nn.Linear(hidden_dim, latent_dim)
        # self.apply(self._init_weights)
    
    def reparametrization(self, mean, log_var):
        # std = torch.sqrt(torch.exp(log_var))
        # dist = ptd.Normal(mean, std**2)
        # return dist.sample()
        std = torch.exp(0.5*log_var)
        epsilon = torch.randn_like(std)
        z = mean + epsilon*std
        
        return z

    def forward(self, x):
        x = self.hidden_layer(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        
        return self.reparametrization(mean, log_var), mean, log_var

    def _init_weights(self, module):
        pass ## To be updated ##

class Decoder(nn.Module):
    def __init__(self, in_feature_dim, hidden_dim, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace= True),
            nn.Linear(hidden_dim, in_feature_dim),
            nn.Sigmoid()
        )
        # self.apply(self._init_weights)

    def forward(self, x):
        return self.decoder(x)

    def _init_weights(self, module):
        pass ## To be updated ##