import torch
import torch.nn as nn

from lgf.models.fully_connected_vector import FullyConnectedVector

class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dims=[512, 512]):
        nn.Module.__init__(self)

        self.MLP = FullyConnectedVector(layer_dims=[state_dim]+hidden_dims,
                                        activations=[nn.ReLU()] * len(hidden_dims))

        self.linear_means = nn.Linear(hidden_dims[-1], latent_dim)
        self.linear_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dims=[512, 512]):
        nn.Module.__init__(self)


        self.MLP = FullyConnectedVector(layer_dims=[latent_dim]+hidden_dims+[state_dim],
                                        activations=[nn.ReLU()] * len(hidden_dims) + [nn.Sigmoid()])

    def forward(self, z):
        x = self.MLP(z)

        return x

class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim, encoder_hidden_dims, decoder_hidden_dims):
        nn.Module.__init__(self)

        self.latent_dim = latent_dim

        self.encoder = Encoder(state_dim=state_dim, latent_dim=latent_dim, hidden_dims=encoder_hidden_dims)
        self.decoder = Decoder(state_dim=state_dim, latent_dim=latent_dim, hidden_dims=decoder_hidden_dims)

    def forward(self, x):
        batch_size = x.size(0)

        means, log_var = self.encoder(x)
        
        device = self.encoder.linear_means.weight.device

        std = torch.exp(0.5 * log_var).to(device)
        eps = torch.randn([batch_size, self.latent_dim]).to(device)

        z = eps * std + means

        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def loss(self, recon_x, x, means, log_var, beta=1.0):
        bs = x.size(0)
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())

        return (BCE + beta * KLD) / bs # VAE

    def encode(self, x):
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        x = self.decoder(z)
        return x
    
