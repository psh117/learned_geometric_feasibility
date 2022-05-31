from turtle import forward
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        nn.Module.__init__(self)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        nn.Module.__init__(self)

        self.MLP = nn.Sequential()

        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            # print (i, in_size, out_size)
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):
        x = self.MLP(z)

        return x

class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        nn.Module.__init__(self)

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size)

    def forward(self, x, c=None):
        # if x.dim() > 2:
        #     x = x.view(-1, 28*28)
        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None, z=None):

        batch_size = n
        if z is None:
            z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x

    def forward_once(self, x):
        means, log_var = self.encoder(x, None)

        std = torch.exp(0.5 * log_var)
        # eps = torch.randn([batch_size, self.latent_size])
        z = means

        recon_x = self.decoder(z, None)

        return recon_x, means, log_var, z

class FullyConnectedNet(nn.Module):
    def __init__(self, layer_sizes, batch_size):
        nn.Module.__init__(self)
        
        def init_weights(m):
            # print(m)
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                # print(m.weight)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            # self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            # print('i:',i)
            # print(len(layer_sizes))
            # print(in_size, out_size)
            if i+2 < len(layer_sizes):
                # self.MLP.add_module(name="BN{:d}".format(i), module=nn.BatchNorm1d(batch_size))
                self.MLP.add_module(
                    # name="D{:d}".format(i), module=nn.Dropout(0.5 -(i) * 0.15))
                    name="D{:d}".format(i), module=nn.Dropout(0.5))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                # print('sigmoid!')
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

        self.MLP.apply(init_weights)

    def forward(self, x):
        x = self.MLP(x)
        
        return x

class ModeNet(nn.Module):
    def __init__(self, fc_layer_sizes, batch_size, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        
        nn.Module.__init__(self)

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.local_occuapncy_vae = VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes)
        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size)

    def forward(self, x):
        """
        x: input pos(3), quat(4), voxel(4096:16x16x16)
        """
        x_pose = x[:,:7]
        x_voxel = x[:,7:]
        recon_x, mean, log_var, z = self.local_occuapncy_vae(x_voxel)
        x_fc = torch.cat([x_pose,z], dim=1)
        y = self.fc(x_fc)
        
        return y, x_voxel, recon_x, mean, log_var
        
class ModeNetVarSize(nn.Module):
    def __init__(self, fc_layer_sizes, batch_size, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        
        nn.Module.__init__(self)

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.local_occuapncy_vae = VAE(encoder_layer_sizes, latent_size, decoder_layer_sizes)
        self.fc = FullyConnectedNet(fc_layer_sizes,batch_size)

    def forward(self, x):
        """
        x: input pos(3), quat(4), voxel(4096:16x16x16)
        """
        x_pose_dim = x[:,:10]
        x_voxel = x[:,10:]
        recon_x, mean, log_var, z = self.local_occuapncy_vae(x_voxel)
        x_fc = torch.cat([x_pose_dim,z], dim=1)
        y = self.fc(x_fc)
        
        return y, x_voxel, recon_x, mean, log_var
        