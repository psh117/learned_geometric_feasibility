import torch
import torch.nn as nn

from lgf.models.fully_connected_vector import FullyConnectedVector
from lgf.models.vae import VAE

class ModeNet(nn.Module):
    def __init__(self, voxel_dim, 
                 voxel_latent_dim=16, 
                 encoder_hidden_dims=[256,256], 
                 decoder_hidden_dims=[256,256],
                 x_dim=7,
                 mode_dim=5,
                 mode_hidden_dims=[512, 512, 512],
                 beta=1.0,
                 dropout=0.5,
                 bce_loss_weight=[1.25, 1.0]):

        super(ModeNet, self).__init__()
        self.x_dim = x_dim
        self.voxel_dim = voxel_dim
        self.beta = beta
        self.bce_loss_weight = bce_loss_weight

        print('beta: ', self.beta)

        self.local_occuapncy_vae = VAE(state_dim=voxel_dim, 
                                       latent_dim=voxel_latent_dim, 
                                       encoder_hidden_dims=encoder_hidden_dims, 
                                       decoder_hidden_dims=decoder_hidden_dims)
        
        self.fc = FullyConnectedVector(layer_dims=[x_dim+voxel_latent_dim]+mode_hidden_dims+[mode_dim],
                                       dropouts=[nn.Dropout(p=dropout)] * (len(mode_hidden_dims)) + [None],
                                       activations=[nn.ReLU()] * len(mode_hidden_dims) + [nn.Sigmoid()])

    def forward(self, x):
        """
        x: input (x_dim), voxel(voxel_dim)
        """
        x_pose = x[:,:self.x_dim]
        voxel = x[:,self.x_dim:]
        
        recon_voxel, mean, log_var, z = self.local_occuapncy_vae(voxel)

        x_fc = torch.cat([x_pose,z], dim=1)
        y = self.fc(x_fc)
        
        return y, voxel, recon_voxel, mean, log_var
    
    def inference(self, x):
        """
        x: input (x_dim), voxel(voxel_dim)
        """
        x_pose = x[:,:self.x_dim]
        voxel = x[:,self.x_dim:]
        
        z = self.local_occuapncy_vae.encode(voxel)

        x_fc = torch.cat([x_pose,z], dim=1)
        y = self.fc(x_fc)
        
        return y

    
    def loss_fc(self, y, y_hat):
        # loss_fc = torch.nn.functional.binary_cross_entropy(y_hat, y, reduction='mean') # sum -> mean, divison remove

        weights = self.bce_loss_weight
        y_hat = torch.clamp(y_hat,min=1e-7,max=1-1e-7)
        bce = - weights[1] * y * torch.log(y_hat) - (1 - y) * weights[0] * torch.log(1 - y_hat)
        return torch.mean(bce)



        # def loss(input, target):


        # return loss_fc
    
    def loss(self, voxel, recon_voxel, y, y_hat, mean, log_var, prefix=''):
        """
        x: input (x_dim), voxel(voxel_dim)
        y: output (mode_dim)
        recon_x: reconstructed voxel (voxel_dim)
        mean: mean of latent vector (voxel_latent_dim)
        log_var: log variance of latent vector (voxel_latent_dim)
        """
        loss_vae = self.local_occuapncy_vae.loss(recon_voxel, voxel, mean, log_var, self.beta)
        loss_fc = self.loss_fc(y, y_hat)

        loss = loss_vae + loss_fc
        
        losses = {prefix+'loss': loss, prefix+'loss_vae': loss_vae, prefix+'loss_fc': loss_fc}

        return losses
    
        
class ModeNetGeometric(ModeNet):
    def __init__(self, 
                 pose_dim=7,
                 geometric_dim=3,
                 **kwargs):
        
        ModeNet.__init__(self, 
                         x_dim=pose_dim+geometric_dim,
                         **kwargs)
        

class ModeNetBox(ModeNetGeometric):
    def __init__(self, 
                 **kwargs):
        
        geometric_dim = 3
        mode_dim = 12
        
        ModeNetGeometric.__init__(self, 
                                  geometric_dim=geometric_dim,
                                  mode_dim=mode_dim,
                                  **kwargs)