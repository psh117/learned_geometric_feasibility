from __future__ import division
import os
import time
import copy
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from torchvision import transforms
import numpy as np
from models import ModeNet

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ModeNetDataset(Dataset):
    """
    data pickle contains dict
        'grid_size' : the size of grid
        'grid_dim'  : total number of the grid dimension
        'pose'      : object pose
        'occupancy' : occupancy data (grid_dim)
        'mode'      : mode data
    """
    def __init__(self, file_name,):
        def data_load():
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
            self.grid_size = dataset['grid_size']
            self.grid_dim = dataset['grid_dim']
            return dataset['pose'], dataset['occupancy'], dataset['mode']
        self.pose, self.occupancy, self.mode = data_load()
        print ('grid_dim:', self.grid_dim)

    def __len__(self):
        return len(self.pose)

    def get_grid_len(self):
        return self.grid_size 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return np.array(self.occupancy[idx],dtype=np.float32), np.array(self.pose[idx],), np.array(self.mode[idx],dtype=np.float32)

def main(args):
    vae_latent_size = args.vae_latent_size
    
    mode_num = args.mode_num

    directory = args.dataset
    log_dir = 'log/' + directory +'/'
    chkpt_dir = 'model/checkpoints/' + directory + '/'

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
    
    suffix = 'lat{}_rnd{}'.format(vae_latent_size, args.seed)

    file_name = "dataset/{}.pkl".format(args.dataset)
    log_file_name = log_dir + 'log_{}'.format(suffix)
    model_name = '{}_{}'.format(args.dataset, suffix)

    """
    layer size = [7+len(z), hidden1, hidden2, mode_num]
    """
    layer_size = [7+vae_latent_size, 256, 256, 256] + [mode_num]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    ts = time.time()

    print('loading data ...')
    read_time = time.time()
    dataset = ModeNetDataset(
        file_name=file_name)
    n_grid_3 = dataset.get_grid_len()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset))
    end_time = time.time()
    
    print('data load done. time took {0}'.format(end_time-read_time))
    print('[data len] total: {} train: {}, test: {}'.format(len(dataset), len(train_dataset), len(test_dataset)))
    
    def loss_fn_vae(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, n_grid_3), x.view(-1, n_grid_3), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    modenet = ModeNet(
        encoder_layer_sizes=[n_grid_3, 256, 256],
        latent_size=vae_latent_size,
        decoder_layer_sizes=[256, 256, n_grid_3],
        fc_layer_sizes=layer_size,
        batch_size=args.batch_size).to(device)

    optimizer = torch.optim.Adam(modenet.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    logs = defaultdict(list)
    # clear log
    with open(log_file_name, 'w'):
        pass
    min_loss = 1e100
    for iteration, (occu, pos, mode) in enumerate(test_data_loader):
        test_pos, test_occu, test_mode = pos.to(device), occu.to(device), mode.to(device)
    for epoch in range(args.epochs):
        modenet.train()

        for iteration, (occu, pos, mode) in enumerate(train_data_loader):
            pos, occu, mode = pos.to(device), occu.to(device), mode.to(device)
            
            std = args.pose_noise
            eps = torch.randn(pos.shape) * std
            pos = pos + eps

            x = torch.cat([pos,occu], dim=1)
            y = mode
            # eps = torch.randn(7) * 0.005
            # x[5:5+7] += eps

            y_hat, x_voxel, recon_x, mean, log_var = modenet(x)

            loss_vae = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid_3
            loss_fc = torch.nn.functional.binary_cross_entropy(
                y_hat.view(-1,mode_num), y.view(-1,mode_num), reduction='mean') # sum -> mean, divison remove

            loss = loss_vae + loss_fc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(train_data_loader)-1:
                print("[Train] Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(train_data_loader)-1, loss.item()))

                y_hat_bin = y_hat > 0.15
                y_bin = y > 0.5

                train_accuracy = (y_bin == y_hat_bin).sum().item()
                
                with torch.no_grad():
                    x = torch.cat([test_pos,test_occu], dim=1)
                    y = test_mode
                    modenet.eval()
                    y_hat, x_voxel, recon_x, mean, log_var = modenet(x)

                    loss_vae_test = loss_fn_vae(recon_x, x_voxel, mean, log_var) / n_grid_3
                    loss_fc_test = torch.nn.functional.binary_cross_entropy(
                            y_hat.view(-1,mode_num), y.view(-1,mode_num), reduction='mean')

                    loss_test = loss_vae_test + loss_fc_test
                    
                    y_hat_bin = (y_hat > 0.15).type(torch.float16)
                    y_bin = (y > 0.5).type(torch.float16)

                    truth_positives = (y_bin == 1).sum().item() 
                    truth_negatives = (y_bin == 0).sum().item() 

                    confusion_vector = y_hat_bin / y_bin
                    
                    true_positives = torch.sum(confusion_vector == 1).item()
                    false_positives = torch.sum(confusion_vector == float('inf')).item()
                    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
                    false_negatives = torch.sum(confusion_vector == 0).item()

                    test_accuracy = (y_bin == y_hat_bin).sum().item()
                    accuracy = {}
                    accuracy['tp'] = true_positives / truth_positives
                    accuracy['fp'] = false_positives / truth_negatives
                    accuracy['tn'] = true_negatives / truth_negatives
                    accuracy['fn'] = false_negatives / truth_positives

                lv = loss_vae_test.item()
                lf = loss_fc_test.item()
                lt = loss_test.item()

                lv0 = loss_vae.item()
                lf0 = loss_fc.item()
                lt0 = loss.item()
                train_accuracy = float(train_accuracy)/pos.size(dim=0)/mode_num
                test_accuracy = float(test_accuracy)/test_pos.size(dim=0)/mode_num
                print("[Test] vae loss: {:.3f} fc loss: {:.3f} total loss: {:.3f}".format(lv,lf,lt))
                # print("[Test] Accuracy: Train: {} / Test: {}".format(train_accuracy,test_accuracy))
                print("[Test] Accuracy: Train: {} / Test: {}".format(train_accuracy, test_accuracy))
                    
                if lt < min_loss:
                    min_loss = loss.item()
                    
                    checkpoint_model_name = chkpt_dir + 'loss_{}_{}_checkpoint_{:02d}_{:04d}_{:.4f}_{}'.format(lt, model_name, epoch, iteration, vae_latent_size, args.seed) + '.pkl'
                    torch.save(modenet.state_dict(), checkpoint_model_name)

                if iteration == 0:
                    with open(log_file_name, 'a') as f:
                        f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(epoch, lv,lf,lt,test_accuracy,lv0,lf0,lt0,train_accuracy,accuracy['tp'],accuracy['fp'],accuracy['tn'],accuracy['fn']))
    torch.save(modenet.state_dict(), 'model/{}.pkl'.format(model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--vae_latent_size", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--mode_num", type=int, default=8)
    parser.add_argument("--log_file_name", type=str, default="box")
    parser.add_argument("--pose_noise", type=float, default=0.00)
    parser.add_argument("--dataset", type=str, default='box')

    args = parser.parse_args()
    main(args)