import os
import torch
import argparse
import numpy as np
from lgf.models import ModeNetBox

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_torch(data):
    voxel = data['voxel']
    pose = data['pose']
    size = data['size']
    mode = data['mode']

    torch_voxel = torch.from_numpy(voxel).float()
    torch_pose = torch.from_numpy(pose).float()
    torch_size = torch.from_numpy(size).float()
    torch_mode = torch.from_numpy(mode).float()
    
    return torch_voxel, torch_pose, torch_size, torch_mode

def get_dataset(scene_name, seed, data_len, n_grid, remove_zero=False):
    dir = os.path.join('datasets', scene_name)
    pose_path = os.path.join(dir, f'poses_{scene_name}_{data_len}_{seed}_{n_grid}.npy')
    voxel_path = os.path.join(dir, f'voxels_{scene_name}_{data_len}_{seed}_{n_grid}.npy')
    mode_path = os.path.join(dir, f'modes_{scene_name}_{data_len}_{seed}_{n_grid}.npy')
    size_path = os.path.join(dir, f'sizes_{scene_name}_{data_len}_{seed}_{n_grid}.npy')

    pose = np.load(pose_path)
    voxel = np.load(voxel_path)
    mode = np.load(mode_path)
    size = np.load(size_path)

    voxel_1 = voxel > 0.5 # use occlusion
    voxel_2 = voxel < 2.5 # use free space

    voxel = voxel_1 * voxel_2
    
    nonzero_cnt = 0
    zero_cnt = 0
    delete_rows = []
    for i in range(mode.shape[0]):
        if mode[i,:].sum() == 0:
            zero_cnt += 1
            if zero_cnt > nonzero_cnt:
                delete_rows.append(i)
        else:
            # pass
            nonzero_cnt += 1
    
    print(f'zero_cnt: {zero_cnt}')
    print(f'zero ratio: {zero_cnt / mode.shape[0]}')

    if remove_zero:
        mode = np.delete(mode, delete_rows, axis=0)
        pose = np.delete(pose, delete_rows, axis=0)
        voxel = np.delete(voxel, delete_rows, axis=0)
        size = np.delete(size, delete_rows, axis=0)

    print(f'mode shape: {mode.shape}')
    
    voxel = voxel.reshape(voxel.shape[0], -1)

    return pose, voxel, mode, size

def get_mutliple_datasets(scene_names, seeds, data_lens, n_grid=16, remove_zero=False):
    poses = np.empty((0,7))
    voxels = np.empty((0,n_grid**3))
    modes = np.empty((0,12))
    sizes = np.empty((0,3))
    for scene_name, seed, data_len in zip(scene_names, seeds, data_lens):
        pose, voxel, mode, size = get_dataset(scene_name, seed, data_len, n_grid, remove_zero)

        assert(voxel.shape[1] == n_grid**3)

        poses = np.concatenate((poses, pose), axis=0)
        voxels = np.concatenate((voxels, voxel), axis=0)
        modes = np.concatenate((modes, mode), axis=0)
        sizes = np.concatenate((sizes, size), axis=0)
    data = {'pose':poses, 'voxel':voxels, 'mode':modes, 'size':sizes}
    return data

class ModeNetLightning(pl.LightningModule):
    def __init__(self, n_grid=16, lr=1e-3, prediction_threshold=0.15, **kwargs):
        
        super().__init__()
        self.model = ModeNetBox(voxel_dim=n_grid**3, **kwargs)
        
        self.lr = lr
        self.prediction_threshold = prediction_threshold

    def forward(self, x):
        """
        x: input (x_dim), voxel(voxel_dim)
        """

        return self.model(x)
    
    def accuracy(self, y_hat, y, prediction_threshold=0.15):
        y_hat = y_hat > prediction_threshold
        return (y_hat == y).float().mean()
    
    def true_positive(self, y_hat, y, prediction_threshold=0.15):
        y_hat = y_hat > prediction_threshold
        return ((y_hat == y) & (y_hat == 1)).float().mean()
    
    def true_negative(self, y_hat, y, prediction_threshold=0.15):
        y_hat = y_hat > prediction_threshold
        return ((y_hat == y) & (y_hat == 0)).float().mean()
    
    def false_positive(self, y_hat, y, prediction_threshold=0.15):
        y_hat = y_hat > prediction_threshold
        return ((y_hat != y) & (y_hat == 1)).float().mean()
    
    def false_negative(self, y_hat, y, prediction_threshold=0.15):
        y_hat = y_hat > prediction_threshold
        return ((y_hat != y) & (y_hat == 0)).float().mean()
    
    def training_step(self, batch, batch_idx):
        voxel_mb, pose_mb, size_mb, y = batch
        
        input_vector = torch.cat((pose_mb, size_mb, voxel_mb), dim=1)
        y_hat, voxel, recon_voxel, mean, log_var = self.model(input_vector)
        losses = self.model.loss(voxel, recon_voxel, y, y_hat, mean, log_var, prefix='train_')
        
        self.log_dict(losses, prog_bar=False, add_dataloader_idx=False)

        return losses['train_loss']
    
    def validation_step(self, batch, batch_idx):
        voxel_mb, pose_mb, size_mb, y = batch
        
        input_vector = torch.cat((pose_mb, size_mb, voxel_mb), dim=1)
        y_hat, voxel, recon_voxel, mean, log_var = self.model(input_vector)

        losses = self.model.loss(voxel, recon_voxel, y, y_hat, mean, log_var, prefix='val_')

        if type(self.prediction_threshold) is list:
            for threshold in self.prediction_threshold:
                accuracy = self.accuracy(y_hat, y, prediction_threshold=threshold)
                self.log(f'val_accuracy_{threshold}', accuracy, prog_bar=False, add_dataloader_idx=False)

                tp = self.true_positive(y_hat, y, prediction_threshold=threshold)
                # self.log(f'val_true_positive_{threshold}', tp, prog_bar=False, add_dataloader_idx=False)

                tn = self.true_negative(y_hat, y, prediction_threshold=threshold)
                # self.log(f'val_true_negative_{threshold}', tn, prog_bar=False, add_dataloader_idx=False)

                fp = self.false_positive(y_hat, y, prediction_threshold=threshold)
                # self.log(f'val_false_positive_{threshold}', fp, prog_bar=False, add_dataloader_idx=False)

                fn = self.false_negative(y_hat, y, prediction_threshold=threshold)
                # self.log(f'val_false_negative_{threshold}', fn, prog_bar=False, add_dataloader_idx=False)

                precision = tp / (tp + fp)
                self.log(f'val_precision_{threshold}', precision, prog_bar=False, add_dataloader_idx=False)

                recall = tp / (tp + fn)
                self.log(f'val_recall_{threshold}', recall, prog_bar=False, add_dataloader_idx=False)

                precision_neg = tn / (tn + fn)
                self.log(f'val_precision_neg_{threshold}', precision_neg, prog_bar=False, add_dataloader_idx=False)

                recall_neg = tn / (tn + fp)
                self.log(f'val_recall_neg_{threshold}', recall_neg, prog_bar=False, add_dataloader_idx=False)

                f1 = 2 * (precision * recall) / (precision + recall)
                self.log(f'val_f1_{threshold}', f1, prog_bar=False, add_dataloader_idx=False)

                f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
                self.log(f'val_f1_neg_{threshold}', f1_neg, prog_bar=False, add_dataloader_idx=False)

        else:
            accuracy = self.accuracy(y_hat, y)
            self.log('val_accuracy', accuracy, prog_bar=True)

        self.log_dict(losses, prog_bar=False, add_dataloader_idx=False)

        return losses['val_loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        voxel_mb, pose_mb, size_mb, y = batch
        
        input_vector = torch.cat((pose_mb, size_mb, voxel_mb), dim=1)
        y_hat, voxel, recon_voxel, mean, log_var = self.model(input_vector)
        losses = self.model.loss(voxel, recon_voxel, y, y_hat, mean, log_var, prefix=f'test_results_{dataloader_idx}/')

        if type(self.prediction_threshold) is list:
                for threshold in self.prediction_threshold:
                    accuracy = self.accuracy(y_hat, y, prediction_threshold=threshold)
                    self.log(f'test_results_{dataloader_idx}/accuracy_{threshold}', accuracy, prog_bar=False, add_dataloader_idx=False)

                    tp = self.true_positive(y_hat, y, prediction_threshold=threshold)
                    # self.log(f'test_results_{dataloader_idx}/true_positive_{threshold}', tp, prog_bar=False, add_dataloader_idx=False)

                    tn = self.true_negative(y_hat, y, prediction_threshold=threshold)
                    # self.log(f'test_results_{dataloader_idx}/true_negative_{threshold}', tn, prog_bar=False, add_dataloader_idx=False)

                    fp = self.false_positive(y_hat, y, prediction_threshold=threshold)
                    # self.log(f'test_results_{dataloader_idx}/false_positive_{threshold}', fp, prog_bar=False, add_dataloader_idx=False)

                    fn = self.false_negative(y_hat, y, prediction_threshold=threshold)
                    # self.log(f'test_results_{dataloader_idx}/false_negative_{threshold}', fn, prog_bar=False, add_dataloader_idx=False)
                    
                    precision = tp / (tp + fp)
                    self.log(f'test_results_{dataloader_idx}/precision_{threshold}', precision, prog_bar=False, add_dataloader_idx=False)

                    recall = tp / (tp + fn)
                    self.log(f'test_results_{dataloader_idx}/recall_{threshold}', recall, prog_bar=False, add_dataloader_idx=False)

                    precision_neg = tn / (tn + fn)
                    self.log(f'test_results_{dataloader_idx}/precision_neg_{threshold}', precision_neg, prog_bar=False, add_dataloader_idx=False)

                    recall_neg = tn / (tn + fp)
                    self.log(f'test_results_{dataloader_idx}/recall_neg_{threshold}', recall_neg, prog_bar=False, add_dataloader_idx=False)

                    f1 = 2 * (precision * recall) / (precision + recall)
                    self.log(f'test_results_{dataloader_idx}/f1_{threshold}', f1, prog_bar=False, add_dataloader_idx=False)

                    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg)
                    self.log(f'test_results_{dataloader_idx}/f1_neg_{threshold}', f1_neg, prog_bar=False, add_dataloader_idx=False)
                    
        else:
            accuracy = self.accuracy(y_hat, y)
            self.log(f'test_results_{dataloader_idx}/accuracy_{dataloader_idx}', accuracy, prog_bar=True)

        self.log_dict(losses, prog_bar=True)

        return losses[f'test_results_{dataloader_idx}/loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer
    
def main(args):
    train_scenes = ['table_box', 'table_side', 'table_middle']
    train_seeds = ['1107', '1107', '1107']
    train_data_lens = ['20000','20000', '20000']

    train_data = get_mutliple_datasets(train_scenes, train_seeds, train_data_lens, n_grid=args.n_grid, remove_zero=False)

    test_data = []
    
    def add_test_data(test_scene, test_seed, test_data_len):
        data = get_mutliple_datasets([test_scene], [test_seed], [test_data_len], n_grid=args.n_grid, remove_zero=False)
        return data
    
    test_data.append(add_test_data('table_bottom', '2000', '200'))
    test_data.append(add_test_data('table_ingolf_bottom', '2000', '200'))
    test_data.append(add_test_data('table_ivar_side', '2000', '200'))

    print('test_data', len(test_data))

    dataset = TensorDataset(*convert_to_torch(train_data))
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_loaders = []
    for test_dataset in test_data:
        test_dataset = TensorDataset(*convert_to_torch(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test_loaders.append(test_loader)

    print('test_loaders', len(test_loaders))

    model = ModeNetLightning(n_grid=args.n_grid, 
                             lr=args.learning_rate,
                             voxel_latent_dim=args.voxel_latent_dim,
                             mode_hidden_dims=args.mode_hidden_dims,
                             encoder_hidden_dims=args.encoder_hidden_dims,
                             decoder_hidden_dims=args.decoder_hidden_dims,
                             prediction_threshold=args.prediction_threshold,
                             dropout=args.dropout,
                             beta=args.beta,
                             bce_loss_weight=[args.bce_loss_weight, 1.0])
    
    ckpt_path = os.path.join('wandb','checkpoints')
    idx = 0
    while True:
        if args.run_name == '':
            run_name = f'modenet_box_{len(train_data["pose"])}_beta_{args.beta}_{idx}'
        else:   
            run_name = args.run_name + f'_{idx}'

        if not os.path.exists(os.path.join(ckpt_path,run_name)):
            break
        idx += 1
        print(f'run_name : {run_name}')
    ckpt_full_path = os.path.join(ckpt_path, run_name)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_full_path,
                                            filename='{epoch}-{val_loss:.2f}',
                                            save_top_k=2,
                                            save_last=True,
                                            monitor='val_loss',
                                            mode='min')

    logger = WandbLogger(name=run_name,project='modenet_box')
    
    logger.experiment.config.update(args)
    logger.experiment.config.update({'n_train_data': len(train_data['pose'])})
    trainer = pl.Trainer(max_epochs=args.epochs,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    trainer.test(dataloaders=test_loaders)

    # save best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = ModeNetLightning.load_from_checkpoint(best_model_path, n_grid=args.n_grid, 
                             lr=args.learning_rate,
                             voxel_latent_dim=args.voxel_latent_dim,
                             mode_hidden_dims=args.mode_hidden_dims,
                             encoder_hidden_dims=args.encoder_hidden_dims,
                             decoder_hidden_dims=args.decoder_hidden_dims,
                             prediction_threshold=args.prediction_threshold,
                             dropout=args.dropout,
                             beta=args.beta)
    torch.save(best_model.model.state_dict(), os.path.join(ckpt_full_path, best_model_path.split('/')[-1].split('.')[0] + '.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0025)
    parser.add_argument("--voxel_latent_dim", type=int, default=4)
    parser.add_argument("--mode_hidden_dims", type=list, default=[512, 512, 512])
    parser.add_argument("--encoder_hidden_dims", type=list, default=[128, 128])
    parser.add_argument("--decoder_hidden_dims", type=list, default=[128, 128])
    parser.add_argument("--prediction_threshold", type=list, default=[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--n_grid", type=int, default=10)
    parser.add_argument("--run_name", type=str, default='')
    parser.add_argument('--dropout', type=float, default=0.12)
    parser.add_argument('--bce_loss_weight', type=float, default=2.0)
    # parser.add_argument("--print_every", type=int, default=100)
    # parser.add_argument("--mode_num", type=int, default=8)
    # parser.add_argument("--log_file_name", type=str, default="box")
    # parser.add_argument("--pose_noise", type=float, default=0.00)
    # parser.add_argument("--dataset", type=str, default='box')

    args = parser.parse_args()
    main(args)