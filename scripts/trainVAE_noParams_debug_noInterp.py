import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import sys
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch import tensor as tt
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchvision.transforms as tforms
from torchvision.transforms import v2
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Timer
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import math
import h5py
import time
import pickle
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
#add to path 
sys.path.insert(0,'/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/ssl-legacysurvey/')
from ssl_legacysurvey.data_loaders import datamodules
from ssl_legacysurvey.utils import format_logger

class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights.unsqueeze(1)
        loss = torch.mean(weighted_squared_errors)
        return loss

# Adding additional terms? Try to re-derive the KL divergence from
# https://ai.stackexchange.com/questions/26366/how-is-this-pytorch-expression-equivalent-to-the-kl-divergence
def ELBO(x_hat, x, mu, logvar, y, beta0, beta1, beta2, device):
    y = y.squeeze() #remove mysterious middle dimension
    x = x
    x_hat = x_hat
    mu = mu
    

    #MSE loss between the image x and the reconstruction x_hat
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)

    mu_obj = torch.zeros([mu.shape[0], mu.shape[1]], dtype=torch.float32,device=device)
    mu_err = torch.zeros([mu.shape[0], mu.shape[1]], dtype=torch.float32,device=device)

    #set the means for our KL-divergence
    #mu_obj[:, 0] = y[:, 0] #redshift
    #mu_obj[:, 1] = y[:, 1] #logmass
    #mu_obj[:, 2] = y[:, 2] #log(SFR)

    #mu_err[:, 0] = y[:, 3]#uncertainty for redshift; placeholder b/c spectroscopic redshift
    #mu_err[:, 1] = y[:, 4] #uncertainty for logmass
    #mu_err[:, 2] = y[:, 5] #uncertainty for SFR
    
    logvar = logvar
    logvar_obj = torch.zeros([logvar.shape[0], logvar.shape[1]], dtype=torch.float32,device=device)

    #KL-divergence between a gaussian and the distribution of latent parameters
    KLD1 = -0.5 * torch.sum(1 + logvar - logvar_obj -  torch.div(torch.subtract(mu, mu_obj).pow(2), logvar_obj.exp()) - torch.div(logvar.exp(), logvar_obj.exp()))
    return beta0*MSE + beta1*KLD1, MSE, KLD1 #+ beta2*MSE_params, MSE, MSE_params


# define the LightningModule
class LitCVAE(pl.LightningModule):
    def __init__(self, d, learning_rate, beta0, beta1, beta2):
        super().__init__()
        self.d = d
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.learning_rate = learning_rate

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1),    # Output: 32x35x35
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, padding=1),   # Output: 64x18x18
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, padding=1),  # Output: 128x9x9
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.enc4 = nn.Sequential(
            nn.Flatten(),                         # Flatten the output
            nn.Linear(128*9*9, d*2)               # Map to 2*d to get mu and logvar for 5 latent dims
        )


        # Decoder
        self.dec1 = nn.Sequential(
            nn.Linear(d, 128*9*9),                # Map from latent space to spatial representation
            nn.Unflatten(1, (128, 9, 9)),         # Unflatten to 3D tensor for convolutional layers
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),  # Output: 64x18x18
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),   # Output: 32x36x36
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, padding=1),                      # Output: 3x69x69
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn(mu.shape, device=self.device)*torch.exp(0.5*logvar)
        else:
            return mu

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x.view(-1, 2, self.d)

    def decode(self, z):
        x_hat = self.dec1(z)
        x_hat = self.dec2(x_hat)
        x_hat = self.dec3(x_hat)
        x_hat = self.dec4(x_hat)
        return x_hat

    def forward(self, x):
        mu_logvar = self.encode(x)

        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]

        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)

        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu, logvar = self.forward(x)
        loss, MSE, KLD = ELBO(x_hat, x, mu, logvar, y, self.beta0[self.current_epoch], self.beta1[self.current_epoch], self.beta2[self.current_epoch], self.device)

        # log sampled images
        sample_imgs = x_hat[:2]
        sample_imgs_true = x[:2]

        grid = torchvision.utils.make_grid(sample_imgs)
        grid_true = torchvision.utils.make_grid(sample_imgs_true)


        if self.current_epoch%5 == 0:
            self.logger.experiment.add_image('generated_images_train', grid, self.current_epoch)
            self.logger.experiment.add_image('true_images_train', grid_true, self.current_epoch)
        
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)
        self.log("train_image_mse", MSE, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)
        self.log("train_kld", KLD, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, mu, logvar = self.forward(x)
        loss, MSE, KLD = ELBO(x_hat, x, mu, logvar, y, self.beta0[self.current_epoch], self.beta1[self.current_epoch], self.beta2[self.current_epoch], self.device)#, self.beta0, self.beta1, self.beta2)

        # log sampled images
        sample_imgs = x_hat[:2]
        sample_imgs_true = x[:2]

        grid = torchvision.utils.make_grid(sample_imgs)
        grid_true = torchvision.utils.make_grid(sample_imgs_true)

        if self.current_epoch%5 == 0:
            self.logger.experiment.add_image('generated_images_val', grid, self.current_epoch)
            self.logger.experiment.add_image('true_images_val', grid_true, self.current_epoch)


        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)
        self.log("val_image_mse", MSE, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)
        self.log("val_kld", KLD, on_step=False, on_epoch=True,sync_dist=True,prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

def main():
    os.chdir("/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder")
    torch.set_float32_matmul_precision('medium')

    params = {}
    params['data_path'] = '/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/TrainSample/segmented/'
    params['val_data_path'] = '/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/TestSample/segmented/'
    params['augmentations'] = 'grrrccgnrg'
    params['val_augmentations'] = 'grccrg'
    #params['jitter_lim'] = 7
    params['max_epochs'] = 5
    params['batch_size'] = 256
    params['checkpoint_every_n_epochs'] = 5
    params['max_num_samples'] = None
    params['max_num_samples_val'] = None
    params['pin_memory'] = True
    params['num_workers'] = 64
    params['output_dir'] = '/n/holyscratch01/iaifi_lab/agagliano/galaxyAutoencoder/models/fullTrain_crop_cyclic_bigBatch_noParams_1epoch/' #decals_checkpoints/'
    params['ckpt_path'] = ''
    params['logfile_name'] = 'cvae_ssl_2M_train_lr5en5_crop_noParams.log'


    d = int(sys.argv[1])
    learning_rate = 5.e-5
    #params['beta0'] = 1. #10.
    #params['beta1'] = 1. #1.
    #params['beta2'] = 1. #50.
    params['beta0'] = np.ones(3000)
    params['beta1'] = np.concatenate([np.linspace(0, 1, 500), np.ones(500), np.linspace(0, 1, 500), np.ones(500), np.linspace(0, 1, 500),np.ones(500)])
    params['beta2'] = np.concatenate([np.ones(500)*.1, np.linspace(0, .1, 500), np.ones(500)*.1, np.linspace(0, .1, 500), np.ones(500)*.1])

    print("learning rate: %.1e"%learning_rate)#, file=f)
    print("number of epochs: %i"%params['max_epochs'])#, file=f)
    print("WeightMSE loss!")#, file=f)

    datamodule = datamodules.DecalsDataModule(params)
    datamodule.train_transforms = None #use default transforms
    datamodule.val_transforms = None #use default transforms

    # init the autoencoder
    if params['ckpt_path']:
        cvae = LitCVAE(d=d, beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'], learning_rate=learning_rate)
        checkpoint = torch.load(params['ckpt_path'])
        cvae.load_state_dict(checkpoint["state_dict"])
        print("Loading checkpoint")
    else:
        cvae = LitCVAE(d=d, beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'], learning_rate=learning_rate)
        print("Training new model")
    cvae = cvae.cuda()

    file_output_head = f"cyclic_noParams_bs{params['batch_size']}_b0_{params['beta0'][0]}_b1_{params['beta1'][0]}_b2_{params['beta2'][0]}"


    logger = format_logger.create_logger(
            filename=os.path.join(params['output_dir'], params['logfile_name']),
        )

    # Log various attributes during training
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=params['output_dir'],
        name=file_output_head,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=params['output_dir'],
        filename=file_output_head+'_{epoch:03d}',
        every_n_epochs=params['checkpoint_every_n_epochs'],
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_on_train_epoch_end=True,
        verbose=True,
        save_last=True,
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='epoch',
    )

    # stop training after 10 minutes
    timer = Timer(duration="00:00:10:00")

    trainer = pl.Trainer(
                max_epochs=params['max_epochs'], 
                devices=torch.cuda.device_count(),
                strategy=DDPStrategy(),
                gradient_clip_val=0.1,
                #profiler='advanced',
                profiler='pytorch',
                enable_model_summary=True,
                callbacks=[checkpoint_callback, lr_monitor, timer],
                logger=[tb_logger], accelerator='gpu')


    logger.info("Training Model")

    # Fit model
    trainer.fit(
        model=cvae,
        datamodule=datamodule,
    )

    results = trainer.validate(model=cvae, datamodule=datamodule)

    # query training/validation/test time (in seconds)
    print("Training Time: ",timer.time_elapsed("train"))
    print("Validation Time: ", timer.time_elapsed("validate"))

    with open('validatedResults_crop_fullTrain_cyclic_bigBatch_noParams_1epoch.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
