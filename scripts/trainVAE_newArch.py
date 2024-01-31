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
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import tensor as tt
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchvision.transforms as tforms
from torchvision.transforms import v2
import math
import time

ts = time.time()
os.chdir("/n/home02/agagliano/galaxyAutoencoder")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ZouGam = np.load("./data/FullSample_26k_photoz_wOrientation_fixorder.npz")
images_rescaled = ZouGam['x'].astype(np.float32)
labels = ZouGam['y'].astype(np.float32)

labels[:, 6] = 0.1

d = int(sys.argv[1])
nepochs = 2000
learning_rate = 5.e-6
#beta0 = np.concatenate([np.linspace(1., 50., 500), [50]*1500])
beta0 = [10.]*2000
beta1 = [1.]*2000
beta2 = [50]*2000
#beta1 = np.concatenate([np.linspace(0, 10., 4000), [10]*1000])
#beta2 = np.concatenate([np.linspace(0, 50., 4000), [50]*1000])

f = open("TrainingOutput_newArch_d%i_%i.txt"%(d, int(ts)), 'w')
print("learning rate: %.1e"%learning_rate, file=f)
print("number of epochs: %i"%nepochs, file=f)
print("using device: %s"% device, file=f)
print("WeightMSE loss!", file=f)
#print("LR scheduler: Exponential, gamma=%.2f"%gamma, file=f)
print("No scheduler used.", file=f)
transform = tforms.ToTensor()

class GalaxyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# random split
N = len(images_rescaled)

np.random.seed(12345)
indices = np.random.permutation(N)
train_idxs = indices[:int(0.8*N)]
valid_idxs = indices[int(0.8*N):]

# transforms and train/valid split
#now, labels should be the sets [redshift, log(stellar mass), log(SFR)]
#transform from 128x128 to 69x69 to work within the pre-existing framework and run quickly
images_train = [F.interpolate(transform(x).unsqueeze(0), size=(100, 100), mode='bicubic').squeeze() for x in images_rescaled[train_idxs]]
images_valid = [F.interpolate(transform(x).unsqueeze(0), size=(100, 100), mode='bicubic').squeeze() for x in images_rescaled[valid_idxs]]

train_ds = GalaxyDataset(images_train, labels[train_idxs], None)
valid_ds = GalaxyDataset(images_valid, labels[valid_idxs], None)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=128,
    num_workers=0,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

def conv2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape

def convtrans2D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv2D
    outshape = ((img_size[0] - 1) * stride[0] - 2 * padding[0] + kernel_size[0],
                (img_size[1] - 1) * stride[1] - 2 * padding[1] + kernel_size[1])
    return outshape

## ---------------------- ResNet VAE ---------------------- ##

class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(100, 100), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst, mu, logvar

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = d     # latent dim extracted by 2D CNN
res_size = 100        # ResNet image size
dropout_p = 0.2       # dropout probability

model = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

def display_images(in_, out, nrows=1, ncols=4, label=None, count=False):
    out_pic = np.array(out.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
    plotCounter = 1
    for row in range(nrows):
        if in_ is not None:
            in_pic = np.array(in_.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1))
            plt.figure(figsize=(18, 4))
            plt.suptitle(label, color='w', fontsize=16)
            for col in range(ncols):
                plt.subplot(1, ncols, plotCounter)
                plt.imshow(in_pic[col + ncols * row])
                plt.axis('off')
                plt.title("Real Image")

                plt.subplot(2, ncols, plotCounter+1)
                plt.imshow(out_pic[col + ncols * row])
                plt.axis('off')
                plt.title("Reconstruction")
                if count: plt.title(str(col + ncols * row), color='w')
                plotCounter += 1

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

#scheduler = ExponentialLR(optimizer, gamma=gamma)


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
def ELBO(x_hat, x, mu, logvar, y, beta0, beta1, beta2):
    #MSE loss between the image x and the reconstruction x_hat
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x)

    mu_obj = torch.zeros([mu.shape[0], mu.shape[1]], dtype=torch.float32).to(device)
    mu_err = torch.zeros([mu.shape[0], mu.shape[1]], dtype=torch.float32).to(device)

    #set the means for our KL-divergence
    mu_obj[:, 0] = y[:, 0] #rotation angle
    mu_obj[:, 1] = y[:, 1] #redshift
    mu_obj[:, 2] = y[:, 2] #logmass
    mu_obj[:, 3] = y[:, 3] #log(SFR)

    mu_err[:, 0] = y[:, 4] #uncertainty for position angle
    mu_err[:, 1] = 1.e-3 #uncertainty for redshift; placeholder b/c spectroscopic redshift
    mu_err[:, 2] = y[:, 5] #uncertainty for logmass
    mu_err[:, 3] = y[:, 6] #uncertainty for SFR

    #decrease the variance for just the three parameters of interest
    logvar_obj = torch.zeros([logvar.shape[0], logvar.shape[1]], dtype=torch.float32).to(device)

    #KL-divergence between a gaussian and the distribution of latent parameters
    KLD1 = -0.5 * torch.sum(1 + logvar[:, 4:] - logvar_obj[:, 4:] -  torch.div(torch.subtract(mu[:, 4:], mu_obj[:, 4:]).pow(2), logvar_obj[:, 4:].exp()) - torch.div(logvar[:, 4:].exp(), logvar_obj[:, 4:].exp()))
    param_sigmas = mu_err[:, 0:4]
    param_sigmas[param_sigmas < 1.e-3] = 1.e-3
    weights = param_sigmas**(-2)
    MSE_params = WeightedMSELoss(weights)(mu[:, 0:4], mu_obj[:, 0:4])
    #MSE_params = torch.nn.MSELoss(reduction='sum')(mu[:, 0:4], mu_obj[:, 0:4])
    return beta0*MSE + beta1*KLD1 + beta2*MSE_params

validation_losses = []
training_losses = []

best_vloss = 100000.
savedEpoch = 0

for epoch in range(0, nepochs):
    # train
    print("Beginning training...")
    model.train()
    train_loss = 0
    print("beta0: %.1f"%beta0[epoch], file=f)
    print("beta1: %.1f"%beta1[epoch], file=f)
    print("beta2: %.1f"%beta2[epoch], file=f)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        x_hat, mu, logvar = model(x)
        loss = ELBO(x_hat, x, mu, logvar, y, beta0[epoch], beta1[epoch], beta2[epoch])
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #scheduler.step()

    # validate
    with torch.no_grad():
        model.eval()
        valid_loss = 0
        for x, y in valid_loader:
            x = x.to(device)
            x_hat, mu, logvar = model(x)
            valid_loss += ELBO(x_hat, x, mu, logvar, y, beta0[epoch], beta1[epoch], beta2[epoch])

    validation_losses.append(valid_loss / len(valid_loader.dataset))
    training_losses.append(train_loss / len(train_loader.dataset))

    print(f' Epoch {epoch} | Train: {train_loss / len(train_loader.dataset):.5f} | Valid: {validation_losses[-1]:.5f}', file=f)

    # display
    if epoch % 50 == 0:
        #print info so that we can see that it's training
        f.flush()

        #see how well it's learning rotation:
        fig, ax =plt.subplots(nrows=5, ncols=10, figsize=(50, 20))
        axs = ax.ravel()

        names = [r'$\phi$', 'z', 'log(M)', 'log(SFR)', 'Latent 1 (Morph.)']
        mu = mu.cpu()
        for j in np.arange(5):
            for i in np.arange(10):
                if j == 1:
                    paramMin = 0
                else:
                    paramMin = np.nanmin(mu[:, j])
                paramMax = np.nanmax(mu[:, j])
                ParamRange = np.linspace(paramMin, paramMax, 10)

                tempSet = np.nanmedian(mu, axis=0)
                tempSet[j] = ParamRange[i]
                x_reconstructed = model.decode(torch.FloatTensor(tempSet).to(device).unsqueeze(0))
                x_img = np.array(x_reconstructed.data.cpu() * 255, dtype=int).transpose((0, 2, 3, 1)).squeeze()
                ax[j, i].imshow(x_img)
                ax[j, i].axis('off')
                ax[j, i].set_title('%s = %.2f'%(names[j], ParamRange[i]))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.savefig("plots/LatentGrid_d%i_newArch_epoch%05d_10b0MSE.png"%(d, epoch),dpi=300, bbox_inches='tight')

    # Track best performance, and save the model's state
    avg_vloss = valid_loss / len(valid_loader.dataset)
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), './cVAE_d%i_newArch_10b0MSE_%i.h5' % (d, int(ts)))
        print("Saving model at epoch %i." %epoch, file=f)
        savedEpoch = epoch

print("Final model saved has val loss = %.2f at epoch %i" % (best_vloss, savedEpoch), file=f)

#check that the loss converged and that the entries for mu are near what we expect them to be near...
mu_list = []
y_list = []
with torch.no_grad():
    model.eval()

    for x, y in valid_loader:
        x = x.to(device)
        x_hat, mu, logvar = model(x)

        mu_list.append(mu)
        y_list.append(y)

mu_list = torch.cat(mu_list).cpu()
y_list = torch.cat(y_list).cpu()


# using the variable axs for multiple Axes
sns.set_context("talk")

try:
    validation_losses = [x.cpu() for x in validation_losses]
    training_losses = [x.cpu() for x in training_losses]
except:
    print("Losses already on cpu!")

#evaluate the losses
plt.figure(figsize=(10,7))
plt.plot(validation_losses, label='Validation Loss')
plt.plot(training_losses, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Batch-Wide Loss")
plt.legend()
plt.savefig("plots/Losses_d%i_newArch_10b0MSE.png"%d,dpi=200, bbox_inches='tight')

fig, axs = plt.subplots(1, 4, figsize=(15, 5))

axs = axs.ravel()

#compare with a random param which should have greater spread
axs[0].plot(y_list[:, 1], mu_list[:, 1], 'o', ms=2);
axs[0].plot([0, 1], [0, 1], ls='--', c='k')
axs[0].set_xlim((0, 1))
axs[0].set_ylim((0, 1))
axs[0].set_xlabel(r"True $z$")
axs[0].set_ylabel(r"Inferred $z$")

axs[1].errorbar(y_list[:, 2], mu_list[:, 2], xerr=y_list[:, 5], fmt='o', ms=2, lw=1, ecolor='gray');
axs[1].plot([8, 12], [8, 12], ls='--', c='k', zorder=100)
axs[1].set_xlim((8, 12))
axs[1].set_ylim((8, 12))
axs[1].set_xlabel(r"True log($M_{*}/M_{\odot}$)")
axs[1].set_ylabel(r"Inferred log($M_{*}/M_{\odot}$)")

axs[2].errorbar(y_list[:, 3], mu_list[:, 3], xerr=y_list[:, 6], fmt='o', ms=2, lw=1, ecolor='gray');
axs[2].plot([7, 11], [7, 11], ls='--', c='k', zorder=100)
axs[2].set_xlim((7, 11))
axs[2].set_ylim((7, 11))
axs[2].set_xlabel(r"True log(SFR/$M_{\odot}$ Gyr$^{-1}$)")
axs[2].set_ylabel(r"Inferred log(SFR/$M_{\odot}$ Gyr$^{-1}$)")
plt.tight_layout()

axs[3].errorbar(y_list[:, 0], mu_list[:, 0], xerr=y_list[:, 4], fmt='o', ms=2, lw=1, ecolor='gray');
axs[3].plot([-1.5, 1.5], [-1.5, 1.5], ls='--', c='k', zorder=100)
axs[3].set_xlim((-1.5, 1.5))
axs[3].set_ylim((-1.5, 1.5))
axs[3].set_xlabel(r"True $\phi$")
axs[3].set_ylabel(r"Inferred $\phi$")

plt.suptitle("VAE-inferred properties, %i latent features" % d, y=1.1)
plt.savefig("plots/VaeProps_vsTrue_d%i_newArch_10b0MSE.png" % d)

f.close()
