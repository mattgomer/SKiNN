# Pytorch modules
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import torch

# Pytorch-Lightning
from pytorch_lightning import LightningModule

import pdb
from torch.utils.cpp_extension import load
from SKiNN.src.model_imp import *


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # create a sequence of transpose + optional batch norm layers
    layers = []
    transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                              kernel_size, stride, padding, bias=False)
    # append transpose convolutional layer
    layers.append(transpose_conv_layer)
    
    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

class Generator(LightningModule):

    def __init__(self,z_size, dec_type='t_conv', conv_dim=32,size_l=5,lr=1e-4,out_channels=5):
        '''method used to define our model parameters'''
        super().__init__()

        self.size_l = size_l
        self.conv_dim = conv_dim
        self.out_channels= out_channels
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        
        # first, fully-connected layer
        self.fc = nn.Linear(z_size, conv_dim*self.size_l*self.size_l)

        if dec_type=='t_conv':
            self.cnn_block= nn.Sequential(
            deconv(conv_dim, int(conv_dim/2), 2),
            nn.ReLU(),
            deconv(int(conv_dim/2), int(conv_dim/4), 2),
            nn.ReLU(),
            deconv(int(conv_dim/4), int(conv_dim/8), 2),
            nn.ReLU(),
            deconv(int(conv_dim/8), int(conv_dim/16), 2),
            nn.ReLU(),
            deconv(int(conv_dim/16), self.out_channels,4, batch_norm=False),
            )
        elif dec_type=='upsampling':
            self.cnn_block= nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Upsample(scale_factor=2.5, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Upsample(scale_factor=2.5, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Upsample(scale_factor=2.162, mode='bilinear'),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Conv2d(512, self.out_channels, 3, padding=1),
            #nn.ReLU(),
            #nn.Conv2d(1024, 2048, 3, padding=1),
            #nn.Conv2d(2048, 5, 3, padding=1),
            )
        else:
            raise NotImplementedError

        # optimizer parameters
        self.lr = lr

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        # fully-connected + reshape 
        out = self.fc(x)
        out = out.view(-1, self.conv_dim, self.size_l, self.size_l) 
    
        # hidden transpose conv layers + relu 
        out = self.cnn_block(out)
        return out

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log training loss
        self.log('train_loss', loss,on_step=False,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss,on_step=False,on_epoch=True,sync_dist=True)

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log test loss
        self.log('test_loss', loss,sync_dist=True)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Generator_imp(LightningModule):

    def __init__(self):
        '''method used to define our model parameters'''
        super().__init__()
        self.fc = nn.Linear(9, 512)
        self.generator = CIPSskip(style_dim=512)
        self.lr = 1e-3
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        # fully-connected + reshape 
        x = self.fc(x)
        #aa = convert_to_coord_format(x.shape[0], 256, 256, integer_values=False).to(self.device)
        aa = convert_to_coord_format(x.shape[0], 276, 276, integer_values=False).to(self.device)
        out,_ = self.generator(aa,[x])
        return out

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        #y = y[:,147:-148,147:-148]
        y = y[:,:276,:276]
        #aa = convert_to_coord_format(1, 256, 256, integer_values=False).to(self.device)
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log training loss
        self.log('train_loss', loss,on_step=False,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        #y = y[:,147:-148,147:-148]
        y = y[:,:276,:276]
        #aa = convert_to_coord_format(1, 256, 256, integer_values=False).to(self.device)
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss,on_step=False,on_epoch=True,sync_dist=True)

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        #y = y[:,147:-148,147:-148]
        y = y[:,:276,:276]
        #aa = convert_to_coord_format(1, 256, 256, integer_values=False).to(self.device)
        out = self(x)
        loss = self.loss_func(out.squeeze(), y.squeeze())

        # Log test loss
        self.log('test_loss', loss,sync_dist=True)
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



if __name__ == "__main__":
    generator = Generator_imp().cuda()
    sample_z = torch.randn(1, 512)
    out ,_ = generator([sample_z.cuda()])
    breakpoint()
