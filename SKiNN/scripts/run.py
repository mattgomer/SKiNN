import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,3,4"

# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch-Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

# Externals
from data_modules import *
from models import *
from data_utils import *




def main():
    # setup data
    data = CosmoDataModule(batch_size=5,new_data=True)
    
    # setup model - choose different hyperparameters per experiment
    name_m ='upsampling'
    #model = Generator(z_size=11,dec_type=name_m,conv_dim=1,size_l=8,lr=1e-4,out_channels=1)
    model = Generator(z_size=9,dec_type=name_m,conv_dim=1,size_l=41,lr=1e-4,out_channels=1)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        dirpath='weights/train_test/',
        filename= name_m +'_cosmo_gen_norm_1channel_new-{epoch:02d}-{valid_loss:.2f}',
        mode='min',
    )


    trainer = Trainer(
        callbacks=[checkpoint_callback],
        logger=wandb_logger,    # W&B integration
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        gpus=4,                # use all GPU's
        max_epochs=15000,
        check_val_every_n_epoch=5,
        #resume_from_checkpoint='/local/home/lbiggio/comso_gen/weights/train_test/upsampling_cosmo_gen_norm_1channel-epoch=4924-valid_loss=0.00.ckpt'
        # precision=16            
        # number of epochs
        )

    trainer.fit(model, data)
    trainer.test(model, datamodule=data)



if __name__ == "__main__":
    wandb.login()
    wandb_logger = WandbLogger(project='Cosmo_gen_1ch')
    main()












