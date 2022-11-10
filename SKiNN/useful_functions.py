import os
import numpy as np
import joblib

def get_module_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'src'))   

def get_scalers_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'scalers'))

def get_weights_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'weights/upsampling_cosmo_gen_norm_1channel_new-epoch=1014-valid_loss=0.00.ckpt'))

def get_scaling_y():
    return np.load(os.path.join(get_scalers_path(),'scaler_y.npy'))

def get_scaling_x():
    return joblib.load(os.path.join(get_scalers_path(),'scaler_x'))