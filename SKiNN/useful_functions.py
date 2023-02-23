import os
import numpy as np
import joblib

def get_module_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'src'))   

def get_scalers_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'scalers'))

def get_weights_path():
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'weights/upsampling_cosmo_gen_norm_1channel_new-epoch=1109-valid_loss=0.00.ckpt'
))

def get_scaling_y():
    return np.load(os.path.join(get_scalers_path(),'scaler_y.npy'))

def get_scaling_x():
    return joblib.load(os.path.join(get_scalers_path(),'scaler_x'))

def mirror_output(prediction_image):
    """ 
    Because the prediction image is only one quarter of the whole image, this function mirrors it to produce the full image
    """
    pred_2d=np.squeeze(prediction_image)
    vert_mirror=pred_2d[::-1,:]
    tall=np.concatenate((pred_2d[:-1,:],vert_mirror),axis=0)
    horz_mirror=tall[:,::-1]
    return np.concatenate((tall[:,:-1],horz_mirror),axis=1)
