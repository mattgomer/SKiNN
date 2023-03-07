import SKiNN
import SKiNN.src.NN_models as NN_models
import sys
import numpy as np
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class Generator(object):
    def __init__(self):
        weights_path = SKiNN.useful_functions.get_weights_path()
        self.model = NN_models.Generator_imp()
        self.net = self.model.load_from_checkpoint(weights_path).cuda()
        self.scaling_y = SKiNN.useful_functions.get_scaling_y()
        self.scaling_x = SKiNN.useful_functions.get_scaling_x()

    def generate_map(self, input_p):
        """Generate velocity maps given input parameters.
        input_p: input parameters (before normalization scaling)
        Returns the velocity map(s)
        """
        self.input_p = self.scaling_x.transform(np.reshape(input_p, (-1, len(input_p))))
        self.input_p = torch.Tensor(self.input_p).cuda()
        self.net.eval()
        with torch.no_grad():
            self.pred = self.net(self.input_p).cpu().numpy()
        scaled_pred=self.pred[0,:,:].squeeze() * self.scaling_y
        mirrored_pred=self.mirror_output(scaled_pred)
        output=np.maximum(mirrored_pred,np.zeros_like(mirrored_pred))
        return output

    def mirror_output(self, prediction_image):
        """
        Because the prediction image is only one quarter of the whole image, this function mirrors it to produce the full image
        """
        pred_2d = np.squeeze(prediction_image)
        vert_mirror = pred_2d[::-1, :]
        tall = np.concatenate((pred_2d[:-1, :], vert_mirror), axis=0)
        horz_mirror = tall[:, ::-1]
        return np.concatenate((tall[:, :-1], horz_mirror), axis=1)