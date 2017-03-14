import caffe
import numpy as np
import skimage.io as imio

class OutputLayer(caffe.Layer):
    """
    Output layer for the deployed net, which will be used to explore overlapping partitions
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 1:
            raise Exception("Need one input (propagated).")

        # Params is a python dictionary with layer parameters.
        if not self.param_str:
            params = dict()
        else:
            params = eval(self.param_str)

        # Loss weight.
        self.loss_weight = params.get('loss_weight', 1.0)
        # Threshold for computing classification error
        self.thresh = 0.5
        self.width = params['width']
        self.height = params['height']
        self.test_sample = 0

    def reshape(self, bottom, top):
        # Difference is shape of inputs.
        pass

    def forward(self, bottom, top):
        prob  = self.sigmoid(bottom[0].data)

        try:
            to_save = np.zeros((self.height, self.width,3), dtype=np.float32)
        except:
            print(self.width + self.height)

        to_save[:, :, 0] = prob
        to_save[:, :, 1] = prob
        to_save[:, :, 2] = prob

        #TODO: This is not ideal. The output images are being saved in a specific format. We do not have access to the
        #relevant information in the OutputLayer to properly format at the moment
        imio.imsave("./output/test_samples/{0}_output.png".format(self.test_sample), to_save)
        self.test_sample += 1
