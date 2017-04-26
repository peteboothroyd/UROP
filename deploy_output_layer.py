import caffe
import numpy as np
import skimage.io as imio
import re
import IO

class DeployOutputLayer(caffe.Layer):
    """
    Output layer for the deployed net, which will be used to explore overlapping partitions
    """

    def setup(self, bottom, top):
        # Check inputs.
        if len(bottom) != 2:
            raise Exception("Need two inputs (propagated, label)")

        # Params is a python dictionary with layer parameters.
        if not self.param_str:
            params = dict()
        else:
            params = eval(self.param_str)

        # Loss weight
        self.loss_weight = params.get('loss_weight', 1.0)
        # Threshold for computing classification error
        self.thresh = 0.5
        self.width = params['width']
        self.height = params['height']
        self.test_sample = 0

        self.im_num = 0
        self.cumulative_im_cerr = 0

        self.output_path = "./output/deploy_output/"
        self.info_file_path = self.output_path + "info.txt"

        self.opened_info_file = False
        print("Set up DeployOutputLayer: ")

    def reshape(self, bottom, top):
        # Difference is shape of inputs.
        if (bottom[0].count != bottom[1].count):
            raise Exception("Inputs must have the same dimension.")

        self.cerr = np.zeros_like(bottom[0].data)

        # Loss outputs are scalar.
        top[0].reshape(1)  # Classification error

    def forward(self, bottom, top):
        if not self.opened_info_file:
            info = IO.read_info_file(self.info_file_path)

            self.num_partitions_per_image, self.image_dim ,self.im_coords = info[0], info[1], info[2]
            self.opened_info_file = True

        current_im_num = int(self.test_sample / self.num_partitions_per_image)
        partition_num = self.test_sample - current_im_num * self.num_partitions_per_image
        print("Image number = " + str(current_im_num) + ". Current partition number = " + str(partition_num))

        if current_im_num > self.im_num: #Have just transitioned to be looking at the next overall image, save
            # old cumulative error and reset cumulative error to 0
            print("Cerr for im " + str(current_im_num) + " = " + str(self.cumulative_im_cerr))
            self.cumulative_im_cerr = 0
            self.im_num = current_im_num

        prob = self.sigmoid(bottom[0].data)
        label = bottom[1].data

        # Classification error.
        self.cerr[...] = ((prob > self.thresh) != (label > self.thresh))

        # Classification error.
        cerr = np.sum(self.cerr)
        top[0].data[...] = cerr
        self.cumulative_im_cerr += cerr

        to_save = np.zeros((self.height, self.width, 3), dtype=np.float32)

        to_save[:, :, 0] = prob
        to_save[:, :, 1] = prob
        to_save[:, :, 2] = prob

        im_path = self.im_files[current_im_num][partition_num]
        image_pattern = r"x(?P<x_offset>\d+)_y(?P<y_offset>\d+).png"
        r = re.findall(image_pattern, im_path)
        x_off, y_off = r[0][0], r[0][1]

        #label_path = self.output_path + "{0}_output_x{1}_y{2}.png".format(image_num, x_off, y_off)
        #imio.imsave(label_path, to_save)

        self.test_sample += 1

    def sigmoid(self, x):
        """Numerically-stable sigmoid function."""
        ret = np.zeros_like(x)
        idx = x >= 0
        z = np.exp(-x[idx])
        ret[idx] = 1 / (1 + z)
        idx = x < 0
        z = np.exp(x[idx])
        ret[idx] = z / (1 + z)
        return ret
