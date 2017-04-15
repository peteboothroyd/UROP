import caffe
import numpy as np
import skimage.io as imio
import ast
import re

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
        prob = self.sigmoid(bottom[0].data)
        label = bottom[1].data

        # Classification error.
        self.cerr[...] = ((prob > self.thresh) != (label > self.thresh))

        # Classification error.
        top[0].data[...] = np.sum(self.cerr)

        if not self.opened_info_file:
            self.read_info_file()

        to_save = np.zeros((self.height, self.width, 3), dtype=np.float32)

        to_save[:, :, 0] = prob
        to_save[:, :, 1] = prob
        to_save[:, :, 2] = prob

        current_im_num = int(self.test_sample / self.num_partitions_per_image)
        partition_num = self.test_sample - current_im_num * self.num_partitions_per_image
        print("Image number = " + str(current_im_num) + ". Current partition number = " + str(partition_num))

        im_path = self.im_files[current_im_num][partition_num]
        image_pattern = r"./output/deploy_output/(?P<image_num>\d+)_image_x(?P<x_offset>\d+)_y(?P<x_offset>\d+).png"
        r = re.findall(image_pattern, im_path)
        image_num, x_off, y_off = r[0][0], r[0][1], r[0][2]

        label_path = self.output_path + "{0}_output_x{1}_y{2}.png".format(image_num, x_off, y_off)
        imio.imsave(label_path, to_save)
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

    def read_info_file(self):
        try:
            #Parsing info file for useful data.
            info_file = open(self.info_file_path, "r")

            num_partitions_line_raw = info_file.readline()
            num_partitions_pattern = r"Number of partitions per image = (?P<num_partitions>\d+)"
            r = re.findall(num_partitions_pattern, num_partitions_line_raw)
            self.num_partitions_per_image = int(r[0])

            image_size_line_raw = info_file.readline()
            image_size_pattern = r"Image size = \[(?P<im_size_x>\d+), (?P<im_size_y>\d+)\]"
            r = re.findall(image_size_pattern, image_size_line_raw)
            self.image_dim = [int(r[0][0]), int(r[0][1])]

            self.im_files = [f.split() for f in info_file.readlines()]
            info_file.close()

            print("number of partitions per image = " + str(self.num_partitions_per_image) + str(type(self.num_partitions_per_image)))

            self.opened_info_file = True

        except IOError:
            print("Could not open info file. Check that it has been created at the path= " + self.info_file_path)
