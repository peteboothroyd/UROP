import caffe
import numpy as np
import skimage.io as imio
import re
import IO
from upconvolveinformation import UpConvolve

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

        self.output_path = "./output/deploy_output/"
        self.info_file_path = self.output_path + "info.txt"

        print("Set up DeployOutputLayer: ")

    def reshape(self, bottom, top):
        # Difference is shape of inputs.
        if (bottom[0].count != bottom[1].count):
            raise Exception("Inputs must have the same dimension.")

        self.cerr = np.zeros_like(bottom[0].data)

        # Loss outputs are scalar.
        top[0].reshape(1)  # Classification error

    def forward(self, bottom, top):
        if self.test_sample == 0:
            self.read_from_info_file()
            self.combined_output = np.zeros((self.image_dim[1], self.image_dim[0]))
            self.combined_label = np.zeros((self.image_dim[1], self.image_dim[0]))

        current_im_num = int(self.test_sample / self.num_partitions_per_image)
        partition_num = self.test_sample - current_im_num * self.num_partitions_per_image
        print("Image number = " + str(current_im_num) + ". Current partition number = " + str(partition_num))

        if current_im_num > self.im_num: #Have just transitioned to be looking at the next overall image, save
            combined_output_path = self.output_path + "{0}_combined_output.png".format(self.im_num)
            combined_label_path = self.output_path + "{0}_combined_label.png".format(self.im_num)

            output_to_save = np.zeros((self.image_dim[1], self.image_dim[0], 3), dtype=np.float32)
            print(output_to_save.shape)
            output_to_save[:,:,0] = self.combined_output
            output_to_save[:,:,1] = self.combined_output
            output_to_save[:,:,2] = self.combined_output
            imio.imsave(combined_output_path, self.combined_output)

            label_to_save = np.zeros((self.image_dim[1], self.image_dim[0], 3), dtype=np.float32)
            print(label_to_save.shape)
            label_to_save[:,:,0] = self.combined_label
            label_to_save[:,:,1] = self.combined_label
            label_to_save[:,:,2] = self.combined_label
            imio.imsave(combined_label_path, label_to_save)

            cerr = np.sum(((self.combined_output > self.thresh) != (self.combined_label > self.thresh)))
            print("Cerr for merged im " + str(self.im_num) + " = " + str(cerr))

            self.im_num = current_im_num
            self.combined_output = np.zeros((self.image_dim[1], self.image_dim[0]))
            self.combined_label = np.zeros((self.image_dim[1], self.image_dim[0]))

        #TODO: Delete this
        assert current_im_num <= 0, "1 test image processed..."

        prob = self.sigmoid(bottom[0].data)
        label = bottom[1].data

        im_coords = self.im_coords[partition_num]
        coord_pattern = r"x(?P<x_offset>\d+)_y(?P<y_offset>\d+)"
        r = re.findall(coord_pattern, im_coords)
        x_off, y_off = int(r[0][0]), int(r[0][1])
        #print("prob dim = " + str(prob.shape))
        #print("weights dim = " + str(self.weights.shape))
        #print("label dim = " + str(label.shape))

        prob_to_save = np.zeros((self.height, self.width, 3), dtype=np.float32)
        prob_to_save[:,:,0] = prob
        prob_to_save[:,:,1] = prob
        prob_to_save[:,:,2] = prob
        imio.imsave(self.output_path + "{0}_output_x{1}_y{2}.png".format(current_im_num, x_off, y_off), prob_to_save)

        for x in range(self.width):
            for y in range(self.height):
                output_label_val = self.element_function(self.weights[y][x]) * prob[0][0][y][x]
                #print("output label val = " + str(output_label_val))
                #print("normalising array[y + y_off][x + x_off] = " + str(self.normalising_array[y + y_off][x + x_off]))
                output_label_normalised = output_label_val / self.normalising_array[y + y_off][x + x_off]
                self.combined_output[y + y_off][x + x_off] += output_label_normalised
                self.combined_label[y + y_off][x + x_off] = label[0][0][y][x]

        # Classification error.
        self.cerr[...] = ((prob > self.thresh) != (label > self.thresh))

        # Classification error.
        cerr = np.sum(self.cerr)
        top[0].data[...] = cerr

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

    def read_from_info_file(self):
        parsed_info = IO.read_info_file(self.info_file_path)

        self.num_partitions_per_image, self.image_dim, self.im_coords = parsed_info[0], parsed_info[1], parsed_info[2]
        self.stride, self.kernel_size, self.num_conv_levels = parsed_info[3], parsed_info[4], parsed_info[5]
        self.opened_info_file = True
        upConvolve = UpConvolve(self.stride, self.kernel_size, [self.width, self.height], self.image_dim, self.num_conv_levels)
        self.weights = upConvolve.generate_weights()
        upConvolve.write_to_im(self.weights, "weights.png")
        self.normalising_array = self.calc_normalising_array(self.weights)
        upConvolve.write_to_im(self.normalising_array, "normalising_array.png")

    def calc_normalising_array(self, weights):
        """
        Calculate the normalising value for each pixel in the array

        Args:
            weights (numpy.array):  A 2D numpy array denoting the weights according to the number of source pixels the array is generated from

        Returns:
            normalising_weights (numpy.array): A 2D numpy array whose entries denote the normalising value of each output pixel

        """
        normalising_weights = np.zeros((self.image_dim[1], self.image_dim[0]))
        #print(normalising_weights.shape)

        for i in range(len(self.im_coords)):
            image_pattern = r"x(?P<x_offset>\d+)_y(?P<y_offset>\d+)"
            r = re.findall(image_pattern, self.im_coords[i])
            #print("normalising r = " + str(r[0]))
            x_off, y_off = int(r[0][0]), int(r[0][1])
            #Need to add numpy array to a smaller numpy array, with an offset given by the x_of and y_off. Note we want to apply some
            #function to each pixel, we do no necessarily think that this relationship is linear
            for x in range(self.width):
                for y in range(self.height):
                    normalising_weights[y + y_off][x + x_off] += self.element_function(weights[y][x])

        #print("normalising array = " + str(normalising_weights))
        return normalising_weights

    def element_function(self, val):
        return val
