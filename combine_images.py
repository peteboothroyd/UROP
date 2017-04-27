import numpy as np
import re
import IO
import skimage.io as imio
import os

from upconvolveinformation import UpConvolve

class combine(object):
    #Want to be able to combine the labels from multiple different partitions into one overall classification
    def __init__(self):
        #TODO: (fixme, dirty)
        self.stride = [2, 2]
        self.kernel_size = [4, 4]
        self.num_conv_levels = 5

        info_file_path = "./output/deploy_output/info.txt"
        self.num_partitions_per_image, self.image_dim, self.output_im_files = IO.read_info_file(info_file_path)

        impath = self.output_im_files[0][0]
        partition_width, partition_height, _ = IO.find_partition_dim(impath)

        upConvolve = UpConvolve(self.stride, self.kernel_size, [partition_width, partition_height], self.image_dim, self.num_conv_levels)
        self.weights = upConvolve.generate_weights()
        self.normalising_array = self.calc_normalising_array(self.weights)

    def calc_normalising_array(self, weights):
        """
        Calculate the normalising value for each pixel in the array

        Returns:
            next_level_weights (numpy.array): A 2D numpy array whose entries denote the normalising value of each output pixel

        """
        image = np.zeros(self.image_dim)

        for i in range(len(self.output_im_files[0])):
            image_pattern = r"./output/deploy_output/(?P<imag_num>\d+)_image_x(?P<x_offset>\d+)_y(?P<x_offset>\d+).png"
            r = re.findall(image_pattern, self.output_im_files[0][i])
            _, x_off, y_off = int(r[0]), int(r[1]), int(r[2])
            #Need to add numpy array to a smaller numpy array, with an offset given by the x_of and y_off. Note we want to apply some
            #function to each pixel, we do no necessarily think that this relationship is linear
            for x in range(self.image_dim[0]):
                for y in range(self.image_dim[1]):
                    image[y + y_off][x + x_off] += self.element_function(weights[y][x])


    def calculate_merged_labels(self):
        #Looping over each image
        image_pattern = r"./output/deploy_output/(?P<imag_num>\d+)_image_x(?P<x_offset>\d+)_y(?P<x_offset>\d+).png"
        for i in range(len(self.output_im_files)):
            combined_output = np.zeros((self.image_dim[1], self.image_dim[0]))
            combined_label = np.zeros((self.image_dim[1], self.image_dim[0]))

            for j in range(len(self.output_im_files[i])):
                r = re.findall(image_pattern, self.output_im_files[i][j])
                im_num, x_off, y_off = int(r[0]), int(r[1]), int(r[2])

                output_path = "./output/deploy_output/{0}_output_x{1}_y{2}.png".format(im_num, x_off, y_off)
                output = imio.imread(output_path)

                label_path = "./output/deploy_output/{0}_label_x{1}_y{2}.png".format(im_num, x_off, y_off)
                label = imio.imread(label_path)

                #Combining output labels and reweighting them appropriately
                for x in range(self.image_dim[0]):
                    for y in range(self.image_dim[1]):
                        output_label_val = self.element_function(self.weights[y][x]) * output[y][x]
                        output_label_normalised = output_label_val / self.normalising_array[y + y_off][x + x_off]
                        combined_output[y + y_off][x + x_off] += output_label_normalised

                        combined_label[y + y_off][x + x_off] = label[y][x]

                combined_output_path = "./output/deploy_output/{0}_combined_output.png".format(im_num)
                combined_label_path = "./output/deploy_output/{0}_combined_label.png".format(im_num)

                imio.imsave(combined_output_path, combined_output)
                imio.imsave(combined_label_path, combined_label)

                cerr = np.sum(((combined_output > self.thresh) != (combined_label > self.thresh)))
                print("Cerr for im " + str(im_num) + " = " + str(cerr))

    def element_function(self, val):
        return val


if __name__=="__main__":
    print("Combining labels and outputs...")
    cb = combine()
    cb.calculate_merged_labels()
